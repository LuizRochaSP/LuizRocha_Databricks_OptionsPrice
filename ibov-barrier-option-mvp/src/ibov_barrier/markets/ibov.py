"""Adaptador IBOV: o pipeline do notebook 01 (main) empacotado no contrato `MarketSnapshot`.

Critério de aceite (fase 4): para a mesma data e os mesmos arquivos B3, este módulo reproduz os números do
notebook 01 do `main` (contagens por filtro, vol implícita por ticker, r, q e vol do alvo, preço da barreira).
Por isso a lógica é cópia fiel do notebook, sem "melhorias": IN e PR lidos em streaming (última publicação de
cada ZIP), spot do IR, curvas DI1/IND do SPRD por `build_market_curves`, prazo T em dias corridos/365, r e q
pelo prazo em DU, mesmos filtros na mesma ordem e a mesma inversão de Black-Scholes (brentq em [1e-4, 5]).

As leituras de XML e as curvas importam `lxml` e `exchange_calendars` só quando chamadas.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from ..market_date import SAO_PAULO
from .base import CleaningReport, ForwardPoint, MarketSnapshot, MarketSpec, TermStructure
from .vol import VolSurface

IBOV_SPEC = MarketSpec(
    code="IBOV", name="Ibovespa", currency="BRL", exchange="B3", timezone=SAO_PAULO,
    close_time=time(18, 0), exercise="european", multiplier=1.0, day_count="BUS/252",
    rate_index="DI1", roots=("IBOV",),
)


@dataclass(frozen=True)
class IbovFilters:
    """Mesmos parâmetros e valores padrão do notebook 01."""
    min_trades: int = 1
    min_open_interest: int = 1
    min_iv: float = 0.01
    max_iv: float = 2.00


# ------------------------------------------------------------------ leitura dos XMLs (cópia do notebook 01)
def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _flatten(element) -> dict:
    row = {}
    for child in element.iter():
        value = (child.text or "").strip()
        if value:
            row.setdefault(_local_name(child.tag), value)
    return row


def _latest_member(archive: ZipFile):
    members = [x for x in archive.infolist() if x.filename.lower().endswith(".xml")]
    if not members:
        raise ValueError("O ZIP não contém XML.")
    return max(members, key=lambda x: x.date_time)


def _stream(zip_path: Path, tag: str, keep) -> tuple[pd.DataFrame, str]:
    from lxml import etree

    rows = []
    with ZipFile(zip_path) as archive:
        member = _latest_member(archive)
        with archive.open(member) as xml_file:
            for _, element in etree.iterparse(xml_file, events=("end",), tag=f"{{*}}{tag}", huge_tree=True):
                row = _flatten(element)
                if keep(row):
                    rows.append(row)
                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]
    return pd.DataFrame(rows), member.filename


def read_ibov_option_catalog(in_zip: Path) -> tuple[pd.DataFrame, str]:
    """Cadastro IN: opções sobre o IBOV com strike e vencimento."""
    catalog, member = _stream(in_zip, "Instrm", lambda r: (
        r.get("Asst") == "IBOV" and r.get("OptnTp") in {"CALL", "PUTT"} and r.get("ExrcPric")
        and r.get("XprtnDt")))
    catalog = catalog.rename(columns={"TckrSymb": "ticker", "ExrcPric": "strike", "XprtnDt": "maturity_date",
                                      "OptnTp": "option_type", "OptnStyle": "option_style"})
    catalog["strike"] = pd.to_numeric(catalog["strike"], errors="coerce")
    catalog["maturity_date"] = pd.to_datetime(catalog["maturity_date"], errors="coerce")
    catalog["option_type"] = catalog["option_type"].replace({"PUTT": "PUT"})
    catalog = catalog.dropna(subset=["ticker", "strike", "maturity_date"])
    return catalog.drop_duplicates("ticker", keep="last"), member


PRICE_COLUMNS = {"TckrSymb": "ticker", "FrstPric": "first_price", "MinPric": "min_price", "MaxPric": "max_price",
                 "TradAvrgPric": "average_price", "LastPric": "last_price", "BestBidPric": "best_bid",
                 "BestAskPric": "best_ask", "RglrTxsQty": "trades", "OpnIntrst": "open_interest"}


def read_ibov_prices(pr_zip: Path) -> tuple[pd.DataFrame, str]:
    """PriceReport: última publicação, tickers que começam com IBOV."""
    prices, member = _stream(pr_zip, "PricRpt", lambda r: r.get("TckrSymb", "").startswith("IBOV"))
    prices = prices.rename(columns=PRICE_COLUMNS)
    for column in list(PRICE_COLUMNS.values())[1:]:
        if column not in prices:
            prices[column] = np.nan
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    return prices.drop_duplicates("ticker", keep="last"), member


def read_ibov_spot(ir_zip: Path) -> float:
    """Fechamento do IBOV no IR (IndxInf com TckrSymb = IBOV)."""
    frame, _ = _stream(ir_zip, "IndxInf", lambda r: True)
    row = frame.loc[frame.get("TckrSymb", pd.Series(dtype=str)).eq("IBOV")]
    if len(row) != 1:
        raise ValueError(f"Esperava um IBOV no IR; encontrei {len(row)}.")
    return float(row.iloc[0]["ClsgPric"])


# ------------------------------------------------------------------------- preço, filtros e vol implícita
def merge_catalog_prices(catalog: pd.DataFrame, prices: pd.DataFrame, valuation_date: date) -> pd.DataFrame:
    """Cruzamento IN x PR, preço de mercado (mid se bid/ask válidos, senão último) e T em dias/365."""
    options = catalog.merge(prices, on="ticker", how="inner", validate="one_to_one")
    valid_spread = (options["best_bid"].notna() & options["best_ask"].notna() & (options["best_bid"] > 0)
                    & (options["best_ask"] >= options["best_bid"]))
    options["market_price"] = np.where(valid_spread, (options["best_bid"] + options["best_ask"]) / 2.0,
                                       options["last_price"])
    options["price_source"] = np.where(valid_spread, "mid", "last")
    options["T"] = (options["maturity_date"] - pd.Timestamp(valuation_date)).dt.days / 365.0
    return options


def black_scholes_price(option_type, spot_value, strike, maturity, rate, dividend_yield, volatility) -> float:
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot_value / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (
        volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot_value * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == "CALL":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def implied_volatility(row, spot: float) -> float:
    discounted_spot = spot * np.exp(-row["q"] * row["T"])
    discounted_strike = row["strike"] * np.exp(-row["r"] * row["T"])
    if row["option_type"] == "CALL":
        lower, upper = max(discounted_spot - discounted_strike, 0.0), discounted_spot
    else:
        lower, upper = max(discounted_strike - discounted_spot, 0.0), discounted_strike
    if not (lower < row["market_price"] < upper):
        return np.nan

    def objective(volatility):
        return black_scholes_price(row["option_type"], spot, row["strike"], row["T"], row["r"], row["q"],
                                   volatility) - row["market_price"]

    try:
        return float(brentq(objective, 1e-4, 5.0, xtol=1e-10, maxiter=200))
    except ValueError:
        return np.nan


def filter_and_invert(options: pd.DataFrame, curves, spot: float, valuation_date: date,
                      filters: IbovFilters = IbovFilters(), *, business_days=None,
                      max_calendar_date=None) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Filtros do notebook 01 na mesma ordem; devolve (filtered, iv_data, relatório)."""
    if business_days is None or max_calendar_date is None:
        from ..di_curve import business_days as _bd, max_calendar_date as _mcd
        business_days = business_days or _bd
        max_calendar_date = max_calendar_date or _mcd
    report = CleaningReport()
    f = options.copy()
    report.register("Cruzamento IN × PR", f)
    f = f.loc[f["option_style"].eq("EURO")]
    report.register("Somente opções europeias", f)
    f = f.loc[f["T"] > 0]
    report.register("Vencimento futuro", f)
    f = f.loc[f["market_price"].gt(0)]
    report.register("Preço positivo", f)
    f = f.loc[f["trades"].fillna(0).ge(filters.min_trades)]
    report.register(f"Ao menos {filters.min_trades} negócio(s)", f)
    f = f.loc[f["open_interest"].fillna(0).ge(filters.min_open_interest)]
    report.register("Posição em aberto positiva", f)
    f = f.loc[f["maturity_date"].dt.date <= max_calendar_date()].copy()
    report.register("Vencimento dentro do calendário BVMF", f)
    f["tenor_bd"] = f["maturity_date"].apply(lambda d: business_days(valuation_date, d.date()))
    du_min, du_max = int(curves.r_curve.tenors_bd.min()), int(curves.r_curve.tenors_bd.max())
    f = f.loc[f["tenor_bd"].between(du_min, du_max)].copy()
    report.register("Prazo coberto pelas curvas", f)
    f["r"] = [curves.r(t) for t in f["tenor_bd"]]
    f["q"] = [curves.q(t) for t in f["tenor_bd"]]
    f["forward"] = spot * np.exp((f["r"] - f["q"]) * f["T"])
    f["log_moneyness"] = np.log(f["strike"] / f["forward"])
    f["implied_vol"] = f.apply(implied_volatility, axis=1, spot=spot) if len(f) else pd.Series(dtype=float)
    iv = f.dropna(subset=["implied_vol"]).copy()
    report.register("Preço dentro dos limites de arbitragem", iv)
    iv = iv.loc[iv["implied_vol"].between(filters.min_iv, filters.max_iv)].copy()
    report.register(f"Volatilidade entre {filters.min_iv:.0%} e {filters.max_iv:.0%}", iv)
    return f, iv, report


def target_inputs(curves, spot: float, iv_data: pd.DataFrame, strike: float, maturity: float) -> dict:
    """r, q, forward e vol do alvo exatamente como o notebook 01 (DU = round(T*252), griddata linear/nearest)."""
    du = round(maturity * 252)
    r, q = curves.r(du), curves.q(du)
    forward = spot * np.exp((r - q) * maturity)
    k = np.log(strike / forward)
    vol, how = VolSurface.from_options(iv_data).vol(k, maturity)
    return {"tenor_bd": du, "r": r, "q": q, "forward": float(forward), "log_moneyness": float(k),
            "volatility": vol, "interpolation": how}


# ---------------------------------------------------------------------------------------- adaptador
def b3_files(folder: Path, market_date: date) -> dict:
    return {p: Path(folder) / f"{p}{market_date:%y%m%d}.zip" for p in ("IN", "IR", "PR", "SPRD")}


class IBOVAdapter:
    """`MarketAdapter` do IBOV sobre os ZIPs da B3 em `data_root/<AAAA-MM-DD>/`."""
    spec = IBOV_SPEC

    def __init__(self, data_root: Path, filters: IbovFilters = IbovFilters(), *, validate: bool = True,
                 curves_builder=None):
        self.data_root = Path(data_root)
        self.filters = filters
        self.validate = validate
        self.curves_builder = curves_builder
        self.last = {}          # objetos intermediários da última carga (catálogo, preços, curvas, ...)

    def market_date(self, now=None) -> date:
        from ..market_date import expected_market_date
        return expected_market_date(now, "B3")

    def load_snapshot(self, market_date: date) -> MarketSnapshot:
        if self.validate:
            from ..market_date import validate_snapshot
            folder = validate_snapshot(self.data_root, market_date)
        else:
            folder = self.data_root / market_date.isoformat()
        files = b3_files(folder, market_date)
        catalog, in_member = read_ibov_option_catalog(files["IN"])
        prices, pr_member = read_ibov_prices(files["PR"])
        spot = read_ibov_spot(files["IR"])
        if self.curves_builder is None:
            from ..di_curve import build_market_curves
            curves = build_market_curves(sprd_zip=files["SPRD"], valuation_date=market_date, spot=spot,
                                         method="flat_forward")
        else:
            curves = self.curves_builder(files["SPRD"], market_date, spot)
        options = merge_catalog_prices(catalog, prices, market_date)
        filtered, iv_data, report = filter_and_invert(options, curves, spot, market_date, self.filters)
        self.last = {"files": files, "in_member": in_member, "pr_member": pr_member, "catalog": catalog,
                     "prices": prices, "options": options, "filtered": filtered, "curves": curves}

        by_expiry = iv_data.groupby("maturity_date").agg(T=("T", "first"), r=("r", "first"), q=("q", "first"),
                                                         forward=("forward", "first"), n=("ticker", "size"))
        by_expiry = by_expiry[by_expiry["T"] > 0].sort_values("T")
        forwards = [ForwardPoint(expiration=d.date(), T=float(row["T"]), forward=float(row["forward"]),
                                 discount_factor=float(np.exp(-row["r"] * row["T"])), n_pairs=int(row["n"]),
                                 residual_std=0.0) for d, row in by_expiry.iterrows()]
        if by_expiry.empty:
            raise ValueError("Nenhuma opção IBOV sobreviveu aos filtros.")
        term = TermStructure(by_expiry["T"].to_numpy(), by_expiry["r"].to_numpy(), by_expiry["q"].to_numpy())
        return MarketSnapshot(spec=IBOV_SPEC, market_date=market_date, spot=spot, spot_source="B3 IR (ClsgPric)",
                              forwards=forwards, term_structure=term, options=iv_data.reset_index(drop=True),
                              report=report)
