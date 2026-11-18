"""Adaptador do S&P 500 (SPX/SPXW, europeias, liquidação financeira em USD).

Fluxo de `SPXAdapter.load_snapshot(D-1)`:
  fornecedor (ThetaData EOD) -> normalização para a cadeia padrão -> prazos AM/PM em ACT/365
  -> limpeza de cotações -> paridade put-call (forward, DF) -> spot e r(T), q(T)
  -> limpeza contra o forward -> volatilidade implícita -> limpeza do smile -> MarketSnapshot.

O acesso ao fornecedor é injetado (`fetch_chain(root, market_date) -> DataFrame bruto`), então os testes
rodam sem rede e a troca de fornecedor não muda nada abaixo da normalização.
"""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Callable

import numpy as np
import pandas as pd

from ..market_date import NEW_YORK, expected_market_date
from .base import CALL, PUT, CleaningReport, MarketSnapshot, MarketSpec, validate_chain
from .credentials import resolve_thetadata_key
from .cleaning import SPX_CLEANING, CleaningConfig, clean_against_forwards, clean_quotes, clean_smiles
from .parity import implied_forwards, implied_spot, term_structure
from .vol import add_implied_vols

SPX_SPEC = MarketSpec(
    code="SPX",
    name="S&P 500",
    currency="USD",
    exchange="NYSE",
    timezone=NEW_YORK,
    close_time=time(16, 0),
    exercise="european",
    multiplier=100.0,
    day_count="ACT/365",
    rate_index="SOFR",
    roots=("SPX", "SPXW"),
    # SPX (mensal, 3a sexta): valor de liquidação pela abertura (SOQ). SPXW: fechamento.
    settlement_times={"AM": time(9, 30), "PM": time(16, 0)},
)
ROOT_SETTLEMENT = {"SPX": "AM", "SPXW": "PM"}
SECONDS_PER_YEAR = 365.0 * 86400.0

# Nomes de coluna aceitos (ThetaData v1.0.x verificada; v2 REST e variações comuns também).
ALIASES = {
    "expiration": ("expiration", "exp", "expiry", "expiration_date"),
    "strike": ("strike", "strike_price"),
    "option_type": ("right", "option_type", "type", "call_put", "put_call"),
    "bid": ("bid", "bid_price", "closing_bid"),
    "ask": ("ask", "ask_price", "closing_ask"),
    "last": ("close", "last", "last_price", "closing_price"),
    "volume": ("volume", "vol"),
    "open_interest": ("open_interest", "oi"),
}
RIGHTS = {"C": CALL, "CALL": CALL, "P": PUT, "PUT": PUT}


def _pick(raw: pd.DataFrame, field: str, required: bool = True):
    lower = {c.lower(): c for c in raw.columns}
    for alias in ALIASES[field]:
        if alias in lower:
            return raw[lower[alias]]
    if required:
        raise ValueError(f"Coluna '{field}' não encontrada; recebidas: {list(raw.columns)}")
    return pd.Series(np.nan, index=raw.index)


def _parse_expiration(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip().str.replace("-", "", regex=False).str[:8]
    return pd.to_datetime(s, format="%Y%m%d").dt.date


def normalize_thetadata_eod(raw: pd.DataFrame, root: str, strike_scale: float | str = "auto") -> pd.DataFrame:
    """Resposta EOD bruta -> cadeia padronizada (CHAIN_COLUMNS).

    `strike_scale="auto"`: a API REST v2 da ThetaData publica strike em milésimos de dólar
    (6000.0 -> 6000000); valores acima de 100 mil são divididos por 1000.
    """
    if raw.empty:
        raise ValueError(f"{root}: resposta EOD vazia.")
    strike = pd.to_numeric(_pick(raw, "strike"), errors="coerce").astype(float)
    if strike_scale == "auto":
        strike_scale = 1000.0 if strike.median() > 100_000 else 1.0
    rights = _pick(raw, "option_type").astype(str).str.strip().str.upper().map(RIGHTS)
    if rights.isna().any():
        raise ValueError(f"{root}: tipo de opção não reconhecido.")
    # A série vem da requisição (uma chamada por série): colunas "symbol" do fornecedor podem trazer
    # o subjacente (SPX) mesmo para opções SPXW.
    out = pd.DataFrame({
        "root": root.upper(),
        "expiration": _parse_expiration(_pick(raw, "expiration")),
        "strike": strike / float(strike_scale),
        "option_type": rights,
        "bid": _pick(raw, "bid"),
        "ask": _pick(raw, "ask"),
        "last": _pick(raw, "last", required=False),
        "volume": _pick(raw, "volume", required=False),
        "open_interest": _pick(raw, "open_interest", required=False),
    })
    unknown = set(out["root"]) - set(ROOT_SETTLEMENT)
    if unknown:
        raise ValueError(f"Série sem regra de liquidação: {sorted(unknown)}")
    out["settlement"] = out["root"].map(ROOT_SETTLEMENT)
    return validate_chain(out)


def expiry_datetime(expiration: date, settlement: str, spec: MarketSpec = SPX_SPEC) -> datetime:
    return datetime.combine(expiration, spec.settlement_times[settlement], tzinfo=spec.timezone)


def year_fraction(start: datetime, end: datetime) -> float:
    """ACT/365 com hora do dia (vencimentos AM e PM no mesmo dia têm prazos diferentes)."""
    return (end - start).total_seconds() / SECONDS_PER_YEAR


class SPXAdapter:
    spec = SPX_SPEC

    def __init__(self, fetch_chain: Callable[[str, date], pd.DataFrame], *,
                 spot_source: Callable[[date], float] | None = None,
                 rate_curve: Callable[[float], float] | None = None,
                 cleaning: CleaningConfig = SPX_CLEANING,
                 assumed_dividend_yield: float = 0.013,
                 strike_scale: float | str = "auto",
                 calendar=None):
        """
        fetch_chain: (root, data) -> EOD bruto do fornecedor para todas as opções da série.
        spot_source: (data) -> fechamento do índice; se ausente, spot implícito pela paridade.
        rate_curve: (T anos) -> r contínuo (SOFR/Treasuries); se ausente, DF implícito pela paridade.
        assumed_dividend_yield: só usado no spot implícito (erro ~ q·T do vencimento mais curto).
        calendar: calendário injetável para o D-1 (testes); padrão XNYS do exchange_calendars.
        """
        self.fetch_chain = fetch_chain
        self.spot_source = spot_source
        self.rate_curve = rate_curve
        self.cleaning = cleaning
        self.assumed_dividend_yield = assumed_dividend_yield
        self.strike_scale = strike_scale
        self.calendar = calendar

    def market_date(self, now=None) -> date:
        return expected_market_date(now, market=self.spec.exchange, calendar=self.calendar)

    def load_chain(self, market_date: date) -> pd.DataFrame:
        frames = [normalize_thetadata_eod(self.fetch_chain(root, market_date), root, self.strike_scale)
                  for root in self.spec.roots]
        chain = pd.concat(frames, ignore_index=True)
        valuation = datetime.combine(market_date, self.spec.close_time, tzinfo=self.spec.timezone)
        chain["T"] = [year_fraction(valuation, expiry_datetime(e, s, self.spec))
                      for e, s in zip(chain["expiration"], chain["settlement"])]
        return chain

    def load_snapshot(self, market_date: date) -> MarketSnapshot:
        report = CleaningReport()
        quotes = clean_quotes(self.load_chain(market_date), self.cleaning, report)

        forwards = implied_forwards(quotes, discount_curve=self.rate_curve, report_warnings=report.warnings)
        if self.spot_source is not None:
            spot, spot_source = float(self.spot_source(market_date)), "fornecedor"
        else:
            spot = implied_spot(forwards, self.assumed_dividend_yield)
            spot_source = "implícito (paridade, vencimento mais curto)"
        ts = term_structure(forwards, spot)

        fwd = pd.DataFrame({
            "expiration": [p.expiration for p in forwards],
            "T": [p.T for p in forwards],
            "forward": [p.forward for p in forwards],
            "r": [p.rate for p in forwards],
        })
        fwd["q"] = fwd["r"] - np.log(fwd["forward"] / spot) / fwd["T"]
        options = quotes.merge(fwd, on=["expiration", "T"], how="inner")
        report.register("Vencimento com forward implícito", options)

        options = clean_against_forwards(options, spot, self.cleaning, report)
        options = add_implied_vols(options, spot)
        options = clean_smiles(options, self.cleaning, report)
        return MarketSnapshot(self.spec, market_date, spot, spot_source, forwards, ts, options, report)


class ThetaDataSource:
    """Ligação com a biblioteca `thetadata` (v1.0.x, Python 3.12+), verificada em 2026-11-18 no plano FREE.

    - `option_history_eod(d, d, root, "*")`: todas as opções da série em D-1. Colunas: symbol, expiration
      (ISO), strike (dólares), right (CALL/PUT), bid, ask, close, volume, ... Sem open interest (endpoint pago).
    - `index_history_eod("SPX", d, d)`: fechamento oficial do índice (coluna close).
    - `interest_rate_history_eod("SOFR", d, d)`: SOFR em % a.a. (coluna rate).
    O EOD de D-1 é gerado por volta de 17:15-17:20 ET (coluna created).

    A chave é resolvida por `credentials.resolve_thetadata_key` (argumento, THETADATA_API_KEY,
    Databricks Secret quantita/thetadata_api_key ou ~/.thetadata/tdk.txt); nunca vem do código.
    """

    def __init__(self, api_key: str | None = None, client=None):
        self.key_source = "cliente injetado"
        if client is None:
            from thetadata import ThetaClient  # requer também python-dotenv (dependência não declarada)
            resolved = resolve_thetadata_key(api_key)
            self.key_source = resolved.source
            client = ThetaClient(api_key=resolved.value, dataframe_type="pandas")
        self.client = client

    def option_chain(self, root: str, market_date: date) -> pd.DataFrame:
        return self.client.option_history_eod(market_date, market_date, root, "*")

    def index_close(self, market_date: date, symbol: str = "SPX") -> float:
        frame = self.client.index_history_eod(symbol, market_date, market_date)
        if len(frame) != 1:
            raise ValueError(f"{symbol}: esperava 1 fechamento em {market_date}, recebi {len(frame)}.")
        return float(frame["close"].iloc[0])

    def sofr(self, market_date: date) -> float:
        """SOFR de D-1 em taxa contínua a.a. (publicado às 08:00 ET de D+0)."""
        frame = self.client.interest_rate_history_eod("SOFR", market_date, market_date)
        simple = float(frame["rate"].iloc[-1]) / 100.0
        return float(np.log1p(simple / 360.0) * 365.0)  # ACT/360 diário composto -> contínuo ACT/365

    def adapter(self, **kw) -> "SPXAdapter":
        kw.setdefault("spot_source", self.index_close)
        return SPXAdapter(self.option_chain, **kw)
