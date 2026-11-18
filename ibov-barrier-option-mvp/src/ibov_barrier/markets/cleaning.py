"""Limpeza da cadeia de opções antes da superfície (calibrada para SPX, reaproveitável em outros índices).

Três estágios, cada etapa registrada no CleaningReport (quantos contratos sobraram):

1. Cotações (`clean_quotes`), sem precisar de forward:
   duplicatas, prazo mínimo/máximo, bid/ask válidos, bid mínimo, spread máximo, liquidez mínima.
2. Contra o forward (`clean_against_forwards`), depois da paridade put-call:
   limites de arbitragem, só OTM (calls K >= F, puts K < F), banda de moneyness,
   monotonicidade e convexidade em strike (sem arbitragem de vertical/butterfly).
3. Smile (`clean_smiles`), depois da volatilidade implícita:
   IV calculável, IV dentro de [min, max], outliers contra a mediana móvel do smile,
   vencimentos com pontos insuficientes.

Preço de mercado: sempre o mid (bid > 0 e ask >= bid já garantidos no estágio 1). No SPX o último
negócio costuma estar defasado do fechamento do índice, então não é usado como fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import CALL, PUT, CleaningReport
from .vol import arbitrage_bounds

SMILE_KEYS = ["expiration", "settlement"]


@dataclass(frozen=True)
class CleaningConfig:
    min_days: float = 7.0              # descarta vencimentos muito curtos (0DTE/semanais: ruído de microestrutura)
    max_days: float = 730.0
    min_bid: float = 0.05              # tick mínimo do SPX: bid zero/abaixo do tick = sem mercado
    max_relative_spread: float = 0.25  # (ask - bid) / mid
    spread_floor: float = 0.20         # spreads até este valor absoluto são sempre aceitos (asas baratas)
    min_open_interest: float = 0.0
    min_volume: float = 0.0
    preferred_roots: tuple = ("SPXW",) # mesmo vencimento em mais de uma série: fica a preferida (PM)
    max_abs_log_moneyness: float = 0.6
    arb_tolerance: float = 0.05        # tolerância (pontos) nas checagens de vertical/butterfly
    min_iv: float = 0.03
    max_iv: float = 2.0
    outlier_window: int = 5
    outlier_mads: float = 5.0
    outlier_floor: float = 0.02        # resíduo mínimo (pontos de vol) para considerar outlier
    min_points_per_expiry: int = 3


SPX_CLEANING = CleaningConfig()


# --------------------------------------------------------------------------- estágio 1
def clean_quotes(chain: pd.DataFrame, config: CleaningConfig, report: CleaningReport) -> pd.DataFrame:
    """Exige coluna T (anos) além do formato padronizado."""
    df = chain.copy()
    report.register("Cadeia recebida", df)

    df = df.drop_duplicates(["root", "expiration", "strike", "option_type"], keep="last")
    if config.preferred_roots:
        rank = df["root"].map({r: i for i, r in enumerate(config.preferred_roots)}).fillna(len(config.preferred_roots))
        best = rank.groupby(df["expiration"]).transform("min")
        df = df.loc[rank.eq(best)]
    report.register("Sem duplicatas (uma série por vencimento)", df)

    df = df.loc[df["T"].between(config.min_days / 365.0, config.max_days / 365.0)]
    report.register(f"Prazo entre {config.min_days:g} e {config.max_days:g} dias", df)

    valid = np.isfinite(df["bid"]) & np.isfinite(df["ask"]) & (df["ask"] >= df["bid"]) & (df["ask"] > 0)
    df = df.loc[valid]
    report.register("Bid/ask presentes e não cruzados", df)

    df = df.loc[df["bid"] >= config.min_bid]
    report.register(f"Bid >= {config.min_bid:g}", df)

    mid = (df["bid"] + df["ask"]) / 2.0
    spread = df["ask"] - df["bid"]
    df = df.loc[(spread <= config.spread_floor) | (spread <= config.max_relative_spread * mid)]
    report.register(f"Spread <= {config.max_relative_spread:.0%} do mid", df)

    if config.min_open_interest > 0:
        df = df.loc[df["open_interest"].fillna(0) >= config.min_open_interest]
        report.register(f"Posição em aberto >= {config.min_open_interest:g}", df)
    if config.min_volume > 0:
        df = df.loc[df["volume"].fillna(0) >= config.min_volume]
        report.register(f"Volume >= {config.min_volume:g}", df)

    df = df.assign(market_price=(df["bid"] + df["ask"]) / 2.0,
                   rel_spread=(df["ask"] - df["bid"]) / ((df["bid"] + df["ask"]) / 2.0))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- estágio 2
def _drop_static_arbitrage(group: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Remove, um a um, pontos que violam monotonicidade ou convexidade em K (mesmo tipo/vencimento).

    Em cada violação sai a cotação menos confiável (maior spread relativo).
    """
    g = group.sort_values("strike")
    is_call = g["option_type"].iloc[0] == CALL
    while len(g) >= 2:
        K = g["strike"].to_numpy(float)
        C = g["market_price"].to_numpy(float)
        rs = g["rel_spread"].to_numpy(float)
        bad = None
        diffs = np.diff(C)
        mono = np.where(diffs > tol)[0] if is_call else np.where(diffs < -tol)[0]
        if len(mono):
            i = mono[0]
            bad = i if rs[i] >= rs[i + 1] else i + 1
        elif len(g) >= 3:
            w = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
            excess = C[1:-1] - (w * C[:-2] + (1 - w) * C[2:])
            conv = np.where(excess > tol)[0]
            if len(conv):
                j = conv[np.argmax(excess[conv])]
                trio = [j, j + 1, j + 2]
                bad = trio[int(np.argmax(rs[trio]))]
        if bad is None:
            break
        g = g.drop(g.index[bad])
    return g


def clean_against_forwards(options: pd.DataFrame, spot: float, config: CleaningConfig,
                           report: CleaningReport) -> pd.DataFrame:
    """Exige colunas T, r, q, forward (anexadas pela paridade put-call)."""
    df = options.copy()
    bounds = [arbitrage_bounds(r.option_type, spot, r.strike, r.T, r.r, r.q) for r in df.itertuples(index=False)]
    lower = np.array([b[0] for b in bounds]) if bounds else np.array([])
    upper = np.array([b[1] for b in bounds]) if bounds else np.array([])
    df = df.loc[(df["market_price"].to_numpy() > lower) & (df["market_price"].to_numpy() < upper)]
    report.register("Preço dentro dos limites de arbitragem", df)

    otm = (df["option_type"].eq(CALL) & (df["strike"] >= df["forward"])) | \
          (df["option_type"].eq(PUT) & (df["strike"] < df["forward"]))
    df = df.loc[otm]
    report.register("Somente OTM (calls K >= F, puts K < F)", df)

    df = df.assign(log_moneyness=np.log(df["strike"] / df["forward"]))
    df = df.loc[df["log_moneyness"].abs() <= config.max_abs_log_moneyness]
    report.register(f"|ln(K/F)| <= {config.max_abs_log_moneyness:g}", df)

    if len(df):
        df = pd.concat([_drop_static_arbitrage(g, config.arb_tolerance)
                        for _, g in df.groupby(SMILE_KEYS + ["option_type"], sort=False)])
    report.register("Sem arbitragem de vertical/butterfly", df)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- estágio 3
def _smile_outliers(group: pd.DataFrame, config: CleaningConfig) -> pd.Series:
    if len(group) < config.outlier_window:
        return pd.Series(False, index=group.index)
    g = group.sort_values("log_moneyness")
    med = g["implied_vol"].rolling(config.outlier_window, center=True, min_periods=3).median()
    resid = g["implied_vol"] - med
    mad = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    flag = (resid.abs() > config.outlier_mads * mad) & (resid.abs() > config.outlier_floor)
    return flag.reindex(group.index).fillna(False)


def clean_smiles(options: pd.DataFrame, config: CleaningConfig, report: CleaningReport) -> pd.DataFrame:
    """Exige coluna implied_vol."""
    df = options.loc[np.isfinite(options["implied_vol"])]
    report.register("Volatilidade implícita calculável", df)

    df = df.loc[df["implied_vol"].between(config.min_iv, config.max_iv)]
    report.register(f"Volatilidade entre {config.min_iv:.0%} e {config.max_iv:.0%}", df)

    if len(df):
        flags = pd.concat([_smile_outliers(g, config) for _, g in df.groupby(SMILE_KEYS, sort=False)])
        df = df.loc[~flags.reindex(df.index).fillna(False).astype(bool)]
    report.register("Sem outliers no smile (mediana móvel/MAD)", df)

    counts = df.groupby(SMILE_KEYS)["strike"].transform("size")
    df = df.loc[counts >= config.min_points_per_expiry]
    report.register(f"Vencimentos com >= {config.min_points_per_expiry} pontos", df)
    return df.reset_index(drop=True)
