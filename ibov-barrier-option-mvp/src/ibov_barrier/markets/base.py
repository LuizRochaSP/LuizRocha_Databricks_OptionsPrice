"""Contrato comum entre os adaptadores de mercado e o núcleo (superfície, motor, risco, Desk).

Cada mercado (IBOV, SPX, NDX, NKY, KOSPI2, SX5E, ...) entrega um `MarketSnapshot` no mesmo formato.
Tudo a partir da superfície de volatilidade é agnóstico ao mercado.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CALL, PUT = "CALL", "PUT"

# Cadeia de opções padronizada: uma linha por contrato, preços na moeda do mercado (por unidade do índice).
CHAIN_COLUMNS = (
    "root",            # série/classe (ex.: SPX, SPXW, IBOV)
    "expiration",      # data de vencimento (datetime.date)
    "strike",          # em pontos do índice
    "option_type",     # "CALL" | "PUT"
    "bid", "ask",      # melhor oferta de compra/venda no fechamento (NaN se ausente)
    "last",            # último preço/fechamento (NaN se ausente)
    "volume",          # contratos negociados no dia
    "open_interest",   # posição em aberto
    "settlement",      # "AM" (abertura do vencimento) | "PM" (fechamento)
)


@dataclass(frozen=True)
class MarketSpec:
    """Metadados de um mercado. Nada aqui depende do fornecedor de dados."""
    code: str                     # "IBOV", "SPX"
    name: str
    currency: str                 # "BRL", "USD"
    exchange: str                 # chave de calendário em market_date.EXCHANGES ("B3", "NYSE")
    timezone: ZoneInfo
    close_time: time              # fechamento do pregão, hora local
    exercise: str                 # "european" | "american"
    multiplier: float             # valor financeiro por ponto do índice
    day_count: str                # "BUS/252" | "ACT/365"
    rate_index: str               # "DI1", "SOFR", ...
    roots: tuple[str, ...]
    settlement_times: dict = field(default_factory=dict)  # {"AM": time, "PM": time}, hora local


@dataclass(frozen=True)
class ForwardPoint:
    """Forward e fator de desconto implícitos de um vencimento."""
    expiration: date
    T: float
    forward: float
    discount_factor: float
    n_pairs: int
    residual_std: float
    fit_rate: float | None = None   # taxa da regressão ampla deste vencimento (antes da suavização)
    in_fit: bool = False            # True se essa taxa entrou no ajuste da curva r(T)

    @property
    def rate(self) -> float:
        return float(-np.log(self.discount_factor) / self.T)


@dataclass(frozen=True)
class TermStructure:
    """r(T) e q(T) contínuos por prazo (anos), interpolados de forma flat forward em r·T e q·T."""
    T: np.ndarray
    r: np.ndarray
    q: np.ndarray

    def __post_init__(self):
        order = np.argsort(self.T)
        object.__setattr__(self, "T", np.asarray(self.T, float)[order])
        object.__setattr__(self, "r", np.asarray(self.r, float)[order])
        object.__setattr__(self, "q", np.asarray(self.q, float)[order])
        if len(self.T) == 0 or np.any(self.T <= 0) or np.any(np.diff(self.T) <= 0):
            raise ValueError("TermStructure exige prazos positivos e distintos.")

    def _interp(self, values: np.ndarray, t: float) -> float:
        t = float(t)
        if t <= 0:
            raise ValueError("Prazo deve ser positivo.")
        if t <= self.T[0]:
            return float(values[0])
        if t >= self.T[-1]:
            return float(values[-1])
        # Interpolação linear da taxa acumulada (log-DF): taxa a termo constante entre vértices.
        return float(np.interp(t, self.T, values * self.T) / t)

    def rate(self, t: float) -> float:
        return self._interp(self.r, t)

    def dividend_yield(self, t: float) -> float:
        return self._interp(self.q, t)


@dataclass
class CleaningReport:
    """Quantidade de contratos após cada etapa (mesmo formato do `register_filter` do notebook IBOV)."""
    steps: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def register(self, stage: str, frame: pd.DataFrame) -> None:
        self.steps.append({"etapa": stage, "quantidade": int(len(frame))})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["etapa", "quantidade"])


@dataclass
class MarketSnapshot:
    """Fotografia D-1 de um mercado, pronta para a superfície de volatilidade."""
    spec: MarketSpec
    market_date: date
    spot: float
    spot_source: str                 # "fornecedor" | "implícito (paridade, vencimento mais curto)"
    forwards: list                   # list[ForwardPoint]
    term_structure: TermStructure
    options: pd.DataFrame            # cadeia limpa com T, r, q, forward, log_moneyness, market_price, implied_vol
    report: CleaningReport


class MarketAdapter(Protocol):
    """O que cada mercado precisa implementar. O resto do pipeline é comum."""
    spec: MarketSpec

    def market_date(self, now=None) -> date: ...

    def load_snapshot(self, market_date: date) -> MarketSnapshot: ...


def validate_chain(frame: pd.DataFrame) -> pd.DataFrame:
    """Confere o formato padronizado e normaliza tipos. Não filtra nada."""
    missing = [c for c in CHAIN_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Cadeia fora do padrão; faltam colunas: {missing}")
    out = frame.loc[:, list(CHAIN_COLUMNS)].copy()
    out["expiration"] = pd.to_datetime(out["expiration"]).dt.date
    for col in ("strike", "bid", "ask", "last", "volume", "open_interest"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    bad_type = ~out["option_type"].isin([CALL, PUT])
    if bad_type.any():
        raise ValueError(f"option_type inválido: {sorted(out.loc[bad_type, 'option_type'].unique())}")
    bad_settle = ~out["settlement"].isin(["AM", "PM"])
    if bad_settle.any():
        raise ValueError(f"settlement inválido: {sorted(out.loc[bad_settle, 'settlement'].unique())}")
    return out.reset_index(drop=True)
