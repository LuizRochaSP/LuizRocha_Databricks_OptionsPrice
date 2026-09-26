"""Wrapper para construir r(T) e q(T) a partir dos dados SPRD da B3.

Convenções:
- r(T) e q(T) retornados em taxa CONTÍNUA (compatível com Black-Scholes).
- Tenores em DU/252 (padrão do mercado brasileiro).
- r(T) vem da curva DI1 (flat forward em log-DF).
- q(T) é inferido do futuro de Ibovespa (IND) via F = S * exp((r - q) * T).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .conventions import BUSINESS_DAYS_PER_YEAR
from .curve import DiscountCurve
from .data_loader import load_di1_ind


@dataclass(frozen=True)
class MarketCurves:
    """Curvas r(T) e q(T) em convenção contínua.

    Attributes:
        r_curve: DiscountCurve de DI1 com taxas efetivas anuais.
        q_tenors_bd: Tenores (em DU) dos contratos IND.
        q_cont_values: Dividend yields contínuos nos vencimentos IND.
        spot: Fechamento do Ibovespa.
    """
    r_curve: DiscountCurve
    q_tenors_bd: np.ndarray
    q_cont_values: np.ndarray
    spot: float

    def r(self, tenor_bd: float) -> float:
        """Taxa livre de risco CONTÍNUA no prazo em DU."""
        return float(np.log1p(self.r_curve.rate(tenor_bd)))

    def q(self, tenor_bd: float) -> float:
        """Dividend yield CONTÍNUO no prazo em DU (interp. linear)."""
        return float(np.interp(tenor_bd, self.q_tenors_bd, self.q_cont_values))

    def forward(self, tenor_bd: float) -> float:
        """F(T) = S * exp((r - q) * T), com T em DU/252 e r, q contínuos."""
        T = tenor_bd / BUSINESS_DAYS_PER_YEAR
        return float(self.spot * np.exp((self.r(tenor_bd) - self.q(tenor_bd)) * T))

    def r_q_at(self, tenor_bd: float) -> tuple[float, float]:
        return self.r(tenor_bd), self.q(tenor_bd)


def build_market_curves(
    sprd_zip: Path,
    valuation_date: date,
    spot: float,
    method: str = "flat_forward",
) -> MarketCurves:
    """Constrói r(T) e q(T) contínuos a partir de DI1 e IND.

    Args:
        sprd_zip: Caminho do ZIP SPRD da B3.
        valuation_date: Data de referência (mercado).
        spot: Fechamento do Ibovespa na data.
        method: Método de interpolação para r(T). Default: flat_forward.

    Returns:
        MarketCurves.

    Raises:
        ValueError: se não houver DI1 ou IND no SPRD, ou se spot <= 0.
    """
    di1, ind = load_di1_ind(sprd_zip, valuation_date)

    if di1.empty:
        raise ValueError("Nenhum contrato DI1 encontrado no SPRD.")
    if ind.empty:
        raise ValueError("Nenhum contrato IND encontrado no SPRD.")
    if spot <= 0:
        raise ValueError(f"Spot deve ser positivo: {spot}")

    r_curve = DiscountCurve.from_di1(
        tenors_bd=di1["tenor_bd"].values,
        rates=di1["rate"].values,
        method=method,
    )

    ind = ind.sort_values("tenor_bd").reset_index(drop=True).copy()
    ind["T_252"] = ind["tenor_bd"] / BUSINESS_DAYS_PER_YEAR
    if (ind["T_252"] <= 0).any():
        raise ValueError("IND contracts com tenor_bd <= 0 encontrados.")

    ind["r_cont"] = np.log1p([r_curve.rate(t) for t in ind["tenor_bd"]])
    ind["q_cont"] = ind["r_cont"] - np.log(ind["future"] / spot) / ind["T_252"]

    return MarketCurves(
        r_curve=r_curve,
        q_tenors_bd=ind["tenor_bd"].values,
        q_cont_values=ind["q_cont"].values,
        spot=float(spot),
    )
