"""Núcleo comum: Black-Scholes, volatilidade implícita e superfície, iguais aos do notebook IBOV.

As funções reproduzem exatamente a seção 5-6 de `databricks/rotinas/01_main_option_pricer.py`
(mesmas fórmulas, mesmo intervalo e tolerância do brentq, mesma interpolação linear com fallback
nearest), agora sem depender de variáveis globais (`spot`) nem do mercado.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import brentq
from scipy.stats import norm

from .base import CALL

IV_LOW, IV_HIGH = 1e-4, 5.0


def black_scholes_price(option_type: str, spot_value: float, strike: float, maturity: float,
                        rate: float, dividend_yield: float, volatility: float) -> float:
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot_value / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (
        volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot_value * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == CALL:
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def arbitrage_bounds(option_type: str, spot_value: float, strike: float, maturity: float,
                     rate: float, dividend_yield: float) -> tuple[float, float]:
    discounted_spot = spot_value * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == CALL:
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


def implied_volatility(option_type: str, price: float, spot_value: float, strike: float,
                       maturity: float, rate: float, dividend_yield: float) -> float:
    lower, upper = arbitrage_bounds(option_type, spot_value, strike, maturity, rate, dividend_yield)
    if not (lower < price < upper):
        return np.nan

    def objective(volatility: float) -> float:
        return black_scholes_price(option_type, spot_value, strike, maturity, rate, dividend_yield,
                                   volatility) - price

    try:
        return float(brentq(objective, IV_LOW, IV_HIGH, xtol=1e-10, maxiter=200))
    except ValueError:
        return np.nan


def add_implied_vols(options: pd.DataFrame, spot_value: float) -> pd.DataFrame:
    """Acrescenta `implied_vol`; exige colunas option_type, market_price, strike, T, r, q."""
    out = options.copy()
    out["implied_vol"] = [
        implied_volatility(row.option_type, row.market_price, spot_value, row.strike, row.T, row.r, row.q)
        for row in out.itertuples(index=False)
    ]
    return out


@dataclass
class VolSurface:
    """Superfície em (log-moneyness forward, T): linear no casco convexo, nearest fora dele."""
    points: np.ndarray   # (n, 2): [log_moneyness, T]
    values: np.ndarray   # (n,)

    @classmethod
    def from_options(cls, iv_data: pd.DataFrame) -> "VolSurface":
        surface_data = iv_data.drop_duplicates(["T", "log_moneyness"])
        if len(surface_data) < 6 or surface_data["T"].nunique() < 2:
            raise ValueError("Há poucos pontos ou vencimentos para construir uma superfície bidimensional.")
        return cls(surface_data[["log_moneyness", "T"]].to_numpy(float),
                   surface_data["implied_vol"].to_numpy(float))

    def vol(self, log_moneyness: float, maturity: float) -> tuple[float, str]:
        target = np.array([[log_moneyness, maturity]])
        iv = griddata(self.points, self.values, target, method="linear")[0]
        if np.isnan(iv):
            return float(griddata(self.points, self.values, target, method="nearest")[0]), \
                "nearest — alvo fora do casco convexo"
        return float(iv), "linear"

    def grid(self, n_t: int = 60, n_k: int = 80):
        """Grade para gráficos (mesmos limites do notebook: quantis 5%-95% em k)."""
        t_grid = np.linspace(self.points[:, 1].min(), self.points[:, 1].max(), n_t)
        k_low, k_high = np.quantile(self.points[:, 0], [0.05, 0.95])
        k_grid = np.linspace(k_low, k_high, n_k)
        K, T = np.meshgrid(k_grid, t_grid)
        return K, T, griddata(self.points, self.values, (K, T), method="linear")
