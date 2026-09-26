"""Curva de juros brasileira: taxa efetiva, DF, forwards, bumps."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conventions import (
    BUSINESS_DAYS_PER_YEAR,
    effective_to_log_df,
    log_df_to_effective,
)
from .interpolators import FlatForwardInterpolator, NSSInterpolator, SplineInterpolator


@dataclass(frozen=True)
class DiscountCurve:
    """Curva de desconto com vértices em DU e taxas efetivas anuais."""
    tenors_bd: np.ndarray
    rates: np.ndarray
    method: str = "flat_forward"
    _interpolator: object = field(default=None, init=False, repr=False)
    _log_df: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        order = np.argsort(self.tenors_bd)
        tb = np.asarray(self.tenors_bd, dtype=float)[order]
        rt = np.asarray(self.rates, dtype=float)[order]
        if np.any(tb <= 0):
            raise ValueError("tenors_bd devem ser positivos")
        if np.any(np.diff(tb) <= 0):
            raise ValueError("tenors_bd devem ser únicos e crescentes")
        object.__setattr__(self, "tenors_bd", tb)
        object.__setattr__(self, "rates", rt)

        log_df = effective_to_log_df(rt, tb)
        object.__setattr__(self, "_log_df", log_df)

        if self.method == "flat_forward":
            interp = FlatForwardInterpolator(tb, log_df)
        elif self.method == "spline":
            interp = SplineInterpolator(tb, log_df)
        elif self.method == "nss":
            interp = NSSInterpolator.calibrate(tb, log_df)
        else:
            raise ValueError(f"método desconhecido: {self.method}")
        object.__setattr__(self, "_interpolator", interp)

    @classmethod
    def from_di1(cls, tenors_bd, rates, method="flat_forward") -> "DiscountCurve":
        return cls(tenors_bd=np.asarray(tenors_bd), rates=np.asarray(rates), method=method)

    def log_df(self, tenor_bd: float) -> float:
        return self._interpolator.at(float(tenor_bd))

    def df(self, tenor_bd: float) -> float:
        return float(np.exp(self.log_df(tenor_bd)))

    def rate(self, tenor_bd: float) -> float:
        return log_df_to_effective(self.log_df(tenor_bd), tenor_bd)

    def forward_rate(self, t1_bd: float, t2_bd: float) -> float:
        """Taxa forward efetiva entre dois prazos."""
        if t2_bd <= t1_bd:
            raise ValueError("t2_bd deve ser maior que t1_bd")
        log_df1 = self.log_df(t1_bd)
        log_df2 = self.log_df(t2_bd)
        T1 = t1_bd / BUSINESS_DAYS_PER_YEAR
        T2 = t2_bd / BUSINESS_DAYS_PER_YEAR
        return float(np.expm1((log_df1 - log_df2) / (T2 - T1)))

    def bump_parallel(self, bump_bp: float) -> "DiscountCurve":
        """Desloca toda a curva em bump_bp pontos-base."""
        bump = bump_bp * 1e-4
        return DiscountCurve(
            self.tenors_bd, self.rates + bump, method=self.method
        )

    def bump_key(self, tenor_bd: float, bump_bp: float) -> "DiscountCurve":
        """Desloca apenas o vértice mais próximo (key-rate bump)."""
        idx = int(np.argmin(np.abs(self.tenors_bd - tenor_bd)))
        new_rates = self.rates.copy()
        new_rates[idx] += bump_bp * 1e-4
        return DiscountCurve(self.tenors_bd, new_rates, method=self.method)