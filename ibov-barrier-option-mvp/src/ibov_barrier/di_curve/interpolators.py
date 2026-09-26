"""Interpoladores de curva: flat forward, spline cúbico, Nelson-Siegel-Svensson."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from .conventions import BUSINESS_DAYS_PER_YEAR


@dataclass(frozen=True)
class FlatForwardInterpolator:
    """Forward instantâneo constante entre vértices. Padrão do mercado brasileiro.

    Equivale a interpolar ln(DF) linearmente em T entre os vértices.
    """
    tenors_bd: np.ndarray
    log_df: np.ndarray

    def __post_init__(self):
        order = np.argsort(self.tenors_bd)
        object.__setattr__(self, "tenors_bd", np.asarray(self.tenors_bd)[order])
        object.__setattr__(self, "log_df", np.asarray(self.log_df)[order])

    def at(self, tenor_bd: float) -> float:
        t = float(tenor_bd)
        T = t / BUSINESS_DAYS_PER_YEAR

        # Caso especial: apenas 1 vértice — forward flat em todo o domínio.
        # log_df(T) = log_df(T0) * (T / T0), o que mantém a taxa zero constante.
        if len(self.tenors_bd) == 1:
            T0 = self.tenors_bd[0] / BUSINESS_DAYS_PER_YEAR
            if T0 <= 0:
                raise ValueError("Vértice único deve ter prazo positivo")
            return float(self.log_df[0] * (T / T0))

        T_v = self.tenors_bd / BUSINESS_DAYS_PER_YEAR

        if t <= self.tenors_bd[0]:
            f = (self.log_df[1] - self.log_df[0]) / (T_v[1] - T_v[0])
            return float(self.log_df[0] + f * (T - T_v[0]))

        if t >= self.tenors_bd[-1]:
            f = (self.log_df[-1] - self.log_df[-2]) / (T_v[-1] - T_v[-2])
            return float(self.log_df[-1] + f * (T - T_v[-1]))

        i = int(np.searchsorted(self.tenors_bd, t) - 1)
        f = (self.log_df[i + 1] - self.log_df[i]) / (T_v[i + 1] - T_v[i])
        return float(self.log_df[i] + f * (T - T_v[i]))


@dataclass(frozen=True)
class SplineInterpolator:
    """Spline cúbico em ln(DF) vs T."""
    tenors_bd: np.ndarray
    log_df: np.ndarray

    def __post_init__(self):
        order = np.argsort(self.tenors_bd)
        tb = np.asarray(self.tenors_bd)[order]
        ld = np.asarray(self.log_df)[order]
        object.__setattr__(self, "tenors_bd", tb)
        object.__setattr__(self, "log_df", ld)
        if len(tb) < 2:
            raise ValueError("SplineInterpolator requer ao menos 2 vértices")
        object.__setattr__(
            self, "_spline",
            CubicSpline(tb / BUSINESS_DAYS_PER_YEAR, ld, bc_type="natural"),
        )

    def at(self, tenor_bd: float) -> float:
        return float(self._spline(tenor_bd / BUSINESS_DAYS_PER_YEAR))


@dataclass(frozen=True)
class NSSInterpolator:
    """Nelson-Siegel-Svensson para taxa zero."""
    params: np.ndarray

    def at(self, tenor_bd: float) -> float:
        T = tenor_bd / BUSINESS_DAYS_PER_YEAR
        if T <= 0:
            return float(self.params[0] + self.params[1])
        b0, b1, b2, b3, t1, t2 = self.params
        x1 = T / t1
        x2 = T / t2
        f1 = (1 - np.exp(-x1)) / x1
        f2 = (1 - np.exp(-x2)) / x2
        y = b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))
        return float(-y * T)

    @classmethod
    def calibrate(
        cls,
        tenors_bd: np.ndarray,
        log_df_target: np.ndarray,
        initial: np.ndarray | None = None,
    ) -> "NSSInterpolator":
        T = np.asarray(tenors_bd, dtype=float) / BUSINESS_DAYS_PER_YEAR
        y_target = -np.asarray(log_df_target, dtype=float) / T

        def zero_rate_residuals(params):
            b0, b1, b2, b3, t1, t2 = params
            x1 = T / t1
            x2 = T / t2
            f1 = (1 - np.exp(-x1)) / x1
            f2 = (1 - np.exp(-x2)) / x2
            y = b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))
            return y - y_target

        if initial is None:
            initial = np.array([y_target[-1], y_target[0] - y_target[-1], 0.0, 0.0, 1.0, 5.0])

        result = least_squares(
            zero_rate_residuals,
            initial,
            bounds=(
                [-0.5, -1.0, -1.0, -1.0, 0.05, 0.1],
                [1.0, 1.0, 1.0, 1.0, 30.0, 30.0],
            ),
            method="trf",
            max_nfev=2000,
        )
        return cls(params=result.x)
