"""Vol local de Dupire a partir da superfície implícita, para o Monte Carlo com barreira.

Cada vencimento listado é ajustado por um SVI bruto na variância total w(k) = sigma_imp^2 T, com
k = ln(K / F(T)). Entre vencimentos, w é linear em T a k fixo (com máximo acumulado em T para não haver
arbitragem de calendário); antes do primeiro vencimento, w = w_1 t / T_1; depois do último, vol implícita
constante. A vol local sai da fórmula de Dupire em variância total (Gatheral):

    sigma_loc^2(k, T) = (dw/dT) / [1 - (k/w) w' + 1/4 (-1/4 - 1/w + k^2/w^2) w'^2 + 1/2 w'']

com as derivadas em k numéricas numa grade fina. O denominador tem piso e a vol local é limitada, para
que pontos de borda mal ajustados não explodam a simulação; `clipped_share` informa quanto foi cortado.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_K_GRID = np.linspace(-1.5, 1.0, 501)
DENOMINATOR_FLOOR = 0.05
LOCAL_VOL_BOUNDS = (0.03, 1.50)


def svi_total_variance(params, k):
    """SVI bruto: w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))."""
    a, b, rho, m, sigma = params
    x = np.asarray(k, dtype=float) - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def fit_svi(k, implied_vol, maturity: float) -> tuple[np.ndarray, float]:
    """Ajusta o SVI a uma fatia (k, vol implícita). Devolve (parâmetros, RMSE em vol)."""
    from scipy.optimize import least_squares

    k = np.asarray(k, dtype=float)
    iv = np.asarray(implied_vol, dtype=float)
    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[ok], iv[ok]
    if len(k) < 5:
        raise ValueError("SVI needs at least 5 points per expiry")
    w_atm = float(np.interp(0.0, np.sort(k), (iv[np.argsort(k)] ** 2) * maturity))

    def resid(p):
        return np.sqrt(np.clip(svi_total_variance(p, k), 1e-10, None) / maturity) - iv

    x0 = [0.5 * w_atm, 0.1, -0.5, 0.0, 0.1]
    lower = [-1.0, 1e-6, -0.999, -1.0, 1e-4]
    upper = [4.0, 10.0, 0.999, 1.0, 3.0]
    fit = least_squares(resid, x0, bounds=(lower, upper), method="trf", max_nfev=5000)
    return fit.x, float(np.sqrt(np.mean(fit.fun ** 2)))


@dataclass
class LocalVolSurface:
    maturities: np.ndarray             # (n_T,) crescentes
    k_grid: np.ndarray                 # (n_k,)
    total_variance: np.ndarray         # (n_T, n_k), não decrescente em T
    fit_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    clipped_share: float = 0.0

    def __post_init__(self):
        self.maturities = np.asarray(self.maturities, dtype=float)
        self.k_grid = np.asarray(self.k_grid, dtype=float)
        w = np.maximum.accumulate(np.clip(np.asarray(self.total_variance, dtype=float), 1e-10, None), axis=0)
        self.total_variance = w
        self._dw = np.gradient(w, self.k_grid, axis=1)
        self._d2w = np.gradient(self._dw, self.k_grid, axis=1)

    # ----------------------------------------------------------------- construção
    @classmethod
    def from_svi_slices(cls, slices: dict[float, np.ndarray], k_grid=None, fit_report=None) -> "LocalVolSurface":
        k_grid = DEFAULT_K_GRID if k_grid is None else np.asarray(k_grid, dtype=float)
        mats = np.array(sorted(slices))
        w = np.vstack([svi_total_variance(slices[t], k_grid) for t in mats])
        return cls(mats, k_grid, w, fit_report if fit_report is not None else pd.DataFrame())

    @classmethod
    def flat(cls, vol: float, maturities=(0.25, 1.0, 3.0), k_grid=None) -> "LocalVolSurface":
        return cls.from_svi_slices({float(t): np.array([vol * vol * t, 0.0, 0.0, 0.0, 0.1]) for t in maturities},
                                   k_grid)

    @classmethod
    def from_options(cls, options: pd.DataFrame, *, min_maturity: float = 7 / 365, max_maturity: float | None = None,
                     min_points: int = 6, k_range=(-1.0, 0.5), max_rmse: float = 0.03,
                     k_grid=None) -> "LocalVolSurface":
        """Ajusta o SVI por vencimento; exige colunas T, log_moneyness e implied_vol."""
        data = options[["T", "log_moneyness", "implied_vol"]].dropna()
        data = data[(data["T"] >= min_maturity) & data["log_moneyness"].between(*k_range) & (data["implied_vol"] > 0)]
        if max_maturity is not None:
            data = data[data["T"] <= max_maturity]
        slices, rows = {}, []
        for t, g in data.groupby("T"):
            g = g.drop_duplicates("log_moneyness")
            if len(g) < min_points:
                continue
            try:
                params, rmse = fit_svi(g["log_moneyness"], g["implied_vol"], float(t))
            except Exception:  # noqa: BLE001 - fatia ruim fica fora
                continue
            if rmse > max_rmse:   # fatia que o SVI não descreve (ruído ou poucos strikes úteis)
                continue
            slices[float(t)] = params
            rows.append({"T": float(t), "dias": round(float(t) * 365), "pontos": len(g), "RMSE vol": rmse,
                         "vol ATM SVI": float(np.sqrt(max(svi_total_variance(params, 0.0), 1e-10) / t)),
                         "a": params[0], "b": params[1], "rho": params[2], "m": params[3], "sigma": params[4]})
        if len(slices) < 2:
            raise ValueError("local vol needs at least two expiries with a valid SVI fit")
        return cls.from_svi_slices(slices, k_grid, pd.DataFrame(rows).set_index("T"))

    # ------------------------------------------------------------------- consulta
    def _slice(self, t: float):
        """w, w', w'' e dw/dT na grade de k no tempo t."""
        mats, w = self.maturities, self.total_variance
        t = max(float(t), 1e-6)
        if t <= mats[0]:
            f = t / mats[0]
            return w[0] * f, self._dw[0] * f, self._d2w[0] * f, w[0] / mats[0]
        if t >= mats[-1]:
            f = t / mats[-1]
            return w[-1] * f, self._dw[-1] * f, self._d2w[-1] * f, w[-1] / mats[-1]
        i = int(np.searchsorted(mats, t)) - 1
        f = (t - mats[i]) / (mats[i + 1] - mats[i])
        lerp = lambda a: (1 - f) * a[i] + f * a[i + 1]  # noqa: E731
        return lerp(w), lerp(self._dw), lerp(self._d2w), (w[i + 1] - w[i]) / (mats[i + 1] - mats[i])

    def implied_vol(self, k, t: float):
        w, *_ = self._slice(t)
        return np.sqrt(np.interp(k, self.k_grid, w) / max(float(t), 1e-6))

    def local_vol_row(self, t: float, k_range=None) -> tuple[np.ndarray, float]:
        """Vol local na grade de k no tempo t e a fração cortada pelos limites (em `k_range`, se dado)."""
        w, dw, d2w, dwdt = self._slice(t)
        k = self.k_grid
        denom = 1 - k / w * dw + 0.25 * (-0.25 - 1 / w + k * k / (w * w)) * dw * dw + 0.5 * d2w
        raw = np.sqrt(np.clip(dwdt, 0.0, None) / np.maximum(denom, DENOMINATOR_FLOOR))
        lo, hi = LOCAL_VOL_BOUNDS
        bad = (raw < lo) | (raw > hi) | (denom < DENOMINATOR_FLOOR)
        if k_range is not None:
            bad = bad[(k >= k_range[0]) & (k <= k_range[1])]
        clipped = float(np.mean(bad)) if bad.size else 0.0
        return np.clip(raw, lo, hi), clipped

    def local_vol(self, k, t: float):
        row, _ = self.local_vol_row(t)
        return np.interp(k, self.k_grid, row)
