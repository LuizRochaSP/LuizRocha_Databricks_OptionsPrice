"""Métricas de risco da curva: DV01 paralelo, key-rate, convexidade, PCA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curve import DiscountCurve
from .swap import price_di_pre_swap


@dataclass(frozen=True)
class KeyRateResult:
    vertices_bd: np.ndarray
    dv01_bp: np.ndarray


@dataclass(frozen=True)
class PCAResult:
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    mean_curve: np.ndarray
    tenors_bd: np.ndarray


def parallel_dv01(curve: DiscountCurve, swap_params: dict, bp: float = 1.0) -> float:
    up = price_di_pre_swap(curve.bump_parallel(bp), **swap_params).npv
    dn = price_di_pre_swap(curve.bump_parallel(-bp), **swap_params).npv
    return float((up - dn) / (2 * bp))


def key_rate_dv01(
    curve: DiscountCurve, swap_params: dict, vertices_bd: np.ndarray, bp: float = 1.0
) -> KeyRateResult:
    dv01 = np.zeros(len(vertices_bd))
    for i, v in enumerate(vertices_bd):
        up = price_di_pre_swap(curve.bump_key(v, bp), **swap_params).npv
        dn = price_di_pre_swap(curve.bump_key(v, -bp), **swap_params).npv
        dv01[i] = (up - dn) / (2 * bp)
    return KeyRateResult(vertices_bd=np.asarray(vertices_bd), dv01_bp=dv01)


def rate_convexity(curve: DiscountCurve, swap_params: dict, bp: float = 1.0) -> float:
    base = price_di_pre_swap(curve, **swap_params).npv
    up = price_di_pre_swap(curve.bump_parallel(bp), **swap_params).npv
    dn = price_di_pre_swap(curve.bump_parallel(-bp), **swap_params).npv
    return float((up - 2 * base + dn) / (bp ** 2))


def pca_decomposition(
    historical_log_df: np.ndarray, tenors_bd: np.ndarray, n_components: int = 3
) -> PCAResult:
    """PCA em ln(DF). historical_log_df: shape (n_days, n_tenors)."""
    mean = historical_log_df.mean(axis=0, keepdims=True)
    centered = historical_log_df - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    var = S ** 2 / (centered.shape[0] - 1)
    ratio = var / var.sum()
    return PCAResult(
        explained_variance_ratio=ratio[:n_components],
        components=Vt[:n_components],
        mean_curve=mean.flatten(),
        tenors_bd=np.asarray(tenors_bd),
    )