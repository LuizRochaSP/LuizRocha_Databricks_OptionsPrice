"""Precificação de swap DI x Pré (bullet, pagamento periódico)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conventions import BUSINESS_DAYS_PER_YEAR
from .curve import DiscountCurve


@dataclass(frozen=True)
class SwapResult:
    npv: float
    fixed_leg_pv: float
    float_leg_pv: float
    dv01: float
    fixed_rate: float
    side: str


def _legs(
    curve: DiscountCurve,
    start_bd: float,
    end_bd: float,
    fixed_rate: float,
    notional: float,
    payments_per_year: int,
) -> tuple[float, float, np.ndarray]:
    n_periods = max(1, int(round((end_bd - start_bd) / (BUSINESS_DAYS_PER_YEAR / payments_per_year))))
    step = (end_bd - start_bd) / n_periods
    payment_tenors = np.array([start_bd + (i + 1) * step for i in range(n_periods)])

    fixed_cashflows = notional * fixed_rate / payments_per_year * np.ones(n_periods)
    fixed_cashflows[-1] += notional
    dfs = np.array([curve.df(t) for t in payment_tenors])
    fixed_pv = float(np.sum(fixed_cashflows * dfs))

    di_cashflows = np.zeros(n_periods)
    prev = start_bd
    for i, t in enumerate(payment_tenors):
        fwd = curve.forward_rate(prev, t)
        di_cashflows[i] = notional * fwd / payments_per_year
        prev = t
    di_cashflows[-1] += notional
    float_pv = float(np.sum(di_cashflows * dfs))

    return fixed_pv, float_pv, payment_tenors


def price_di_pre_swap(
    curve: DiscountCurve,
    start_bd: float,
    end_bd: float,
    fixed_rate: float,
    *,
    notional: float = 100_000.0,
    payments_per_year: int = 1,
    side: str = "pay_fixed",
    dv01_bp: float = 1.0,
) -> SwapResult:
    """Precifica swap DI x Pré bullet.

    side = "pay_fixed": paga taxa fixa, recebe DI. NPV = VP(DI) - VP(Pré).
    side = "receive_fixed": recebe taxa fixa, paga DI. NPV = VP(Pré) - VP(DI).
    """
    if side not in {"pay_fixed", "receive_fixed"}:
        raise ValueError("side deve ser 'pay_fixed' ou 'receive_fixed'")
    if end_bd <= start_bd:
        raise ValueError("end_bd deve ser maior que start_bd")

    fixed_pv, float_pv, _ = _legs(
        curve, start_bd, end_bd, fixed_rate, notional, payments_per_year
    )
    npv = float_pv - fixed_pv if side == "pay_fixed" else fixed_pv - float_pv

    dv01 = 0.0
    if dv01_bp > 0:
        up = _legs(curve.bump_parallel(dv01_bp), start_bd, end_bd, fixed_rate, notional, payments_per_year)
        dn = _legs(curve.bump_parallel(-dv01_bp), start_bd, end_bd, fixed_rate, notional, payments_per_year)
        npv_up = up[1] - up[0] if side == "pay_fixed" else up[0] - up[1]
        npv_dn = dn[1] - dn[0] if side == "pay_fixed" else dn[0] - dn[1]
        dv01 = (npv_up - npv_dn) / (2 * dv01_bp)

    return SwapResult(
        npv=float(npv),
        fixed_leg_pv=fixed_pv,
        float_leg_pv=float_pv,
        dv01=float(dv01),
        fixed_rate=fixed_rate,
        side=side,
    )