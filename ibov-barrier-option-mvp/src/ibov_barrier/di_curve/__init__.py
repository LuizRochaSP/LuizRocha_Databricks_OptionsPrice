"""Módulo de curva de juros brasileira: DI1, swap, risco e validação."""
from .conventions import (
    BUSINESS_DAYS_PER_YEAR,
    add_business_days,
    business_days,
    effective_to_log_df,
    first_business_day,
    is_session,
    log_df_to_effective,
    nearest_wednesday,
)
from .curve import DiscountCurve
from .data_loader import load_anbima_ettj, load_di1_ind
from .risk import (
    KeyRateResult,
    PCAResult,
    key_rate_dv01,
    parallel_dv01,
    pca_decomposition,
    rate_convexity,
)
from .swap import SwapResult, price_di_pre_swap
from .validation import (
    check_monotonic_df,
    check_positive_forwards,
    compare_with_anbima,
)

__all__ = [
    "BUSINESS_DAYS_PER_YEAR",
    "DiscountCurve",
    "KeyRateResult",
    "PCAResult",
    "SwapResult",
    "add_business_days",
    "business_days",
    "check_monotonic_df",
    "check_positive_forwards",
    "compare_with_anbima",
    "effective_to_log_df",
    "first_business_day",
    "is_session",
    "key_rate_dv01",
    "load_anbima_ettj",
    "load_di1_ind",
    "log_df_to_effective",
    "nearest_wednesday",
    "parallel_dv01",
    "pca_decomposition",
    "price_di_pre_swap",
    "rate_convexity",
]