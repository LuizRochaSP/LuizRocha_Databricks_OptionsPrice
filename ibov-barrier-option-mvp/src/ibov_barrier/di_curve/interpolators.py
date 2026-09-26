"""Convenções de mercado brasileiro: DU/252, calendário BVMF, day count."""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars
import numpy as np

BUSINESS_DAYS_PER_YEAR = 252
CALENDAR_NAME = "BVMF"


@lru_cache(maxsize=1)
def bvmf_calendar():
    """Retorna o calendário BVMF (cacheado)."""
    return exchange_calendars.get_calendar(CALENDAR_NAME)


def is_session(d: date) -> bool:
    """True se d é dia de sessão na B3."""
    return bool(bvmf_calendar().is_session(d.isoformat()))


def business_days(start: date, end: date) -> int:
    """Dias úteis em (start, end]. Convenção DI1."""
    if end <= start:
        raise ValueError(f"end ({end}) deve ser posterior a start ({start})")
    sessions = bvmf_calendar().sessions_in_range(start.isoformat(), end.isoformat())
    return int(sum(1 for s in sessions if start < s.date() <= end))


def add_business_days(start: date, n: int) -> date:
    """Avança n dias úteis a partir de start."""
    return bvmf_calendar().session_offset(start.isoformat(), n).date()


def first_business_day(year: int, month: int) -> date:
    """Primeiro dia útil do mês. Usado para vencimento do DI1."""
    d = date(year, month, 1)
    last_day = _last_day_of_month(year, month)
    while d.day <= last_day:
        if is_session(d):
            return d
        d = date(year, month, d.day + 1)
    raise ValueError(f"Nenhum dia útil em {year}-{month:02d}")


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def nearest_wednesday(year: int, month: int) -> date:
    """4ª feira mais próxima do dia 15. Usado para vencimento do IND."""
    candidates = [
        date(year, month, d)
        for d in range(12, 19)
        if date(year, month, d).weekday() == 2
    ]
    return min(candidates, key=lambda x: abs(x.day - 15))


def effective_to_log_df(rate: float, tenor_bd: float) -> float:
    """ln(DF) a partir de taxa efetiva anual e prazo em DU."""
    return -tenor_bd / BUSINESS_DAYS_PER_YEAR * np.log1p(rate)


def log_df_to_effective(log_df: float, tenor_bd: float) -> float:
    """Taxa efetiva anual a partir de ln(DF) e prazo em DU."""
    if tenor_bd <= 0:
        raise ValueError("tenor_bd deve ser positivo")
    T = tenor_bd / BUSINESS_DAYS_PER_YEAR
    return float(np.expm1(-log_df / T))