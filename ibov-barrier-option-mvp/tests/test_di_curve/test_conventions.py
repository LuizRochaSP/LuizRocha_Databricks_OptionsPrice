from datetime import date

from ibov_barrier.di_curve.conventions import (
    business_days,
    is_session,
)


def test_weekend_not_session():
    assert not is_session(date(2026, 9, 26))
    assert not is_session(date(2026, 9, 27))


def test_business_days_positive():
    n = business_days(date(2026, 9, 24), date(2026, 10, 24))
    assert 18 <= n <= 23


def test_business_days_raises_on_invalid_range():
    try:
        business_days(date(2026, 10, 1), date(2026, 9, 1))
    except ValueError:
        return
    assert False, "deveria ter levantado ValueError"