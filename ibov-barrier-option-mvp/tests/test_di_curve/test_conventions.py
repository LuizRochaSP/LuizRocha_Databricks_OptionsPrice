"""Testes das convenções de mercado brasileiro."""
import unittest
from datetime import date

from ibov_barrier.di_curve.conventions import (
    business_days,
    is_session,
)


class TestConventions(unittest.TestCase):
    def test_weekend_not_session(self):
        """Sábado e domingo não são dias de sessão na B3."""
        self.assertFalse(is_session(date(2026, 9, 26)))  # sábado
        self.assertFalse(is_session(date(2026, 9, 27)))  # domingo

    def test_business_days_positive(self):
        """Um mês entre 24/09 e 24/10 deve ter entre 18 e 23 dias úteis."""
        n = business_days(date(2026, 9, 24), date(2026, 10, 24))
        self.assertGreaterEqual(n, 18)
        self.assertLessEqual(n, 23)

    def test_business_days_raises_on_invalid_range(self):
        """Intervalo invertido deve levantar ValueError."""
        with self.assertRaises(ValueError):
            business_days(date(2026, 10, 1), date(2026, 9, 1))


if __name__ == "__main__":
    unittest.main()