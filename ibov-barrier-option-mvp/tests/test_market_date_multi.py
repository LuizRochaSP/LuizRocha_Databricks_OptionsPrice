"""D-1 por mercado: fuso de São Paulo (B3) vs Nova York (NYSE), feriados de cada bolsa."""
from datetime import date, datetime, timedelta, timezone
import importlib.util
import unittest

import pandas as pd

from ibov_barrier.market_date import (EXCHANGES, NEW_YORK, SAO_PAULO, execution_date, expected_market_date,
                                      require_market_date)

HAS_CALENDARS = importlib.util.find_spec("exchange_calendars") is not None


class WeekdayCalendar:
    def date_to_session(self, iso, direction="previous"):
        d = date.fromisoformat(iso)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return pd.Timestamp(d)


CAL = WeekdayCalendar()


class TimezoneTests(unittest.TestCase):
    def test_exchanges_configured(self):
        self.assertEqual(EXCHANGES["B3"].timezone, SAO_PAULO)
        self.assertEqual(EXCHANGES["NYSE"].timezone, NEW_YORK)
        self.assertEqual(EXCHANGES["NYSE"].calendar_name, "XNYS")
        with self.assertRaises(ValueError):
            expected_market_date(date(2026, 9, 22), market="LSE", calendar=CAL)

    def test_after_midnight_sao_paulo_before_midnight_new_york(self):
        # 22/09/2026 03:30 UTC = 00:30 de terça em SP (UTC-3) e 23:30 de segunda em NY (EDT, UTC-4).
        now = datetime(2026, 9, 22, 3, 30, tzinfo=timezone.utc)
        self.assertEqual(execution_date(now, SAO_PAULO), date(2026, 9, 22))
        self.assertEqual(execution_date(now, NEW_YORK), date(2026, 9, 21))
        self.assertEqual(expected_market_date(now, "B3", CAL), date(2026, 9, 21))
        self.assertEqual(expected_market_date(now, "NYSE", CAL), date(2026, 9, 18))

    def test_window_is_wider_in_new_york_winter(self):
        # Inverno americano (EST, UTC-5): 04:30 UTC = 01:30 em SP e 23:30 em NY.
        now = datetime(2026, 12, 15, 4, 30, tzinfo=timezone.utc)
        self.assertEqual(expected_market_date(now, "B3", CAL), date(2026, 12, 14))
        self.assertEqual(expected_market_date(now, "NYSE", CAL), date(2026, 12, 11))

    def test_same_day_in_business_hours(self):
        # Job das 10:00 em SP (09:00 em NY): mesmo D-1 nos dois mercados.
        now = datetime(2026, 9, 22, 10, tzinfo=SAO_PAULO)
        self.assertEqual(expected_market_date(now, "B3", CAL), expected_market_date(now, "NYSE", CAL))

    def test_naive_datetime_rejected(self):
        with self.assertRaises(ValueError):
            expected_market_date(datetime(2026, 9, 22, 10), "NYSE", CAL)

    def test_require_market_date_names_market(self):
        with self.assertRaisesRegex(RuntimeError, "NYSE"):
            require_market_date("2026-09-22", date(2026, 9, 22), "NYSE", CAL)
        self.assertEqual(require_market_date("2026-09-21", date(2026, 9, 22), "NYSE", CAL), date(2026, 9, 21))

    def test_b3_default_unchanged(self):
        self.assertEqual(expected_market_date(date(2026, 9, 22), calendar=CAL), date(2026, 9, 21))


@unittest.skipUnless(HAS_CALENDARS, "exchange_calendars não instalado")
class RealHolidayTests(unittest.TestCase):
    def test_independence_day_observed(self):
        # 04/07/2026 é sábado: NYSE fecha sexta 03/07; B3 abre.
        self.assertEqual(expected_market_date(date(2026, 7, 6), "NYSE"), date(2026, 7, 2))
        self.assertEqual(expected_market_date(date(2026, 7, 6), "B3"), date(2026, 7, 3))

    def test_thanksgiving(self):
        self.assertEqual(expected_market_date(date(2026, 11, 27), "NYSE"), date(2026, 11, 25))
        self.assertEqual(expected_market_date(date(2026, 11, 27), "B3"), date(2026, 11, 26))

    def test_carnival_vs_presidents_day(self):
        # 16/02/2026: Presidents Day (NYSE fechada) e Carnaval (B3 fechada); 17/02 só a B3 fechada.
        self.assertEqual(expected_market_date(date(2026, 2, 18), "NYSE"), date(2026, 2, 17))
        self.assertEqual(expected_market_date(date(2026, 2, 18), "B3"), date(2026, 2, 13))

    def test_real_calendar_midnight_window(self):
        now = datetime(2026, 9, 22, 3, 30, tzinfo=timezone.utc)
        self.assertEqual(expected_market_date(now, "B3"), date(2026, 9, 21))
        self.assertEqual(expected_market_date(now, "NYSE"), date(2026, 9, 18))


if __name__ == "__main__":
    unittest.main()
