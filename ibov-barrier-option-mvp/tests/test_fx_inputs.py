import tempfile
import unittest
from datetime import date
from math import exp, log
from pathlib import Path

import pandas as pd

from ibov_barrier.markets.fx_inputs import (
    continuous_from_effective,
    ettj_rate,
    fetch_sgs,
    index_closes,
    latest_sprd,
    parse_sgs,
    value_on_or_before,
)


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout, params):
        self.calls.append(params)
        d = pd.to_datetime(params["dataInicial"], format="%d/%m/%Y")
        return _Resp([{"data": d.strftime("%d/%m/%Y"), "valor": "5,40"}])


class SgsTests(unittest.TestCase):
    def test_parse_handles_comma_and_sorts(self):
        s = parse_sgs([{"data": "21/11/2026", "valor": "5.3812"}, {"data": "19/11/2026", "valor": "5,4010"}])
        self.assertEqual(list(s.index), [date(2026, 11, 19), date(2026, 11, 21)])
        self.assertAlmostEqual(s.iloc[0], 5.401)

    def test_parse_rejects_empty(self):
        with self.assertRaises(ValueError):
            parse_sgs([])

    def test_fetch_splits_in_ten_year_blocks(self):
        session = _Session()
        out = fetch_sgs(1, date(2005, 3, 1), date(2026, 11, 20), session=session)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(session.calls[0]["dataInicial"], "01/03/2005")
        self.assertEqual(session.calls[0]["dataFinal"], "28/02/2015")
        self.assertEqual(session.calls[-1]["dataFinal"], "20/11/2026")
        self.assertEqual(len(out), 3)

    def test_value_on_or_before_skips_holiday(self):
        s = pd.Series([5.40, 5.38], index=[date(2026, 11, 19), date(2026, 11, 21)])
        self.assertEqual(value_on_or_before(s, date(2026, 11, 20)), (date(2026, 11, 19), 5.40))
        with self.assertRaises(ValueError):
            value_on_or_before(s, date(2026, 1, 1))


class BrlRateTests(unittest.TestCase):
    def test_continuous_from_effective_preserves_discount_factor(self):
        r_c = continuous_from_effective(0.14, 126, 0.5)
        self.assertAlmostEqual(exp(-r_c * 0.5), (1.14) ** (-126 / 252), places=12)
        with self.assertRaises(ValueError):
            continuous_from_effective(0.14, 0, 0.5)

    def test_ettj_formats_and_flat_forward_interpolation(self):
        pct = pd.DataFrame({"Dias Úteis": [126, 252, 504], "DI x pré 252": ["14,00", "13,50", "13,00"]})
        dec = pd.DataFrame({"vertice_du": [126, 252, 504], "taxa": [0.14, 0.135, 0.13]})
        for frame in (pct, dec):
            self.assertAlmostEqual(ettj_rate(frame, 252), 0.135, places=12)
            r = ettj_rate(frame, 189)
            log_df = 0.5 * (-126 / 252 * log(1.14) - log(1.135))
            self.assertAlmostEqual(r, exp(-log_df * 252 / 189) - 1, places=12)
            self.assertAlmostEqual(ettj_rate(frame, 1000), 0.13)

    def test_ettj_rejects_calendar_days_only(self):
        with self.assertRaises(ValueError):
            ettj_rate(pd.DataFrame({"Dias Corridos": [30, 60], "Taxa": [14.0, 13.9]}), 40)

    def test_latest_sprd_walks_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "2026-11-19"
            folder.mkdir()
            (folder / "SPRD261119.zip").write_bytes(b"x")
            d, path = latest_sprd(Path(tmp), date(2026, 11, 21))
            self.assertEqual(d, date(2026, 11, 19))
            self.assertEqual(path.name, "SPRD261119.zip")
            with self.assertRaises(FileNotFoundError):
                latest_sprd(Path(tmp), date(2026, 11, 18))


class IndexClosesTests(unittest.TestCase):
    def test_reads_date_column_and_drops_bad_rows(self):
        frame = pd.DataFrame({"created": ["2026-11-19T17:16:00", "2026-11-20T17:15:00", "2026-11-20T17:20:00"],
                              "close": [7175.83, 0.0, 6910.39]})
        s = index_closes(frame)
        self.assertEqual(list(s.index), [date(2026, 11, 19), date(2026, 11, 20)])
        self.assertAlmostEqual(s.iloc[-1], 6910.39)

    def test_requires_close(self):
        with self.assertRaises(ValueError):
            index_closes(pd.DataFrame({"date": ["2026-11-19"], "last": [1.0]}))


if __name__ == "__main__":
    unittest.main()
