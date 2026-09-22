from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from ibov_barrier.market_date import (
    expected_market_date, require_market_date, validate_snapshot, PREFIXES,
)


class MarketDateTests(unittest.TestCase):
    def test_previous_trading_day(self):
        self.assertEqual(expected_market_date(date(2026, 9, 22)), date(2026, 9, 21))

    def test_monday(self):
        self.assertEqual(expected_market_date(date(2026, 9, 21)), date(2026, 9, 18))

    def test_holiday(self):
        self.assertEqual(expected_market_date(date(2026, 9, 8)), date(2026, 9, 4))

    def test_carnival_and_ash_wednesday(self):
        self.assertEqual(expected_market_date(date(2026, 2, 18)), date(2026, 2, 13))
        self.assertEqual(expected_market_date(date(2026, 2, 19)), date(2026, 2, 18))

    def test_year_boundary(self):
        self.assertEqual(expected_market_date(date(2026, 1, 2)), date(2025, 12, 30))

    def test_christmas(self):
        self.assertEqual(expected_market_date(date(2026, 12, 28)), date(2026, 12, 23))

    def test_sao_paulo_not_utc(self):
        now = datetime(2026, 9, 22, 1, tzinfo=timezone.utc)
        self.assertEqual(expected_market_date(now), date(2026, 9, 18))

    def test_naive_datetime_rejected(self):
        with self.assertRaises(ValueError):
            expected_market_date(datetime(2026, 9, 22))

    def test_stale_or_current_date_rejected(self):
        for value in ("2026-09-18", "2026-09-22", "2026-09-23"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                require_market_date(value, date(2026, 9, 22))
        self.assertEqual(require_market_date("2026-09-21", date(2026, 9, 22)), date(2026, 9, 21))

    def snapshot(self, root, reference, xml_date=None):
        folder = root / reference.isoformat()
        folder.mkdir()
        for prefix in PREFIXES:
            parent = "RptDtAndTm" if prefix == "IN" else "TradDt"
            xml = f'<Document xmlns="urn:test"><{parent}><Dt>{xml_date or reference.isoformat()}</Dt></{parent}></Document>'
            if prefix == "PR":
                xml = f'<Document xmlns="urn:test"><PricRpt><TradDt><Dt>{xml_date or reference.isoformat()}</Dt></TradDt><SctyId><TckrSymb>IBOVTEST</TckrSymb></SctyId></PricRpt></Document>'
            with ZipFile(folder / f"{prefix}{reference:%y%m%d}.zip", "w") as archive:
                archive.writestr("report.xml", xml)
        return folder

    def test_d0_folder_is_ignored(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self.snapshot(root, date(2026, 9, 21))
            self.snapshot(root, date(2026, 9, 22))
            self.assertEqual(validate_snapshot(root, expected_market_date(date(2026, 9, 22))), old)

    def test_missing_d1_does_not_fall_back(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.snapshot(root, date(2026, 9, 18))
            self.snapshot(root, date(2026, 9, 22))
            with self.assertRaises(FileNotFoundError):
                validate_snapshot(root, date(2026, 9, 21))

    def test_wrong_internal_date(self):
        with TemporaryDirectory() as tmp:
            self.snapshot(Path(tmp), date(2026, 9, 21), "2026-09-18")
            with self.assertRaises(RuntimeError):
                validate_snapshot(Path(tmp), date(2026, 9, 21))

    def test_incomplete_snapshot(self):
        with TemporaryDirectory() as tmp:
            folder = self.snapshot(Path(tmp), date(2026, 9, 21))
            (folder / "IR260921.zip").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_snapshot(Path(tmp), date(2026, 9, 21))

    def test_empty_zip_rejected(self):
        with TemporaryDirectory() as tmp:
            folder = self.snapshot(Path(tmp), date(2026, 9, 21))
            with ZipFile(folder / "IN260921.zip", "w"):
                pass
            with self.assertRaises(RuntimeError):
                validate_snapshot(Path(tmp), date(2026, 9, 21))

    def test_pr_other_market_date_not_used_by_pricer(self):
        with TemporaryDirectory() as tmp:
            folder = self.snapshot(Path(tmp), date(2026, 9, 21))
            xml = '<Document xmlns="urn:test">'
            for ticker, day in [("IBOVTEST", "2026-09-21"), ("BGIU26", "2026-09-22")]:
                xml += f'<PricRpt><TradDt><Dt>{day}</Dt></TradDt><SctyId><TckrSymb>{ticker}</TckrSymb></SctyId></PricRpt>'
            with ZipFile(folder / "PR260921.zip", "w") as archive:
                archive.writestr("report.xml", xml + '</Document>')
            self.assertEqual(validate_snapshot(Path(tmp), date(2026, 9, 21)), folder)

    def test_pr_ibov_wrong_date_rejected(self):
        with TemporaryDirectory() as tmp:
            folder = self.snapshot(Path(tmp), date(2026, 9, 21))
            xml = '<Document xmlns="urn:test"><PricRpt><TradDt><Dt>2026-09-22</Dt></TradDt><SctyId><TckrSymb>IBOVTEST</TckrSymb></SctyId></PricRpt></Document>'
            with ZipFile(folder / "PR260921.zip", "w") as archive:
                archive.writestr("report.xml", xml)
            with self.assertRaises(RuntimeError):
                validate_snapshot(Path(tmp), date(2026, 9, 21))
