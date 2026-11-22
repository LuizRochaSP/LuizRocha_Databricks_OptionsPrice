"""Adaptador IBOV contra o próprio notebook 01: mesmos arquivos sintéticos, mesmos números.

O teste executa as células do notebook 01 (seções 2 a 5, código idêntico ao `main`) e o adaptador sobre os
mesmos ZIPs, e exige igualdade exata das contagens e das vols implícitas. Só as curvas DI1/IND e o
calendário BVMF são substituídos por versões sintéticas, iguais para os dois lados.
"""
import importlib.util
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ibov_barrier.markets.ibov import (IBOVAdapter, black_scholes_price, filter_and_invert, merge_catalog_prices,
                                       read_ibov_option_catalog, read_ibov_prices, read_ibov_spot, target_inputs)

HAS_LXML = importlib.util.find_spec("lxml") is not None
NOTEBOOK = Path(__file__).resolve().parents[1] / "databricks" / "rotinas" / "01_main_option_pricer.py"
VALUATION = date(2026, 11, 19)
SPOT = 155_000.0
NS = "urn:bvmf.052.01.xsd"


class FakeCurves:
    """r e q contínuos simples em função do prazo em DU (mesma interface de MarketCurves)."""
    r_curve = SimpleNamespace(tenors_bd=np.array([1.0, 700.0]))
    q_tenors_bd = np.array([20.0, 250.0])

    def r(self, du):
        return 0.13 + 0.00002 * du

    def q(self, du):
        return 0.02 - 0.00001 * du


def fake_business_days(start, end):
    return int(np.busday_count(start, end))


def fake_max_calendar_date():
    return date(2028, 12, 31)


def _xml(tag, rows):
    body = "".join("<{0}>{1}</{0}>".format(tag, "".join(f"<{k}>{v}</{k}>" for k, v in r.items())) for r in rows)
    return f'<?xml version="1.0" encoding="UTF-8"?><Doc xmlns="{NS}">{body}</Doc>'


def _zip(path, xml):
    with ZipFile(path, "w") as z:
        z.writestr(path.stem + ".xml", xml)


def build_files(folder: Path):
    """IN, PR, IR e SPRD sintéticos com casos que cada filtro do notebook remove."""
    rng = np.random.default_rng(7)
    folder.mkdir(parents=True)
    curves = FakeCurves()
    catalog, prices = [], []
    expiries = [VALUATION + timedelta(days=d) for d in (28, 91, 182, 364)]
    n = 0
    for exp in expiries:
        t = (exp - VALUATION).days / 365.0
        du = fake_business_days(VALUATION, exp)
        r, q = curves.r(du), curves.q(du)
        for strike in np.arange(130_000, 180_001, 5_000):
            for opt in ("CALL", "PUT"):
                n += 1
                ticker = f"IBOV{opt[0]}{n:04d}"
                vol = 0.20 + 0.15 * np.log(SPOT / strike)
                price = black_scholes_price(opt, SPOT, strike, t, r, q, vol)
                catalog.append({"TckrSymb": ticker, "Asst": "IBOV", "OptnTp": "CALL" if opt == "CALL" else "PUTT",
                                "ExrcPric": f"{strike:.2f}", "XprtnDt": exp.isoformat(), "OptnStyle": "EURO"})
                spread = max(price * 0.02, 1.0)
                row = {"TckrSymb": ticker, "BestBidPric": f"{price - spread / 2:.2f}",
                       "BestAskPric": f"{price + spread / 2:.2f}", "LastPric": f"{price * 1.01:.2f}",
                       "RglrTxsQty": str(int(rng.integers(0, 5))), "OpnIntrst": str(int(rng.integers(0, 50)))}
                if n % 9 == 0:
                    row.pop("BestBidPric")             # sem bid: usa o último
                prices.append(row)
    catalog.append({"TckrSymb": "IBOVA9999", "Asst": "IBOV", "OptnTp": "CALL", "ExrcPric": "150000.00",
                    "XprtnDt": expiries[1].isoformat(), "OptnStyle": "AMER"})
    prices.append({"TckrSymb": "IBOVA9999", "LastPric": "9000", "RglrTxsQty": "3", "OpnIntrst": "10"})
    catalog.append({"TckrSymb": "IBOVC8888", "Asst": "IBOV", "OptnTp": "CALL", "ExrcPric": "150000.00",
                    "XprtnDt": "2030-01-16", "OptnStyle": "EURO"})          # além do calendário
    prices.append({"TckrSymb": "IBOVC8888", "LastPric": "20000", "RglrTxsQty": "3", "OpnIntrst": "10"})
    catalog.append({"TckrSymb": "PETRC1", "Asst": "PETR", "OptnTp": "CALL", "ExrcPric": "30",
                    "XprtnDt": expiries[0].isoformat(), "OptnStyle": "EURO"})
    tag = VALUATION.strftime("%y%m%d")
    _zip(folder / f"IN{tag}.zip", _xml("Instrm", catalog))
    _zip(folder / f"PR{tag}.zip", _xml("PricRpt", prices))
    _zip(folder / f"IR{tag}.zip", _xml("IndxInf", [{"TckrSymb": "IBXX", "ClsgPric": "60000"},
                                                   {"TckrSymb": "IBOV", "ClsgPric": f"{SPOT:.2f}"}]))
    _zip(folder / f"SPRD{tag}.zip", _xml("PricRpt", [{"TckrSymb": "DI1F27", "AdjstdQt": "99000"}]))
    return {p: folder / f"{p}{tag}.zip" for p in ("IN", "IR", "PR", "SPRD")}


def run_notebook_01(files):
    """Executa as células do notebook 01 da leitura dos XMLs até a vol implícita."""
    from zipfile import ZipFile as _ZipFile

    from lxml import etree
    from scipy.optimize import brentq
    from scipy.stats import norm

    cells = NOTEBOOK.read_text(encoding="utf-8").split("# COMMAND ----------")
    start = next(i for i, c in enumerate(cells) if "def local_name(" in c)
    stop = next(i for i, c in enumerate(cells) if 'filtered["implied_vol"] = filtered.apply' in c)
    ns = {"np": np, "pd": pd, "etree": etree, "ZipFile": _ZipFile, "brentq": brentq, "norm": norm,
          "display": lambda *a, **k: None, "print": lambda *a, **k: None, "date": date, "Path": Path,
          "in_zip": files["IN"], "ir_zip": files["IR"], "pr_zip": files["PR"], "sprd_zip": files["SPRD"],
          "valuation_date": VALUATION, "valuation_ts": pd.Timestamp(VALUATION),
          "MIN_TRADES": 1, "MIN_OPEN_INTEREST": 1, "MIN_IV": 0.01, "MAX_IV": 2.00,
          "build_market_curves": lambda **kw: FakeCurves(), "business_days": fake_business_days,
          "max_calendar_date": fake_max_calendar_date}
    for cell in cells[start:stop + 1]:
        exec(cell, ns)
    return ns


@unittest.skipUnless(HAS_LXML, "lxml ausente")
class IbovAgainstNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.files = build_files(Path(cls.tmp.name) / VALUATION.isoformat())
        cls.nb = run_notebook_01(cls.files)
        catalog, _ = read_ibov_option_catalog(cls.files["IN"])
        prices, _ = read_ibov_prices(cls.files["PR"])
        cls.spot = read_ibov_spot(cls.files["IR"])
        options = merge_catalog_prices(catalog, prices, VALUATION)
        cls.filtered, cls.iv, cls.report = filter_and_invert(
            options, FakeCurves(), cls.spot, VALUATION, business_days=fake_business_days,
            max_calendar_date=fake_max_calendar_date)
        cls.catalog, cls.prices, cls.options = catalog, prices, options

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_readers_match(self):
        self.assertEqual(self.spot, self.nb["spot"])
        pd.testing.assert_frame_equal(self.catalog.reset_index(drop=True), self.nb["catalog"].reset_index(drop=True))
        pd.testing.assert_frame_equal(self.prices.reset_index(drop=True), self.nb["prices"].reset_index(drop=True))
        pd.testing.assert_frame_equal(self.options, self.nb["options"])

    def test_filter_counts_match(self):
        expected = pd.DataFrame(self.nb["filter_summary"]).drop_duplicates("etapa", keep="last")
        self.assertEqual(self.report.to_frame().to_dict("records"), expected.reset_index(drop=True).to_dict("records"))
        counts = self.report.to_frame().set_index("etapa")["quantidade"]
        self.assertLess(counts.iloc[-1], counts.iloc[0])      # os casos sintéticos exercitam os filtros

    def test_implied_vols_match_exactly(self):
        mine = self.iv.set_index("ticker")[["T", "r", "q", "forward", "implied_vol"]].sort_index()
        theirs = self.nb["iv_data"].set_index("ticker")[["T", "r", "q", "forward", "implied_vol"]].sort_index()
        pd.testing.assert_frame_equal(mine, theirs, check_exact=True)
        pd.testing.assert_series_equal(self.filtered["log_moneyness"], self.nb["filtered"]["log_moneyness"]
                                       if "log_moneyness" in self.nb["filtered"] else self.filtered["log_moneyness"])

    def test_target_matches_notebook_griddata(self):
        from scipy.interpolate import griddata

        out = target_inputs(FakeCurves(), self.spot, self.iv, 160_000.0, 0.5)
        du = round(0.5 * 252)
        r, q = FakeCurves().r(du), FakeCurves().q(du)
        k = np.log(160_000.0 / (self.spot * np.exp((r - q) * 0.5)))
        surface = self.nb["iv_data"].drop_duplicates(["T", "log_moneyness"]) \
            if "log_moneyness" in self.nb["iv_data"] else None
        if surface is None:
            surface = self.nb["iv_data"].assign(log_moneyness=np.log(self.nb["iv_data"]["strike"]
                                                                     / self.nb["iv_data"]["forward"]))
            surface = surface.drop_duplicates(["T", "log_moneyness"])
        expected = griddata(surface[["log_moneyness", "T"]].to_numpy(), surface["implied_vol"].to_numpy(),
                            np.array([[k, 0.5]]), method="linear")[0]
        self.assertEqual(out["volatility"], float(expected))
        self.assertEqual((out["r"], out["q"]), (r, q))

    def test_recovers_generating_vols(self):
        mids = self.iv[self.iv["price_source"].eq("mid")]
        expected = 0.20 + 0.15 * np.log(SPOT / mids["strike"])
        # Mid é o preço teórico (spread simétrico; arredondamento de 2 casas).
        np.testing.assert_allclose(mids["implied_vol"], expected, atol=2e-3)


@unittest.skipUnless(HAS_LXML, "lxml ausente")
class IbovAdapterTests(unittest.TestCase):
    def test_snapshot_contract(self):
        import ibov_barrier.markets.ibov as ibov

        with tempfile.TemporaryDirectory() as tmp:
            build_files(Path(tmp) / VALUATION.isoformat())
            adapter = IBOVAdapter(tmp, validate=False, curves_builder=lambda *a: FakeCurves())
            original = ibov.filter_and_invert
            ibov.filter_and_invert = lambda *a, **k: original(*a, business_days=fake_business_days,
                                                               max_calendar_date=fake_max_calendar_date, **k)
            try:
                snap = adapter.load_snapshot(VALUATION)
            finally:
                ibov.filter_and_invert = original
        self.assertEqual(snap.spec.code, "IBOV")
        self.assertEqual(snap.spot, SPOT)
        self.assertEqual(len(snap.forwards), 4)
        self.assertTrue({"T", "r", "q", "forward", "log_moneyness", "market_price", "implied_vol"}
                        <= set(snap.options.columns))
        f = snap.forwards[1]
        du = fake_business_days(VALUATION, f.expiration)
        self.assertAlmostEqual(f.rate, FakeCurves().r(du), places=12)
        self.assertAlmostEqual(snap.term_structure.rate(f.T), FakeCurves().r(du), places=12)


if __name__ == "__main__":
    unittest.main()
