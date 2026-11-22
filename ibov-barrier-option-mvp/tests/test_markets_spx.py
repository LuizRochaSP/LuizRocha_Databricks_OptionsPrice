"""Testes sintéticos da camada multi-mercado e do adaptador SPX (sem rede, sem fornecedor real).

A cadeia é gerada por Black-Scholes com spot, juros, dividendos e smile conhecidos; o pipeline
precisa recuperá-los. Contratos com problemas são injetados e precisam ser removidos pela limpeza.
"""
from datetime import date, datetime, timedelta
import unittest

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import brentq
from scipy.stats import norm

from ibov_barrier.markets import CALL, PUT, CleaningReport, SPXAdapter, TermStructure, VolSurface
from ibov_barrier.markets.cleaning import SPX_CLEANING, clean_smiles
from ibov_barrier.markets.parity import implied_forward, implied_spot, term_structure
from ibov_barrier.markets.spx import (SPX_SPEC, expiry_datetime, normalize_thetadata_eod, year_fraction)
from ibov_barrier.markets.vol import (arbitrage_bounds, black_scholes_price, implied_volatility)

S0, R, Q = 6000.0, 0.043, 0.013
MARKET_DATE = date(2026, 9, 21)  # segunda-feira
PM_EXPIRIES = [date(2026, 9, 23), date(2026, 10, 2), date(2026, 10, 16), date(2026, 11, 20),
               date(2026, 12, 18), date(2027, 3, 19), date(2027, 6, 18)]
AM_EXPIRIES = [date(2026, 10, 16), date(2026, 11, 20), date(2026, 12, 18)]  # 3as sextas (SPX mensal)


def true_vol(k, T):
    return 0.17 - 0.12 * k + 0.40 * k**2 + 0.01 * np.sqrt(T)


def _T(expiration, settlement):
    valuation = datetime.combine(MARKET_DATE, SPX_SPEC.close_time, tzinfo=SPX_SPEC.timezone)
    return year_fraction(valuation, expiry_datetime(expiration, settlement))


def synthetic_raw(root, expiries, settlement, tick=None):
    """EOD no formato da API REST v2 da ThetaData: strike em milésimos, right C/P, expiration YYYYMMDD."""
    rows = []
    for exp in expiries:
        T = _T(exp, settlement)
        F = S0 * np.exp((R - Q) * T)
        for K in np.arange(4200.0, 7801.0, 25.0):
            k = np.log(K / F)
            for right, otype in (("C", CALL), ("P", PUT)):
                mid = black_scholes_price(otype, S0, K, T, R, Q, true_vol(k, T))
                spread = max(0.10, 0.02 * mid)
                bid, ask = mid - spread / 2, mid + spread / 2
                if tick:
                    bid, ask = np.floor(bid / tick) * tick, np.ceil(ask / tick) * tick
                rows.append({"root": root, "expiration": int(exp.strftime("%Y%m%d")),
                             "strike": int(round(K * 1000)), "right": right,
                             "bid": max(bid, 0.0), "ask": ask, "close": mid,
                             "volume": 10, "open_interest": 100})
    return pd.DataFrame(rows)


class FakeCalendar:
    """Dias úteis menos feriados; mesmo contrato de `date_to_session` do exchange_calendars."""
    def __init__(self, holidays=()):
        self.holidays = set(holidays)

    def date_to_session(self, iso, direction="previous"):
        d = date.fromisoformat(iso)
        while d.weekday() >= 5 or d in self.holidays:
            d -= timedelta(days=1)
        return pd.Timestamp(d)


def adapter(raw_by_root, **kw):
    return SPXAdapter(lambda root, d: raw_by_root[root].copy(), calendar=FakeCalendar(), **kw)


def clean_raw(tick=None):
    return {"SPX": synthetic_raw("SPX", AM_EXPIRIES, "AM", tick),
            "SPXW": synthetic_raw("SPXW", PM_EXPIRIES, "PM", tick)}


class NormalizationTests(unittest.TestCase):
    def test_thetadata_v2_format(self):
        chain = normalize_thetadata_eod(synthetic_raw("SPXW", PM_EXPIRIES[:1], "PM"), "SPXW")
        self.assertEqual(chain["strike"].min(), 4200.0)
        self.assertEqual(set(chain["option_type"]), {CALL, PUT})
        self.assertEqual(set(chain["settlement"]), {"PM"})
        self.assertEqual(chain["expiration"].iloc[0], date(2026, 9, 23))

    def test_dollar_strikes_and_long_names(self):
        raw = pd.DataFrame({"symbol": ["SPX"], "expiration": ["2026-10-16"], "strike": [6000.0],
                            "right": ["call"], "bid": [10.0], "ask": [11.0]})
        chain = normalize_thetadata_eod(raw, "SPX")
        self.assertEqual(chain.loc[0, "strike"], 6000.0)
        self.assertEqual(chain.loc[0, "settlement"], "AM")
        self.assertTrue(np.isnan(chain.loc[0, "open_interest"]))

    def test_invalid_input(self):
        raw = pd.DataFrame({"expiration": [20261016], "strike": [6000000], "right": ["X"],
                            "bid": [1.0], "ask": [2.0]})
        with self.assertRaises(ValueError):
            normalize_thetadata_eod(raw, "SPXW")
        with self.assertRaises(ValueError):
            normalize_thetadata_eod(raw.assign(right="C"), "NDXP")  # série sem regra AM/PM
        with self.assertRaises(ValueError):
            normalize_thetadata_eod(raw.drop(columns="bid"), "SPXW")

    def test_am_settlement_is_shorter(self):
        exp = date(2026, 10, 16)
        diff = _T(exp, "PM") - _T(exp, "AM")
        self.assertAlmostEqual(diff * 365 * 24, 6.5, places=9)


class VolCoreMatchesIbovNotebook(unittest.TestCase):
    """O núcleo comum precisa ser idêntico às funções da seção 5 do notebook IBOV (cópia literal abaixo)."""

    @staticmethod
    def notebook_iv(row, spot):
        def black_scholes_price(option_type, spot_value, strike, maturity, rate, dividend_yield, volatility):
            sqrt_t = np.sqrt(maturity)
            d1 = (np.log(spot_value / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (
                volatility * sqrt_t)
            d2 = d1 - volatility * sqrt_t
            discounted_spot = spot_value * np.exp(-dividend_yield * maturity)
            discounted_strike = strike * np.exp(-rate * maturity)
            if option_type == "CALL":
                return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
            return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))

        def arbitrage_bounds(row):
            discounted_spot = spot * np.exp(-row["q"] * row["T"])
            discounted_strike = row["strike"] * np.exp(-row["r"] * row["T"])
            if row["option_type"] == "CALL":
                return max(discounted_spot - discounted_strike, 0.0), discounted_spot
            return max(discounted_strike - discounted_spot, 0.0), discounted_strike

        lower, upper = arbitrage_bounds(row)
        if not (lower < row["market_price"] < upper):
            return np.nan

        def objective(volatility):
            return black_scholes_price(row["option_type"], spot, row["strike"], row["T"], row["r"], row["q"],
                                       volatility) - row["market_price"]
        try:
            return float(brentq(objective, 1e-4, 5.0, xtol=1e-10, maxiter=200))
        except ValueError:
            return np.nan

    def test_identical_implied_vols(self):
        rng = np.random.default_rng(3)
        spot = 128_000.0
        for _ in range(200):
            row = {"option_type": rng.choice([CALL, PUT]), "strike": spot * rng.uniform(0.6, 1.4),
                   "T": rng.uniform(0.02, 2.0), "r": rng.uniform(0.08, 0.15), "q": rng.uniform(0.0, 0.05)}
            row["market_price"] = rng.uniform(0.0, 1.2) * black_scholes_price(
                row["option_type"], spot, row["strike"], row["T"], row["r"], row["q"], rng.uniform(0.1, 0.6))
            ours = implied_volatility(row["option_type"], row["market_price"], spot, row["strike"], row["T"],
                                      row["r"], row["q"])
            ref = self.notebook_iv(row, spot)
            if np.isnan(ref):
                self.assertTrue(np.isnan(ours))
            else:
                self.assertEqual(ours, ref)

    def test_surface_matches_notebook_interpolation(self):
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({"log_moneyness": rng.uniform(-0.3, 0.3, 60),
                              "T": rng.choice([0.1, 0.25, 0.5, 1.0], 60)})
        frame["implied_vol"] = true_vol(frame["log_moneyness"], frame["T"])
        surface = VolSurface.from_options(frame)
        pts, vals = frame[["log_moneyness", "T"]].to_numpy(), frame["implied_vol"].to_numpy()
        iv, method = surface.vol(0.02, 0.4)
        self.assertEqual(method, "linear")
        self.assertEqual(iv, griddata(pts, vals, np.array([[0.02, 0.4]]), method="linear")[0])
        iv, method = surface.vol(0.0, 3.0)
        self.assertTrue(method.startswith("nearest"))
        self.assertEqual(iv, griddata(pts, vals, np.array([[0.0, 3.0]]), method="nearest")[0])
        with self.assertRaises(ValueError):
            VolSurface.from_options(frame.head(4))


class ParityTests(unittest.TestCase):
    def setUp(self):
        exp = date(2026, 12, 18)
        self.T = _T(exp, "PM")
        self.quotes = normalize_thetadata_eod(synthetic_raw("SPXW", [exp], "PM"), "SPXW")
        self.exp = exp

    def test_regression_recovers_forward_and_discount(self):
        p = implied_forward(self.quotes, self.T, self.exp)
        self.assertAlmostEqual(p.forward, S0 * np.exp((R - Q) * self.T), delta=1e-6)
        self.assertAlmostEqual(p.rate, R, delta=1e-9)
        self.assertEqual(p.n_pairs, 10)

    def test_given_discount_factor(self):
        p = implied_forward(self.quotes, self.T, self.exp, discount_factor=np.exp(-R * self.T))
        self.assertAlmostEqual(p.forward, S0 * np.exp((R - Q) * self.T), delta=1e-6)

    def test_spot_and_term_structure(self):
        exps = [date(2026, 10, 2), date(2026, 12, 18), date(2027, 6, 18)]
        pts = [implied_forward(normalize_thetadata_eod(synthetic_raw("SPXW", [e], "PM"), "SPXW"), _T(e, "PM"), e)
               for e in exps]
        self.assertAlmostEqual(implied_spot(pts, Q), S0, delta=1e-5)
        self.assertAlmostEqual(implied_spot(pts, 0.0), S0, delta=S0 * Q * pts[0].T * 1.01)
        ts = term_structure(pts, S0)
        for t in (0.01, 0.2, 0.5, 2.0):
            self.assertAlmostEqual(ts.rate(t), R, delta=1e-8)
            self.assertAlmostEqual(ts.dividend_yield(t), Q, delta=1e-7)

    def test_too_few_pairs(self):
        with self.assertRaises(ValueError):
            implied_forward(self.quotes.head(4), self.T, self.exp)

    def test_term_structure_flat_forward(self):
        ts = TermStructure(np.array([1.0, 2.0]), np.array([0.10, 0.12]), np.array([0.0, 0.0]))
        self.assertAlmostEqual(ts.rate(1.5) * 1.5, (0.10 * 1.0 + 0.12 * 2.0) / 2, places=12)
        self.assertEqual(ts.rate(0.5), 0.10)
        self.assertEqual(ts.rate(5.0), 0.12)
        with self.assertRaises(ValueError):
            TermStructure(np.array([1.0, 1.0]), np.array([0.1, 0.1]), np.array([0.0, 0.0]))


class CleaningTests(unittest.TestCase):
    def test_injected_problems_are_removed(self):
        raw = clean_raw()
        w = raw["SPXW"]
        exp = 20261218
        T = _T(date(2026, 12, 18), "PM")
        F = S0 * np.exp((R - Q) * T)

        def row(strike, right):
            return (w["expiration"].eq(exp) & w["strike"].eq(strike * 1000) & w["right"].eq(right))

        def price(K, otype):
            return black_scholes_price(otype, S0, K, T, R, Q, true_vol(np.log(K / F), T))

        bad = {}
        w.loc[row(5000, "P"), ["bid", "ask"]] = [12.0, 11.0];          bad["crossed"] = (5000, PUT)
        w.loc[row(5100, "P"), ["bid", "ask"]] = [0.0, 3.0];            bad["zero_bid"] = (5100, PUT)
        w.loc[row(5200, "P"), ["bid", "ask"]] = [5.0, 60.0];           bad["wide"] = (5200, PUT)
        w.loc[row(4500, "P"), ["bid", "ask"]] = [9000.0, 9001.0];      bad["absurd"] = (4500, PUT)
        # Call OTM mais cara que a do strike anterior (vertical negativa), cotação larga.
        up = price(6475, CALL) + 3.0
        w.loc[row(6500, "C"), ["bid", "ask"]] = [up * 0.9, up * 1.1];  bad["vertical"] = (6500, CALL)
        # Butterfly: monotônica, mas acima da corda entre os vizinhos.
        c0, c2 = price(6550, CALL), price(6600, CALL)
        mid = (c0 + c2) / 2 + 1.5
        w.loc[row(6575, "C"), ["bid", "ask"]] = [mid * 0.9, mid * 1.1]; bad["butterfly"] = (6575, CALL)

        snap = adapter(raw, spot_source=lambda d: S0).load_snapshot(MARKET_DATE)
        out = snap.options
        dec = out.loc[out["expiration"].eq(date(2026, 12, 18))]
        for name, (K, otype) in bad.items():
            with self.subTest(name=name):
                self.assertFalse(((dec["strike"] == K) & (dec["option_type"] == otype)).any())
        # Vizinhos saudáveis continuam.
        for K, otype in ((6525, CALL), (6550, CALL), (6600, CALL), (5300, PUT)):
            self.assertTrue(((dec["strike"] == K) & (dec["option_type"] == otype)).any(), (K, otype))
        # Uma série por vencimento (SPXW preferida), sem vencimentos < 7 dias, só OTM.
        self.assertEqual(set(out["root"]), {"SPXW"})
        self.assertNotIn(date(2026, 9, 23), set(out["expiration"]))
        otm = np.where(out["option_type"].eq(CALL), out["strike"] >= out["forward"], out["strike"] < out["forward"])
        self.assertTrue(otm.all())
        counts = snap.report.to_frame()["quantidade"].to_numpy()
        self.assertTrue(np.all(np.diff(counts) <= 0))
        self.assertEqual(snap.report.warnings, [])

    def test_smile_outlier(self):
        k = np.linspace(-0.2, 0.2, 21)
        frame = pd.DataFrame({"expiration": date(2026, 12, 18), "settlement": "PM", "strike": 6000 * np.exp(k),
                              "log_moneyness": k, "implied_vol": true_vol(k, 0.24)})
        frame.loc[7, "implied_vol"] += 0.05
        out = clean_smiles(frame, SPX_CLEANING, CleaningReport())
        self.assertEqual(len(out), 20)
        self.assertNotIn(frame.loc[7, "strike"], set(out["strike"]))

    def test_smooth_smile_untouched(self):
        k = np.linspace(-0.3, 0.3, 25)
        frame = pd.DataFrame({"expiration": date(2026, 12, 18), "settlement": "PM", "strike": 6000 * np.exp(k),
                              "log_moneyness": k, "implied_vol": true_vol(k, 0.24)})
        self.assertEqual(len(clean_smiles(frame, SPX_CLEANING, CleaningReport())), 25)


class AdapterEndToEndTests(unittest.TestCase):
    def test_exact_recovery(self):
        snap = adapter(clean_raw(), assumed_dividend_yield=Q).load_snapshot(MARKET_DATE)
        self.assertAlmostEqual(snap.spot, S0, delta=1e-4)
        self.assertTrue(snap.spot_source.startswith("implícito"))
        opts = snap.options
        err = opts["implied_vol"] - true_vol(opts["log_moneyness"], opts["T"])
        self.assertLess(err.abs().max(), 1e-6)
        self.assertGreater(len(opts), 500)
        self.assertEqual(set(opts["T"].round(6)), {round(_T(e, "PM"), 6) for e in PM_EXPIRIES[1:]})
        for t in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(snap.term_structure.rate(t), R, delta=1e-7)
            self.assertAlmostEqual(snap.term_structure.dividend_yield(t), Q, delta=1e-6)
        surface = VolSurface.from_options(opts)
        iv, method = surface.vol(-0.05, 0.5)
        self.assertEqual(method, "linear")
        self.assertAlmostEqual(iv, true_vol(-0.05, 0.5), delta=2e-3)

    def test_tick_rounded_quotes(self):
        """Cotações arredondadas ao tick de 0,05: ruído realista, recuperação aproximada."""
        snap = adapter(clean_raw(tick=0.05), spot_source=lambda d: S0,
                       rate_curve=lambda t: R).load_snapshot(MARKET_DATE)
        for p in snap.forwards:
            self.assertAlmostEqual(p.forward, S0 * np.exp((R - Q) * p.T), delta=0.05)
        opts = snap.options
        near = opts["log_moneyness"].abs() < 0.1
        err = (opts["implied_vol"] - true_vol(opts["log_moneyness"], opts["T"]))[near]
        self.assertLess(err.abs().max(), 0.005)

    def test_short_expiries_borrow_rate_from_reliable_tenor(self):
        """Sem curva externa e com cotações no tick, prazos curtos não podem gerar taxas absurdas."""
        snap = adapter(clean_raw(tick=0.05), spot_source=lambda d: S0).load_snapshot(MARKET_DATE)
        short = [p for p in snap.forwards if p.T < 0.25]
        self.assertTrue(short)
        for p in snap.forwards:
            self.assertAlmostEqual(p.rate, R, delta=0.005)
            self.assertAlmostEqual(p.forward, S0 * np.exp((R - Q) * p.T), delta=0.1)
        longest_short = max(short, key=lambda p: p.T)
        first_reliable = min((p for p in snap.forwards if p.T >= 0.25), key=lambda p: p.T)
        self.assertAlmostEqual(longest_short.rate, first_reliable.rate, places=12)

    def test_market_date_uses_new_york(self):
        a = adapter(clean_raw())
        self.assertEqual(a.market_date(date(2026, 9, 22)), date(2026, 9, 21))
        self.assertEqual(a.spec.currency, "USD")

    def test_bounds_helper_consistent(self):
        lo, hi = arbitrage_bounds(CALL, S0, 5000.0, 0.5, R, Q)
        self.assertAlmostEqual(hi, S0 * np.exp(-Q * 0.5))
        self.assertGreater(lo, 0)


if __name__ == "__main__":
    unittest.main()


class SmoothRateTests(unittest.TestCase):
    """Curva r(T) suavizada, regressão ampla e descarte de vencimentos incoerentes."""

    def test_fit_rate_curve_linear_with_outlier(self):
        from ibov_barrier.markets.parity import fit_rate_curve
        T = np.array([0.25, 0.4, 0.5, 0.75, 1.0, 1.2, 1.6])
        r = 0.045 + 0.004 * T
        r[2] -= 0.25  # vencimento quebrado: -2.500 bps
        curve, used = fit_rate_curve(T, r)
        self.assertFalse(used[2])
        self.assertTrue(used[[0, 1, 3, 4, 5, 6]].all())
        for t in (0.3, 1.0, 1.5):
            self.assertAlmostEqual(curve(t), 0.045 + 0.004 * t, places=10)
        self.assertAlmostEqual(curve(0.05), curve(0.25), places=12)  # constante abaixo do menor prazo
        self.assertAlmostEqual(curve(3.0), curve(1.6), places=12)    # e acima do maior

    def test_fit_rate_curve_few_points_is_median(self):
        from ibov_barrier.markets.parity import fit_rate_curve
        curve, used = fit_rate_curve([0.3, 0.6], [0.04, 0.05])
        self.assertTrue(used.all())
        self.assertAlmostEqual(curve(0.1), 0.045)
        self.assertAlmostEqual(curve(2.0), 0.045)

    def test_wide_regression_recovers_discount(self):
        from ibov_barrier.markets.parity import implied_discount_factor
        exp = date(2027, 3, 19)
        T = _T(exp, "PM")
        quotes = normalize_thetadata_eod(synthetic_raw("SPXW", [exp], "PM"), "SPXW")
        df, n = implied_discount_factor(quotes, exp)
        self.assertAlmostEqual(-np.log(df) / T, R, delta=1e-9)
        self.assertGreater(n, 30)

    def test_smooth_is_default_and_fills_diagnostics(self):
        snap = adapter(clean_raw(tick=0.05), spot_source=lambda d: S0).load_snapshot(MARKET_DATE)
        long_ = [p for p in snap.forwards if p.T >= 0.25]
        self.assertTrue(long_)
        for p in long_:
            self.assertIsNotNone(p.fit_rate)
            self.assertTrue(p.in_fit)
        for p in snap.forwards:
            self.assertAlmostEqual(p.rate, R, delta=0.002)
            self.assertAlmostEqual(p.forward, S0 * np.exp((R - Q) * p.T), delta=0.1)
        self.assertEqual(snap.report.warnings, [])

    def test_atm_method_still_available(self):
        snap = adapter(clean_raw(tick=0.05), spot_source=lambda d: S0, rate_method="atm").load_snapshot(MARKET_DATE)
        short = [p for p in snap.forwards if p.T < 0.25]
        first_reliable = min((p for p in snap.forwards if p.T >= 0.25), key=lambda p: p.T)
        for p in short:
            self.assertAlmostEqual(p.rate, first_reliable.rate, places=12)
            self.assertIsNone(p.fit_rate)

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            adapter(clean_raw(), spot_source=lambda d: S0, rate_method="xyz").load_snapshot(MARKET_DATE)

    def test_incoherent_expiry_is_dropped_with_warning(self):
        """Cotações das calls perto do ATM deslocadas ±15 pontos (spreads normais): forward incoerente."""
        raw = clean_raw()
        w = raw["SPXW"]
        bad_exp = 20270319
        near = w["expiration"].eq(bad_exp) & w["right"].eq("C") & w["strike"].between(5500_000, 6800_000)
        shift = np.where((w.loc[near, "strike"] // 25_000) % 2 == 0, 15.0, -15.0)
        w.loc[near, "bid"] += shift
        w.loc[near, "ask"] += shift
        snap = adapter(raw, spot_source=lambda d: S0).load_snapshot(MARKET_DATE)
        self.assertNotIn(date(2027, 3, 19), {p.expiration for p in snap.forwards})
        self.assertTrue(any("2027-03-19" in m and "descartado" in m for m in snap.report.warnings))
        self.assertNotIn(date(2027, 3, 19), set(snap.options["expiration"]))
        self.assertIn(date(2027, 6, 18), {p.expiration for p in snap.forwards})


class CoverageTests(unittest.TestCase):
    def test_flags_low_days_against_previous_median(self):
        from ibov_barrier.markets.quality import low_coverage_flags
        pts = pd.Series([8700, 8600, 8800, 8700, 8650, 8750, 4500, 8700, 5000])
        out = low_coverage_flags(pts)
        self.assertFalse(out["baixa cobertura"].iloc[:6].any())  # sem histórico ou normais
        self.assertTrue(out.loc[6, "baixa cobertura"])
        self.assertFalse(out.loc[7, "baixa cobertura"])
        self.assertTrue(out.loc[8, "baixa cobertura"])

    def test_flagged_days_do_not_lower_reference(self):
        from ibov_barrier.markets.quality import low_coverage_flags
        # 6 dias normais, 10 dias ruins seguidos e depois um dia parcial: ele precisa ser marcado.
        pts = pd.Series([8700] * 6 + [5000] * 10 + [8700, 5400])
        out = low_coverage_flags(pts)
        self.assertTrue(out["baixa cobertura"].iloc[6:16].all())
        self.assertFalse(out["baixa cobertura"].iloc[16])
        self.assertTrue(out["baixa cobertura"].iloc[17])
        self.assertAlmostEqual(out["referência"].iloc[17], 8700)

    def test_warning_ignores_flagged_history(self):
        from ibov_barrier.markets.quality import coverage_warning
        hist = pd.Series([8700] * 5 + [5000] * 10)
        flagged = pd.Series([False] * 5 + [True] * 10)
        self.assertIsNone(coverage_warning(5400, hist))  # sem as marcas, a referência fica contaminada
        self.assertIn("Baixa cobertura", coverage_warning(5400, hist, flagged))

    def test_warning_message(self):
        from ibov_barrier.markets.quality import coverage_warning
        hist = pd.Series([8700, 8600, 8800, 8700, 8650])
        self.assertIsNone(coverage_warning(8000, hist))
        self.assertIn("Baixa cobertura", coverage_warning(5000, hist))
        self.assertIsNone(coverage_warning(100, hist.head(3)))  # histórico curto: sem referência
