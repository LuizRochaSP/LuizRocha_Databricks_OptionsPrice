"""Rebate nas funções Knock-Out originais e taxa de accrual no Range Accrual."""
import unittest
from math import exp

from ibov_barrier import pricing_barriers as pb
from ibov_barrier.pricing import BarrierContract, MarketData, price_down_and_out_call
from ibov_barrier.pricing_extensions import (
    price_range_accrual,
    price_single_barrier,
    range_accrual_analytical,
)

MARKET = MarketData(spot=150_000, rate=0.12, dividend_yield=0.02, volatility=0.22)
DOWN = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)
UP = BarrierContract(strike=155_000, barrier=180_000, maturity=0.5)
FAST = dict(paths=20_000, steps=63, seed=11)

LEGACY_KO = [
    (price_down_and_out_call, "call", "down", DOWN),
    (pb.price_up_and_out_call, "call", "up", UP),
    (pb.price_down_and_out_put, "put", "down", DOWN),
    (pb.price_up_and_out_put, "put", "up", UP),
]


class TestLegacyKnockOutRebate(unittest.TestCase):

    def test_zero_rebate_keeps_original_price(self):
        for fn, _, _, contract in LEGACY_KO:
            with self.subTest(fn=fn.__name__):
                self.assertEqual(fn(MARKET, contract, rebate=0.0, **FAST).price,
                                 fn(MARKET, contract, **FAST).price)

    def test_rebate_matches_generic_engine_exactly(self):
        for fn, opt, direction, contract in LEGACY_KO:
            with self.subTest(fn=fn.__name__):
                legacy = fn(MARKET, contract, rebate=500.0, **FAST)
                generic = price_single_barrier(MARKET, contract, option_type=opt, direction=direction,
                                               knock="out", rebate=500.0, **FAST)
                self.assertEqual(legacy.price, generic.price)

    def test_rebate_adds_discounted_rebate_times_knock_probability(self):
        for fn, _, _, contract in LEGACY_KO:
            with self.subTest(fn=fn.__name__):
                base = fn(MARKET, contract, **FAST)
                with_rebate = fn(MARKET, contract, rebate=500.0, **FAST)
                expected = base.price + 500.0 * exp(-MARKET.rate * contract.maturity) * base.knock_out_probability
                self.assertAlmostEqual(with_rebate.price, expected,
                                       delta=4 * with_rebate.standard_error + 1.0)

    def test_rebate_with_injected_randoms_rejected(self):
        import numpy as np
        normals = np.zeros((4, 3))
        with self.assertRaises(ValueError):
            price_down_and_out_call(MARKET, DOWN, paths=4, steps=3, normals=normals, rebate=10.0)


class TestRangeAccrualRate(unittest.TestCase):

    KW = dict(lower=135_000, upper=165_000, notional=1_000_000.0, paths=20_000, steps=126, seed=5)

    def test_default_keeps_notional_times_fraction(self):
        old = price_range_accrual(MARKET, 0.5, **self.KW)
        self.assertAlmostEqual(old.price, 1_000_000.0 * exp(-MARKET.rate * 0.5) * old.expected_fraction,
                               places=6)

    def test_accrual_rate_scales_by_rate_times_maturity(self):
        base = price_range_accrual(MARKET, 0.5, **self.KW)
        coupon = price_range_accrual(MARKET, 0.5, accrual_rate=0.10, **self.KW)
        self.assertAlmostEqual(coupon.price, base.price * 0.10 * 0.5, places=6)
        self.assertAlmostEqual(coupon.analytical_price, base.analytical_price * 0.10 * 0.5, places=6)
        self.assertAlmostEqual(coupon.expected_fraction, base.expected_fraction, places=12)

    def test_accrual_rate_matches_analytical(self):
        for maturity in (0.25, 0.5, 1.0):
            with self.subTest(maturity=maturity):
                steps = int(round(maturity * 252))
                kw = dict(self.KW, steps=steps)
                mc = price_range_accrual(MARKET, maturity, accrual_rate=0.10, **kw)
                ref = range_accrual_analytical(MARKET, maturity, lower=135_000, upper=165_000,
                                               notional=1_000_000.0, observations=steps, accrual_rate=0.10)
                self.assertLess(abs(mc.price - ref), 4 * mc.standard_error)

    def test_invalid_accrual_rate(self):
        for bad in (0.0, -0.05, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                price_range_accrual(MARKET, 0.5, accrual_rate=bad, **self.KW)


if __name__ == "__main__":
    unittest.main()
