"""Testes das lacunas fáceis e médias do roadmap.

Rebate (A), variantes Up de Collar/Spreads/Strangle (C), Knock-In Straddle e
Forward (C), One/No-Touch Down (D), Range Accrual (D) e Double One/No-Touch (D).
Cada produto é validado contra um benchmark analítico ou uma identidade exata.
"""
import unittest
from math import exp

from ibov_barrier.pricing import BarrierContract, MarketData, price_down_and_out_call
from ibov_barrier import pricing_barriers as pb
from ibov_barrier.pricing_extensions import (
    barrier_hit_probability_analytical,
    double_no_touch_probability_analytical,
    price_double_no_touch,
    price_double_one_touch,
    price_knock_in_forward,
    price_knock_in_straddle,
    price_no_touch,
    price_no_touch_down,
    price_one_touch,
    price_one_touch_down,
    price_range_accrual,
    price_single_barrier,
    range_accrual_analytical,
)

MARKET = MarketData(spot=150_000, rate=0.12, dividend_yield=0.02, volatility=0.22)
DOWN = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)
UP = BarrierContract(strike=155_000, barrier=180_000, maturity=0.5)
FAST = dict(paths=20_000, steps=63, seed=11)
DISCOUNT = exp(-MARKET.rate * DOWN.maturity)


def hit_probability(barrier: float) -> float:
    return barrier_hit_probability_analytical(
        MARKET.spot, barrier, DOWN.maturity, MARKET.rate, MARKET.dividend_yield, MARKET.volatility
    )


class TestSingleBarrierWithRebate(unittest.TestCase):

    def test_zero_rebate_reproduces_all_eight_legacy_pricers(self):
        legacy = {
            ("call", "down", "out", DOWN): price_down_and_out_call,
            ("call", "down", "in", DOWN): pb.price_down_and_in_call,
            ("put", "down", "out", DOWN): pb.price_down_and_out_put,
            ("put", "down", "in", DOWN): pb.price_down_and_in_put,
            ("call", "up", "out", UP): pb.price_up_and_out_call,
            ("call", "up", "in", UP): pb.price_up_and_in_call,
            ("put", "up", "out", UP): pb.price_up_and_out_put,
            ("put", "up", "in", UP): pb.price_up_and_in_put,
        }
        for monitoring in ("brownian_bridge", "discrete"):
            for (opt, direction, knock, contract), fn in legacy.items():
                with self.subTest(opt=opt, direction=direction, knock=knock, monitoring=monitoring):
                    old = fn(MARKET, contract, monitoring=monitoring, **FAST)
                    new = price_single_barrier(
                        MARKET, contract, option_type=opt, direction=direction, knock=knock,
                        monitoring=monitoring, **FAST,
                    )
                    self.assertEqual(old.price, new.price)
                    self.assertEqual(old.standard_error, new.standard_error)
                    self.assertEqual(old.knock_out_probability, new.knock_out_probability)

    def test_knock_out_rebate_adds_discounted_hit_probability(self):
        rebate = 1_000.0
        base = price_single_barrier(MARKET, DOWN, paths=100_000, steps=126, seed=3)
        with_rebate = price_single_barrier(MARKET, DOWN, rebate=rebate, paths=100_000, steps=126, seed=3)
        expected = rebate * DISCOUNT * hit_probability(DOWN.barrier)
        self.assertAlmostEqual(with_rebate.price - base.price, expected, delta=0.03 * expected)

    def test_knock_in_rebate_paid_when_never_activated(self):
        rebate = 1_000.0
        base = price_single_barrier(MARKET, DOWN, knock="in", paths=100_000, steps=126, seed=3)
        with_rebate = price_single_barrier(MARKET, DOWN, knock="in", rebate=rebate,
                                           paths=100_000, steps=126, seed=3)
        expected = rebate * DISCOUNT * (1.0 - hit_probability(DOWN.barrier))
        self.assertAlmostEqual(with_rebate.price - base.price, expected, delta=0.01 * expected)

    def test_already_knocked_out_pays_only_discounted_rebate(self):
        dead = BarrierContract(strike=155_000, barrier=160_000, maturity=0.5)
        result = price_single_barrier(MARKET, dead, rebate=500.0, **FAST)
        self.assertAlmostEqual(result.price, 500.0 * DISCOUNT)
        self.assertEqual(result.knock_out_probability, 1.0)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            price_single_barrier(MARKET, DOWN, rebate=-1.0, **FAST)
        with self.assertRaises(ValueError):
            price_single_barrier(MARKET, DOWN, option_type="digital", **FAST)
        with self.assertRaises(ValueError):
            price_single_barrier(MARKET, DOWN, direction="sideways", **FAST)


class TestTouchBinaries(unittest.TestCase):

    def test_generic_up_matches_legacy_one_touch_up(self):
        legacy = pb.price_one_touch_up(MARKET, UP, 100.0, **FAST)
        new = price_one_touch(MARKET, UP, direction="up", payout=100.0, **FAST)
        self.assertEqual(legacy.price, new.price)

    def test_one_touch_down_matches_first_passage_formula(self):
        result = price_one_touch_down(MARKET, DOWN, payout=100.0, paths=200_000, steps=252)
        expected = 100.0 * DISCOUNT * hit_probability(DOWN.barrier)
        self.assertAlmostEqual(result.price, expected, delta=4 * result.standard_error)

    def test_one_touch_plus_no_touch_equals_discounted_payout(self):
        ot = price_one_touch_down(MARKET, DOWN, payout=250.0, **FAST)
        nt = price_no_touch_down(MARKET, DOWN, payout=250.0, **FAST)
        self.assertAlmostEqual(ot.price + nt.price, 250.0 * DISCOUNT, places=8)
        self.assertAlmostEqual(ot.knock_out_probability + nt.knock_out_probability, 1.0)

    def test_down_barrier_already_touched(self):
        touched = BarrierContract(strike=1.0, barrier=150_000, maturity=0.5)
        self.assertAlmostEqual(price_one_touch_down(MARKET, touched, **FAST).price, 100.0 * DISCOUNT)
        self.assertAlmostEqual(price_no_touch(MARKET, touched, direction="down", **FAST).price, 0.0)


class TestKnockInStructures(unittest.TestCase):

    def test_knock_in_straddle_is_sum_of_legs(self):
        straddle = price_knock_in_straddle(MARKET, DOWN, **FAST)
        call = pb.price_down_and_in_call(MARKET, DOWN, **FAST)
        put = pb.price_down_and_in_put(MARKET, DOWN, **FAST)
        self.assertAlmostEqual(straddle.price, call.price + put.price, places=8)

    def test_in_plus_out_forward_equals_vanilla_forward(self):
        kif = price_knock_in_forward(MARKET, DOWN, **FAST)
        kof = pb.price_knock_out_forward(MARKET, DOWN, **FAST)
        vanilla_forward = (MARKET.spot * exp(-MARKET.dividend_yield * DOWN.maturity)
                           - DOWN.strike * DISCOUNT)
        self.assertAlmostEqual(kif.price + kof.price, vanilla_forward, delta=1e-6 * MARKET.spot)

    def test_up_knock_in_straddle(self):
        straddle = price_knock_in_straddle(MARKET, UP, direction="up", **FAST)
        call = pb.price_up_and_in_call(MARKET, UP, **FAST)
        put = pb.price_up_and_in_put(MARKET, UP, **FAST)
        self.assertAlmostEqual(straddle.price, call.price + put.price, places=8)


class TestUpVariantsOfCombinedStructures(unittest.TestCase):

    def test_up_structures_use_up_and_out_legs(self):
        k2 = 140_000.0
        uoc_k1 = pb.price_up_and_out_call(MARKET, UP, **FAST).price
        uop_k1 = pb.price_up_and_out_put(MARKET, UP, **FAST).price
        uoc_k2 = pb.price_up_and_out_call(MARKET, BarrierContract(k2, UP.barrier, UP.maturity), **FAST).price
        uop_k2 = pb.price_up_and_out_put(MARKET, BarrierContract(k2, UP.barrier, UP.maturity), **FAST).price
        expected = {
            pb.price_barrier_strangle: uoc_k1 + uop_k2,
            pb.price_barrier_collar: uop_k1 - uoc_k2,
            pb.price_barrier_call_spread: uoc_k1 - uoc_k2,
            pb.price_barrier_put_spread: uop_k1 - uop_k2,
        }
        for fn, value in expected.items():
            with self.subTest(fn=fn.__name__):
                self.assertAlmostEqual(fn(MARKET, UP, k2, direction="up", **FAST).price, value, places=8)

    def test_default_direction_is_unchanged(self):
        k2 = 140_000.0
        doc = price_down_and_out_call(MARKET, DOWN, **FAST).price
        dop_k2 = pb.price_down_and_out_put(MARKET, BarrierContract(k2, DOWN.barrier, DOWN.maturity), **FAST).price
        self.assertAlmostEqual(pb.price_barrier_strangle(MARKET, DOWN, k2, **FAST).price, doc + dop_k2, places=8)

    def test_invalid_direction_rejected(self):
        with self.assertRaises(ValueError):
            pb.price_barrier_collar(MARKET, DOWN, 160_000, direction="flat", **FAST)


class TestRangeAccrual(unittest.TestCase):

    def test_corridor_matches_exact_expectation(self):
        result = price_range_accrual(MARKET, 0.5, lower=135_000, upper=170_000, notional=100.0,
                                     paths=100_000, steps=126)
        self.assertAlmostEqual(result.price, result.analytical_price, delta=4 * result.standard_error)
        self.assertTrue(0.0 < result.expected_fraction < 1.0)

    def test_down_and_up_single_sided_are_complementary(self):
        level = 145_000
        down = price_range_accrual(MARKET, 0.5, lower=level, upper=None, **FAST)
        up = price_range_accrual(MARKET, 0.5, lower=None, upper=level, **FAST)
        self.assertAlmostEqual(down.price + up.price, 100.0 * DISCOUNT, places=8)
        self.assertAlmostEqual(down.analytical_price + up.analytical_price, 100.0 * DISCOUNT, places=8)

    def test_infinite_corridor_accrues_everything(self):
        value = range_accrual_analytical(MARKET, 0.5, lower=1.0, upper=1e12, notional=100.0, observations=10)
        self.assertAlmostEqual(value, 100.0 * DISCOUNT, places=6)

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            price_range_accrual(MARKET, 0.5, lower=None, upper=None, **FAST)
        with self.assertRaises(ValueError):
            price_range_accrual(MARKET, 0.5, lower=170_000, upper=135_000, **FAST)


class TestDoubleTouch(unittest.TestCase):
    LOWER, UPPER = 135_000.0, 170_000.0

    def analytic_dnt(self) -> float:
        return double_no_touch_probability_analytical(
            MARKET.spot, self.LOWER, self.UPPER, 0.5, MARKET.rate, MARKET.dividend_yield, MARKET.volatility
        )

    def test_fourier_series_collapses_to_single_barrier(self):
        series = double_no_touch_probability_analytical(
            MARKET.spot, DOWN.barrier, 1e9, 0.5, MARKET.rate, MARKET.dividend_yield, MARKET.volatility
        )
        self.assertAlmostEqual(series, 1.0 - hit_probability(DOWN.barrier), places=8)

    def test_double_no_touch_matches_fourier_series(self):
        result = price_double_no_touch(MARKET, lower=self.LOWER, upper=self.UPPER, maturity=0.5,
                                       paths=200_000, steps=252)
        expected = 100.0 * DISCOUNT * self.analytic_dnt()
        self.assertAlmostEqual(result.price, expected, delta=4 * result.standard_error)

    def test_double_one_touch_plus_no_touch_equals_payout(self):
        kwargs = dict(lower=self.LOWER, upper=self.UPPER, maturity=0.5, payout=100.0, **FAST)
        dot = price_double_one_touch(MARKET, **kwargs)
        dnt = price_double_no_touch(MARKET, **kwargs)
        self.assertAlmostEqual(dot.price + dnt.price, 100.0 * DISCOUNT, places=8)

    def test_spot_outside_corridor(self):
        kwargs = dict(lower=155_000, upper=170_000, maturity=0.5, **FAST)
        self.assertAlmostEqual(price_double_one_touch(MARKET, **kwargs).price, 100.0 * DISCOUNT)
        self.assertAlmostEqual(price_double_no_touch(MARKET, **kwargs).price, 0.0)


if __name__ == "__main__":
    unittest.main()
