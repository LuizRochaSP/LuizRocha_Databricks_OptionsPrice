"""Testes das barreiras dependentes da trajetória: Parisian, Step e Soft (janela).

O Monte Carlo é validado contra benchmarks semi-analíticos independentes
(cadeia de Markov discreta para a Parisian; densidade de transição morta na
barreira, encadeada por segmentos, para Step e Soft), contra Reiner-Rubinstein
e contra identidades exatas nos mesmos caminhos.
"""
import unittest

from ibov_barrier.pricing import BarrierContract, MarketData
from ibov_barrier.pricing_double import _vanilla, single_barrier_out_analytical
from ibov_barrier.pricing_extensions import price_single_barrier
from ibov_barrier.pricing_path_barriers import (
    parisian_barrier_quadrature,
    price_parisian_barrier,
    price_soft_barrier,
    price_step_barrier,
    soft_barrier_quadrature,
    step_barrier_quadrature,
)

S0, K, T, R, Q, VOL = 150_000.0, 150_000.0, 0.5, 0.12, 0.02, 0.22
MARKET = MarketData(spot=S0, rate=R, dividend_yield=Q, volatility=VOL)
STEPS = 126
FAST = dict(paths=40_000, steps=STEPS, seed=7)
CASES = [("call", "down", 130_000.0), ("put", "up", 170_000.0)]
MKT = (T, R, Q, VOL)


class TestParisian(unittest.TestCase):

    def test_window_one_equals_discrete_single_barrier(self):
        for opt, direction, h in CASES:
            for knock in ("out", "in"):
                with self.subTest(opt=opt, knock=knock):
                    par = price_parisian_barrier(MARKET, strike=K, barrier=h, maturity=T, window=1,
                                                 option_type=opt, direction=direction, knock=knock, **FAST)
                    single = price_single_barrier(MARKET, BarrierContract(K, h, T), option_type=opt,
                                                  direction=direction, knock=knock, monitoring="discrete",
                                                  **FAST)
                    self.assertEqual(par.price, single.price)

    def test_matches_markov_chain_quadrature(self):
        for opt, direction, h in CASES:
            for knock in ("out", "in"):
                for window, cumulative in ((5, False), (10, True)):
                    with self.subTest(opt=opt, knock=knock, window=window, cumulative=cumulative):
                        kw = dict(window=window, cumulative=cumulative, option_type=opt,
                                  direction=direction, knock=knock, rebate=300.0)
                        mc = price_parisian_barrier(MARKET, strike=K, barrier=h, maturity=T, **kw, **FAST)
                        ref = parisian_barrier_quadrature(S0, K, h, *MKT, steps=STEPS, **kw)
                        self.assertLess(abs(mc.price - ref), 4 * mc.standard_error)

    def test_knock_in_plus_knock_out_is_vanilla(self):
        for opt, direction, h in CASES:
            with self.subTest(opt=opt):
                kw = dict(strike=K, barrier=h, maturity=T, window=5, option_type=opt, direction=direction)
                out = price_parisian_barrier(MARKET, knock="out", **kw, **FAST)
                kin = price_parisian_barrier(MARKET, knock="in", **kw, **FAST)
                self.assertAlmostEqual(out.price + kin.price, out.vanilla_price, places=6)

    def test_longer_window_is_worth_more_and_cumulative_less(self):
        kw = dict(steps=STEPS, option_type="call", direction="down")
        values = [parisian_barrier_quadrature(S0, K, 130_000.0, *MKT, window=w, **kw) for w in (1, 3, 10, 30)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[-1], _vanilla(S0, K, *MKT, "call"))
        self.assertLess(parisian_barrier_quadrature(S0, K, 130_000.0, *MKT, window=10, cumulative=True, **kw),
                        parisian_barrier_quadrature(S0, K, 130_000.0, *MKT, window=10, **kw))

    def test_invalid_window(self):
        for bad in (0, -2, 2.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                price_parisian_barrier(MARKET, strike=K, barrier=130_000.0, maturity=T, window=bad, **FAST)


class TestStepBarrier(unittest.TestCase):

    def test_single_level_equals_single_barrier(self):
        for opt, direction, h in CASES:
            for knock in ("out", "in"):
                with self.subTest(opt=opt, knock=knock):
                    step = price_step_barrier(MARKET, strike=K, schedule=[(0.0, h)], maturity=T,
                                              option_type=opt, direction=direction, knock=knock, **FAST)
                    single = price_single_barrier(MARKET, BarrierContract(K, h, T), option_type=opt,
                                                  direction=direction, knock=knock, **FAST)
                    self.assertAlmostEqual(step.price, single.price, places=8)

    def test_quadrature_single_level_matches_reiner_rubinstein(self):
        for opt, direction, h in CASES:
            with self.subTest(opt=opt):
                quad = step_barrier_quadrature(S0, K, *MKT, schedule=[(0.0, h)], option_type=opt,
                                               direction=direction)
                rr = single_barrier_out_analytical(S0, K, h, *MKT, option_type=opt, direction=direction)
                self.assertAlmostEqual(quad, rr, delta=1e-4 * rr)

    def test_two_levels_match_quadrature(self):
        for opt, direction, h in CASES:
            second = h * (1.08 if direction == "down" else 0.95)
            schedule = [(0.0, h), (T / 2, second)]
            for knock in ("out", "in"):
                with self.subTest(opt=opt, knock=knock):
                    kw = dict(option_type=opt, direction=direction, knock=knock, rebate=300.0)
                    mc = price_step_barrier(MARKET, strike=K, schedule=schedule, maturity=T, **kw, **FAST)
                    ref = step_barrier_quadrature(S0, K, *MKT, schedule=schedule, **kw)
                    self.assertLess(abs(mc.price - ref), 4 * mc.standard_error)

    def test_tighter_step_is_worth_less(self):
        base = step_barrier_quadrature(S0, K, *MKT, schedule=[(0.0, 130_000.0)])
        tighter = step_barrier_quadrature(S0, K, *MKT, schedule=[(0.0, 130_000.0), (T / 2, 140_000.0)])
        looser = step_barrier_quadrature(S0, K, *MKT, schedule=[(0.0, 130_000.0), (T / 2, 120_000.0)])
        self.assertLess(tighter, base)
        self.assertGreater(looser, base)

    def test_invalid_schedule(self):
        bad_schedules = ([], [(0.1, 130_000.0)], [(0.0, 130_000.0), (0.0, 120_000.0)],
                         [(0.0, 130_000.0), (T, 120_000.0)], [(0.0, -1.0)])
        for schedule in bad_schedules:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                price_step_barrier(MARKET, strike=K, schedule=schedule, maturity=T, **FAST)


class TestSoftBarrier(unittest.TestCase):

    def test_full_window_equals_single_barrier(self):
        for opt, direction, h in CASES:
            for knock in ("out", "in"):
                with self.subTest(opt=opt, knock=knock):
                    soft = price_soft_barrier(MARKET, strike=K, barrier=h, maturity=T, windows=[(0.0, T)],
                                              option_type=opt, direction=direction, knock=knock, **FAST)
                    single = price_single_barrier(MARKET, BarrierContract(K, h, T), option_type=opt,
                                                  direction=direction, knock=knock, **FAST)
                    self.assertAlmostEqual(soft.price, single.price, places=8)

    def test_windows_match_quadrature(self):
        windows_list = ([(0.0, T / 2)], [(T / 2, T)], [(0.0, T / 4), (T / 2, 3 * T / 4)])
        for opt, direction, h in CASES:
            for windows in windows_list:
                for knock in ("out", "in"):
                    with self.subTest(opt=opt, windows=windows, knock=knock):
                        kw = dict(option_type=opt, direction=direction, knock=knock, rebate=300.0)
                        mc = price_soft_barrier(MARKET, strike=K, barrier=h, maturity=T, windows=windows,
                                                **kw, **FAST)
                        ref = soft_barrier_quadrature(S0, K, h, *MKT, windows=windows, **kw)
                        self.assertLess(abs(mc.price - ref), 4 * mc.standard_error)

    def test_window_value_between_full_barrier_and_vanilla(self):
        for opt, direction, h in CASES:
            with self.subTest(opt=opt):
                full = single_barrier_out_analytical(S0, K, h, *MKT, option_type=opt, direction=direction)
                part = soft_barrier_quadrature(S0, K, h, *MKT, windows=[(T / 2, T)], option_type=opt,
                                               direction=direction)
                self.assertLess(full, part)
                self.assertLess(part, _vanilla(S0, K, *MKT, opt))

    def test_invalid_windows(self):
        for windows in ([], [(0.3, 0.2)], [(0.0, T + 0.1)], [(0.0, 0.3), (0.2, 0.4)]):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                price_soft_barrier(MARKET, strike=K, barrier=130_000.0, maturity=T, windows=windows, **FAST)


if __name__ == "__main__":
    unittest.main()
