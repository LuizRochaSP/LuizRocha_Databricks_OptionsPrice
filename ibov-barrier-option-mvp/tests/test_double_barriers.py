"""Testes da Família B completa: DKO, DKI, KIKO e KOKI.

Cada produto é validado contra um benchmark analítico independente
(Ikeda-Kunitomo, Reiner-Rubinstein, identidade KIKO e integral de primeiro
toque para o KOKI) ou contra uma identidade exata nos mesmos caminhos.
"""
import unittest
from math import exp

from ibov_barrier.analytical import price_down_and_out_call_analytical
from ibov_barrier.pricing import MarketData
from scipy.integrate import quad

from ibov_barrier.pricing_double import (
    CONVENTIONS,
    double_barrier_option_analytical,
    exit_density_analytical,
    price_double_barrier_option,
    double_knock_out_analytical,
    kiko_analytical,
    koki_analytical,
    price_double_barrier,
    price_double_knock_in,
    price_double_knock_out,
    price_kiko,
    price_koki,
    single_barrier_out_analytical,
)
from ibov_barrier.pricing_double import _vanilla
from ibov_barrier.pricing_extensions import double_no_touch_probability_analytical, price_double_one_touch

MARKET = MarketData(spot=150_000, rate=0.12, dividend_yield=0.02, volatility=0.22)
T = 0.5
L, U = 120_000, 185_000
ARGS = (MARKET.rate, MARKET.dividend_yield, MARKET.volatility)
MC = dict(paths=60_000, steps=252, seed=7)


def assert_close_to(test, result, benchmark, n_se=4.0, floor=1.0):
    test.assertLess(abs(result.price - benchmark), n_se * result.standard_error + floor,
                    msg=f"MC {result.price:.2f} ± {result.standard_error:.2f} vs {benchmark:.2f}")


class TestAnalyticalBenchmarks(unittest.TestCase):

    def test_single_out_matches_existing_reiner_rubinstein(self):
        for strike in (110_000, 155_000):
            with self.subTest(strike=strike):
                ref = price_down_and_out_call_analytical(MARKET.spot, strike, 120_000, T, *ARGS)
                new = single_barrier_out_analytical(MARKET.spot, strike, 120_000, T, *ARGS,
                                                    option_type="call", direction="down")
                self.assertAlmostEqual(new, ref, places=6)

    def test_dko_collapses_to_single_barrier_when_other_barrier_is_far(self):
        cases = [("call", "down", 120_000, 1e7, 120_000), ("put", "down", 120_000, 1e7, 120_000),
                 ("call", "up", 1.0, 185_000, 185_000), ("put", "up", 1.0, 185_000, 185_000)]
        for opt, direction, lo, hi, barrier in cases:
            with self.subTest(opt=opt, direction=direction):
                dko = double_knock_out_analytical(MARKET.spot, 150_000, lo, hi, T, *ARGS, option_type=opt)
                single = single_barrier_out_analytical(MARKET.spot, 150_000, barrier, T, *ARGS,
                                                       option_type=opt, direction=direction)
                self.assertAlmostEqual(dko, single, delta=1e-6 * max(single, 1.0))

    def test_dko_bounded_by_each_single_knock_out(self):
        for opt in ("call", "put"):
            dko = double_knock_out_analytical(MARKET.spot, 150_000, L, U, T, *ARGS, option_type=opt)
            down = single_barrier_out_analytical(MARKET.spot, 150_000, L, T, *ARGS,
                                                 option_type=opt, direction="down")
            up = single_barrier_out_analytical(MARKET.spot, 150_000, U, T, *ARGS,
                                               option_type=opt, direction="up")
            self.assertLessEqual(dko, min(down, up) + 1e-9)

    def test_kiko_not_above_koki_not_above_knock_in(self):
        ki, ko = 125_000, 175_000
        for opt in ("call", "put"):
            kiko = kiko_analytical(MARKET.spot, 150_000, ki, ko, T, *ARGS, option_type=opt)
            koki = koki_analytical(MARKET.spot, 150_000, ki, ko, T, *ARGS, option_type=opt)
            vanilla = _vanilla(MARKET.spot, 150_000, T, *ARGS, opt)
            knock_in = vanilla - single_barrier_out_analytical(MARKET.spot, 150_000, ki, T, *ARGS,
                                                               option_type=opt, direction="down")
            self.assertLessEqual(kiko, koki + 1e-6)
            self.assertLessEqual(koki, knock_in + 1e-6)


class TestDoubleKnockOutIn(unittest.TestCase):

    def test_dko_matches_ikeda_kunitomo(self):
        for opt in ("call", "put"):
            for strike in (135_000, 155_000):
                with self.subTest(opt=opt, strike=strike):
                    mc = price_double_knock_out(MARKET, strike=strike, lower=L, upper=U, maturity=T,
                                                option_type=opt, **MC)
                    ref = double_knock_out_analytical(MARKET.spot, strike, L, U, T, *ARGS, option_type=opt)
                    assert_close_to(self, mc, ref)

    def test_dko_plus_dki_equals_vanilla_exactly(self):
        for opt in ("call", "put"):
            dko = price_double_knock_out(MARKET, strike=150_000, lower=L, upper=U, maturity=T,
                                         option_type=opt, **MC)
            dki = price_double_knock_in(MARKET, strike=150_000, lower=L, upper=U, maturity=T,
                                        option_type=opt, **MC)
            self.assertAlmostEqual(dko.price + dki.price, dko.vanilla_price, places=6)

    def test_touch_probability_matches_double_one_touch_on_same_paths(self):
        dko = price_double_knock_out(MARKET, strike=150_000, lower=L, upper=U, maturity=T, **MC)
        dot = price_double_one_touch(MARKET, lower=L, upper=U, maturity=T, **MC)
        self.assertAlmostEqual(dko.knock_out_probability, dot.knock_out_probability, places=12)

    def test_rebate_adds_discounted_rebate_times_touch_probability(self):
        base = price_double_knock_out(MARKET, strike=150_000, lower=L, upper=U, maturity=T, **MC)
        with_rebate = price_double_barrier(MARKET, strike=150_000, lower=L, upper=U, maturity=T,
                                           rebate=1_000.0, **MC)
        expected = base.price + 1_000.0 * exp(-MARKET.rate * T) * base.knock_out_probability
        self.assertAlmostEqual(with_rebate.price, expected, delta=3 * with_rebate.standard_error)

    def test_spot_outside_tunnel(self):
        out = price_double_knock_out(MARKET, strike=150_000, lower=155_000, upper=U, maturity=T)
        self.assertEqual(out.price, 0.0)
        dki = price_double_knock_in(MARKET, strike=150_000, lower=155_000, upper=U, maturity=T)
        self.assertAlmostEqual(dki.price, dki.vanilla_price)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            price_double_barrier(MARKET, strike=150_000, lower=U, upper=L, maturity=T)
        with self.assertRaises(ValueError):
            price_double_barrier(MARKET, strike=150_000, lower=L, upper=U, maturity=T, knock="up")
        with self.assertRaises(ValueError):
            price_double_barrier(MARKET, strike=150_000, lower=L, upper=U, maturity=T, rebate=-1.0)


class TestKikoKoki(unittest.TestCase):

    CASES = [("call", 125_000, 175_000), ("call", 175_000, 125_000),
             ("put", 125_000, 175_000), ("put", 175_000, 125_000)]

    def test_kiko_matches_exact_identity(self):
        for opt, ki, ko in self.CASES:
            with self.subTest(opt=opt, ki=ki, ko=ko):
                mc = price_kiko(MARKET, strike=150_000, knock_in=ki, knock_out=ko, maturity=T,
                                option_type=opt, **MC)
                ref = kiko_analytical(MARKET.spot, 150_000, ki, ko, T, *ARGS, option_type=opt)
                assert_close_to(self, mc, ref)

    def test_koki_matches_first_passage_integral(self):
        for opt, ki, ko in self.CASES:
            with self.subTest(opt=opt, ki=ki, ko=ko):
                mc = price_koki(MARKET, strike=150_000, knock_in=ki, knock_out=ko, maturity=T,
                                option_type=opt, **MC)
                ref = koki_analytical(MARKET.spot, 150_000, ki, ko, T, *ARGS, option_type=opt)
                assert_close_to(self, mc, ref)

    def test_koki_dominates_kiko(self):
        # KO antes da ativação mata o KIKO mas não o KOKI: barreiras próximas evidenciam.
        kwargs = dict(strike=150_000, knock_in=135_000, knock_out=160_000, maturity=T, **MC)
        kiko = price_kiko(MARKET, **kwargs)
        koki = price_koki(MARKET, **kwargs)
        self.assertGreater(koki.price, kiko.price)
        ref_gap = (koki_analytical(MARKET.spot, 150_000, 135_000, 160_000, T, *ARGS, option_type="call")
                   - kiko_analytical(MARKET.spot, 150_000, 135_000, 160_000, T, *ARGS, option_type="call"))
        self.assertGreater(ref_gap, 0.0)

    def test_every_convention_matches_its_benchmark(self):
        for convention in CONVENTIONS:
            for opt, ki, ko in self.CASES:
                with self.subTest(convention=convention, opt=opt, ki=ki, ko=ko):
                    mc = price_double_barrier_option(
                        MARKET, strike=150_000, knock_in=ki, knock_out=ko, maturity=T,
                        option_type=opt, convention=convention, **MC,
                    )
                    ref = double_barrier_option_analytical(
                        MARKET.spot, 150_000, ki, ko, T, *ARGS, option_type=opt, convention=convention,
                    )
                    assert_close_to(self, mc, ref)

    def test_exit_densities_integrate_to_touch_probability(self):
        for lo, hi in ((L, U), (140_000, 160_000)):
            mass = sum(
                quad(lambda t, s=side: exit_density_analytical(t, MARKET.spot, lo, hi, *ARGS, side=s),
                     0.0, T, limit=200)[0]
                for side in ("lower", "upper")
            )
            expected = 1.0 - double_no_touch_probability_analytical(MARKET.spot, lo, hi, T, *ARGS)
            self.assertAlmostEqual(mass, expected, places=6)

    def test_koki_plus_sequential_is_knock_in_on_same_paths(self):
        alive = {}
        for convention in ("KO_after_KI", "sequential"):
            r = price_double_barrier_option(MARKET, strike=150_000, knock_in=125_000, knock_out=175_000,
                                            maturity=T, convention=convention, **MC)
            alive[convention] = 1.0 - r.knock_out_probability
        dko = price_double_knock_out(MARKET, strike=150_000, lower=125_000, upper=175_000, maturity=T, **MC)
        kiko = price_kiko(MARKET, strike=150_000, knock_in=125_000, knock_out=175_000, maturity=T, **MC)
        # P(KI tocado) = P(KOKI viva) + P(sequential viva), exato nos mesmos caminhos.
        p_ki = alive["KO_after_KI"] + alive["sequential"]
        self.assertGreaterEqual(p_ki, 1.0 - kiko.knock_out_probability)
        self.assertLessEqual(p_ki, dko.knock_out_probability + 1e-12)

    def test_ko_first_spans_knock_out_and_vanilla(self):
        # KO_first vale mais que o KO simples (o KI pode "desligar" o KO) e menos que a vanilla.
        for opt, ki, ko in self.CASES:
            with self.subTest(opt=opt, ki=ki, ko=ko):
                ko_first = double_barrier_option_analytical(MARKET.spot, 150_000, ki, ko, T, *ARGS,
                                                            option_type=opt, convention="KO_first")
                single_ko = single_barrier_out_analytical(MARKET.spot, 150_000, ko, T, *ARGS, option_type=opt,
                                                          direction="up" if ko > MARKET.spot else "down")
                vanilla = _vanilla(MARKET.spot, 150_000, T, *ARGS, opt)
                self.assertGreater(ko_first, single_ko)
                self.assertLess(ko_first, vanilla)

    def test_unknown_convention_rejected(self):
        with self.assertRaises(ValueError):
            price_double_barrier_option(MARKET, strike=150_000, knock_in=125_000, knock_out=175_000,
                                        maturity=T, convention="KO_last")

    def test_barriers_on_same_side_rejected(self):
        with self.assertRaises(ValueError):
            price_kiko(MARKET, strike=150_000, knock_in=120_000, knock_out=130_000, maturity=T)
        with self.assertRaises(ValueError):
            koki_analytical(MARKET.spot, 150_000, 160_000, 170_000, T, *ARGS, option_type="call")


if __name__ == "__main__":
    unittest.main()
