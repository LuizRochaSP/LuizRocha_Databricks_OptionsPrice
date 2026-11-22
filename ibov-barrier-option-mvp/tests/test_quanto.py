import unittest
from dataclasses import replace
from math import exp

import numpy as np
import pandas as pd

from ibov_barrier.analytical import price_down_and_out_call_analytical
from ibov_barrier.pricing import BarrierContract, black_scholes_call
from ibov_barrier.pricing_quanto import (
    QuantoMarket,
    historical_fx_inputs,
    price_compo_vanilla,
    price_converted_vanilla,
    price_quanto_vanilla,
    price_two_factor_mc,
    quanto_forward,
    quanto_market_data,
)

# Ordem de grandeza de nov/2026: SPX ~7.000, r_usd ~ SOFR + 90 bps, DI ~ 14%, vol USDBRL ~ 13%.
M = QuantoMarket(spot=7000.0, rate_usd=0.048, dividend_yield=0.009, vol_asset=0.18, fx_spot=5.40,
                 rate_brl=0.135, vol_fx=0.13, correlation=-0.40)
K, T = 7100.0, 0.75


class QuantoClosedFormTests(unittest.TestCase):
    def test_no_correlation_and_equal_rates_is_converted_option(self):
        m = replace(M, correlation=0.0, rate_brl=M.rate_usd)
        self.assertAlmostEqual(price_quanto_vanilla(m, K, T), price_converted_vanilla(m, K, T), places=8)

    def test_put_call_parity(self):
        call = price_quanto_vanilla(M, K, T, option_type="call", fx_fixed=1.0)
        put = price_quanto_vanilla(M, K, T, option_type="put", fx_fixed=1.0)
        self.assertAlmostEqual(call - put, exp(-M.rate_brl * T) * (quanto_forward(M, T) - K), places=8)

    def test_negative_correlation_raises_quanto_forward(self):
        # rho < 0: ajuste -rho sigma_s sigma_x > 0, forward Quanto acima do forward em USD.
        self.assertGreater(M.quanto_adjustment, 0)
        self.assertAlmostEqual(M.quanto_adjustment, 0.40 * 0.18 * 0.13, places=12)
        high = price_quanto_vanilla(M, K, T)
        low = price_quanto_vanilla(replace(M, correlation=0.40), K, T)
        self.assertGreater(high, low)

    def test_market_data_mapping_reuses_engine(self):
        md = quanto_market_data(M)
        engine = black_scholes_call(md, BarrierContract(strike=K, barrier=1.0, maturity=T))
        self.assertAlmostEqual(M.fx_spot * engine, price_quanto_vanilla(M, K, T), places=6)
        self.assertAlmostEqual(md.rate - md.dividend_yield, M.rate_usd - M.dividend_yield + M.quanto_adjustment)

    def test_compo_parity_and_volatility(self):
        call = price_compo_vanilla(M, K * M.fx_spot, T, option_type="call")
        put = price_compo_vanilla(M, K * M.fx_spot, T, option_type="put")
        expected = M.spot * M.fx_spot * exp(-M.dividend_yield * T) - K * M.fx_spot * exp(-M.rate_brl * T)
        self.assertAlmostEqual(call - put, expected, places=6)
        self.assertAlmostEqual(M.compo_volatility, np.sqrt(0.18**2 + 0.13**2 - 2 * 0.4 * 0.18 * 0.13))

    def test_compo_without_fx_vol_is_black_scholes_in_brl(self):
        m = replace(M, vol_fx=0.0)
        md = replace(quanto_market_data(m), spot=m.spot * m.fx_spot, dividend_yield=m.dividend_yield)
        bs = black_scholes_call(md, BarrierContract(strike=K * m.fx_spot, barrier=1.0, maturity=T))
        self.assertAlmostEqual(price_compo_vanilla(m, K * m.fx_spot, T), bs, places=6)

    def test_converted_does_not_depend_on_fx_parameters(self):
        base = price_converted_vanilla(M, K, T)
        self.assertAlmostEqual(base, price_converted_vanilla(replace(M, correlation=0.9, vol_fx=0.3), K, T))

    def test_validation(self):
        with self.assertRaises(ValueError):
            replace(M, correlation=1.2)
        with self.assertRaises(ValueError):
            replace(M, fx_spot=0.0)
        with self.assertRaises(ValueError):
            price_quanto_vanilla(M, K, T, option_type="digital")
        with self.assertRaises(ValueError):
            price_two_factor_mc(M, K, T, product="flexo")


class TwoFactorMonteCarloTests(unittest.TestCase):
    def assertWithinMC(self, result, target, n_se=4.0, rel=0.0):
        tol = n_se * result.standard_error + rel * abs(target)
        self.assertLess(abs(result.price - target), tol, f"MC {result.price:.4f} vs {target:.4f} (tol {tol:.4f})")

    def test_simulated_dynamics_match_closed_forms(self):
        """Payoff descontado cru (sem variável de controle) bate com as três fórmulas fechadas."""
        closed = {
            ("quanto", "call"): price_quanto_vanilla(M, K, T),
            ("quanto", "put"): price_quanto_vanilla(M, K, T, option_type="put"),
            ("compo", "call"): price_compo_vanilla(M, K * M.fx_spot, T),
            ("compo", "put"): price_compo_vanilla(M, K * M.fx_spot, T, option_type="put"),
            ("converted", "call"): price_converted_vanilla(M, K, T),
            ("converted", "put"): price_converted_vanilla(M, K, T, option_type="put"),
        }
        for (product, option_type), target in closed.items():
            strike = K * M.fx_spot if product == "compo" else K
            mean, se = _raw_mc(M, strike, T, product, option_type)
            self.assertLess(abs(mean - target), 4 * se, f"{product} {option_type}: {mean:.4f} vs {target:.4f}")

    def test_converted_down_and_out_matches_usd_analytical(self):
        """X_T * DOC(S) vale X_0 * DOC em USD: valida a simulação conjunta de S e X com barreira."""
        barrier = 6000.0
        closed = M.fx_spot * price_down_and_out_call_analytical(M.spot, K, barrier, T, M.rate_usd,
                                                                M.dividend_yield, M.vol_asset)
        mc = price_two_factor_mc(M, K, T, product="converted", barrier=barrier, paths=200_000, steps=126,
                                 seed=9)
        self.assertWithinMC(mc, closed, rel=0.005)

    def test_quanto_down_and_out_matches_analytical_with_mapping(self):
        md = quanto_market_data(M)
        barrier = 6000.0
        closed = M.fx_spot * price_down_and_out_call_analytical(M.spot, K, barrier, T, md.rate,
                                                                md.dividend_yield, md.volatility)
        mc = price_two_factor_mc(M, K, T, product="quanto", barrier=barrier, paths=200_000, steps=126, seed=3)
        self.assertWithinMC(mc, closed, rel=0.005)
        self.assertLess(mc.price, mc.vanilla_price)
        self.assertGreater(mc.knock_out_probability, 0.0)

    def test_breached_barrier_is_worth_zero(self):
        mc = price_two_factor_mc(M, K, T, barrier=7100.0, barrier_direction="down", paths=100, steps=2)
        self.assertEqual(mc.price, 0.0)
        self.assertEqual(mc.knock_out_probability, 1.0)

    def test_up_and_out_compo_is_below_vanilla(self):
        mc = price_two_factor_mc(M, K * M.fx_spot, T, product="compo", barrier=8500.0, barrier_direction="up",
                                 paths=50_000, steps=63, seed=5)
        self.assertLess(mc.price, mc.vanilla_price)
        self.assertGreater(mc.price, 0.0)


def _raw_mc(m, strike, maturity, product, option_type, paths=400_000, seed=21):
    """Payoff descontado sem variável de controle, simulado direto no vencimento."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(paths)
    z2 = m.correlation * z1 + np.sqrt(1 - m.correlation**2) * rng.standard_normal(paths)
    s = m.spot * np.exp((m.rate_usd - m.dividend_yield + m.quanto_adjustment - 0.5 * m.vol_asset**2) * maturity
                        + m.vol_asset * np.sqrt(maturity) * z1)
    x = m.fx_spot * np.exp((m.rate_brl - m.rate_usd - 0.5 * m.vol_fx**2) * maturity
                           + m.vol_fx * np.sqrt(maturity) * z2)
    sign = 1.0 if option_type == "call" else -1.0
    payoff = {"quanto": m.fx_spot * np.maximum(sign * (s - strike), 0),
              "compo": np.maximum(sign * (s * x - strike), 0),
              "converted": x * np.maximum(sign * (s - strike), 0)}[product]
    pv = np.exp(-m.rate_brl * maturity) * payoff
    return float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(paths))


class HistoricalInputsTests(unittest.TestCase):
    def test_recovers_vol_and_correlation_from_simulated_closes(self):
        rng = np.random.default_rng(0)
        n, dt, rho = 252 * 6, 1 / 252, -0.4
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        idx = pd.bdate_range("2020-01-01", periods=n)
        s = pd.Series(7000 * np.exp(np.cumsum(0.18 * np.sqrt(dt) * z1)), index=idx)
        x = pd.Series(5.4 * np.exp(np.cumsum(0.13 * np.sqrt(dt) * z2)), index=idx)
        weekly = historical_fx_inputs(s, x)
        daily = historical_fx_inputs(s, x, frequency=None)
        for out, tol in [(weekly, 0.08), (daily, 0.04)]:
            self.assertAlmostEqual(out["correlation"], rho, delta=tol)
            self.assertAlmostEqual(out["vol_fx"], 0.13, delta=0.015)
            self.assertAlmostEqual(out["vol_asset"], 0.18, delta=0.02)
        self.assertGreater(daily["observations"], 4 * weekly["observations"])

    def test_needs_enough_history(self):
        idx = pd.bdate_range("2026-01-01", periods=5)
        with self.assertRaises(ValueError):
            historical_fx_inputs(pd.Series(range(1, 6), index=idx), pd.Series(range(1, 6), index=idx),
                                 frequency=None)


if __name__ == "__main__":
    unittest.main()
