import unittest
from dataclasses import replace
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

from ibov_barrier.analytical import price_down_and_out_call_analytical
from ibov_barrier.local_vol import LocalVolSurface, fit_svi, svi_total_variance
from ibov_barrier.pricing_quanto import QuantoMarket, price_compo_vanilla, price_two_factor_mc, quanto_market_data

M = QuantoMarket(spot=7000.0, rate_usd=0.048, dividend_yield=0.009, vol_asset=0.20, fx_spot=5.40,
                 rate_brl=0.135, vol_fx=0.13, correlation=-0.30)
T = 0.5
BASE = np.array([0.02, 0.08, -0.7, 0.05, 0.15])     # SVI de 1 ano com skew de índice


def scaled(t):
    # w(k, T) = T * w_1y(k): sem arbitragem de calendário e com skew parecido com o do SPX.
    return BASE * np.array([t, t, 1, 1, 1])


def skew_surface():
    return LocalVolSurface.from_svi_slices({t: scaled(t) for t in (0.1, 0.25, 0.5, 1.0)})


def bs_usd(s, k, t, r, q, vol):
    f = s * exp((r - q) * t)
    d1 = (log(f / k) + 0.5 * vol * vol * t) / (vol * sqrt(t))
    return exp(-r * t) * (f * norm.cdf(d1) - k * norm.cdf(d1 - vol * sqrt(t)))


class SviTests(unittest.TestCase):
    def test_fit_recovers_smile(self):
        k = np.linspace(-0.5, 0.3, 25)
        iv = np.sqrt(svi_total_variance(scaled(0.5), k) / 0.5)
        params, rmse = fit_svi(k, iv, 0.5)
        self.assertLess(rmse, 1e-4)
        np.testing.assert_allclose(np.sqrt(svi_total_variance(params, k) / 0.5), iv, atol=1e-4)

    def test_from_options_and_implied_roundtrip(self):
        rows = [{"T": t, "log_moneyness": k, "implied_vol": sqrt(svi_total_variance(scaled(t), k) / t)}
                for t in (0.25, 0.5, 1.0) for k in np.linspace(-0.4, 0.2, 15)]
        lv = LocalVolSurface.from_options(pd.DataFrame(rows))
        self.assertEqual(len(lv.fit_report), 3)
        expected = sqrt(svi_total_variance(scaled(0.5), -0.1) / 0.5)
        self.assertAlmostEqual(float(lv.implied_vol(-0.1, 0.5)), expected, places=3)

    def test_needs_two_expiries(self):
        rows = [{"T": 0.5, "log_moneyness": k, "implied_vol": 0.2} for k in np.linspace(-0.3, 0.2, 10)]
        with self.assertRaises(ValueError):
            LocalVolSurface.from_options(pd.DataFrame(rows))


class DupireTests(unittest.TestCase):
    def test_flat_surface_gives_flat_local_vol(self):
        lv = LocalVolSurface.flat(0.2)
        for t in (0.05, 0.4, 2.0, 5.0):
            np.testing.assert_allclose(lv.local_vol(np.array([-0.5, 0.0, 0.3]), t), 0.2, atol=1e-6)

    def test_skew_local_vol_is_steeper_than_implied(self):
        lv = skew_surface()
        k = np.array([-0.1, 0.0])
        loc, imp = lv.local_vol(k, 0.5), lv.implied_vol(k, 0.5)
        # Regra prática: perto do dinheiro a inclinação da vol local é cerca do dobro da implícita.
        self.assertGreater(loc[0] - loc[1], 1.5 * (imp[0] - imp[1]))


class LocalVolMonteCarloTests(unittest.TestCase):
    def test_flat_local_vol_matches_closed_forms(self):
        lv = LocalVolSurface.flat(M.vol_asset)
        md = quanto_market_data(M)
        H, K = 6000.0, 7100.0
        doc = M.fx_spot * price_down_and_out_call_analytical(M.spot, K, H, T, md.rate, md.dividend_yield,
                                                             md.volatility)
        mc = price_two_factor_mc(M, K, T, barrier=H, paths=100_000, steps=100, seed=4, local_vol=lv)
        self.assertLess(abs(mc.price - doc), 4 * mc.standard_error + 0.005 * doc)
        compo = price_two_factor_mc(M, K * M.fx_spot, T, product="compo", paths=100_000, steps=50, seed=8,
                                    local_vol=lv)
        target = price_compo_vanilla(M, K * M.fx_spot, T)
        self.assertLess(abs(compo.price - target), 4 * compo.standard_error + 0.005 * target)

    def test_dupire_reprices_the_smile(self):
        """Convertida = X_0 * vanilla USD: com vol local o MC tem que devolver o preço da vol implícita."""
        lv = skew_surface()
        f = M.spot * exp((M.rate_usd - M.dividend_yield) * T)
        for k in (-0.15, 0.0, 0.08):
            strike = f * exp(k)
            target = M.fx_spot * bs_usd(M.spot, strike, T, M.rate_usd, M.dividend_yield,
                                        float(lv.implied_vol(k, T)))
            mc = price_two_factor_mc(M, strike, T, product="converted", paths=150_000, steps=100, seed=11,
                                     local_vol=lv)
            self.assertLess(abs(mc.price - target), 4 * mc.standard_error + 0.01 * target,
                            f"k={k}: {mc.price:.2f} vs {target:.2f}")

    def test_skew_makes_down_and_out_call_cheaper_than_flat_at_strike(self):
        lv = skew_surface()
        f = M.spot * exp((M.rate_usd - M.dividend_yield) * T)
        flat = replace(M, vol_asset=float(lv.implied_vol(log(M.spot / f), T)))
        kw = dict(barrier=0.85 * M.spot, paths=100_000, steps=100, seed=2)
        p_flat = price_two_factor_mc(flat, M.spot, T, **kw)
        p_lv = price_two_factor_mc(M, M.spot, T, local_vol=lv, **kw)
        self.assertLess(p_lv.price, p_flat.price)
        self.assertGreater(p_lv.knock_out_probability, p_flat.knock_out_probability)


if __name__ == "__main__":
    unittest.main()
