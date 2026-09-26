"""Testes do módulo market_curves."""
import unittest

import numpy as np

from ibov_barrier.di_curve.curve import DiscountCurve
from ibov_barrier.di_curve.market_curves import MarketCurves


class TestMarketCurves(unittest.TestCase):
    def _build_synthetic(self):
        tenors = np.array([252.0, 504.0, 756.0])
        r_flat = DiscountCurve.from_di1([252, 504, 756], [0.10, 0.10, 0.10])
        q_tenors = np.array([252.0, 504.0, 756.0])
        q_values = np.array([0.02, 0.02, 0.02])
        return MarketCurves(
            r_curve=r_flat,
            q_tenors_bd=q_tenors,
            q_values=q_values,
            spot=150_000.0,
        )

    def test_r_returns_flat_rate(self):
        mc = self._build_synthetic()
        for t in [126, 252, 504, 756]:
            self.assertAlmostEqual(mc.r(t), 0.10, places=10)

    def test_q_returns_flat_dividend(self):
        mc = self._build_synthetic()
        for t in [126, 252, 504, 756]:
            self.assertAlmostEqual(mc.q(t), 0.02, places=10)

    def test_forward_matches_formula(self):
        mc = self._build_synthetic()
        T = 252.0 / 252.0
        expected = 150_000.0 * np.exp((0.10 - 0.02) * T)
        self.assertAlmostEqual(mc.forward(252), expected, places=6)

    def test_r_q_at(self):
        mc = self._build_synthetic()
        r, q = mc.r_q_at(252)
        self.assertAlmostEqual(r, 0.10, places=10)
        self.assertAlmostEqual(q, 0.02, places=10)

    def test_spot_positive(self):
        mc = self._build_synthetic()
        self.assertGreater(mc.spot, 0)


if __name__ == "__main__":
    unittest.main()
