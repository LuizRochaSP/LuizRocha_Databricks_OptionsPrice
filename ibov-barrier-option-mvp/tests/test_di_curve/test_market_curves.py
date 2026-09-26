"""Testes do módulo market_curves."""
import unittest

import numpy as np

from ibov_barrier.di_curve.curve import DiscountCurve
from ibov_barrier.di_curve.market_curves import MarketCurves


class TestMarketCurves(unittest.TestCase):
    def _build_synthetic(self):
        # Curva r efetiva flat em 10% a.a.; q contínuo flat em 2%.
        r_flat = DiscountCurve.from_di1([252, 504, 756], [0.10, 0.10, 0.10])
        q_tenors = np.array([252.0, 504.0, 756.0])
        q_cont = np.array([0.02, 0.02, 0.02])
        return MarketCurves(
            r_curve=r_flat,
            q_tenors_bd=q_tenors,
            q_cont_values=q_cont,
            spot=150_000.0,
        )

    def test_r_returns_continuous_rate(self):
        # r efetiva flat em 10% → r contínua = log(1.10) ≈ 0.0953102
        mc = self._build_synthetic()
        expected = float(np.log(1.10))
        for t in [126, 252, 504, 756]:
            self.assertAlmostEqual(mc.r(t), expected, places=10)

    def test_q_returns_continuous_dividend(self):
        mc = self._build_synthetic()
        for t in [126, 252, 504, 756]:
            self.assertAlmostEqual(mc.q(t), 0.02, places=10)

    def test_forward_matches_formula(self):
        mc = self._build_synthetic()
        T = 1.0
        r_cont = float(np.log(1.10))
        expected = 150_000.0 * np.exp((r_cont - 0.02) * T)
        self.assertAlmostEqual(mc.forward(252), expected, places=6)

    def test_r_q_at(self):
        mc = self._build_synthetic()
        r, q = mc.r_q_at(252)
        self.assertAlmostEqual(r, float(np.log(1.10)), places=10)
        self.assertAlmostEqual(q, 0.02, places=10)

    def test_spot_positive(self):
        mc = self._build_synthetic()
        self.assertGreater(mc.spot, 0)


if __name__ == "__main__":
    unittest.main()
