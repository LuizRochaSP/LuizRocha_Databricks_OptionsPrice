"""Testes de precificação do swap DI x Pré."""
import unittest

from ibov_barrier.di_curve import DiscountCurve, price_di_pre_swap


class TestSwap(unittest.TestCase):
    def test_swap_npv_sign(self):
        """Pagar taxa fixa acima da curva gera NPV negativo; receber, positivo."""
        curve = DiscountCurve.from_di1([252, 504], [0.10, 0.10])
        pay_fixed = price_di_pre_swap(curve, 0, 252, 0.12, side="pay_fixed")
        receive_fixed = price_di_pre_swap(curve, 0, 252, 0.12, side="receive_fixed")
        self.assertLess(pay_fixed.npv, 0)
        self.assertGreater(receive_fixed.npv, 0)
        # Simetria: as duas pontas devem se anular
        self.assertAlmostEqual(pay_fixed.npv + receive_fixed.npv, 0.0, places=6)

    def test_dv01_finite(self):
        """DV01 deve ser não-nulo para swap com exposição a juros."""
        curve = DiscountCurve.from_di1([252, 504], [0.10, 0.11])
        result = price_di_pre_swap(curve, 0, 504, 0.11, side="pay_fixed")
        self.assertGreater(abs(result.dv01), 0)


if __name__ == "__main__":
    unittest.main()
