"""Testes da DiscountCurve."""
import unittest

from ibov_barrier.di_curve import DiscountCurve


class TestDiscountCurve(unittest.TestCase):
    def test_flat_curve_returns_constant_rate(self):
        """Curva flat deve retornar a mesma taxa em qualquer prazo."""
        curve = DiscountCurve.from_di1([252, 504, 756], [0.10, 0.10, 0.10])
        for t in [100, 252, 400, 600, 756]:
            self.assertAlmostEqual(curve.rate(t), 0.10, places=10)

    def test_df_at_vertex_matches_input(self):
        """DF no vértice deve bater com a fórmula (1+r)^(-T)."""
        rates = [0.11, 0.12, 0.13]
        tenors = [252, 504, 756]
        curve = DiscountCurve.from_di1(tenors, rates)
        for t, r in zip(tenors, rates):
            expected_df = (1 + r) ** (-t / 252)
            self.assertAlmostEqual(curve.df(t), expected_df, places=12)

    def test_df_monotonic(self):
        """DF deve ser estritamente decrescente em T."""
        curve = DiscountCurve.from_di1([252, 504, 756, 1008], [0.10, 0.11, 0.12, 0.13])
        dfs = [curve.df(t) for t in range(100, 1009, 50)]
        for i in range(len(dfs) - 1):
            self.assertGreater(dfs[i], dfs[i + 1])

    def test_forward_rate_positive(self):
        """Forward entre dois prazos deve ser positivo."""
        curve = DiscountCurve.from_di1([252, 504, 756], [0.10, 0.12, 0.14])
        self.assertGreater(curve.forward_rate(252, 504), 0)

    def test_parallel_bump(self):
        """Choque paralelo de +100 bp adiciona 0.01 a todas as taxas."""
        curve = DiscountCurve.from_di1([252, 504], [0.10, 0.10])
        bumped = curve.bump_parallel(100)
        self.assertAlmostEqual(bumped.rate(252), 0.11, places=10)
        self.assertAlmostEqual(bumped.rate(504), 0.11, places=10)

    def test_key_rate_bump_isolated(self):
        """Key-rate bump afeta apenas o vértice mais próximo."""
        curve = DiscountCurve.from_di1([252, 504, 756], [0.10, 0.11, 0.12])
        bumped = curve.bump_key(504, 100)
        self.assertAlmostEqual(bumped.rate(252), 0.10, places=10)
        self.assertAlmostEqual(bumped.rate(504), 0.12, places=10)
        self.assertAlmostEqual(bumped.rate(756), 0.12, places=10)


if __name__ == "__main__":
    unittest.main()