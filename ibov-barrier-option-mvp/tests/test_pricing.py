import math
import unittest

import numpy as np

from ibov_barrier.pricing import (
    BarrierContract,
    MarketData,
    black_scholes_call,
    greeks,
    price_down_and_out_call,
)


MARKET = MarketData(spot=150_000, rate=0.12, dividend_yield=0.0, volatility=0.22)
CONTRACT = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)


def price(contract=CONTRACT, market=MARKET, **kwargs):
    return price_down_and_out_call(market, contract, paths=40_000, steps=63, seed=7, **kwargs)


class PricingTests(unittest.TestCase):
    # ========================================================================
    # Testes originais (preservados)
    # ========================================================================

    def test_barrier_price_is_bounded_by_vanilla(self):
        result = price()
        self.assertLessEqual(0.0, result.price)
        self.assertLessEqual(result.price, result.vanilla_price)

    def test_already_breached_is_worth_zero(self):
        result = price(BarrierContract(strike=155_000, barrier=150_000, maturity=0.5))
        self.assertEqual(result.price, 0.0)
        self.assertEqual(result.knock_out_probability, 1.0)

    def test_very_low_barrier_is_close_to_vanilla(self):
        result = price(BarrierContract(strike=155_000, barrier=1_000, maturity=0.5))
        self.assertAlmostEqual(result.price / result.vanilla_price, 1.0, delta=0.03)

    def test_higher_barrier_reduces_value(self):
        low = price(BarrierContract(strike=155_000, barrier=110_000, maturity=0.5)).price
        high = price(BarrierContract(strike=155_000, barrier=140_000, maturity=0.5)).price
        self.assertLess(high, low)

    # ========================================================================
    # 1. Reprodutibilidade
    # ========================================================================

    def test_reproducibility_same_seed_same_result(self):
        """Duas chamadas com mesma seed devem retornar exatamente os mesmos resultados."""
        result1 = price_down_and_out_call(MARKET, CONTRACT, paths=10_000, steps=50, seed=42)
        result2 = price_down_and_out_call(MARKET, CONTRACT, paths=10_000, steps=50, seed=42)
        self.assertEqual(result1.price, result2.price)
        self.assertEqual(result1.standard_error, result2.standard_error)
        self.assertEqual(result1.ci_low, result2.ci_low)
        self.assertEqual(result1.ci_high, result2.ci_high)
        self.assertEqual(result1.knock_out_probability, result2.knock_out_probability)
        self.assertEqual(result1.vanilla_price, result2.vanilla_price)

    # ========================================================================
    # 2. Black-Scholes no vencimento (T=0)
    # ========================================================================

    def test_black_scholes_at_maturity_itm(self):
        """T=0 e spot > strike: deve retornar max(S-K, 0)."""
        market = MarketData(spot=160_000, rate=0.12, dividend_yield=0.0, volatility=0.22)
        contract = BarrierContract(strike=155_000, barrier=120_000, maturity=0.0)
        vanilla = black_scholes_call(market, contract)
        expected = 160_000 - 155_000
        self.assertAlmostEqual(vanilla, expected, places=2)

    def test_black_scholes_at_maturity_otm(self):
        """T=0 e spot < strike: deve retornar 0."""
        market = MarketData(spot=150_000, rate=0.12, dividend_yield=0.0, volatility=0.22)
        contract = BarrierContract(strike=155_000, barrier=120_000, maturity=0.0)
        vanilla = black_scholes_call(market, contract)
        self.assertEqual(vanilla, 0.0)

    # ========================================================================
    # 3. Black-Scholes com volatilidade zero
    # ========================================================================

    def test_black_scholes_zero_volatility_itm(self):
        """Vol=0 e terminal > strike: retorna valor descontado."""
        market = MarketData(spot=160_000, rate=0.12, dividend_yield=0.0, volatility=0.0)
        contract = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)
        vanilla = black_scholes_call(market, contract)
        # Terminal = 160000 * exp(0.12 * 0.5) = 160000 * exp(0.06)
        terminal = 160_000 * math.exp(0.06)
        expected = math.exp(-0.12 * 0.5) * max(terminal - 155_000, 0.0)
        self.assertAlmostEqual(vanilla, expected, places=2)

    def test_black_scholes_zero_volatility_otm(self):
        """Vol=0 e terminal < strike: retorna 0."""
        market = MarketData(spot=150_000, rate=0.12, dividend_yield=0.0, volatility=0.0)
        contract = BarrierContract(strike=160_000, barrier=120_000, maturity=0.5)
        vanilla = black_scholes_call(market, contract)
        # Terminal = 150000 * exp(0.06) ≈ 159277 < 160000
        self.assertAlmostEqual(vanilla, 0.0, places=2)

    # ========================================================================
    # 4. Resultado estatístico
    # ========================================================================

    def test_price_and_standard_error_are_finite(self):
        """Preço e erro-padrão devem ser finitos."""
        result = price()
        self.assertTrue(math.isfinite(result.price))
        self.assertTrue(math.isfinite(result.standard_error))

    def test_standard_error_is_non_negative(self):
        """Erro-padrão não pode ser negativo."""
        result = price()
        self.assertGreaterEqual(result.standard_error, 0.0)

    def test_confidence_interval_bounds_price(self):
        """O preço deve estar dentro do intervalo de confiança."""
        result = price()
        self.assertLessEqual(result.ci_low, result.price)
        self.assertLessEqual(result.price, result.ci_high)

    def test_confidence_interval_is_symmetric(self):
        """IC deve ser aproximadamente simétrico em torno do preço."""
        result = price()
        half_width = 1.96 * result.standard_error
        self.assertAlmostEqual(result.price - result.ci_low, half_width, places=2)
        self.assertAlmostEqual(result.ci_high - result.price, half_width, places=2)

    # ========================================================================
    # 5. Validação de entradas
    # ========================================================================

    def test_negative_spot_raises_error(self):
        """Spot negativo deve gerar ValueError."""
        market = MarketData(spot=-100, rate=0.12, dividend_yield=0.0, volatility=0.22)
        with self.assertRaises(ValueError):
            price_down_and_out_call(market, CONTRACT, paths=1000, steps=10, seed=42)

    def test_negative_strike_raises_error(self):
        """Strike negativo deve gerar ValueError."""
        contract = BarrierContract(strike=-1000, barrier=120_000, maturity=0.5)
        with self.assertRaises(ValueError):
            price_down_and_out_call(MARKET, contract, paths=1000, steps=10, seed=42)

    def test_negative_barrier_raises_error(self):
        """Barreira negativa deve gerar ValueError."""
        contract = BarrierContract(strike=155_000, barrier=-1000, maturity=0.5)
        with self.assertRaises(ValueError):
            price_down_and_out_call(MARKET, contract, paths=1000, steps=10, seed=42)

    def test_negative_maturity_raises_error(self):
        """Prazo negativo deve gerar ValueError."""
        contract = BarrierContract(strike=155_000, barrier=120_000, maturity=-0.5)
        with self.assertRaises(ValueError):
            price_down_and_out_call(MARKET, contract, paths=1000, steps=10, seed=42)

    def test_negative_volatility_raises_error(self):
        """Volatilidade negativa deve gerar ValueError."""
        market = MarketData(spot=150_000, rate=0.12, dividend_yield=0.0, volatility=-0.1)
        with self.assertRaises(ValueError):
            price_down_and_out_call(market, CONTRACT, paths=1000, steps=10, seed=42)

    def test_zero_paths_raises_error(self):
        """paths=0 deve gerar ValueError."""
        with self.assertRaises(ValueError):
            price_down_and_out_call(MARKET, CONTRACT, paths=0, steps=10, seed=42)

    def test_one_path_raises_error(self):
        """paths=1 deve gerar ValueError (mínimo 2)."""
        with self.assertRaises(ValueError):
            price_down_and_out_call(MARKET, CONTRACT, paths=1, steps=10, seed=42)

    def test_zero_steps_raises_error(self):
        """steps=0 deve gerar ValueError."""
        with self.assertRaises(ValueError):
            price_down_and_out_call(MARKET, CONTRACT, paths=1000, steps=0, seed=42)

    def test_invalid_monitoring_raises_error(self):
        """monitoring inválido deve gerar ValueError."""
        with self.assertRaises(ValueError):
            price_down_and_out_call(MARKET, CONTRACT, paths=1000, steps=10, seed=42, monitoring="invalid")

    # ========================================================================
    # 6. Shapes de números externos
    # ========================================================================

    def test_normals_wrong_shape_raises_error(self):
        """normals com shape errado deve gerar ValueError."""
        normals = np.random.randn(100, 50)  # Errado: deveria ser (1000, 10)
        with self.assertRaises(ValueError):
            price_down_and_out_call(MARKET, CONTRACT, paths=1000, steps=10, seed=42, normals=normals)

    def test_uniforms_wrong_shape_raises_error(self):
        """uniforms com shape errado deve gerar ValueError."""
        normals = np.random.randn(1000, 10)
        uniforms = np.random.rand(100, 50)  # Errado
        with self.assertRaises(ValueError):
            price_down_and_out_call(
                MARKET, CONTRACT, paths=1000, steps=10, seed=42,
                normals=normals, uniforms=uniforms, monitoring="brownian_bridge"
            )

    def test_normals_correct_shape_accepted(self):
        """normals com shape correto deve ser aceito."""
        normals = np.random.randn(1000, 10)
        result = price_down_and_out_call(MARKET, CONTRACT, paths=1000, steps=10, seed=42, normals=normals)
        self.assertTrue(math.isfinite(result.price))

    def test_uniforms_correct_shape_accepted(self):
        """uniforms com shape correto deve ser aceito."""
        normals = np.random.randn(1000, 10)
        uniforms = np.random.rand(1000, 10)
        result = price_down_and_out_call(
            MARKET, CONTRACT, paths=1000, steps=10, seed=42,
            normals=normals, uniforms=uniforms, monitoring="brownian_bridge"
        )
        self.assertTrue(math.isfinite(result.price))

    # ========================================================================
    # 7. Brownian Bridge versus monitoramento discreto
    # ========================================================================

    def test_bridge_vs_discrete_with_common_rng(self):
        """Bridge e discreto com mesmos números devem produzir resultados coerentes."""
        rng = np.random.default_rng(42)
        normals = rng.standard_normal((10_000, 50))
        uniforms = rng.random((10_000, 50))

        result_discrete = price_down_and_out_call(
            MARKET, CONTRACT, paths=10_000, steps=50, seed=42,
            normals=normals, uniforms=None, monitoring="discrete"
        )

        result_bridge = price_down_and_out_call(
            MARKET, CONTRACT, paths=10_000, steps=50, seed=42,
            normals=normals, uniforms=uniforms, monitoring="brownian_bridge"
        )

        # Bridge detecta mais knock-outs (crossing entre steps), logo preço menor ou igual
        # Mas com variável de controle, não é garantido monotônico exato
        # Verificar que ambos são finitos e razoáveis
        self.assertTrue(math.isfinite(result_discrete.price))
        self.assertTrue(math.isfinite(result_bridge.price))
        self.assertGreaterEqual(result_discrete.price, 0.0)
        self.assertGreaterEqual(result_bridge.price, 0.0)

        # Bridge deve detectar >= knock-outs que discreto
        self.assertGreaterEqual(result_bridge.knock_out_probability, result_discrete.knock_out_probability)

        # Vanilla deve ser idêntico
        self.assertEqual(result_discrete.vanilla_price, result_bridge.vanilla_price)

    def test_bridge_detects_more_knockouts_than_discrete(self):
        """Bridge deve detectar pelo menos tantos knock-outs quanto discreto."""
        # Usar barreira relativamente alta para aumentar probabilidade de crossing
        contract_high_barrier = BarrierContract(strike=155_000, barrier=140_000, maturity=0.5)
        rng = np.random.default_rng(123)
        normals = rng.standard_normal((20_000, 100))
        uniforms = rng.random((20_000, 100))

        result_discrete = price_down_and_out_call(
            MARKET, contract_high_barrier, paths=20_000, steps=100, seed=123,
            normals=normals, uniforms=None, monitoring="discrete"
        )

        result_bridge = price_down_and_out_call(
            MARKET, contract_high_barrier, paths=20_000, steps=100, seed=123,
            normals=normals, uniforms=uniforms, monitoring="brownian_bridge"
        )

        self.assertGreaterEqual(result_bridge.knock_out_probability, result_discrete.knock_out_probability)

    # ========================================================================
    # 8. Gregas
    # ========================================================================

    def test_greeks_are_finite(self):
        """Todas as gregas devem ser finitas."""
        result_greeks = greeks(MARKET, CONTRACT, paths=50_000, steps=50, seed=42)
        for name, value in result_greeks.items():
            self.assertTrue(math.isfinite(value), f"{name} não é finito: {value}")

    def test_greeks_reproducibility(self):
        """Gregas com mesma seed devem ser reprodutíveis."""
        result1 = greeks(MARKET, CONTRACT, paths=50_000, steps=50, seed=42)
        result2 = greeks(MARKET, CONTRACT, paths=50_000, steps=50, seed=42)
        for name in result1:
            self.assertEqual(result1[name], result2[name], f"{name} não é reprodutível")

    def test_delta_is_finite_and_reproducible(self):
        """Delta deve ser finito e reprodutível (sem assumir range ou sinal)."""
        result1 = greeks(MARKET, CONTRACT, paths=50_000, steps=50, seed=42)
        result2 = greeks(MARKET, CONTRACT, paths=50_000, steps=50, seed=42)
        # Finito
        self.assertTrue(math.isfinite(result1["delta"]))
        # Reprodutível
        self.assertEqual(result1["delta"], result2["delta"])

    def test_greeks_do_not_assume_signs(self):
        """Não assumir genericamente sinais de Gamma ou Vega para barrier."""
        result_greeks = greeks(MARKET, CONTRACT, paths=50_000, steps=50, seed=42)
        # Apenas verificar finitude
        self.assertTrue(math.isfinite(result_greeks["gamma"]))
        self.assertTrue(math.isfinite(result_greeks["vega_1pct"]))
        self.assertTrue(math.isfinite(result_greeks["rho_1pct"]))
        self.assertTrue(math.isfinite(result_greeks["theta_1day"]))

    # ========================================================================
    # 9. Knock-out imediato (ampliado)
    # ========================================================================

    def test_spot_below_barrier_is_zero(self):
        """Spot abaixo da barreira: preço = 0, knock-out = 100%."""
        market = MarketData(spot=115_000, rate=0.12, dividend_yield=0.0, volatility=0.22)
        contract = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)
        result = price_down_and_out_call(market, contract, paths=1000, steps=10, seed=42)
        self.assertEqual(result.price, 0.0)
        self.assertEqual(result.knock_out_probability, 1.0)
        self.assertEqual(result.standard_error, 0.0)
        self.assertEqual(result.ci_low, 0.0)
        self.assertEqual(result.ci_high, 0.0)

    def test_spot_equal_barrier_is_zero(self):
        """Spot exatamente igual à barreira: preço = 0."""
        market = MarketData(spot=120_000, rate=0.12, dividend_yield=0.0, volatility=0.22)
        contract = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)
        result = price_down_and_out_call(market, contract, paths=1000, steps=10, seed=42)
        self.assertEqual(result.price, 0.0)
        self.assertEqual(result.knock_out_probability, 1.0)

    # ========================================================================
    # 10. Monotonicidade da barreira (preservado e ampliado)
    # ========================================================================

    def test_barrier_monotonicity_with_common_seed(self):
        """Barreira mais alta deve reduzir o preço, usando mesma seed."""
        seed = 999
        low_barrier = BarrierContract(strike=155_000, barrier=110_000, maturity=0.5)
        high_barrier = BarrierContract(strike=155_000, barrier=140_000, maturity=0.5)

        result_low = price_down_and_out_call(MARKET, low_barrier, paths=50_000, steps=50, seed=seed)
        result_high = price_down_and_out_call(MARKET, high_barrier, paths=50_000, steps=50, seed=seed)

        self.assertLess(result_high.price, result_low.price)
        self.assertGreater(result_high.knock_out_probability, result_low.knock_out_probability)


if __name__ == "__main__":
    unittest.main()
