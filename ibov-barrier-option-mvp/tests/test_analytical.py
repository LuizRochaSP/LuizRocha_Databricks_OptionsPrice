import unittest
from ibov_barrier.pricing import MarketData, BarrierContract, price_down_and_out_call
from ibov_barrier.analytical import price_down_and_out_call_analytical

class TestAnalyticalBenchmark(unittest.TestCase):
    def test_mc_vs_analytical(self):
        """Compara o preço do Monte Carlo com a fórmula fechada de Reiner-Rubinstein."""
        # Contrato padrão
        market = MarketData(spot=150_000, rate=0.12, dividend_yield=0.02, volatility=0.22)
        contract = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)

        # 1. Preço Analítico (Fórmula Fechada)
        analytical_price = price_down_and_out_call_analytical(
            spot=market.spot, strike=contract.strike, barrier=contract.barrier,
            maturity=contract.maturity, rate=market.rate, 
            dividend_yield=market.dividend_yield, volatility=market.volatility
        )

        # 2. Preço Monte Carlo (usando muitos passos para reduzir o viés de discretização)
        mc_result = price_down_and_out_call(
            market, contract, paths=200_000, steps=252, seed=42, monitoring="brownian_bridge"
        )
        
        # 3. Comparação (tolerância de 1%)
        error_pct = abs(mc_result.price - analytical_price) / analytical_price
        print(f"\nAnalítico (Reiner-Rubinstein): R$ {analytical_price:.2f}")
        print(f"Monte Carlo (Brownian Bridge): R$ {mc_result.price:.2f}")
        print(f"Erro relativo: {error_pct:.4%}")
        
        self.assertLess(error_pct, 0.01, "Monte Carlo divergiu mais de 1% do analítico")

if __name__ == "__main__":
    unittest.main()
