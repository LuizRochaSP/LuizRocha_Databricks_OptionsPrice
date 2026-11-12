"""Testes de paridade In/Out para os novos tipos de barreira.

Prova a relação matemática: Vanilla = Down-and-Out + Down-and-In.
Com a mesma seed, os caminhos (e uniforms do Brownian Bridge) são idênticos
nas duas precificações, e a soma das variáveis de controle fecha a paridade.
"""
import unittest

from ibov_barrier.pricing import (
    BarrierContract,
    MarketData,
    black_scholes_call,
    price_down_and_out_call,
)
from ibov_barrier.pricing_barriers import price_down_and_in_call


class TestInOutParity(unittest.TestCase):

    def test_vanilla_equals_down_and_out_plus_down_and_in(self):
        """Vanilla (BS) deve igualar DOC (MC) + DIC (MC) com a mesma seed."""
        market = MarketData(spot=150_000, rate=0.12, dividend_yield=0.02, volatility=0.22)
        contract = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)

        vanilla = black_scholes_call(market, contract)

        down_and_out = price_down_and_out_call(
            market, contract, paths=200_000, steps=252, seed=42,
            monitoring="brownian_bridge",
        )
        down_and_in = price_down_and_in_call(
            market, contract, paths=200_000, steps=252, seed=42,
            monitoring="brownian_bridge",
        )

        soma = down_and_out.price + down_and_in.price
        erro = abs(soma - vanilla) / vanilla

        print(f"\nVanilla (Black-Scholes):  R$ {vanilla:,.2f}")
        print(f"Down-and-Out (MC):       R$ {down_and_out.price:,.2f} (SE {down_and_out.standard_error:.2f})")
        print(f"Down-and-In (MC):        R$ {down_and_in.price:,.2f} (SE {down_and_in.standard_error:.2f})")
        print(f"Soma DOC + DIC:          R$ {soma:,.2f}")
        print(f"Erro da paridade:        {erro:.4%}")

        self.assertLess(erro, 0.02, "Paridade In-Out violada: DOC + DIC != Vanilla (tolerância 2%)")


if __name__ == "__main__":
    unittest.main()