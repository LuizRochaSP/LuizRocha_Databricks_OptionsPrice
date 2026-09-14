import unittest

from ibov_barrier.pricing import BarrierContract, MarketData, price_down_and_out_call


MARKET = MarketData(spot=150_000, rate=0.12, dividend_yield=0.0, volatility=0.22)
CONTRACT = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)


def price(contract=CONTRACT, market=MARKET):
    return price_down_and_out_call(market, contract, paths=40_000, steps=63, seed=7)


class PricingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
