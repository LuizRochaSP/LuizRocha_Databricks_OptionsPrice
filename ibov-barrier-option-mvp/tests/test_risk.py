import math
import unittest

from ibov_barrier import (
    BarrierContract,
    MarketData,
    RiskConfig,
    calculate_risk_metrics,
    delta_gamma_var_es,
    rate_risk,
)


MARKET = MarketData(spot=150_000, rate=0.12, dividend_yield=0.01, volatility=0.22)
CONTRACT = BarrierContract(strike=155_000, barrier=120_000, maturity=0.5)


class RiskTests(unittest.TestCase):
    def test_delta_gamma_var_es_is_reproducible(self):
        config = RiskConfig(scenarios=20_000, seed=11)
        first = delta_gamma_var_es(
            spot=MARKET.spot, volatility=MARKET.volatility,
            delta=0.5, gamma=1e-5, config=config,
        )
        second = delta_gamma_var_es(
            spot=MARKET.spot, volatility=MARKET.volatility,
            delta=0.5, gamma=1e-5, config=config,
        )
        self.assertEqual(first, second)

    def test_expected_shortfall_is_not_below_var(self):
        result = delta_gamma_var_es(
            spot=MARKET.spot, volatility=MARKET.volatility,
            delta=0.5, gamma=1e-5,
            config=RiskConfig(scenarios=30_000, seed=7),
        )
        self.assertGreaterEqual(result["expected_shortfall"], result["var"])
        self.assertGreaterEqual(result["var"], 0.0)

    def test_risk_config_validation(self):
        invalid = [
            RiskConfig(horizon_days=0),
            RiskConfig(confidence_level=0.0),
            RiskConfig(confidence_level=1.0),
            RiskConfig(scenarios=1),
            RiskConfig(annual_drift=math.nan),
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                delta_gamma_var_es(
                    spot=MARKET.spot, volatility=MARKET.volatility,
                    delta=0.5, gamma=1e-5, config=config,
                )

    def test_delta_gamma_inputs_must_be_valid(self):
        config = RiskConfig(scenarios=100)
        for kwargs in [
            {"spot": 0.0, "volatility": 0.2, "delta": 0.5, "gamma": 0.0},
            {"spot": 100.0, "volatility": -0.1, "delta": 0.5, "gamma": 0.0},
            {"spot": 100.0, "volatility": 0.2, "delta": math.nan, "gamma": 0.0},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                delta_gamma_var_es(config=config, **kwargs)

    def test_rate_risk_is_finite_and_reproducible(self):
        kwargs = dict(paths=5_000, steps=20, seed=5)
        first = rate_risk(MARKET, CONTRACT, **kwargs)
        second = rate_risk(MARKET, CONTRACT, **kwargs)
        self.assertEqual(first, second)
        self.assertTrue(all(math.isfinite(value) for value in first.values()))

    def test_rate_risk_rejects_invalid_bump(self):
        with self.assertRaises(ValueError):
            rate_risk(MARKET, CONTRACT, paths=100, steps=5, bump_bp=0.0)

    def test_calculate_risk_metrics_returns_consistent_result(self):
        result = calculate_risk_metrics(
            MARKET,
            CONTRACT,
            paths=5_000,
            steps=20,
            seed=9,
            config=RiskConfig(scenarios=10_000, seed=9),
        )
        self.assertTrue(all(math.isfinite(value) for value in [
            result.price, result.delta, result.gamma, result.vega_1pct,
            result.theta_1day, result.rho_1pct, result.dv01,
            result.rate_convexity_1bp, result.var, result.expected_shortfall,
        ]))
        self.assertGreaterEqual(result.expected_shortfall, result.var)
        self.assertEqual(result.horizon_days, 1)
        self.assertEqual(result.confidence_level, 0.99)


if __name__ == "__main__":
    unittest.main()
