from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from .pricing import BarrierContract, MarketData, greeks, price_down_and_out_call


@dataclass(frozen=True)
class RiskConfig:
    horizon_days: int = 1
    confidence_level: float = 0.99
    scenarios: int = 100_000
    seed: int = 42
    annual_drift: float = 0.0


@dataclass(frozen=True)
class RiskResult:
    price: float
    delta: float
    gamma: float
    vega_1pct: float
    theta_1day: float
    rho_1pct: float
    dv01: float
    rate_convexity_1bp: float
    var: float
    expected_shortfall: float
    pnl_mean: float
    pnl_std: float
    horizon_days: int
    confidence_level: float
    scenarios: int
    seed: int
    methodology: str


def _validate_risk_config(config: RiskConfig) -> None:
    values = np.asarray([config.confidence_level, config.annual_drift], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("risk configuration values must be finite")
    if config.horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if not 0.0 < config.confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if config.scenarios < 2:
        raise ValueError("scenarios must be >= 2")


def rate_risk(
    market: MarketData,
    contract: BarrierContract,
    *,
    paths: int = 200_000,
    steps: int = 126,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    bump_bp: float = 1.0,
) -> dict[str, float]:
    """Signed DV01 and discrete rate convexity using common random numbers.

    ``dv01`` is the central estimate of the price change for a +1 bp parallel
    rate move. ``rate_convexity_1bp`` is ``V(r+1bp)-2V(r)+V(r-1bp)``.
    """
    if not np.isfinite(bump_bp) or bump_bp <= 0.0:
        raise ValueError("bump_bp must be finite and positive")
    bump = bump_bp * 1e-4
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((paths, steps))
    uniforms = rng.random((paths, steps)) if monitoring == "brownian_bridge" else None

    def value(rate: float) -> float:
        bumped = MarketData(
            spot=market.spot,
            rate=rate,
            dividend_yield=market.dividend_yield,
            volatility=market.volatility,
        )
        return price_down_and_out_call(
            bumped,
            contract,
            paths=paths,
            steps=steps,
            seed=seed,
            monitoring=monitoring,
            normals=normals,
            uniforms=uniforms,
        ).price

    base = value(market.rate)
    up = value(market.rate + bump)
    down = value(market.rate - bump)
    return {
        "dv01": (up - down) / (2.0 * bump_bp),
        "rate_convexity_1bp": (up - 2.0 * base + down) / (bump_bp**2),
        "price_rate_up": up,
        "price_rate_down": down,
    }


def delta_gamma_var_es(
    *,
    spot: float,
    volatility: float,
    delta: float,
    gamma: float,
    config: RiskConfig,
) -> dict[str, float]:
    """One-position VaR/ES from delta-gamma P&L under lognormal spot shocks.

    The distribution is model-based, not historical: annual drift and current
    annualized volatility are converted to the configured business-day horizon.
    Positive VaR and ES represent losses.
    """
    _validate_risk_config(config)
    values = np.asarray([spot, volatility, delta, gamma], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("spot, volatility, delta and gamma must be finite")
    if spot <= 0.0 or volatility < 0.0:
        raise ValueError("spot must be positive and volatility non-negative")

    horizon = config.horizon_days / 252.0
    rng = np.random.default_rng(config.seed)
    z = rng.standard_normal(config.scenarios)
    shocked_spot = spot * np.exp(
        (config.annual_drift - 0.5 * volatility**2) * horizon
        + volatility * sqrt(horizon) * z
    )
    ds = shocked_spot - spot
    pnl = delta * ds + 0.5 * gamma * ds**2
    losses = -pnl
    var = float(np.quantile(losses, config.confidence_level))
    tail = losses[losses >= var]
    expected_shortfall = float(tail.mean()) if tail.size else var
    return {
        "var": max(var, 0.0),
        "expected_shortfall": max(expected_shortfall, 0.0),
        "pnl_mean": float(pnl.mean()),
        "pnl_std": float(pnl.std(ddof=1)),
    }


def calculate_risk_metrics(
    market: MarketData,
    contract: BarrierContract,
    *,
    paths: int = 200_000,
    steps: int = 126,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    config: RiskConfig | None = None,
) -> RiskResult:
    """Calculate Greeks, DV01, rate convexity and delta-gamma VaR/ES."""
    config = config or RiskConfig(seed=seed)
    _validate_risk_config(config)
    price = price_down_and_out_call(
        market,
        contract,
        paths=paths,
        steps=steps,
        seed=seed,
        monitoring=monitoring,
    ).price
    greek_values = greeks(
        market,
        contract,
        paths=paths,
        steps=steps,
        seed=seed,
        monitoring=monitoring,
    )
    rates = rate_risk(
        market,
        contract,
        paths=paths,
        steps=steps,
        seed=seed,
        monitoring=monitoring,
    )
    tail = delta_gamma_var_es(
        spot=market.spot,
        volatility=market.volatility,
        delta=greek_values["delta"],
        gamma=greek_values["gamma"],
        config=config,
    )
    return RiskResult(
        price=price,
        delta=greek_values["delta"],
        gamma=greek_values["gamma"],
        vega_1pct=greek_values["vega_1pct"],
        theta_1day=greek_values["theta_1day"],
        rho_1pct=greek_values["rho_1pct"],
        dv01=rates["dv01"],
        rate_convexity_1bp=rates["rate_convexity_1bp"],
        var=tail["var"],
        expected_shortfall=tail["expected_shortfall"],
        pnl_mean=tail["pnl_mean"],
        pnl_std=tail["pnl_std"],
        horizon_days=config.horizon_days,
        confidence_level=config.confidence_level,
        scenarios=config.scenarios,
        seed=config.seed,
        methodology="delta-gamma Monte Carlo with lognormal spot shocks",
    )
