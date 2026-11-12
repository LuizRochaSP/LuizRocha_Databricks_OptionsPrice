from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class MarketData:
    spot: float
    rate: float
    dividend_yield: float
    volatility: float


@dataclass(frozen=True)
class BarrierContract:
    strike: float
    barrier: float
    maturity: float


@dataclass(frozen=True)
class PriceResult:
    price: float
    standard_error: float
    ci_low: float
    ci_high: float
    knock_out_probability: float
    vanilla_price: float


def black_scholes_call(market: MarketData, contract: BarrierContract) -> float:
    s, k, t = market.spot, contract.strike, contract.maturity
    r, q, sigma = market.rate, market.dividend_yield, market.volatility
    parameters = np.asarray([s, k, t, r, q, sigma], dtype=float)
    if not np.isfinite(parameters).all():
        raise ValueError("market and contract parameters must be finite")
    if s <= 0 or k <= 0:
        raise ValueError("spot and strike must be positive")
    if t < 0 or sigma < 0:
        raise ValueError("maturity and volatility must be non-negative")
    if t <= 0:
        return max(s - k, 0.0)
    if sigma <= 0:
        terminal = s * exp((r - q) * t)
        return exp(-r * t) * max(terminal - k, 0.0)
    d1 = (log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    return s * exp(-q * t) * norm.cdf(d1) - k * exp(-r * t) * norm.cdf(d2)


def price_down_and_out_call(
    market: MarketData,
    contract: BarrierContract,
    *,
    paths: int = 500_000,
    steps: int = 126,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    normals: np.ndarray | None = None,
    uniforms: np.ndarray | None = None,
) -> PriceResult:
    """Price a no-rebate down-and-out call with constant parameters.

    Brownian-bridge monitoring samples the conditional probability of crossing
    the log-barrier between consecutive simulated observations.
    """
    s0, h, t = market.spot, contract.barrier, contract.maturity
    sigma = market.volatility
    parameters = np.asarray(
        [s0, h, contract.strike, t, sigma, market.rate, market.dividend_yield],
        dtype=float,
    )
    if not np.isfinite(parameters).all():
        raise ValueError("market and contract parameters must be finite")
    if min(s0, h, contract.strike, t, sigma) <= 0:
        raise ValueError("spot, strike, barrier, maturity and volatility must be positive")
    if paths < 2 or steps < 1:
        raise ValueError("paths must be >= 2 and steps must be >= 1")
    if monitoring not in {"discrete", "brownian_bridge"}:
        raise ValueError("monitoring must be 'discrete' or 'brownian_bridge'")
    vanilla = black_scholes_call(market, contract)
    if s0 <= h:
        return PriceResult(0.0, 0.0, 0.0, 0.0, 1.0, vanilla)

    rng = np.random.default_rng(seed)
    dt = t / steps
    z = np.asarray(normals, dtype=float) if normals is not None else rng.standard_normal((paths, steps))
    if z.shape != (paths, steps):
        raise ValueError("normals must have shape (paths, steps)")
    if not np.isfinite(z).all():
        raise ValueError("normals must contain only finite values")
    if uniforms is None and monitoring == "brownian_bridge":
        uniforms = rng.random((paths, steps))
    if uniforms is not None:
        uniforms = np.asarray(uniforms, dtype=float)
        if uniforms.shape != (paths, steps):
            raise ValueError("uniforms must have shape (paths, steps)")
        if not np.isfinite(uniforms).all() or np.any((uniforms < 0.0) | (uniforms > 1.0)):
            raise ValueError("uniforms must contain finite values in [0, 1]")

    drift = (market.rate - market.dividend_yield - 0.5 * sigma**2) * dt
    diffusion = sigma * sqrt(dt)
    log_paths = log(s0) + np.cumsum(drift + diffusion * z, axis=1)
    log_h = log(h)
    previous = np.concatenate((np.full((paths, 1), log(s0)), log_paths[:, :-1]), axis=1)
    alive = np.ones(paths, dtype=bool)

    if monitoring == "discrete":
        alive = np.all(log_paths > log_h, axis=1)
    else:
        endpoints_above = (previous > log_h) & (log_paths > log_h)
        crossing_probability = np.ones_like(log_paths)
        crossing_probability[endpoints_above] = np.exp(
            -2.0
            * (previous[endpoints_above] - log_h)
            * (log_paths[endpoints_above] - log_h)
            / (sigma**2 * dt)
        )
        alive = np.all(endpoints_above & (uniforms > crossing_probability), axis=1)

    terminal = np.exp(log_paths[:, -1])
    discount = exp(-market.rate * t)
    vanilla_payoffs = discount * np.maximum(terminal - contract.strike, 0.0)
    barrier_payoffs = vanilla_payoffs * alive

    # The corresponding vanilla payoff is an effective control variate because
    # its exact expectation is known from Black-Scholes and it is highly
    # correlated with the barrier payoff.
    vanilla_variance = float(vanilla_payoffs.var(ddof=1))
    beta = (
        float(np.cov(barrier_payoffs, vanilla_payoffs, ddof=1)[0, 1]) / vanilla_variance
        if vanilla_variance > 0.0
        else 0.0
    )
    adjusted_payoffs = barrier_payoffs - beta * (vanilla_payoffs - vanilla)
    price = float(adjusted_payoffs.mean())
    standard_error = float(adjusted_payoffs.std(ddof=1) / sqrt(paths))
    return PriceResult(
        price=price,
        standard_error=standard_error,
        ci_low=price - 1.96 * standard_error,
        ci_high=price + 1.96 * standard_error,
        knock_out_probability=float(1.0 - alive.mean()),
        vanilla_price=vanilla,
    )


def greeks(
    market: MarketData,
    contract: BarrierContract,
    *,
    paths: int = 200_000,
    steps: int = 126,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> dict[str, float]:
    """Central finite-difference Greeks with common random numbers."""
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((paths, steps))
    uniforms = rng.random((paths, steps))

    def value(m: MarketData, c: BarrierContract = contract) -> float:
        return price_down_and_out_call(
            m, c, paths=paths, steps=steps, seed=seed,
            monitoring=monitoring, normals=normals, uniforms=uniforms,
        ).price

    ds = max(1.0, market.spot * 0.001)
    dv = 0.001
    dr = 0.0001
    dt = min(1.0 / 365.0, contract.maturity / 4.0)
    up_s, down_s = replace(market, spot=market.spot + ds), replace(market, spot=market.spot - ds)
    base = value(market)
    p_up, p_down = value(up_s), value(down_s)
    return {
        "delta": (p_up - p_down) / (2.0 * ds),
        "gamma": (p_up - 2.0 * base + p_down) / ds**2,
        "vega_1pct": (value(replace(market, volatility=market.volatility + dv)) - value(replace(market, volatility=market.volatility - dv))) / (2.0 * dv) * 0.01,
        "rho_1pct": (value(replace(market, rate=market.rate + dr)) - value(replace(market, rate=market.rate - dr))) / (2.0 * dr) * 0.01,
        "theta_1day": value(market, replace(contract, maturity=contract.maturity - dt)) - base,
    }
