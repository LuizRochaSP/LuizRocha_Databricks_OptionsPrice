"""Extensões do motor de precificação: barreiras Up-and-Out e Down-and-In.

Reutiliza as estruturas de dados, o gerador de caminhos, o monitoramento
Brownian Bridge e a variável de controle vanilla do módulo `pricing`, alterando
apenas a lógica de ativação/knock-out da barreira.
"""
from __future__ import annotations

from math import exp, log, sqrt

import numpy as np

from ibov_barrier.pricing import (
    BarrierContract,
    MarketData,
    PriceResult,
    black_scholes_call,
)


def price_up_and_out_call(
    market: MarketData,
    contract: BarrierContract,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Preço de uma call up-and-out sem rebate (barreira superior).

    A opção morre se o ativo SUBIR e tocar a barreira superior. Exige spot
    estritamente inferior à barreira. O monitoramento Brownian Bridge amostra a
    probabilidade condicional de cruzar a barreira logarítmica para cima entre
    observações consecutivas.
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
    if s0 >= h:
        return PriceResult(0.0, 0.0, 0.0, 0.0, 1.0, vanilla)

    rng = np.random.default_rng(seed)
    dt = t / steps
    z = rng.standard_normal((paths, steps))
    uniforms = rng.random((paths, steps)) if monitoring == "brownian_bridge" else None

    drift = (market.rate - market.dividend_yield - 0.5 * sigma**2) * dt
    diffusion = sigma * sqrt(dt)
    log_paths = log(s0) + np.cumsum(drift + diffusion * z, axis=1)
    log_h = log(h)
    previous = np.concatenate((np.full((paths, 1), log(s0)), log_paths[:, :-1]), axis=1)

    if monitoring == "discrete":
        alive = np.all(log_paths < log_h, axis=1)
    else:
        endpoints_below = (previous < log_h) & (log_paths < log_h)
        crossing_probability = np.ones_like(log_paths)
        crossing_probability[endpoints_below] = np.exp(
            -2.0
            * (log_h - previous[endpoints_below])
            * (log_h - log_paths[endpoints_below])
            / (sigma**2 * dt)
        )
        alive = np.all(endpoints_below & (uniforms > crossing_probability), axis=1)

    terminal = np.exp(log_paths[:, -1])
    discount = exp(-market.rate * t)
    vanilla_payoffs = discount * np.maximum(terminal - contract.strike, 0.0)
    barrier_payoffs = vanilla_payoffs * alive

    # A payoff vanilla correspondente é uma variável de controle efetiva: sua
    # expectação exata é conhecida (Black-Scholes) e ela é altamente correlacionada
    # com o payoff da barreira.
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


def price_down_and_in_call(
    market: MarketData,
    contract: BarrierContract,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Preço de uma call down-and-in sem rebate (barreira inferior).

    A opção SÓ paga se o ativo DESCER e tocar a barreira inferior em algum
    momento até o vencimento; caso contrário expira sem valor. Se o spot já
    nasce na barreira ou abaixo dela, a opção já está ativa e equivale à call
    vanilla. Neste caso `knock_out_probability` reporta a probabilidade de a
    opção morrer sem ativação (nunca tocar a barreira).
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
        return PriceResult(vanilla, 0.0, vanilla, vanilla, 0.0, vanilla)

    rng = np.random.default_rng(seed)
    dt = t / steps
    z = rng.standard_normal((paths, steps))
    uniforms = rng.random((paths, steps)) if monitoring == "brownian_bridge" else None

    drift = (market.rate - market.dividend_yield - 0.5 * sigma**2) * dt
    diffusion = sigma * sqrt(dt)
    log_paths = log(s0) + np.cumsum(drift + diffusion * z, axis=1)
    log_h = log(h)
    previous = np.concatenate((np.full((paths, 1), log(s0)), log_paths[:, :-1]), axis=1)

    if monitoring == "discrete":
        knocked_in = np.any(log_paths <= log_h, axis=1)
    else:
        endpoints_above = (previous > log_h) & (log_paths > log_h)
        crossing_probability = np.ones_like(log_paths)
        crossing_probability[endpoints_above] = np.exp(
            -2.0
            * (previous[endpoints_above] - log_h)
            * (log_paths[endpoints_above] - log_h)
            / (sigma**2 * dt)
        )
        survived = np.all(endpoints_above & (uniforms > crossing_probability), axis=1)
        knocked_in = ~survived

    terminal = np.exp(log_paths[:, -1])
    discount = exp(-market.rate * t)
    vanilla_payoffs = discount * np.maximum(terminal - contract.strike, 0.0)
    barrier_payoffs = vanilla_payoffs * knocked_in

    # Variável de controle: mesma payoff vanilla com expectação exata conhecida.
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
        knock_out_probability=float(1.0 - knocked_in.mean()),
        vanilla_price=vanilla,
    )