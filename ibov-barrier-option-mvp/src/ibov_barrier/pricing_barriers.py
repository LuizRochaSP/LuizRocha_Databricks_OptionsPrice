"""Extensões do motor de precificação: Família A completa de barreiras.

Calls e Puts, Down/Up, In/Out (DOC, UOC, DIC, UIC, DOP, UOP, DIP, UIP) —
reutiliza as estruturas de dados, o gerador de caminhos, o monitoramento
Brownian Bridge e a variável de controle vanilla do módulo `pricing`, alterando
apenas a lógica de ativação/knock-out da barreira e o payoff final.

O preço da put vanilla (necessário para a variável de controle) é obtido pela
paridade put-call: P = C - S·e^(-qT) + K·e^(-rT).
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from math import exp, log, sqrt

import numpy as np

from ibov_barrier.pricing import (
    BarrierContract,
    MarketData,
    PriceResult,
    black_scholes_call,
    price_down_and_out_call,
)


def price_up_and_out_call(
    market: MarketData,
    contract: BarrierContract,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    rebate: float = 0.0,
) -> PriceResult:
    """Preço de uma call up-and-out sem rebate (barreira superior).

    A opção morre se o ativo SUBIR e tocar a barreira superior. Exige spot
    estritamente inferior à barreira. O monitoramento Brownian Bridge amostra a
    probabilidade condicional de cruzar a barreira logarítmica para cima entre
    observações consecutivas.
    """
    if rebate:
        # Rebate pago no vencimento se a barreira for tocada: delega ao motor
        # genérico (mesma sequência aleatória; rebate=0 segue o caminho original).
        from ibov_barrier.pricing_extensions import price_single_barrier
        return price_single_barrier(
            market, contract, option_type="call", direction="up", knock="out",
            rebate=rebate, paths=paths, steps=steps, seed=seed, monitoring=monitoring,
        )
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


def _black_scholes_put_by_parity(
    market: MarketData,
    contract: BarrierContract,
    vanilla_call: float,
) -> float:
    """Preço da put vanilla via paridade put-call: P = C - S·e^(-qT) + K·e^(-rT)."""
    t = contract.maturity
    return (
        vanilla_call
        - market.spot * exp(-market.dividend_yield * t)
        + contract.strike * exp(-market.rate * t)
    )


def price_down_and_out_put(
    market: MarketData,
    contract: BarrierContract,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    rebate: float = 0.0,
) -> PriceResult:
    """Preço de uma put down-and-out sem rebate (barreira inferior).

    A opção morre se o ativo DESCER e tocar a barreira inferior. Exige spot
    estritamente superior à barreira. O Monte Carlo (caminhos, Brownian Bridge
    e knock-out) é idêntico ao das calls; a única mudança é o payoff final:
    np.maximum(strike - terminal, 0.0). A put vanilla da variável de controle
    vem da paridade put-call.
    """
    if rebate:
        # Rebate pago no vencimento se a barreira for tocada: delega ao motor
        # genérico (mesma sequência aleatória; rebate=0 segue o caminho original).
        from ibov_barrier.pricing_extensions import price_single_barrier
        return price_single_barrier(
            market, contract, option_type="put", direction="down", knock="out",
            rebate=rebate, paths=paths, steps=steps, seed=seed, monitoring=monitoring,
        )
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
    vanilla = _black_scholes_put_by_parity(market, contract, black_scholes_call(market, contract))
    if s0 <= h:
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
    vanilla_payoffs = discount * np.maximum(contract.strike - terminal, 0.0)
    barrier_payoffs = vanilla_payoffs * alive

    # Variável de controle: payoff da put vanilla, com expectação exata
    # conhecida via paridade put-call.
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


def price_up_and_out_put(
    market: MarketData,
    contract: BarrierContract,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    rebate: float = 0.0,
) -> PriceResult:
    """Preço de uma put up-and-out sem rebate (barreira superior).

    A opção morre se o ativo SUBIR e tocar a barreira superior. Exige spot
    estritamente inferior à barreira. O Monte Carlo (caminhos, Brownian Bridge
    e knock-out) é idêntico ao das calls; a única mudança é o payoff final:
    np.maximum(strike - terminal, 0.0). A put vanilla da variável de controle
    vem da paridade put-call.
    """
    if rebate:
        # Rebate pago no vencimento se a barreira for tocada: delega ao motor
        # genérico (mesma sequência aleatória; rebate=0 segue o caminho original).
        from ibov_barrier.pricing_extensions import price_single_barrier
        return price_single_barrier(
            market, contract, option_type="put", direction="up", knock="out",
            rebate=rebate, paths=paths, steps=steps, seed=seed, monitoring=monitoring,
        )
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
    vanilla = _black_scholes_put_by_parity(market, contract, black_scholes_call(market, contract))
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
    vanilla_payoffs = discount * np.maximum(contract.strike - terminal, 0.0)
    barrier_payoffs = vanilla_payoffs * alive

    # Variável de controle: payoff da put vanilla, com expectação exata
    # conhecida via paridade put-call.
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


def price_down_and_in_put(
    market: MarketData,
    contract: BarrierContract,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Preço de uma put down-and-in sem rebate (barreira inferior).

    A opção SÓ paga se o ativo DESCER e tocar a barreira inferior em algum
    momento até o vencimento; caso contrário expira sem valor. Se o spot já
    nasce na barreira ou abaixo dela, a opção já está ativa e equivale à put
    vanilla (obtida por paridade put-call). `knock_out_probability` reporta a
    probabilidade de a opção morrer sem ativação (nunca tocar a barreira).
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
    vanilla = _black_scholes_put_by_parity(market, contract, black_scholes_call(market, contract))
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
    vanilla_payoffs = discount * np.maximum(contract.strike - terminal, 0.0)
    barrier_payoffs = vanilla_payoffs * knocked_in

    # Variável de controle: payoff da put vanilla, com expectação exata
    # conhecida via paridade put-call.
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


def price_up_and_in_call(
    market: MarketData,
    contract: BarrierContract,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Preço de uma call up-and-in sem rebate (barreira superior).

    A opção SÓ paga se o ativo SUBIR e tocar a barreira superior em algum
    momento até o vencimento; caso contrário expira sem valor. Se o spot já
    nasce na barreira ou acima dela, a opção já está ativa e equivale à call
    vanilla. `knock_out_probability` reporta a probabilidade de a opção morrer
    sem ativação (nunca tocar a barreira).
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
        knocked_in = np.any(log_paths >= log_h, axis=1)
    else:
        endpoints_below = (previous < log_h) & (log_paths < log_h)
        crossing_probability = np.ones_like(log_paths)
        crossing_probability[endpoints_below] = np.exp(
            -2.0
            * (log_h - previous[endpoints_below])
            * (log_h - log_paths[endpoints_below])
            / (sigma**2 * dt)
        )
        survived = np.all(endpoints_below & (uniforms > crossing_probability), axis=1)
        knocked_in = ~survived

    terminal = np.exp(log_paths[:, -1])
    discount = exp(-market.rate * t)
    vanilla_payoffs = discount * np.maximum(terminal - contract.strike, 0.0)
    barrier_payoffs = vanilla_payoffs * knocked_in

    # Variável de controle: payoff da call vanilla, com expectação exata
    # conhecida (Black-Scholes).
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


def price_up_and_in_put(
    market: MarketData,
    contract: BarrierContract,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Preço de uma put up-and-in sem rebate (barreira superior).

    A opção SÓ paga se o ativo SUBIR e tocar a barreira superior em algum
    momento até o vencimento; caso contrário expira sem valor. Se o spot já
    nasce na barreira ou acima dela, a opção já está ativa e equivale à put
    vanilla (obtida por paridade put-call). `knock_out_probability` reporta a
    probabilidade de a opção morrer sem ativação (nunca tocar a barreira).
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
    vanilla = _black_scholes_put_by_parity(market, contract, black_scholes_call(market, contract))
    if s0 >= h:
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
        knocked_in = np.any(log_paths >= log_h, axis=1)
    else:
        endpoints_below = (previous < log_h) & (log_paths < log_h)
        crossing_probability = np.ones_like(log_paths)
        crossing_probability[endpoints_below] = np.exp(
            -2.0
            * (log_h - previous[endpoints_below])
            * (log_h - log_paths[endpoints_below])
            / (sigma**2 * dt)
        )
        survived = np.all(endpoints_below & (uniforms > crossing_probability), axis=1)
        knocked_in = ~survived

    terminal = np.exp(log_paths[:, -1])
    discount = exp(-market.rate * t)
    vanilla_payoffs = discount * np.maximum(contract.strike - terminal, 0.0)
    barrier_payoffs = vanilla_payoffs * knocked_in

    # Variável de controle: payoff da put vanilla, com expectação exata
    # conhecida via paridade put-call.
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


def _knock_out_legs(direction: str):
    """(call KO, put KO) para a direção da barreira: 'down' (DOC/DOP) ou 'up' (UOC/UOP)."""
    if direction == "down":
        return price_down_and_out_call, price_down_and_out_put
    if direction == "up":
        return price_up_and_out_call, price_up_and_out_put
    raise ValueError("direction must be 'down' or 'up'")


def price_barrier_strangle(
    market: MarketData,
    contract: BarrierContract,
    strike_2: float,
    *,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    direction: str = "down",
) -> PriceResult:
    """Preço de um Strangle com barreira Down (Família C — estrutura combinada).

    Compra uma Call Down-and-Out (Strike 1 = `contract.strike`) e uma Put
    Down-and-Out (Strike 2 = `strike_2`), ambas com a MESMA barreira inferior.
    Mesma seed → mesmos caminhos e mesmos eventos de knock-out nas duas
    pernas, logo `knock_out_probability` é idêntica entre as pernas.

    O erro padrão reportado é a soma em quadratura dos SEs das pernas
    (limite conservador: ignora a correlação das pernas, que reduz a variância
    verdadeira da soma).

    `direction='up'` monta a variante com barreira SUPERIOR (pernas
    Up-and-Out); o padrão 'down' preserva o comportamento original.
    """
    if not np.isfinite(strike_2) or strike_2 <= 0:
        raise ValueError("strike_2 must be a positive finite number")
    call_ko, put_ko = _knock_out_legs(direction)
    call_leg = call_ko(
        market, contract, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    put_leg = put_ko(
        market,
        replace(contract, strike=strike_2),
        paths=paths,
        steps=steps,
        seed=seed,
        monitoring=monitoring,
    )
    price = call_leg.price + put_leg.price
    standard_error = (call_leg.standard_error**2 + put_leg.standard_error**2) ** 0.5
    return SimpleNamespace(
        price=price,
        standard_error=standard_error,
        ci_low=price - 1.96 * standard_error,
        ci_high=price + 1.96 * standard_error,
        knock_out_probability=call_leg.knock_out_probability,
        vanilla_price=call_leg.vanilla_price + put_leg.vanilla_price,
    )


def price_barrier_risk_reversal(
    market: MarketData,
    contract: BarrierContract,
    strike_2: float,
    *,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Preço de um Risk Reversal com barreira Down (Família C — estrutura combinada).

    Compra uma Call Down-and-Out (Strike 1 = `contract.strike`) e VENDE uma
    Put Down-and-Out (Strike 2 = `strike_2`), ambas com a MESMA barreira
    inferior. Preço líquido = call − put: NEGATIVO significa crédito líquido
    RECEBIDO pela venda da perna put. O payoff delta das pernas usa a mesma
    seed (mesmos caminhos e knock-outs).

    O erro padrão reportado é a soma em quadratura dos SEs das pernas
    (limite conservador para a diferença de pernas correlacionadas).
    """
    if not np.isfinite(strike_2) or strike_2 <= 0:
        raise ValueError("strike_2 must be a positive finite number")
    call_leg = price_down_and_out_call(
        market, contract, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    put_leg = price_down_and_out_put(
        market,
        replace(contract, strike=strike_2),
        paths=paths,
        steps=steps,
        seed=seed,
        monitoring=monitoring,
    )
    price = call_leg.price - put_leg.price
    standard_error = (call_leg.standard_error**2 + put_leg.standard_error**2) ** 0.5
    return SimpleNamespace(
        price=price,
        standard_error=standard_error,
        ci_low=price - 1.96 * standard_error,
        ci_high=price + 1.96 * standard_error,
        knock_out_probability=call_leg.knock_out_probability,
        vanilla_price=call_leg.vanilla_price - put_leg.vanilla_price,
    )


def price_one_touch_up(
    market: MarketData,
    contract: BarrierContract,
    payout: float = 100.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
) -> PriceResult:
    """One-Touch Up: Paga o 'payout' se o ativo tocar a barreira superior em QUALQUER momento.

    Binário de barreira (Família D): payoff fixo descontado multiplicado pela
    probabilidade de toque. O campo `knock_out_probability` reporta P(tocado)
    nesta função (convenção binária). Monitoramento Brownian Bridge sobre a
    barreira logarítmica, payoff liquidado no vencimento.
    """
    s0, h, t = market.spot, contract.barrier, contract.maturity
    sigma = market.volatility
    if s0 >= h:
        # Barreira já tocada: paga o payout com certeza (P(tocado) = 1).
        return PriceResult(
            payout * exp(-market.rate * t),
            0.0,
            payout * exp(-market.rate * t),
            payout * exp(-market.rate * t),
            1.0,
            0.0,
        )

    rng = np.random.default_rng(seed)
    dt = t / steps
    z = rng.standard_normal((paths, steps))
    uniforms = rng.random((paths, steps))

    drift = (market.rate - market.dividend_yield - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_paths = np.log(s0) + np.cumsum(drift + diffusion * z, axis=1)
    log_h = np.log(h)
    previous = np.concatenate((np.full((paths, 1), np.log(s0)), log_paths[:, :-1]), axis=1)

    endpoints_below = (previous < log_h) & (log_paths < log_h)
    crossing_probability = np.ones_like(log_paths)
    crossing_probability[endpoints_below] = np.exp(
        -2.0 * (log_h - previous[endpoints_below]) * (log_h - log_paths[endpoints_below]) / (sigma**2 * dt)
    )
    survived = np.all(endpoints_below & (uniforms > crossing_probability), axis=1)
    touched = ~survived

    discount = exp(-market.rate * t)
    payoffs = discount * payout * touched

    price = float(payoffs.mean())
    standard_error = float(payoffs.std(ddof=1) / sqrt(paths))
    return PriceResult(
        price,
        standard_error,
        price - 1.96 * standard_error,
        price + 1.96 * standard_error,
        float(touched.mean()),
        0.0,
    )


def price_no_touch_up(
    market: MarketData,
    contract: BarrierContract,
    payout: float = 100.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
) -> PriceResult:
    """No-Touch Up: Paga o 'payout' se o ativo NÃO tocar a barreira superior em momento algum.

    Complemento binário do One-Touch. O erro padrão herdado do One-Touch não é
    aproximação: payoffs binários complementares têm variância idêntica
    (p(1-p)), logo o SE do No-Touch é EXATAMENTE o SE do One-Touch.
    O campo `knock_out_probability` reporta P(não tocado).
    """
    # Roda o One-Touch para pegar os caminhos idênticos
    res_ot = price_one_touch_up(market, contract, payout, paths, steps, seed)
    p_not_touch = 1.0 - res_ot.knock_out_probability
    discount = exp(-market.rate * contract.maturity)
    price = discount * payout * p_not_touch
    standard_error = res_ot.standard_error  # Exato: variância de Bernoulli complementar
    return PriceResult(
        price,
        standard_error,
        price - 1.96 * standard_error,
        price + 1.96 * standard_error,
        p_not_touch,
        0.0,
    )


def price_barrier_collar(
    market: MarketData,
    contract: BarrierContract,
    strike_2: float,
    *,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    direction: str = "down",
):
    """Collar com barreira Down (Família C — lacuna fácil).

    Compra uma Put Down-and-Out (Strike 1 = `contract.strike`) e VENDE uma
    Call Down-and-Out (Strike 2 = `strike_2`), ambas com a MESMA barreira
    inferior. Preço líquido = put − call (negativo = crédito líquido).
    Mesma seed → mesmos caminhos e knock-outs; SE em quadratura
    (conservador, ignora a correlação das pernas).

    `direction='up'` monta a variante com barreira SUPERIOR (pernas
    Up-and-Out); o padrão 'down' preserva o comportamento original.
    """
    if not np.isfinite(strike_2) or strike_2 <= 0:
        raise ValueError("strike_2 must be a positive finite number")
    call_ko, put_ko = _knock_out_legs(direction)
    put_leg = put_ko(market, contract, paths=paths, steps=steps, seed=seed, monitoring=monitoring)
    call_leg = call_ko(
        market, replace(contract, strike=strike_2), paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    price = put_leg.price - call_leg.price
    se = (put_leg.standard_error**2 + call_leg.standard_error**2) ** 0.5
    return SimpleNamespace(
        price=price,
        standard_error=se,
        ci_low=price - 1.96 * se,
        ci_high=price + 1.96 * se,
        knock_out_probability=put_leg.knock_out_probability,
        vanilla_price=put_leg.vanilla_price - call_leg.vanilla_price,
    )


def price_barrier_call_spread(
    market: MarketData,
    contract: BarrierContract,
    strike_2: float,
    *,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    direction: str = "down",
):
    """Call Spread com barreira Down (Família C — lacuna fácil).

    Compra uma Call Down-and-Out (Strike 1) e VENDE uma Call Down-and-Out
    (Strike 2, tipicamente acima do Strike 1 para débito líquido).
    Payoff limitado nos dois lados: risco e retorno known em troca de
    prêmio menor que a call avulsa.

    `direction='up'` monta a variante com barreira SUPERIOR (pernas
    Up-and-Out); o padrão 'down' preserva o comportamento original.
    """
    if not np.isfinite(strike_2) or strike_2 <= 0:
        raise ValueError("strike_2 must be a positive finite number")
    call_ko, put_ko = _knock_out_legs(direction)
    call_1 = call_ko(market, contract, paths=paths, steps=steps, seed=seed, monitoring=monitoring)
    call_2 = call_ko(
        market, replace(contract, strike=strike_2), paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    price = call_1.price - call_2.price
    se = (call_1.standard_error**2 + call_2.standard_error**2) ** 0.5
    return SimpleNamespace(
        price=price,
        standard_error=se,
        ci_low=price - 1.96 * se,
        ci_high=price + 1.96 * se,
        knock_out_probability=call_1.knock_out_probability,
        vanilla_price=call_1.vanilla_price - call_2.vanilla_price,
    )


def price_barrier_put_spread(
    market: MarketData,
    contract: BarrierContract,
    strike_2: float,
    *,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
    direction: str = "down",
):
    """Put Spread com barreira Down (Família C — lacuna fácil).

    Compra uma Put Down-and-Out (Strike 1) e VENDE uma Put Down-and-Out
    (Strike 2). Com Strike 2 abaixo do Strike 1, é o spread de put
    clássico de débito (proteção limitada ao crash, mais barato que a
    put avulsa — e a barreira Down corta ainda o custo da perna longa).

    `direction='up'` monta a variante com barreira SUPERIOR (pernas
    Up-and-Out); o padrão 'down' preserva o comportamento original.
    """
    if not np.isfinite(strike_2) or strike_2 <= 0:
        raise ValueError("strike_2 must be a positive finite number")
    call_ko, put_ko = _knock_out_legs(direction)
    put_1 = put_ko(market, contract, paths=paths, steps=steps, seed=seed, monitoring=monitoring)
    put_2 = put_ko(
        market, replace(contract, strike=strike_2), paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    price = put_1.price - put_2.price
    se = (put_1.standard_error**2 + put_2.standard_error**2) ** 0.5
    return SimpleNamespace(
        price=price,
        standard_error=se,
        ci_low=price - 1.96 * se,
        ci_high=price + 1.96 * se,
        knock_out_probability=put_1.knock_out_probability,
        vanilla_price=put_1.vanilla_price - put_2.vanilla_price,
    )


def price_knock_out_forward(
    market: MarketData,
    contract: BarrierContract,
    *,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
):
    """Forward sintético com barreira Down (Família C — lacuna fácil).

    Compra uma Call Down-and-Out e VENDE uma Put Down-and-Out com o MESMO
    Strike e a MESMA barreira (paridade put-call barrier). O valor forward
    vanilla seria S0·e^(−qT) − K·e^(−rT); a barreira Down AGREGA valor a
    essa parcela: no knock-out as duas pernas morrem e o payoff vira zero
    em vez de S_T − K < 0, eliminando o downside do crash. Por isso o
    KO Forward custa MAIS que o forward vanilla (prêmio da proteção
    implícita) — validado em smoke: +271,77 com barreira a 70%.
    """
    call = price_down_and_out_call(market, contract, paths=paths, steps=steps, seed=seed, monitoring=monitoring)
    put = price_down_and_out_put(market, contract, paths=paths, steps=steps, seed=seed, monitoring=monitoring)
    price = call.price - put.price
    se = (call.standard_error**2 + put.standard_error**2) ** 0.5
    return SimpleNamespace(
        price=price,
        standard_error=se,
        ci_low=price - 1.96 * se,
        ci_high=price + 1.96 * se,
        knock_out_probability=call.knock_out_probability,
        vanilla_price=call.vanilla_price - put.vanilla_price,
    )