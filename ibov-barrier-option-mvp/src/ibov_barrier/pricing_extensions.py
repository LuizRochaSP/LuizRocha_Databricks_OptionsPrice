"""Extensões do motor: rebate, binários Down, Knock-In combinadas, Range Accrual
e binários de dupla barreira (lacunas fáceis e médias do roadmap).

Diferente de `pricing_barriers`, este módulo centraliza a simulação em dois
blocos reutilizáveis:

- `_simulate_log_paths`: gera os log-caminhos GBM com a MESMA sequência de
  números aleatórios do motor original (normais primeiro, depois uniformes do
  Brownian Bridge). Por isso `price_single_barrier(..., rebate=0)` reproduz
  exatamente os preços das funções DOC/UOC/DIC/UIC/DOP/UOP/DIP/UIP.
- `_touched`: indica, por caminho, se a barreira foi tocada (monitoramento
  discreto ou Brownian Bridge).

Cada produto novo vem com um benchmark analítico (fórmulas fechadas sob GBM
com monitoramento contínuo), usado nos testes para validar o Monte Carlo.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm

from ibov_barrier.pricing import (
    BarrierContract,
    MarketData,
    PriceResult,
    black_scholes_call,
)
from ibov_barrier.pricing_barriers import _black_scholes_put_by_parity

_OPTION_TYPES = {"call", "put"}
_DIRECTIONS = {"down", "up"}
_KNOCKS = {"out", "in"}
_MONITORING = {"discrete", "brownian_bridge"}


# ---------------------------------------------------------------------------
# Blocos de simulação compartilhados
# ---------------------------------------------------------------------------

def _validate_market(market: MarketData, maturity: float, *levels: float) -> None:
    values = np.asarray(
        [market.spot, market.rate, market.dividend_yield, market.volatility, maturity, *levels],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("market and contract parameters must be finite")
    if min(market.spot, market.volatility, maturity, *levels) <= 0:
        raise ValueError("spot, levels, maturity and volatility must be positive")


def _validate_simulation(paths: int, steps: int, monitoring: str) -> None:
    if paths < 2 or steps < 1:
        raise ValueError("paths must be >= 2 and steps must be >= 1")
    if monitoring not in _MONITORING:
        raise ValueError("monitoring must be 'discrete' or 'brownian_bridge'")


def _simulate_log_paths(
    market: MarketData,
    maturity: float,
    *,
    paths: int,
    steps: int,
    seed: int,
    monitoring: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """Retorna (log_paths, previous, uniforms, dt) com a ordem de sorteio do motor original."""
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    z = rng.standard_normal((paths, steps))
    uniforms = rng.random((paths, steps)) if monitoring == "brownian_bridge" else None
    sigma = market.volatility
    drift = (market.rate - market.dividend_yield - 0.5 * sigma**2) * dt
    log_s0 = log(market.spot)
    log_paths = log_s0 + np.cumsum(drift + sigma * sqrt(dt) * z, axis=1)
    previous = np.concatenate((np.full((paths, 1), log_s0), log_paths[:, :-1]), axis=1)
    return log_paths, previous, uniforms, dt


def _crossing_probability(
    previous: np.ndarray,
    log_paths: np.ndarray,
    log_h: float,
    direction: str,
    sigma: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidade Brownian Bridge de cruzar a barreira em cada passo.

    Retorna (inside, prob): `inside` indica os passos com as duas pontas do
    lado seguro da barreira; `prob` vale 1 nos demais (toque observado).
    """
    if direction == "down":
        inside = (previous > log_h) & (log_paths > log_h)
        dist_a, dist_b = previous - log_h, log_paths - log_h
    else:
        inside = (previous < log_h) & (log_paths < log_h)
        dist_a, dist_b = log_h - previous, log_h - log_paths
    prob = np.ones_like(log_paths)
    prob[inside] = np.exp(-2.0 * dist_a[inside] * dist_b[inside] / (sigma**2 * dt))
    return inside, prob


def _touched(
    log_paths: np.ndarray,
    previous: np.ndarray,
    uniforms: np.ndarray | None,
    barrier: float,
    direction: str,
    sigma: float,
    dt: float,
    monitoring: str,
) -> np.ndarray:
    """Vetor booleano: o caminho tocou a barreira em algum momento?"""
    log_h = log(barrier)
    if monitoring == "discrete":
        if direction == "down":
            return ~np.all(log_paths > log_h, axis=1)
        return ~np.all(log_paths < log_h, axis=1)
    inside, prob = _crossing_probability(previous, log_paths, log_h, direction, sigma, dt)
    return ~np.all(inside & (uniforms > prob), axis=1)


def _summary(payoffs: np.ndarray, probability: float, vanilla: float) -> PriceResult:
    price = float(payoffs.mean())
    se = float(payoffs.std(ddof=1) / sqrt(payoffs.size))
    return PriceResult(price, se, price - 1.96 * se, price + 1.96 * se, probability, vanilla)


def _combine(legs: list[tuple[float, PriceResult]], probability: float) -> PriceResult:
    """Combina pernas com pesos (+1 compra, −1 venda). SE em quadratura (conservador)."""
    price = sum(w * leg.price for w, leg in legs)
    se = sqrt(sum(leg.standard_error**2 for _, leg in legs))
    vanilla = sum(w * leg.vanilla_price for w, leg in legs)
    return PriceResult(price, se, price - 1.96 * se, price + 1.96 * se, probability, vanilla)


# ---------------------------------------------------------------------------
# Benchmarks analíticos (GBM, monitoramento contínuo)
# ---------------------------------------------------------------------------

def barrier_hit_probability_analytical(
    spot: float,
    barrier: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """P(tocar a barreira até T) para GBM sob a medida neutra ao risco.

    Fórmula do princípio da reflexão para Browniano com drift
    μ = r − q − σ²/2 e b = ln(H/S₀):
      Down (b<0): N((b − μT)/σ√T) + e^{2μb/σ²}·N((b + μT)/σ√T)
      Up   (b>0): N((−b + μT)/σ√T) + e^{2μb/σ²}·N((−b − μT)/σ√T)
    """
    if barrier == spot:
        return 1.0
    mu = rate - dividend_yield - 0.5 * volatility**2
    b = log(barrier / spot)
    s = volatility * sqrt(maturity)
    weight = exp(2.0 * mu * b / volatility**2)
    if b < 0:
        p = norm.cdf((b - mu * maturity) / s) + weight * norm.cdf((b + mu * maturity) / s)
    else:
        p = norm.cdf((-b + mu * maturity) / s) + weight * norm.cdf((-b - mu * maturity) / s)
    return float(min(max(p, 0.0), 1.0))


def double_no_touch_probability_analytical(
    spot: float,
    lower: float,
    upper: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    terms: int = 200,
) -> float:
    """P(não tocar nenhuma das barreiras L < S₀ < U até T) — série de Fourier.

    Densidade do Browniano com drift μ morto em 0 e ℓ = ln(U/L), partindo de
    x = ln(S₀/L) (expansão em autofunções sin(nπy/ℓ)). Integrando em y:

      P = (2/ℓ)·e^{−a x − μ²T/(2σ²)} · Σ sin(bₙx)·e^{−σ²bₙ²T/2}·bₙ(1 − (−1)ⁿe^{aℓ})/(a² + bₙ²)

    com a = μ/σ² e bₙ = nπ/ℓ.
    """
    if not lower < spot < upper:
        return 0.0
    sigma2 = volatility**2
    mu = rate - dividend_yield - 0.5 * sigma2
    a = mu / sigma2
    width = log(upper / lower)
    x = log(spot / lower)
    n = np.arange(1, terms + 1)
    b = n * np.pi / width
    series = (
        np.sin(b * x)
        * np.exp(-0.5 * sigma2 * b**2 * maturity)
        * b * (1.0 - (-1.0) ** n * exp(a * width))
        / (a**2 + b**2)
    )
    p = (2.0 / width) * exp(-a * x - mu**2 * maturity / (2.0 * sigma2)) * float(series.sum())
    return float(min(max(p, 0.0), 1.0))


def _accrual_amount(notional: float, accrual_rate: float | None, maturity: float) -> float:
    """Valor máximo pago (100% dos fixings na faixa).

    Sem `accrual_rate`: Notional (convenção original, payoff = Notional × fração).
    Com `accrual_rate` (decimal a.a., ex. 0.10): cupom = Notional × taxa × T.
    """
    if not np.isfinite(notional) or notional <= 0:
        raise ValueError("notional must be a positive finite number")
    if accrual_rate is None:
        return notional
    if not np.isfinite(accrual_rate) or accrual_rate <= 0:
        raise ValueError("accrual_rate must be a positive finite number")
    return notional * accrual_rate * maturity


def range_accrual_analytical(
    market: MarketData,
    maturity: float,
    *,
    lower: float | None,
    upper: float | None,
    notional: float = 100.0,
    observations: int = 252,
    accrual_rate: float | None = None,
) -> float:
    """Valor exato do Range Accrual sob GBM (linearidade da esperança).

    E[fração] = (1/N)·Σᵢ P(L ≤ S_{tᵢ} ≤ U), com S_{tᵢ} lognormal, tᵢ = i·T/N.
    Preço = Notional · e^{−rT} · E[fração]  (ou Notional · taxa · T · e^{−rT} · E[fração]
    quando `accrual_rate` é informado).
    """
    lo, hi = _range_bounds(lower, upper)
    sigma = market.volatility
    mu = market.rate - market.dividend_yield - 0.5 * sigma**2
    t = maturity * np.arange(1, observations + 1) / observations
    scale = sigma * np.sqrt(t)

    def cdf(level: float) -> np.ndarray:
        if level == np.inf:
            return np.ones_like(t)
        if level == 0.0:
            return np.zeros_like(t)
        return norm.cdf((log(level / market.spot) - mu * t) / scale)

    fraction = float(np.mean(cdf(hi) - cdf(lo)))
    return _accrual_amount(notional, accrual_rate, maturity) * exp(-market.rate * maturity) * fraction


# ---------------------------------------------------------------------------
# Família A com Rebate
# ---------------------------------------------------------------------------

def price_single_barrier(
    market: MarketData,
    contract: BarrierContract,
    *,
    option_type: str = "call",
    direction: str = "down",
    knock: str = "out",
    rebate: float = 0.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Precificador genérico de barreira simples (DOC, UOC, DIC, UIC, DOP, UOP, DIP, UIP) com rebate.

    Rebate (pago no vencimento):
    - Knock-Out: o cliente recebe `rebate` se a barreira for tocada (a opção morreu).
    - Knock-In: o cliente recebe `rebate` se a barreira NUNCA for tocada (a opção não ativou).

    Com `rebate=0` reproduz exatamente as funções dedicadas de `pricing` e
    `pricing_barriers` (mesmos caminhos, mesma variável de controle vanilla).
    `knock_out_probability` segue a convenção existente: P(tocar) para
    Knock-Out e P(não ativar) para Knock-In.
    """
    if option_type not in _OPTION_TYPES:
        raise ValueError("option_type must be 'call' or 'put'")
    if direction not in _DIRECTIONS:
        raise ValueError("direction must be 'down' or 'up'")
    if knock not in _KNOCKS:
        raise ValueError("knock must be 'out' or 'in'")
    if not np.isfinite(rebate) or rebate < 0:
        raise ValueError("rebate must be a non-negative finite number")
    _validate_market(market, contract.maturity, contract.strike, contract.barrier)
    _validate_simulation(paths, steps, monitoring)

    t = contract.maturity
    discount = exp(-market.rate * t)
    vanilla = black_scholes_call(market, contract)
    if option_type == "put":
        vanilla = _black_scholes_put_by_parity(market, contract, vanilla)

    already_touched = (
        market.spot <= contract.barrier if direction == "down" else market.spot >= contract.barrier
    )
    if already_touched:
        # Knock-Out já morto: só resta o rebate. Knock-In já ativo: vira vanilla.
        value = rebate * discount if knock == "out" else vanilla
        return PriceResult(value, 0.0, value, value, 1.0 if knock == "out" else 0.0, vanilla)

    log_paths, previous, uniforms, dt = _simulate_log_paths(
        market, t, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    touched = _touched(
        log_paths, previous, uniforms, contract.barrier, direction, market.volatility, dt, monitoring
    )
    active = ~touched if knock == "out" else touched

    terminal = np.exp(log_paths[:, -1])
    intrinsic = terminal - contract.strike if option_type == "call" else contract.strike - terminal
    vanilla_payoffs = discount * np.maximum(intrinsic, 0.0)
    payoffs = vanilla_payoffs * active + rebate * discount * ~active

    # Variável de controle vanilla (expectação exata conhecida).
    vanilla_variance = float(vanilla_payoffs.var(ddof=1))
    beta = (
        float(np.cov(payoffs, vanilla_payoffs, ddof=1)[0, 1]) / vanilla_variance
        if vanilla_variance > 0.0
        else 0.0
    )
    adjusted = payoffs - beta * (vanilla_payoffs - vanilla)
    return _summary(adjusted, float(1.0 - active.mean()), vanilla)


# ---------------------------------------------------------------------------
# Família D: binários de toque (Down e genéricos)
# ---------------------------------------------------------------------------

def price_one_touch(
    market: MarketData,
    contract: BarrierContract,
    *,
    direction: str,
    payout: float = 100.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """One-Touch (pago no vencimento): `payout` se a barreira for tocada até T.

    `knock_out_probability` reporta P(tocado). `contract.strike` é ignorado.
    """
    if direction not in _DIRECTIONS:
        raise ValueError("direction must be 'down' or 'up'")
    if not np.isfinite(payout) or payout <= 0:
        raise ValueError("payout must be a positive finite number")
    _validate_market(market, contract.maturity, contract.barrier)
    _validate_simulation(paths, steps, monitoring)
    discount = exp(-market.rate * contract.maturity)
    already_touched = (
        market.spot <= contract.barrier if direction == "down" else market.spot >= contract.barrier
    )
    if already_touched:
        value = payout * discount
        return PriceResult(value, 0.0, value, value, 1.0, 0.0)

    log_paths, previous, uniforms, dt = _simulate_log_paths(
        market, contract.maturity, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    touched = _touched(
        log_paths, previous, uniforms, contract.barrier, direction, market.volatility, dt, monitoring
    )
    return _summary(discount * payout * touched, float(touched.mean()), 0.0)


def price_no_touch(
    market: MarketData,
    contract: BarrierContract,
    *,
    direction: str,
    payout: float = 100.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """No-Touch (pago no vencimento): `payout` se a barreira NUNCA for tocada.

    Complemento exato do One-Touch nos mesmos caminhos: NT = payout·e^{−rT} − OT,
    com o mesmo erro-padrão (variância de Bernoulli complementar).
    `knock_out_probability` reporta P(não tocado).
    """
    one_touch = price_one_touch(
        market, contract, direction=direction, payout=payout,
        paths=paths, steps=steps, seed=seed, monitoring=monitoring,
    )
    full = payout * exp(-market.rate * contract.maturity)
    price = full - one_touch.price
    se = one_touch.standard_error
    return PriceResult(price, se, price - 1.96 * se, price + 1.96 * se,
                       1.0 - one_touch.knock_out_probability, 0.0)


def price_one_touch_down(market, contract, payout=100.0, paths=200_000, steps=252, seed=42,
                         monitoring="brownian_bridge") -> PriceResult:
    """One-Touch Down: paga `payout` se o ativo tocar a barreira INFERIOR."""
    return price_one_touch(market, contract, direction="down", payout=payout,
                           paths=paths, steps=steps, seed=seed, monitoring=monitoring)


def price_no_touch_down(market, contract, payout=100.0, paths=200_000, steps=252, seed=42,
                        monitoring="brownian_bridge") -> PriceResult:
    """No-Touch Down: paga `payout` se o ativo NUNCA tocar a barreira INFERIOR."""
    return price_no_touch(market, contract, direction="down", payout=payout,
                          paths=paths, steps=steps, seed=seed, monitoring=monitoring)


# ---------------------------------------------------------------------------
# Família C: estruturas Knock-In
# ---------------------------------------------------------------------------

def _knock_in_legs(market, contract, direction, paths, steps, seed, monitoring):
    kwargs = dict(direction=direction, knock="in", paths=paths, steps=steps, seed=seed,
                  monitoring=monitoring)
    call = price_single_barrier(market, contract, option_type="call", **kwargs)
    put = price_single_barrier(market, contract, option_type="put", **kwargs)
    return call, put


def price_knock_in_straddle(
    market: MarketData,
    contract: BarrierContract,
    *,
    direction: str = "down",
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Knock-In Straddle: compra Call KI + Put KI, mesmo strike e mesma barreira.

    A estrutura só nasce se a barreira for tocada — aposta em volatilidade
    condicionada a um movimento extremo. `knock_out_probability` = P(não ativar).
    """
    call, put = _knock_in_legs(market, contract, direction, paths, steps, seed, monitoring)
    return _combine([(1.0, call), (1.0, put)], call.knock_out_probability)


def price_knock_in_forward(
    market: MarketData,
    contract: BarrierContract,
    *,
    direction: str = "down",
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Knock-In Forward: compra Call KI − vende Put KI, mesmo strike e barreira.

    Forward sintético que só passa a existir se a barreira for tocada. Por
    paridade In/Out: KI Forward + KO Forward = forward vanilla
    S₀·e^{−qT} − K·e^{−rT}. `knock_out_probability` = P(não ativar).
    """
    call, put = _knock_in_legs(market, contract, direction, paths, steps, seed, monitoring)
    return _combine([(1.0, call), (-1.0, put)], call.knock_out_probability)


# ---------------------------------------------------------------------------
# Família D: Range Accrual
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeAccrualResult:
    price: float
    standard_error: float
    ci_low: float
    ci_high: float
    expected_fraction: float
    analytical_price: float
    observations: int


def _range_bounds(lower: float | None, upper: float | None) -> tuple[float, float]:
    if lower is None and upper is None:
        raise ValueError("at least one of lower/upper must be provided")
    lo = 0.0 if lower is None else float(lower)
    hi = np.inf if upper is None else float(upper)
    if lower is not None and (not np.isfinite(lo) or lo <= 0):
        raise ValueError("lower must be a positive finite number")
    if upper is not None and (not np.isfinite(hi) or hi <= 0):
        raise ValueError("upper must be a positive finite number")
    if lo >= hi:
        raise ValueError("lower must be strictly below upper")
    return lo, hi


def price_range_accrual(
    market: MarketData,
    maturity: float,
    *,
    lower: float | None,
    upper: float | None,
    notional: float = 100.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    accrual_rate: float | None = None,
) -> RangeAccrualResult:
    """Range Accrual: paga Notional × (fixings dentro da faixa) / (total de fixings), no vencimento.

    Com `accrual_rate` (decimal a.a., ex. 0.10 = 10% a.a.) o pagamento vira o
    cupom de mercado: Notional × taxa × T × fração (a taxa anual é proporcional
    ao prazo; sem o fator T um cupom de 6 meses seria pago como se fosse de 1 ano).

    Cada passo da simulação é uma data de fixing (use `steps` = dias úteis do
    prazo para acúmulo diário). Variantes:
    - túnel: `lower` e `upper` informados (acumula enquanto L ≤ S ≤ U);
    - Down: só `lower` (acumula enquanto S ≥ L);
    - Up: só `upper` (acumula enquanto S ≤ U).

    O benchmark exato `analytical_price` é retornado junto para auditoria.
    """
    lo, hi = _range_bounds(lower, upper)
    amount = _accrual_amount(notional, accrual_rate, maturity)
    _validate_market(market, maturity)
    _validate_simulation(paths, steps, "discrete")

    log_paths, _, _, _ = _simulate_log_paths(
        market, maturity, paths=paths, steps=steps, seed=seed, monitoring="discrete"
    )
    spots = np.exp(log_paths)
    fraction = ((spots >= lo) & (spots <= hi)).mean(axis=1)
    payoffs = amount * exp(-market.rate * maturity) * fraction
    base = _summary(payoffs, 0.0, 0.0)
    analytical = range_accrual_analytical(
        market, maturity, lower=lower, upper=upper, notional=notional, observations=steps,
        accrual_rate=accrual_rate,
    )
    return RangeAccrualResult(
        price=base.price,
        standard_error=base.standard_error,
        ci_low=base.ci_low,
        ci_high=base.ci_high,
        expected_fraction=float(fraction.mean()),
        analytical_price=analytical,
        observations=steps,
    )


# ---------------------------------------------------------------------------
# Família D: binários de dupla barreira
# ---------------------------------------------------------------------------

def _double_touched(
    market: MarketData,
    lower: float,
    upper: float,
    maturity: float,
    *,
    paths: int,
    steps: int,
    seed: int,
    monitoring: str,
) -> np.ndarray:
    """Toque em qualquer das barreiras L < S₀ < U.

    Brownian Bridge de dupla barreira: num passo com as duas pontas dentro do
    túnel, P(sair) ≈ p_L + p_U. Os termos omitidos da série exata exigem
    cruzar as DUAS barreiras no mesmo passo (ordem e^{−2ℓ²/σ²dt}) e são
    desprezíveis para os passos usados. Um único uniforme por passo decide o toque.
    """
    log_paths, previous, uniforms, dt = _simulate_log_paths(
        market, maturity, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    log_l, log_u = log(lower), log(upper)
    if monitoring == "discrete":
        return ~np.all((log_paths > log_l) & (log_paths < log_u), axis=1)
    sigma = market.volatility
    inside_l, prob_l = _crossing_probability(previous, log_paths, log_l, "down", sigma, dt)
    inside_u, prob_u = _crossing_probability(previous, log_paths, log_u, "up", sigma, dt)
    inside = inside_l & inside_u
    prob = np.minimum(prob_l + prob_u, 1.0)
    return ~np.all(inside & (uniforms > prob), axis=1)


def _validate_double(market, lower, upper, maturity, payout, paths, steps, monitoring):
    _validate_market(market, maturity, lower, upper)
    _validate_simulation(paths, steps, monitoring)
    if not lower < upper:
        raise ValueError("lower must be strictly below upper")
    if not np.isfinite(payout) or payout <= 0:
        raise ValueError("payout must be a positive finite number")


def price_double_one_touch(
    market: MarketData,
    *,
    lower: float,
    upper: float,
    maturity: float,
    payout: float = 100.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Double One-Touch: `payout` no vencimento se tocar L OU U até T.

    `knock_out_probability` reporta P(tocado).
    """
    _validate_double(market, lower, upper, maturity, payout, paths, steps, monitoring)
    discount = exp(-market.rate * maturity)
    if not lower < market.spot < upper:
        value = payout * discount
        return PriceResult(value, 0.0, value, value, 1.0, 0.0)
    touched = _double_touched(
        market, lower, upper, maturity, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    return _summary(discount * payout * touched, float(touched.mean()), 0.0)


def price_double_no_touch(
    market: MarketData,
    *,
    lower: float,
    upper: float,
    maturity: float,
    payout: float = 100.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Double No-Touch: `payout` no vencimento se o ativo NUNCA sair do túnel (L, U).

    Complemento exato do Double One-Touch. `knock_out_probability` reporta P(não tocado).
    """
    dot = price_double_one_touch(
        market, lower=lower, upper=upper, maturity=maturity, payout=payout,
        paths=paths, steps=steps, seed=seed, monitoring=monitoring,
    )
    full = payout * exp(-market.rate * maturity)
    price = full - dot.price
    se = dot.standard_error
    return PriceResult(price, se, price - 1.96 * se, price + 1.96 * se,
                       1.0 - dot.knock_out_probability, 0.0)
