"""Barreiras dependentes da trajetória (lacunas difíceis nº 8, 9 e 10 do roadmap).

Produtos (call ou put, barreira Down ou Up, Knock-Out ou Knock-In, rebate no vencimento):

- Parisian (`price_parisian_barrier`): o evento de barreira só acontece se o
  ativo FECHAR além da barreira por `window` fechamentos CONSECUTIVOS
  (ex.: 5 dias úteis seguidos abaixo de 80% do spot). Com `cumulative=True`
  vira a variante "ParAsian": conta o total de fechamentos além da barreira,
  sem zerar o contador quando o ativo volta. Monitoramento discreto por
  construção (1 fechamento por passo), que é como o produto é escrito no mercado.
- Step (`price_step_barrier`): a barreira muda de nível em datas
  pré-definidas, `schedule = [(0.0, H1), (t1, H2), ...]` (tempo em anos).
- Soft / janela (`price_soft_barrier`): a barreira só é monitorada dentro de
  janelas de tempo, `windows = [(t_ini, t_fim), ...]` (partial-time / window
  barrier). Fora das janelas o ativo pode cruzar o nível sem consequência.
  Obs.: na literatura (Hart-Ross, Haug) "soft barrier" também designa a
  barreira com knock-out proporcional dentro de uma faixa; aqui seguimos a
  definição do roadmap do projeto (monitoramento em janelas).

Motor: os mesmos log-caminhos GBM do motor original (`_simulate_log_paths`).
Step e Soft usam uma barreira POR PASSO (vetor de log-níveis, com ±inf para
"barreira desligada") no Brownian Bridge, então são contínuas dentro de cada
passo; a data de mudança/janela é aplicada pelo ponto médio de cada passo
(use datas múltiplas de T/steps para alinhamento exato). Preço com variável
de controle vanilla, como no restante do motor.

Benchmarks semi-analíticos independentes do Monte Carlo (GBM):
- `parisian_barrier_quadrature`: cadeia de Markov exata do monitoramento
  discreto em grade de log-preço, com estado aumentado pelo contador de
  fechamentos além da barreira (convolução gaussiana passo a passo).
- `segmented_barrier_quadrature`: monitoramento CONTÍNUO por segmentos de
  tempo; em cada segmento usa a densidade de transição exata do movimento
  browniano com drift morto na barreira (princípio da reflexão) e encadeia os
  segmentos por quadratura. Cobre Step e Soft; com um único segmento reproduz
  Reiner-Rubinstein.
"""
from __future__ import annotations

from math import ceil, exp, log, pi, sqrt
from typing import Sequence

import numpy as np
from scipy.signal import fftconvolve

from ibov_barrier.pricing import MarketData, PriceResult
from ibov_barrier.pricing_double import _controlled, _vanilla
from ibov_barrier.pricing_extensions import (
    _crossing_probability,
    _simulate_log_paths,
    _validate_market,
    _validate_simulation,
)

_OPTION_TYPES = {"call", "put"}
_DIRECTIONS = {"down", "up"}
_KNOCKS = {"out", "in"}


# ---------------------------------------------------------------------------
# Validação e utilidades
# ---------------------------------------------------------------------------

def _validate_product(option_type: str, direction: str, knock: str, rebate: float) -> None:
    if option_type not in _OPTION_TYPES:
        raise ValueError("option_type must be 'call' or 'put'")
    if direction not in _DIRECTIONS:
        raise ValueError("direction must be 'down' or 'up'")
    if knock not in _KNOCKS:
        raise ValueError("knock must be 'out' or 'in'")
    if not np.isfinite(rebate) or rebate < 0:
        raise ValueError("rebate must be a non-negative finite number")


def _beyond(x: np.ndarray, log_h: float, direction: str) -> np.ndarray:
    """Lado "proibido" da barreira (fechamento além dela)."""
    return x <= log_h if direction == "down" else x >= log_h


def _normalize_schedule(schedule: Sequence[tuple[float, float]], maturity: float) -> list[tuple[float, float]]:
    items = [(float(t), float(level)) for t, level in schedule]
    if not items:
        raise ValueError("schedule must have at least one (time, level) pair")
    if abs(items[0][0]) > 1e-12:
        raise ValueError("schedule must start at time 0")
    times = [t for t, _ in items]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("schedule times must be strictly increasing")
    if times[-1] >= maturity:
        raise ValueError("schedule times must be before maturity")
    if not all(np.isfinite(level) and level > 0 for _, level in items):
        raise ValueError("schedule levels must be positive and finite")
    return items


def _normalize_windows(windows: Sequence[tuple[float, float]], maturity: float) -> list[tuple[float, float]]:
    items = sorted((float(a), float(b)) for a, b in windows)
    if not items:
        raise ValueError("windows must have at least one (start, end) pair")
    for a, b in items:
        if not (0.0 <= a < b <= maturity + 1e-12):
            raise ValueError("each window must satisfy 0 <= start < end <= maturity")
    if any(nxt[0] < cur[1] for cur, nxt in zip(items, items[1:])):
        raise ValueError("windows must not overlap")
    return items


def _schedule_segments(schedule, maturity):
    """[(t_ini, t_fim, nível)] a partir de [(t_ini, nível)]."""
    ends = [t for t, _ in schedule[1:]] + [maturity]
    return [(t, end, level) for (t, level), end in zip(schedule, ends)]


def _window_segments(windows, barrier, maturity):
    """[(t_ini, t_fim, nível ou None)] cobrindo [0, T]: None = barreira desligada."""
    segments, cursor = [], 0.0
    for a, b in windows:
        if a > cursor + 1e-12:
            segments.append((cursor, a, None))
        segments.append((a, min(b, maturity), barrier))
        cursor = min(b, maturity)
    if cursor < maturity - 1e-12:
        segments.append((cursor, maturity, None))
    return segments


def _step_log_levels(segments, maturity: float, steps: int, direction: str) -> np.ndarray:
    """Log-barreira de cada passo pelo ponto médio do passo (±inf = desligada)."""
    mids = (np.arange(steps) + 0.5) * (maturity / steps)
    off = -np.inf if direction == "down" else np.inf
    levels = np.full(steps, off)
    for start, end, level in segments:
        if level is None:
            continue
        mask = (mids >= start) & (mids < end)
        levels[mask] = log(level)
    return levels


def _finish(alive, terminal_log, strike, option_type, discount, vanilla, rebate, knock):
    alive_mask = alive if knock == "out" else ~alive
    return _controlled(alive_mask, terminal_log, strike, option_type, discount, vanilla, rebate * discount)


# ---------------------------------------------------------------------------
# Precificadores Monte Carlo
# ---------------------------------------------------------------------------

def price_parisian_barrier(
    market: MarketData,
    *,
    strike: float,
    barrier: float,
    maturity: float,
    window: int = 5,
    cumulative: bool = False,
    option_type: str = "call",
    direction: str = "down",
    knock: str = "out",
    rebate: float = 0.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
) -> PriceResult:
    """Opção Parisian: o evento de barreira exige `window` fechamentos além da barreira.

    - consecutive (padrão): `window` fechamentos SEGUIDOS além da barreira.
    - cumulative=True (ParAsian): `window` fechamentos no total, consecutivos ou não.
    Um fechamento por passo (steps = dias úteis do prazo → janela em dias úteis).
    `window=1` reproduz exatamente a barreira simples com monitoramento discreto.
    Knock-Out: vanilla se o evento NÃO ocorrer; Knock-In: vanilla se ocorrer.
    `knock_out_probability`: P(evento) no Knock-Out e P(sem ativação) no Knock-In.
    """
    _validate_product(option_type, direction, knock, rebate)
    _validate_market(market, maturity, strike, barrier)
    _validate_simulation(paths, steps, "discrete")
    if int(window) != window or window < 1:
        raise ValueError("window must be a positive integer number of closes")
    window = int(window)

    log_paths, _, _, _ = _simulate_log_paths(
        market, maturity, paths=paths, steps=steps, seed=seed, monitoring="discrete"
    )
    beyond = _beyond(log_paths, log(barrier), direction)
    counter = np.zeros(paths, dtype=np.int32)
    knocked = np.zeros(paths, dtype=bool)
    for i in range(steps):
        if cumulative:
            counter += beyond[:, i]
        else:
            counter = (counter + 1) * beyond[:, i]
        knocked |= counter >= window

    discount = exp(-market.rate * maturity)
    vanilla = _vanilla(market.spot, strike, maturity, market.rate, market.dividend_yield,
                       market.volatility, option_type)
    return _finish(~knocked, log_paths[:, -1], strike, option_type, discount, vanilla, rebate, knock)


def _price_segmented(market, segments, *, strike, maturity, option_type, direction, knock, rebate,
                     paths, steps, seed, monitoring):
    log_levels = _step_log_levels(segments, maturity, steps, direction)
    log_paths, previous, uniforms, dt = _simulate_log_paths(
        market, maturity, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    if monitoring == "discrete":
        touched = _beyond(log_paths, log_levels, direction).any(axis=1)
    else:
        with np.errstate(invalid="ignore", over="ignore"):
            inside, prob = _crossing_probability(previous, log_paths, log_levels, direction,
                                                 market.volatility, dt)
        touched = ~np.all(inside & (uniforms > prob), axis=1)

    discount = exp(-market.rate * maturity)
    vanilla = _vanilla(market.spot, strike, maturity, market.rate, market.dividend_yield,
                       market.volatility, option_type)
    return _finish(~touched, log_paths[:, -1], strike, option_type, discount, vanilla, rebate, knock)


def price_step_barrier(
    market: MarketData,
    *,
    strike: float,
    schedule: Sequence[tuple[float, float]],
    maturity: float,
    option_type: str = "call",
    direction: str = "down",
    knock: str = "out",
    rebate: float = 0.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Step Barrier: barreira por degraus, `schedule = [(0.0, H1), (t1, H2), ...]`.

    O nível H_k vale de t_k até a próxima data (ou o vencimento). Na data de
    mudança, se o ativo já estiver além do novo nível, a barreira é tocada.
    Com um único degrau reproduz exatamente `price_single_barrier`.
    """
    _validate_product(option_type, direction, knock, rebate)
    items = _normalize_schedule(schedule, maturity)
    _validate_market(market, maturity, strike, *[level for _, level in items])
    _validate_simulation(paths, steps, monitoring)
    return _price_segmented(market, _schedule_segments(items, maturity), strike=strike,
                            maturity=maturity, option_type=option_type, direction=direction,
                            knock=knock, rebate=rebate, paths=paths, steps=steps, seed=seed,
                            monitoring=monitoring)


def price_soft_barrier(
    market: MarketData,
    *,
    strike: float,
    barrier: float,
    maturity: float,
    windows: Sequence[tuple[float, float]],
    option_type: str = "call",
    direction: str = "down",
    knock: str = "out",
    rebate: float = 0.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Soft Barrier (janela): barreira monitorada só dentro de `windows = [(t_ini, t_fim), ...]`.

    Fora das janelas o ativo cruza o nível sem consequência. Uma janela
    [0, T] reproduz exatamente `price_single_barrier`.
    """
    _validate_product(option_type, direction, knock, rebate)
    items = _normalize_windows(windows, maturity)
    _validate_market(market, maturity, strike, barrier)
    _validate_simulation(paths, steps, monitoring)
    return _price_segmented(market, _window_segments(items, barrier, maturity), strike=strike,
                            maturity=maturity, option_type=option_type, direction=direction,
                            knock=knock, rebate=rebate, paths=paths, steps=steps, seed=seed,
                            monitoring=monitoring)


# ---------------------------------------------------------------------------
# Benchmarks semi-analíticos (quadratura em grade de log-preço)
# ---------------------------------------------------------------------------

def _grid(spot, maturity, rate, dividend_yield, volatility, anchor, points):
    """Grade uniforme em log-preço com `anchor` (log-barreira) no meio de duas células."""
    nu = rate - dividend_yield - 0.5 * volatility**2
    center = log(spot) + nu * maturity
    half = 8.0 * volatility * sqrt(maturity) + abs(nu) * maturity
    lo, hi = min(center - half, log(spot) - half), max(center + half, log(spot) + half)
    h = (hi - lo) / (points - 1)
    shift = ((anchor - lo) / h) % 1.0 - 0.5
    x = lo + shift * h + h * np.arange(points)
    return x, h, nu


def _gauss(y, var):
    return np.exp(-0.5 * y * y / var) / sqrt(2.0 * pi * var)


def _terminal_value(density, x, h, strike, option_type, rate, maturity):
    terminal = np.exp(x)
    intrinsic = terminal - strike if option_type == "call" else strike - terminal
    alive_value = exp(-rate * maturity) * float(np.sum(density * np.maximum(intrinsic, 0.0)) * h)
    return alive_value, float(np.sum(density) * h)


def _assemble(alive_value, p_alive, vanilla, rebate, rate, maturity, knock):
    rebate_pv = rebate * exp(-rate * maturity)
    if knock == "out":
        return alive_value + rebate_pv * (1.0 - p_alive)
    return vanilla - alive_value + rebate_pv * p_alive


def parisian_barrier_quadrature(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    steps: int,
    window: int,
    cumulative: bool = False,
    option_type: str = "call",
    direction: str = "down",
    knock: str = "out",
    rebate: float = 0.0,
    points: int = 1201,
) -> float:
    """Preço exato (a menos da grade) da Parisian com monitoramento discreto.

    Estado aumentado (log-preço, contador): a cada fechamento a densidade é
    convoluída com o núcleo gaussiano do GBM e redistribuída entre os
    contadores; massa com contador = `window` é o evento de barreira.
    """
    _validate_product(option_type, direction, knock, rebate)
    window = int(window)
    if window < 1 or steps < 1:
        raise ValueError("window and steps must be positive")
    dt = maturity / steps
    x, h, nu = _grid(spot, maturity, rate, dividend_yield, volatility, log(barrier), points)
    var = volatility**2 * dt
    beyond = _beyond(x, log(barrier), direction)

    m = int(ceil((abs(nu) * dt + 9.0 * sqrt(var)) / h))
    kernel = _gauss(h * np.arange(-m, m + 1) - nu * dt, var) * h

    states = np.zeros((window + 1, points))  # linha `window` = absorvente (evento ocorreu)
    states[0] = _gauss(x - log(spot) - nu * dt, var)
    states[1] += states[0] * beyond
    states[0] *= ~beyond
    for _ in range(steps - 1):
        moved = fftconvolve(states[:window], kernel[None, :], mode="same", axes=1)
        moved = np.clip(moved, 0.0, None)
        new = np.zeros_like(states)
        if cumulative:
            new[0] = moved[0] * ~beyond
            new[1:window] += moved[1:window] * ~beyond
            new[1:window + 1] += moved[:window] * beyond
        else:
            new[0] = moved.sum(axis=0) * ~beyond
            new[1:window + 1] = moved[:window] * beyond
        states = new
    alive_value, p_alive = _terminal_value(states[:window].sum(axis=0), x, h, strike, option_type,
                                           rate, maturity)
    vanilla = _vanilla(spot, strike, maturity, rate, dividend_yield, volatility, option_type)
    return _assemble(alive_value, p_alive, vanilla, rebate, rate, maturity, knock)


def _killed_kernel(x_from, x_to, tau, nu, volatility, log_h, direction):
    """Densidade de transição em `tau` sem tocar a barreira (reflexão com drift)."""
    var = volatility**2 * tau
    dx = x_to[None, :] - x_from[:, None]
    free = _gauss(dx - nu * tau, var)
    if log_h is None:
        return free
    reflected = _gauss(x_to[None, :] + x_from[:, None] - 2.0 * log_h - nu * tau, var)
    with np.errstate(over="ignore"):
        weight = np.exp(2.0 * nu * (log_h - x_from[:, None]) / volatility**2)
    kernel = free - weight * reflected
    safe_from = ~_beyond(x_from, log_h, direction)
    safe_to = ~_beyond(x_to, log_h, direction)
    kernel = np.where(safe_from[:, None] & safe_to[None, :], kernel, 0.0)
    return np.clip(kernel, 0.0, None)


def segmented_barrier_quadrature(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    segments: Sequence[tuple[float, float, float | None]],
    option_type: str = "call",
    direction: str = "down",
    knock: str = "out",
    rebate: float = 0.0,
    points: int = 1501,
) -> float:
    """Preço com monitoramento contínuo por segmentos [(t_ini, t_fim, nível ou None)].

    Encadeia a densidade de transição morta na barreira de cada segmento
    (exata sob GBM) por quadratura na grade de log-preço. Base dos
    benchmarks de Step e Soft.
    """
    _validate_product(option_type, direction, knock, rebate)
    levels = [level for _, _, level in segments if level is not None]
    anchor = log(levels[0]) if levels else log(spot)
    x, h, nu = _grid(spot, maturity, rate, dividend_yield, volatility, anchor, points)
    density = None
    for start, end, level in segments:
        tau = end - start
        if tau <= 1e-14:
            continue
        log_h = None if level is None else log(level)
        if density is None:
            density = _killed_kernel(np.array([log(spot)]), x, tau, nu, volatility, log_h, direction)[0]
        else:
            density = (density * h) @ _killed_kernel(x, x, tau, nu, volatility, log_h, direction)
    alive_value, p_alive = _terminal_value(density, x, h, strike, option_type, rate, maturity)
    vanilla = _vanilla(spot, strike, maturity, rate, dividend_yield, volatility, option_type)
    return _assemble(alive_value, p_alive, vanilla, rebate, rate, maturity, knock)


def step_barrier_quadrature(spot, strike, maturity, rate, dividend_yield, volatility, *, schedule,
                            **kwargs) -> float:
    """Benchmark contínuo da Step Barrier."""
    items = _normalize_schedule(schedule, maturity)
    return segmented_barrier_quadrature(spot, strike, maturity, rate, dividend_yield, volatility,
                                        segments=_schedule_segments(items, maturity), **kwargs)


def soft_barrier_quadrature(spot, strike, barrier, maturity, rate, dividend_yield, volatility, *,
                            windows, **kwargs) -> float:
    """Benchmark contínuo da Soft Barrier (janela)."""
    items = _normalize_windows(windows, maturity)
    return segmented_barrier_quadrature(spot, strike, maturity, rate, dividend_yield, volatility,
                                        segments=_window_segments(items, barrier, maturity), **kwargs)
