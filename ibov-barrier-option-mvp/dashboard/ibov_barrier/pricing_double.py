"""Família B completa: opções de dupla barreira (lacuna difícil nº 7 do roadmap).

Produtos (todos pagos no vencimento, monitoramento de teto e chão simultâneo):

- DKO (Double Knock-Out): vanilla viva só se o ativo NUNCA sair do túnel (L, U).
- DKI (Double Knock-In): vanilla que só nasce se o ativo tocar L OU U.
- KIKO (Knock-In Knock-Out): nasce ao tocar a barreira KI, mas morre se a
  barreira KO for tocada a QUALQUER momento (antes ou depois da ativação).
- KOKI (Knock-Out após Knock-In): nasce ao tocar a barreira KI e só a partir
  daí a barreira KO passa a ser monitorada.
- Convenções de ordem parametrizáveis em `price_double_barrier_option`
  (`convention` = KI_first | KO_after_KI | KO_first | sequential). Todas
  dependem da ORDEM dos toques: em vez de "tocou / não tocou" por caminho,
  o simulador registra o INSTANTE (passo) do primeiro toque em cada barreira.

Motor: `_barrier_hits` devolve, para cada caminho e cada passo, se L e U foram
tocadas naquele intervalo. No modo Brownian Bridge usa um único uniforme por
passo particionado em [0, p_L) → toque em L e [p_L, p_L + p_U) → toque em U,
o que reproduz exatamente `_double_touched` (Double One/No-Touch) com a mesma
semente. Os caminhos seguem a mesma ordem de sorteio do motor original.

Benchmarks analíticos (GBM, monitoramento contínuo):
- `single_barrier_out_analytical`: Reiner-Rubinstein generalizado (8 casos Out).
- `double_knock_out_analytical`: série de Ikeda-Kunitomo (1992), barreiras planas.
- `kiko_analytical`: identidade exata KIKO = KO simples(H_KO) − DKO(L, U).
- `koki_analytical`: propriedade forte de Markov — integra a densidade do
  primeiro toque em H_KI contra o preço Reiner-Rubinstein da barreira KO
  partindo de H_KI com o prazo remanescente.
- `ko_first_analytical`: densidade de saída do túnel pela barreira KO (série
  de Fourier) integrada contra a vanilla a partir de H_KO.
- `sequential_analytical`: identidade exata Knock-In simples − KOKI.
"""
from __future__ import annotations

from math import exp, log, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from ibov_barrier.pricing import BarrierContract, MarketData, PriceResult, black_scholes_call
from ibov_barrier.pricing_barriers import _black_scholes_put_by_parity
from ibov_barrier.pricing_extensions import (
    _crossing_probability,
    _simulate_log_paths,
    _summary,
    _validate_market,
    _validate_simulation,
)

_OPTION_TYPES = {"call", "put"}
_KNOCKS = {"out", "in"}


# ---------------------------------------------------------------------------
# Benchmarks analíticos
# ---------------------------------------------------------------------------

def single_barrier_out_analytical(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    option_type: str,
    direction: str,
) -> float:
    """Knock-Out simples (sem rebate) — Reiner-Rubinstein na notação de Haug.

    Cobre DOC, DOP, UOC, UOP. Com prazo zero devolve o intrínseco (se vivo).
    """
    phi = 1.0 if option_type == "call" else -1.0
    eta = 1.0 if direction == "down" else -1.0
    knocked = spot <= barrier if direction == "down" else spot >= barrier
    if knocked:
        return 0.0
    if maturity <= 1e-12:
        return max(phi * (spot - strike), 0.0)

    b = rate - dividend_yield
    v = volatility * sqrt(maturity)
    mu = (b - 0.5 * volatility**2) / volatility**2
    carry = exp((b - rate) * maturity)
    disc = exp(-rate * maturity)
    h_s = barrier / spot

    x1 = log(spot / strike) / v + (1 + mu) * v
    x2 = log(spot / barrier) / v + (1 + mu) * v
    y1 = log(barrier**2 / (spot * strike)) / v + (1 + mu) * v
    y2 = log(barrier / spot) / v + (1 + mu) * v

    A = phi * spot * carry * norm.cdf(phi * x1) - phi * strike * disc * norm.cdf(phi * x1 - phi * v)
    B = phi * spot * carry * norm.cdf(phi * x2) - phi * strike * disc * norm.cdf(phi * x2 - phi * v)
    C = (phi * spot * carry * h_s ** (2 * (mu + 1)) * norm.cdf(eta * y1)
         - phi * strike * disc * h_s ** (2 * mu) * norm.cdf(eta * y1 - eta * v))
    D = (phi * spot * carry * h_s ** (2 * (mu + 1)) * norm.cdf(eta * y2)
         - phi * strike * disc * h_s ** (2 * mu) * norm.cdf(eta * y2 - eta * v))

    above = strike > barrier
    if direction == "down" and option_type == "call":
        price = A - C if above else B - D
    elif direction == "down":
        price = A - B + C - D if above else 0.0
    elif option_type == "call":
        price = 0.0 if above else A - B + C - D
    else:
        price = B - D if above else A - C
    return float(max(price, 0.0))


def double_knock_out_analytical(
    spot: float,
    strike: float,
    lower: float,
    upper: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    option_type: str,
    terms: int = 10,
) -> float:
    """Double Knock-Out (barreiras planas, sem rebate) — série de Ikeda-Kunitomo.

    O preço é e^{−rT}·E[(φ(S_T − K))⁺ · 1{nunca saiu de (L, U)}]. A densidade
    "morta" nas duas barreiras é uma soma de imagens refletidas (n = −N..N);
    integrando o payoff na região de exercício [a, b] ⊂ (L, U):

      SA = S·e^{(b_c−r)T}·Σ[(U/L)^{nμ}(N(d1)−N(d2)) − (L^{n+1}/(UⁿS))^{μ}(N(d3)−N(d4))]
      KP = K·e^{−rT}·Σ[(U/L)^{n(μ−2)}(N(d1−v)−N(d2−v)) − (L^{n+1}/(UⁿS))^{μ−2}(N(d3−v)−N(d4−v))]

    com μ = 2b_c/σ² + 1, v = σ√T. Call = SA − KP com [a, b] = [max(K, L), U];
    Put = KP − SA com [a, b] = [L, min(K, U)]. A série converge muito rápido
    (termos ~ e^{−2n²ℓ²/σ²T}); N = 10 é mais que suficiente.
    """
    if not lower < spot < upper:
        return 0.0
    if option_type == "call":
        a, b_reg = max(strike, lower), upper
    else:
        a, b_reg = lower, min(strike, upper)
    if a >= b_reg:
        return 0.0

    b_c = rate - dividend_yield
    v = volatility * sqrt(maturity)
    m = 2.0 * b_c / volatility**2 + 1.0
    drift = (b_c + 0.5 * volatility**2) * maturity
    n = np.arange(-terms, terms + 1, dtype=float)

    # Logs para evitar overflow em U^{2n}.
    ln_s, ln_l, ln_u = log(spot), log(lower), log(upper)
    ln_a, ln_b = log(a), log(b_reg)
    d1 = (ln_s + 2 * n * ln_u - ln_a - 2 * n * ln_l + drift) / v
    d2 = (ln_s + 2 * n * ln_u - ln_b - 2 * n * ln_l + drift) / v
    d3 = ((2 * n + 2) * ln_l - ln_a - ln_s - 2 * n * ln_u + drift) / v
    d4 = ((2 * n + 2) * ln_l - ln_b - ln_s - 2 * n * ln_u + drift) / v
    ln_ratio = n * (ln_u - ln_l)                       # ln (U/L)^n
    ln_refl = (n + 1) * ln_l - n * ln_u - ln_s         # ln L^{n+1}/(UⁿS)

    sa = spot * exp((b_c - rate) * maturity) * float(np.sum(
        np.exp(m * ln_ratio) * (norm.cdf(d1) - norm.cdf(d2))
        - np.exp(m * ln_refl) * (norm.cdf(d3) - norm.cdf(d4))
    ))
    kp = strike * exp(-rate * maturity) * float(np.sum(
        np.exp((m - 2) * ln_ratio) * (norm.cdf(d1 - v) - norm.cdf(d2 - v))
        - np.exp((m - 2) * ln_refl) * (norm.cdf(d3 - v) - norm.cdf(d4 - v))
    ))
    price = sa - kp if option_type == "call" else kp - sa
    return float(max(price, 0.0))


def _vanilla(spot, strike, maturity, rate, dividend_yield, volatility, option_type) -> float:
    if maturity <= 1e-12:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    market = MarketData(spot=spot, rate=rate, dividend_yield=dividend_yield, volatility=volatility)
    contract = BarrierContract(strike=strike, barrier=spot, maturity=maturity)
    call = black_scholes_call(market, contract)
    return call if option_type == "call" else _black_scholes_put_by_parity(market, contract, call)


def _split_barriers(spot: float, knock_in: float, knock_out: float) -> tuple[float, float, str, str]:
    """Valida KI e KO em lados opostos do spot; devolve (L, U, dir_KI, dir_KO)."""
    if knock_in < spot < knock_out:
        return knock_in, knock_out, "down", "up"
    if knock_out < spot < knock_in:
        return knock_out, knock_in, "up", "down"
    raise ValueError("knock_in and knock_out must lie on opposite sides of the spot")


def kiko_analytical(
    spot: float,
    strike: float,
    knock_in: float,
    knock_out: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    option_type: str,
) -> float:
    """KIKO exato: 1{tocou KI}·1{nunca tocou KO} = 1{nunca KO} − 1{nunca KO e nunca KI}.

    Logo KIKO = KO simples na barreira KO − DKO no túnel (L, U).
    """
    lower, upper, _, dir_ko = _split_barriers(spot, knock_in, knock_out)
    single = single_barrier_out_analytical(
        spot, strike, knock_out, maturity, rate, dividend_yield, volatility,
        option_type=option_type, direction=dir_ko,
    )
    dko = double_knock_out_analytical(
        spot, strike, lower, upper, maturity, rate, dividend_yield, volatility, option_type=option_type,
    )
    return float(max(single - dko, 0.0))


def koki_analytical(
    spot: float,
    strike: float,
    knock_in: float,
    knock_out: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    option_type: str,
) -> float:
    """KOKI por propriedade forte de Markov.

    No primeiro toque τ em H_KI a estrutura vira um Knock-Out simples na
    barreira KO, com spot = H_KI e prazo T − τ. Então

      KOKI = ∫₀ᵀ f_τ(t)·e^{−rt}·KO(H_KI, K, H_KO, T − t) dt,

    com f_τ a densidade do primeiro toque do Browniano com drift
    μ = r − q − σ²/2 no nível β = ln(H_KI/S₀):
      f_τ(t) = |β| / (σ√(2πt³)) · exp(−(β − μt)² / (2σ²t)).
    """
    _, _, _, dir_ko = _split_barriers(spot, knock_in, knock_out)
    mu = rate - dividend_yield - 0.5 * volatility**2
    beta = log(knock_in / spot)

    def integrand(t: float) -> float:
        if t <= 0.0:
            return 0.0
        density = abs(beta) / (volatility * sqrt(2 * pi * t**3)) * exp(
            -((beta - mu * t) ** 2) / (2 * volatility**2 * t)
        )
        ko = single_barrier_out_analytical(
            knock_in, strike, knock_out, maturity - t, rate, dividend_yield, volatility,
            option_type=option_type, direction=dir_ko,
        )
        return density * exp(-rate * t) * ko

    value, _ = quad(integrand, 0.0, maturity, limit=200, epsabs=1e-10, epsrel=1e-8)
    return float(max(value, 0.0))


def exit_density_analytical(
    t: float,
    spot: float,
    lower: float,
    upper: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    side: str,
) -> float:
    """Densidade do instante de SAÍDA do túnel (L, U) pela barreira `side` ('lower'/'upper').

    É o fluxo de probabilidade na fronteira da densidade "morta" nas duas
    barreiras (expansão em sin(nπy/ℓ), ℓ = ln(U/L), x = ln(S₀/L)):

      g_L(t) = (σ²π/ℓ²)·e^{−μx/σ² − μ²t/2σ²}·Σ n·sin(nπx/ℓ)·e^{−σ²n²π²t/2ℓ²}
      g_U(t) = (σ²π/ℓ²)·e^{μ(ℓ−x)/σ² − μ²t/2σ²}·Σ (−1)^{n+1}·n·sin(nπx/ℓ)·e^{−σ²n²π²t/2ℓ²}

    ∫₀ᵀ (g_L + g_U) dt = 1 − P(Double No-Touch). Para t muito pequeno (saída
    com probabilidade < e^{−50}) devolve 0, evitando a série mal condicionada.
    """
    if side not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    if t <= 0.0 or not lower < spot < upper:
        return 0.0
    sigma2 = volatility**2
    mu = rate - dividend_yield - 0.5 * sigma2
    width = log(upper / lower)
    x = log(spot / lower)
    gap = x if side == "lower" else width - x
    if gap**2 / (2.0 * sigma2 * t) > 50.0:
        return 0.0
    n_max = int(np.ceil(sqrt(80.0 * width**2 / (sigma2 * pi**2 * t)))) + 10
    n = np.arange(1, n_max + 1, dtype=float)
    decay = np.exp(-sigma2 * n**2 * pi**2 * t / (2.0 * width**2))
    terms = n * np.sin(n * pi * x / width) * decay
    if side == "lower":
        level = -mu * x / sigma2
    else:
        level = mu * (width - x) / sigma2
        terms = terms * (-1.0) ** (n + 1)
    value = sigma2 * pi / width**2 * exp(level - mu**2 * t / (2.0 * sigma2)) * float(terms.sum())
    return max(value, 0.0)


def ko_first_analytical(
    spot: float,
    strike: float,
    knock_in: float,
    knock_out: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    option_type: str,
) -> float:
    """Convenção KO_first: a opção só morre se o KO for tocado ANTES do KI.

    KO_first = Vanilla − E[e^{−rτ}·Vanilla(H_KO, T − τ); saída do túnel pelo KO em τ ≤ T],
    com τ o primeiro instante de saída (densidade `exit_density_analytical`).
    """
    lower, upper, _, dir_ko = _split_barriers(spot, knock_in, knock_out)
    side = "upper" if dir_ko == "up" else "lower"

    def integrand(t: float) -> float:
        g = exit_density_analytical(t, spot, lower, upper, rate, dividend_yield, volatility, side=side)
        if g == 0.0:
            return 0.0
        return g * exp(-rate * t) * _vanilla(knock_out, strike, maturity - t, rate, dividend_yield,
                                             volatility, option_type)

    lost, _ = quad(integrand, 0.0, maturity, limit=200, epsabs=1e-10, epsrel=1e-8)
    vanilla = _vanilla(spot, strike, maturity, rate, dividend_yield, volatility, option_type)
    return float(max(vanilla - lost, 0.0))


def sequential_analytical(
    spot: float,
    strike: float,
    knock_in: float,
    knock_out: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    option_type: str,
) -> float:
    """Convenção sequential: paga só se tocar o KI e DEPOIS o KO.

    1{KI e KO depois} = 1{KI} − 1{KI e nenhum KO depois} ⇒ Sequential = Knock-In simples − KOKI.
    """
    _, _, dir_ki, _ = _split_barriers(spot, knock_in, knock_out)
    args = (rate, dividend_yield, volatility)
    knock_in_price = _vanilla(spot, strike, maturity, *args, option_type) - single_barrier_out_analytical(
        spot, strike, knock_in, maturity, *args, option_type=option_type, direction=dir_ki,
    )
    koki = koki_analytical(spot, strike, knock_in, knock_out, maturity, *args, option_type=option_type)
    return float(max(knock_in_price - koki, 0.0))


CONVENTIONS = ("KI_first", "KO_after_KI", "KO_first", "sequential")


def double_barrier_option_analytical(
    spot: float,
    strike: float,
    knock_in: float,
    knock_out: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    *,
    option_type: str,
    convention: str = "KI_first",
) -> float:
    """Benchmark analítico de `price_double_barrier_option` para cada convenção."""
    fn = {
        "KI_first": kiko_analytical,
        "KO_after_KI": koki_analytical,
        "KO_first": ko_first_analytical,
        "sequential": sequential_analytical,
    }.get(convention)
    if fn is None:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    return fn(spot, strike, knock_in, knock_out, maturity, rate, dividend_yield, volatility,
              option_type=option_type)


# ---------------------------------------------------------------------------
# Motor: toques por passo (com ordem temporal)
# ---------------------------------------------------------------------------

def _barrier_hits(
    market: MarketData,
    lower: float,
    upper: float,
    maturity: float,
    *,
    paths: int,
    steps: int,
    seed: int,
    monitoring: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matrizes (paths × steps) de toque em L e em U por passo + log-spot terminal.

    Brownian Bridge: com as duas pontas dentro do túnel, um único uniforme u
    decide o passo — u < p_L toca L; p_L ≤ u < p_L + p_U toca U (cruzar as
    duas no mesmo passo tem ordem e^{−2ℓ²/σ²dt}, desprezível). Se uma ponta
    está fora de uma barreira, aquela barreira é tocada com certeza e a outra
    usa sua própria probabilidade de ponte com o mesmo u.
    """
    log_paths, previous, uniforms, dt = _simulate_log_paths(
        market, maturity, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    log_l, log_u = log(lower), log(upper)
    if monitoring == "discrete":
        return log_paths <= log_l, log_paths >= log_u, log_paths[:, -1]

    sigma = market.volatility
    inside_l, prob_l = _crossing_probability(previous, log_paths, log_l, "down", sigma, dt)
    inside_u, prob_u = _crossing_probability(previous, log_paths, log_u, "up", sigma, dt)
    both = inside_l & inside_u
    hit_l = uniforms < prob_l
    hit_u = np.where(both, (uniforms >= prob_l) & (uniforms < prob_l + prob_u), uniforms < prob_u)
    return hit_l, hit_u, log_paths[:, -1]


def _first_hit(hits: np.ndarray) -> np.ndarray:
    """Índice do primeiro passo com toque (ou `steps` se nunca tocou)."""
    any_hit = hits.any(axis=1)
    return np.where(any_hit, hits.argmax(axis=1), hits.shape[1])


def _controlled(
    alive: np.ndarray,
    terminal_log: np.ndarray,
    strike: float,
    option_type: str,
    discount: float,
    vanilla: float,
    dead_value: float = 0.0,
) -> PriceResult:
    """Payoff vivo × vanilla (+ valor fixo se morto) com variável de controle vanilla."""
    terminal = np.exp(terminal_log)
    intrinsic = terminal - strike if option_type == "call" else strike - terminal
    vanilla_payoffs = discount * np.maximum(intrinsic, 0.0)
    payoffs = vanilla_payoffs * alive + dead_value * ~alive
    variance = float(vanilla_payoffs.var(ddof=1))
    beta = float(np.cov(payoffs, vanilla_payoffs, ddof=1)[0, 1]) / variance if variance > 0 else 0.0
    adjusted = payoffs - beta * (vanilla_payoffs - vanilla)
    return _summary(adjusted, float(1.0 - alive.mean()), vanilla)


def _validate_common(market, strike, maturity, option_type, paths, steps, monitoring, *levels):
    if option_type not in _OPTION_TYPES:
        raise ValueError("option_type must be 'call' or 'put'")
    _validate_market(market, maturity, strike, *levels)
    _validate_simulation(paths, steps, monitoring)


# ---------------------------------------------------------------------------
# Precificadores
# ---------------------------------------------------------------------------

def price_double_barrier(
    market: MarketData,
    *,
    strike: float,
    lower: float,
    upper: float,
    maturity: float,
    option_type: str = "call",
    knock: str = "out",
    rebate: float = 0.0,
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """DKO / DKI (Call ou Put) com rebate pago no vencimento.

    - knock='out': vanilla se nunca tocar L nem U; rebate se tocar alguma.
    - knock='in':  vanilla se tocar L ou U; rebate se nunca tocar.
    `knock_out_probability`: P(tocado) para DKO e P(não ativar) para DKI.
    Paridade (rebate 0, mesmos caminhos): DKO + DKI = vanilla.
    """
    if knock not in _KNOCKS:
        raise ValueError("knock must be 'out' or 'in'")
    if not np.isfinite(rebate) or rebate < 0:
        raise ValueError("rebate must be a non-negative finite number")
    _validate_common(market, strike, maturity, option_type, paths, steps, monitoring, lower, upper)
    if not lower < upper:
        raise ValueError("lower must be strictly below upper")

    discount = exp(-market.rate * maturity)
    vanilla = _vanilla(market.spot, strike, maturity, market.rate, market.dividend_yield,
                       market.volatility, option_type)
    if not lower < market.spot < upper:
        value = rebate * discount if knock == "out" else vanilla
        return PriceResult(value, 0.0, value, value, 1.0 if knock == "out" else 0.0, vanilla)

    hit_l, hit_u, terminal_log = _barrier_hits(
        market, lower, upper, maturity, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    touched = (hit_l | hit_u).any(axis=1)
    alive = ~touched if knock == "out" else touched
    return _controlled(alive, terminal_log, strike, option_type, discount, vanilla, rebate * discount)


def price_double_knock_out(market, *, strike, lower, upper, maturity, option_type="call", **kwargs):
    """Double Knock-Out: atalho para `price_double_barrier(knock='out')`."""
    return price_double_barrier(market, strike=strike, lower=lower, upper=upper, maturity=maturity,
                                option_type=option_type, knock="out", **kwargs)


def price_double_knock_in(market, *, strike, lower, upper, maturity, option_type="call", **kwargs):
    """Double Knock-In: atalho para `price_double_barrier(knock='in')`."""
    return price_double_barrier(market, strike=strike, lower=lower, upper=upper, maturity=maturity,
                                option_type=option_type, knock="in", **kwargs)


def price_double_barrier_option(
    market: MarketData,
    *,
    strike: float,
    knock_in: float,
    knock_out: float,
    maturity: float,
    option_type: str = "call",
    convention: str = "KI_first",
    paths: int = 200_000,
    steps: int = 252,
    seed: int = 42,
    monitoring: str = "brownian_bridge",
) -> PriceResult:
    """Opção de dupla barreira KI/KO com ordem de ativação parametrizável.

    KI e KO ficam em lados opostos do spot. `convention` define o que a ordem
    dos toques faz com a opção (τ_KI, τ_KO = primeiros toques):

    - "KI_first" (KIKO): só nasce se tocar KI; o KO mata a qualquer momento,
      antes ou depois da ativação. Viva ⇔ τ_KI ≤ T e KO nunca tocado.
    - "KO_after_KI" (KOKI): só nasce se tocar KI; o KO só é monitorado após a
      ativação. Viva ⇔ τ_KI ≤ T e nenhum toque em KO depois de τ_KI.
    - "KO_first": nasce viva como Knock-Out; se o KI for tocado primeiro o KO
      deixa de existir (vira vanilla). Viva ⇔ NÃO (τ_KO < τ_KI).
    - "sequential": só paga se tocar o KI e depois o KO (duplo toque
      sequencial). Viva ⇔ τ_KI ≤ T e algum toque em KO depois de τ_KI.

    Identidades: KO_after_KI + sequential = Knock-In simples (mesmos caminhos);
    KI_first ≤ KO_after_KI. Toques no mesmo passo são tratados de forma
    conservadora (contam como KO). `knock_out_probability` = P(não viva em T).
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    _validate_common(market, strike, maturity, option_type, paths, steps, monitoring, knock_in, knock_out)
    lower, upper, dir_ki, _ = _split_barriers(market.spot, knock_in, knock_out)
    discount = exp(-market.rate * maturity)
    vanilla = _vanilla(market.spot, strike, maturity, market.rate, market.dividend_yield,
                       market.volatility, option_type)

    hit_l, hit_u, terminal_log = _barrier_hits(
        market, lower, upper, maturity, paths=paths, steps=steps, seed=seed, monitoring=monitoring
    )
    ki_hits, ko_hits = (hit_l, hit_u) if dir_ki == "down" else (hit_u, hit_l)
    ki_step = _first_hit(ki_hits)
    ko_step = _first_hit(ko_hits)
    activated = ki_step < steps

    if convention == "KI_first":
        alive = activated & (ko_step == steps)
    elif convention == "KO_first":
        alive = ~((ko_step < steps) & (ko_step <= ki_step))
    else:
        # KO a partir do passo da ativação (empate no mesmo passo conta como KO).
        step_index = np.arange(steps)
        ko_after = (ko_hits & (step_index[None, :] >= ki_step[:, None])).any(axis=1)
        alive = activated & (~ko_after if convention == "KO_after_KI" else ko_after)
    return _controlled(alive, terminal_log, strike, option_type, discount, vanilla)


def price_kiko(market, *, strike, knock_in, knock_out, maturity, option_type="call", **kwargs):
    """KIKO = `price_double_barrier_option(convention='KI_first')`."""
    return price_double_barrier_option(market, strike=strike, knock_in=knock_in, knock_out=knock_out,
                                       maturity=maturity, option_type=option_type,
                                       convention="KI_first", **kwargs)


def price_koki(market, *, strike, knock_in, knock_out, maturity, option_type="call", **kwargs):
    """KOKI = `price_double_barrier_option(convention='KO_after_KI')`."""
    return price_double_barrier_option(market, strike=strike, knock_in=knock_in, knock_out=knock_out,
                                       maturity=maturity, option_type=option_type,
                                       convention="KO_after_KI", **kwargs)
