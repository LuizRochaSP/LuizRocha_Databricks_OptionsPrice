"""Opções sobre ativo em USD (ex.: S&P 500) pagas em BRL: Quanto e Compo.

Convenções
----------
- S: ativo em USD (pontos do índice). X: câmbio USDBRL, em BRL por 1 USD.
- Taxas contínuas ACT/365: r_usd (taxa implícita do SPX ou SOFR), r_brl (curva DI), q (dividendos de S).
- rho = correlação entre os log-retornos de S e de X. Para SPX x USDBRL costuma ser negativa: quando a bolsa
  americana cai, o real tende a se desvalorizar (X sobe).

Sob a medida neutra ao risco em BRL:
    dX/X = (r_brl - r_usd) dt + sigma_x dW_x
    dS/S = (r_usd - q - rho sigma_s sigma_x) dt + sigma_s dW_s,   corr(dW_s, dW_x) = rho

Produtos
--------
- **Quanto:** paga `fx_fixed * max(S_T - K, 0)` em BRL, com K em USD e câmbio fixado no contrato.
  Black-Scholes com drift `r_usd - q - rho sigma_s sigma_x`, vol sigma_s, desconto em r_brl.
  Como a barreira é observada no próprio S, qualquer barreira do motor vale para o Quanto trocando o
  `MarketData` (`quanto_market_data`) e multiplicando o preço por `fx_fixed`.
- **Compo:** paga `max(S_T X_T - K_brl, 0)` em BRL, com strike em BRL. O ativo em BRL `S X` tem drift
  `r_brl - q` e vol `sqrt(sigma_s^2 + sigma_x^2 + 2 rho sigma_s sigma_x)`.
- **Convertida:** opção em USD convertida no câmbio do vencimento, `X_T max(S_T - K, 0)`. Vale
  `X_0 * BS_usd`, sem ajuste de correlação. Serve de referência.

O Monte Carlo de dois fatores (`price_two_factor_mc`) simula S e X juntos e valida as fórmulas fechadas,
inclusive com barreira em S (monitoramento discreto ou ponte browniana).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

from ibov_barrier.pricing import MarketData, PriceResult


@dataclass(frozen=True)
class QuantoMarket:
    spot: float            # S_0 em USD
    rate_usd: float        # r em USD, contínua
    dividend_yield: float  # q de S, contínua
    vol_asset: float       # sigma_s (superfície do SPX no strike e prazo do contrato)
    fx_spot: float         # X_0, BRL por USD
    rate_brl: float        # r em BRL, contínua (curva DI no prazo do contrato)
    vol_fx: float          # sigma_x do USDBRL
    correlation: float     # rho entre log-retornos de S e X

    def __post_init__(self):
        values = [self.spot, self.rate_usd, self.dividend_yield, self.vol_asset, self.fx_spot,
                  self.rate_brl, self.vol_fx, self.correlation]
        if not np.isfinite(values).all():
            raise ValueError("market parameters must be finite")
        if self.spot <= 0 or self.fx_spot <= 0:
            raise ValueError("spot and fx_spot must be positive")
        if self.vol_asset < 0 or self.vol_fx < 0:
            raise ValueError("volatilities must be non-negative")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must be in [-1, 1]")

    @property
    def quanto_adjustment(self) -> float:
        """Ajuste de drift do Quanto, `-rho sigma_s sigma_x` (contínuo, ao ano)."""
        return -self.correlation * self.vol_asset * self.vol_fx

    @property
    def compo_volatility(self) -> float:
        var = self.vol_asset**2 + self.vol_fx**2 + 2.0 * self.correlation * self.vol_asset * self.vol_fx
        return sqrt(max(var, 0.0))

    def fx_forward(self, maturity: float) -> float:
        return self.fx_spot * exp((self.rate_brl - self.rate_usd) * maturity)


def _black(forward: float, strike: float, maturity: float, discount: float, vol: float, option_type: str) -> float:
    """Black-76 sobre o forward; `discount` é o fator de desconto do vencimento."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if forward <= 0 or strike <= 0:
        raise ValueError("forward and strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    sign = 1.0 if option_type == "call" else -1.0
    if maturity == 0 or vol == 0:
        return discount * max(sign * (forward - strike), 0.0)
    sd = vol * sqrt(maturity)
    d1 = (log(forward / strike) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    return discount * sign * (forward * norm.cdf(sign * d1) - strike * norm.cdf(sign * d2))


def quanto_forward(market: QuantoMarket, maturity: float) -> float:
    """Forward de S sob a medida BRL: `S_0 exp((r_usd - q - rho sigma_s sigma_x) T)`."""
    drift = market.rate_usd - market.dividend_yield + market.quanto_adjustment
    return market.spot * exp(drift * maturity)


def price_quanto_vanilla(market: QuantoMarket, strike: float, maturity: float, *, option_type: str = "call",
                         fx_fixed: float | None = None) -> float:
    """Quanto europeia em BRL: `fx_fixed * max(S_T - K, 0)`. Sem `fx_fixed`, usa o câmbio de hoje."""
    fx = market.fx_spot if fx_fixed is None else fx_fixed
    return fx * _black(quanto_forward(market, maturity), strike, maturity, exp(-market.rate_brl * maturity),
                       market.vol_asset, option_type)


def quanto_market_data(market: QuantoMarket) -> MarketData:
    """`MarketData` para reutilizar qualquer pricer do motor (barreiras inclusive) como Quanto.

    O pricer desconta em `rate` e usa drift `rate - dividend_yield`. Com rate = r_brl e
    dividend_yield = r_brl - (r_usd - q - rho sigma_s sigma_x), o drift de S fica o do Quanto e o desconto
    fica em BRL. O resultado está em USD por ponto: multiplique por `fx_fixed`.
    """
    drift = market.rate_usd - market.dividend_yield + market.quanto_adjustment
    return MarketData(spot=market.spot, rate=market.rate_brl, dividend_yield=market.rate_brl - drift,
                      volatility=market.vol_asset)


def price_compo_vanilla(market: QuantoMarket, strike_brl: float, maturity: float, *,
                        option_type: str = "call") -> float:
    """Compo europeia: `max(S_T X_T - K_brl, 0)` em BRL, vol composta sobre o ativo em BRL."""
    forward = market.spot * market.fx_spot * exp((market.rate_brl - market.dividend_yield) * maturity)
    return _black(forward, strike_brl, maturity, exp(-market.rate_brl * maturity), market.compo_volatility,
                  option_type)


def price_converted_vanilla(market: QuantoMarket, strike: float, maturity: float, *,
                            option_type: str = "call") -> float:
    """Opção em USD convertida no câmbio do vencimento: `X_0 * BS_usd` (não depende de rho nem de sigma_x)."""
    forward = market.spot * exp((market.rate_usd - market.dividend_yield) * maturity)
    return market.fx_spot * _black(forward, strike, maturity, exp(-market.rate_usd * maturity),
                                   market.vol_asset, option_type)


@dataclass(frozen=True)
class TwoFactorResult(PriceResult):
    """Preço em BRL do Monte Carlo de dois fatores; `vanilla_price` é a fórmula fechada sem barreira."""


def price_two_factor_mc(market: QuantoMarket, strike: float, maturity: float, *, product: str = "quanto",
                        option_type: str = "call", fx_fixed: float | None = None,
                        barrier: float | None = None, barrier_direction: str = "down",
                        monitoring: str = "brownian_bridge", paths: int = 200_000, steps: int = 126,
                        seed: int = 42, local_vol=None) -> TwoFactorResult:
    """Monte Carlo conjunto de S (USD) e X (USDBRL) sob a medida BRL.

    product: 'quanto' (strike em USD, câmbio fixo), 'compo' (strike em BRL sobre S X) ou 'converted'
    (strike em USD, convertido em X_T). A barreira, se houver, é knock-out observada em S (USD). O vanilla
    correspondente entra como variável de controle, como no motor principal.

    local_vol: `LocalVolSurface` opcional (vol local de Dupire em k = ln(S/F_usd(t))). Nesse caso
    `vol_asset` é ignorado na simulação, o ajuste Quanto usa a vol local de cada passo e a variável de
    controle passa a ser S_T X_T (martingal com valor conhecido sob qualquer vol); `vanilla_price` é o
    vanilla simulado com vol local, sem barreira.
    """
    if product not in {"quanto", "compo", "converted"}:
        raise ValueError("product must be 'quanto', 'compo' or 'converted'")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if barrier_direction not in {"down", "up"}:
        raise ValueError("barrier_direction must be 'down' or 'up'")
    if monitoring not in {"discrete", "brownian_bridge"}:
        raise ValueError("monitoring must be 'discrete' or 'brownian_bridge'")
    if maturity <= 0 or strike <= 0 or paths < 2 or steps < 1:
        raise ValueError("maturity and strike must be positive; paths >= 2 and steps >= 1")
    if barrier is not None and barrier <= 0:
        raise ValueError("barrier must be positive")

    m = market
    fx = m.fx_spot if fx_fixed is None else fx_fixed
    if product == "quanto":
        vanilla = price_quanto_vanilla(m, strike, maturity, option_type=option_type, fx_fixed=fx)
    elif product == "compo":
        vanilla = price_compo_vanilla(m, strike, maturity, option_type=option_type)
    else:
        vanilla = price_converted_vanilla(m, strike, maturity, option_type=option_type)

    breached = barrier is not None and (m.spot <= barrier if barrier_direction == "down" else m.spot >= barrier)
    if breached:
        return TwoFactorResult(0.0, 0.0, 0.0, 0.0, 1.0, vanilla)

    if local_vol is not None:
        return _two_factor_local_vol(m, strike, maturity, product, option_type, fx, barrier, barrier_direction,
                                     monitoring, paths, steps, seed, local_vol)

    rng = np.random.default_rng(seed)
    dt = maturity / steps
    z1 = rng.standard_normal((paths, steps))
    z2 = m.correlation * z1 + sqrt(max(1.0 - m.correlation**2, 0.0)) * rng.standard_normal((paths, steps))
    s_drift = (m.rate_usd - m.dividend_yield + m.quanto_adjustment - 0.5 * m.vol_asset**2) * dt
    x_drift = (m.rate_brl - m.rate_usd - 0.5 * m.vol_fx**2) * dt
    log_s = log(m.spot) + np.cumsum(s_drift + m.vol_asset * sqrt(dt) * z1, axis=1)
    log_x_t = log(m.fx_spot) + (x_drift * steps + m.vol_fx * sqrt(dt) * z2.sum(axis=1))

    alive = np.ones(paths, dtype=bool)
    if barrier is not None:
        log_h = log(barrier)
        sign = 1.0 if barrier_direction == "down" else -1.0
        dist = sign * (log_s - log_h)          # > 0 do lado vivo
        prev = np.concatenate((np.full((paths, 1), sign * (log(m.spot) - log_h)), dist[:, :-1]), axis=1)
        alive = np.all(dist > 0, axis=1)
        if monitoring == "brownian_bridge" and m.vol_asset > 0:
            cross = np.exp(-2.0 * np.clip(prev, 0, None) * np.clip(dist, 0, None) / (m.vol_asset**2 * dt))
            alive &= np.all(rng.random((paths, steps)) > cross, axis=1)

    s_t, x_t = np.exp(log_s[:, -1]), np.exp(log_x_t)
    sign = 1.0 if option_type == "call" else -1.0
    if product == "quanto":
        vanilla_payoff = fx * np.maximum(sign * (s_t - strike), 0.0)
    elif product == "compo":
        vanilla_payoff = np.maximum(sign * (s_t * x_t - strike), 0.0)
    else:
        vanilla_payoff = x_t * np.maximum(sign * (s_t - strike), 0.0)
    disc = exp(-m.rate_brl * maturity)
    vanilla_pv = disc * vanilla_payoff
    barrier_pv = vanilla_pv * alive

    var = float(vanilla_pv.var(ddof=1))
    beta = float(np.cov(barrier_pv, vanilla_pv, ddof=1)[0, 1]) / var if var > 0 else 0.0
    adjusted = barrier_pv - beta * (vanilla_pv - vanilla)
    price = float(adjusted.mean())
    se = float(adjusted.std(ddof=1) / sqrt(paths))
    return TwoFactorResult(price, se, price - 1.96 * se, price + 1.96 * se, float(1.0 - alive.mean()), vanilla)


def _two_factor_local_vol(m, strike, maturity, product, option_type, fx, barrier, barrier_direction,
                          monitoring, paths, steps, seed, surface) -> TwoFactorResult:
    """Euler em log, passo a passo: S com vol local de Dupire, X lognormal, sob a medida BRL."""
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    carry = m.rate_usd - m.dividend_yield
    log_s = np.full(paths, log(m.spot))
    log_x = np.full(paths, log(m.fx_spot))
    alive = np.ones(paths, dtype=bool)
    sign_b = 1.0 if barrier_direction == "down" else -1.0
    log_h = log(barrier) if barrier is not None else None
    rho_c = sqrt(max(1.0 - m.correlation**2, 0.0))
    for i in range(steps):
        t = i * dt
        row, _ = surface.local_vol_row(t + 0.5 * dt)
        sig = np.interp(log_s - (log(m.spot) + carry * t), surface.k_grid, row)
        z1 = rng.standard_normal(paths)
        z2 = m.correlation * z1 + rho_c * rng.standard_normal(paths)
        new_s = log_s + (carry - m.correlation * sig * m.vol_fx - 0.5 * sig * sig) * dt + sig * sqrt(dt) * z1
        log_x += (m.rate_brl - m.rate_usd - 0.5 * m.vol_fx**2) * dt + m.vol_fx * sqrt(dt) * z2
        if barrier is not None:
            prev, dist = sign_b * (log_s - log_h), sign_b * (new_s - log_h)
            alive &= dist > 0
            if monitoring == "brownian_bridge":
                cross = np.exp(-2.0 * np.clip(prev, 0, None) * np.clip(dist, 0, None) / (sig * sig * dt))
                alive &= rng.random(paths) > cross
        log_s = new_s

    s_t, x_t = np.exp(log_s), np.exp(log_x)
    sign = 1.0 if option_type == "call" else -1.0
    if product == "quanto":
        payoff = fx * np.maximum(sign * (s_t - strike), 0.0)
    elif product == "compo":
        payoff = np.maximum(sign * (s_t * x_t - strike), 0.0)
    else:
        payoff = x_t * np.maximum(sign * (s_t - strike), 0.0)
    disc = exp(-m.rate_brl * maturity)
    control = disc * s_t * x_t
    control_mean = m.spot * m.fx_spot * exp(-m.dividend_yield * maturity)
    var = float(control.var(ddof=1))

    def controlled(pv):
        beta = float(np.cov(pv, control, ddof=1)[0, 1]) / var if var > 0 else 0.0
        adj = pv - beta * (control - control_mean)
        return float(adj.mean()), float(adj.std(ddof=1) / sqrt(paths))

    vanilla, _ = controlled(disc * payoff)
    price, se = controlled(disc * payoff * alive)
    return TwoFactorResult(price, se, price - 1.96 * se, price + 1.96 * se, float(1.0 - alive.mean()), vanilla)


def historical_fx_inputs(asset_close: pd.Series, fx_close: pd.Series, *, frequency: str = "W-FRI",
                         window: int | None = None, periods_per_year: float | None = None) -> dict[str, float]:
    """Vol do câmbio e correlação histórica entre S e X a partir de fechamentos (índice = datas).

    O SPX fecha às 17:00 ET e o USDBRL (PTAX) às 13:00 BRT: com retornos diários os fechamentos não são
    simultâneos, o que puxa a correlação para zero. Por isso o padrão é retorno semanal (`W-FRI`); use
    `frequency=None` para diário. `window` limita aos últimos N retornos.
    """
    df = pd.concat({"s": pd.Series(asset_close, dtype=float), "x": pd.Series(fx_close, dtype=float)},
                   axis=1).dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if frequency is not None:
        df = df.resample(frequency).last().dropna()
    rets = np.log(df).diff().dropna()
    if window is not None:
        rets = rets.tail(window)
    if len(rets) < 10:
        raise ValueError("at least 10 aligned returns are needed")
    if periods_per_year is None:
        periods_per_year = 52.0 if frequency and frequency.startswith("W") else 252.0
    ann = sqrt(periods_per_year)
    return {"vol_asset": float(rets["s"].std(ddof=1) * ann), "vol_fx": float(rets["x"].std(ddof=1) * ann),
            "correlation": float(rets["s"].corr(rets["x"])), "observations": int(len(rets)),
            "start": rets.index[0].date().isoformat(), "end": rets.index[-1].date().isoformat()}
