"""Adaptadores dos pricers atuais para a interface `Produto` (etapa A1 do plano de COE).

Os pricers do motor não mudam. Cada adaptador lê do `EstadoMercado` o spot, a taxa, os dividendos e a vol no
prazo restante, monta os argumentos do pricer original e devolve o mesmo número (testes "mesmo número" em
`tests/test_produtos.py`).

| Adaptador | Pricer original | Avaliação vetorial |
| :--- | :--- | :--- |
| `Vanilla` | `pricing.black_scholes_call` (put pela paridade) | numpy, fórmula fechada |
| `CallSpread` | duas `Vanilla` | numpy, fórmula fechada |
| `Digital` | `analytical._bs_digital` | numpy, fórmula fechada |
| `BarreiraSimples` | `pricing_extensions.price_single_barrier` (com rebate) | Monte Carlo, um cenário por vez |
| `OneTouch`, `NoTouch` | `pricing_extensions.price_one_touch` / `price_no_touch` | Monte Carlo, um cenário por vez |
| qualquer um com `quanto=Quanto(...)` | `pricing_quanto.price_quanto_vanilla` / `quanto_market_data` | idem |

**Vol usada.** Opções com strike: a vol no strike. Toques (sem strike): a vol na barreira.

**Prazo.** `vencimento` é contado da data de referência (anos, na convenção do ativo); o prazo restante é
`vencimento - estado.tempo`. Vencido, o produto europeu vale o payoff no spot; o dependente de trajetória
levanta erro, porque o valor depende de um histórico que o estado não guarda.

**Quanto.** Ativo em moeda estrangeira (S&P 500 em USD) pago em BRL. O drift do ativo ganha o ajuste
-rho sigma_S sigma_X e o desconto é na curva BRL, a mesma conta do `quanto_market_data`. Payoffs em pontos
(vanilla, barreira, rebate) são convertidos pelo câmbio fixo do contrato; pagamentos em dinheiro (digital,
toques) já estão em BRL e não são convertidos.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.stats import norm

from ibov_barrier.pricing import BarrierContract, MarketData, PriceResult
from ibov_barrier.pricing_extensions import price_no_touch, price_one_touch, price_single_barrier
from ibov_barrier.produtos.base import Composicao, Fluxo, Observacao, por_cenario, saida
from ibov_barrier.produtos.mercado import EstadoMercado, Numero

_TIPOS = {"call", "put"}
_DIRECOES = {"down", "up"}
_KNOCKS = {"out", "in"}
_MONITORAMENTOS = {"discrete", "brownian_bridge"}


@dataclass(frozen=True)
class Quanto:
    """Liquidação Quanto: o ativo cotado em `par` base (USD) paga na moeda de cotação (BRL).

    `cambio_fixo` converte payoffs em pontos para BRL. Produtos que pagam valor fixo em dinheiro não usam.
    """

    par: str = "USDBRL"
    cambio_fixo: float | None = None

    def __post_init__(self):
        if self.cambio_fixo is not None and (not np.isfinite(self.cambio_fixo) or self.cambio_fixo <= 0):
            raise ValueError("cambio_fixo deve ser positivo")


@dataclass(frozen=True)
class _Parametros:
    spot: Numero
    taxa: Numero
    dividendos: Numero
    vol: Numero
    fator: float          # câmbio fixo no Quanto (payoffs em pontos), 1 nos demais
    moeda: str


def _parametros(estado: EstadoMercado, ativo: str, nivel_vol: float, prazo: float,
                quanto: Quanto | None) -> _Parametros:
    """Spot, r, q, vol e fator de conversão no prazo restante, na medida da moeda de liquidação."""
    m = estado.ativo(ativo)
    q = m.dividendos_em(prazo)
    vol = m.vol_em(nivel_vol, prazo)
    r_ativo = estado.curva(m.moeda).taxa_em(prazo)
    if quanto is None:
        return _Parametros(m.spot, r_ativo, q, vol, 1.0, m.moeda)
    cambio = estado.cambio(quanto.par)
    if cambio.base != m.moeda:
        raise ValueError(f"o par {quanto.par} não tem {m.moeda} como moeda base")
    r_dom = estado.curva(cambio.cotacao).taxa_em(prazo)
    rho = estado.correlacao(ativo, quanto.par)
    # Mesma conta do `quanto_market_data`: drift = r_ext - q - rho sigma_S sigma_X; q_eq = r_dom - drift.
    drift = r_ativo - q + (-rho * vol * cambio.vol)
    fator = 1.0 if quanto.cambio_fixo is None else float(quanto.cambio_fixo)
    return _Parametros(m.spot, r_dom, r_dom - drift, vol, fator, cambio.cotacao)


def _market_data(p: _Parametros) -> MarketData:
    return MarketData(spot=float(p.spot), rate=float(p.taxa), dividend_yield=float(p.dividendos),
                      volatility=float(p.vol))


def _escalar(resultado: PriceResult, fator: float) -> PriceResult:
    if fator == 1.0:
        return resultado
    return PriceResult(resultado.price * fator, resultado.standard_error * fator, resultado.ci_low * fator,
                       resultado.ci_high * fator, resultado.knock_out_probability, resultado.vanilla_price * fator)


def _validar_comum(ativo: str, vencimento: float, *niveis: float) -> None:
    if not ativo:
        raise ValueError("ativo não informado")
    if not np.isfinite(vencimento) or vencimento <= 0:
        raise ValueError("vencimento deve ser positivo")
    for nivel in niveis:
        if not np.isfinite(nivel) or nivel <= 0:
            raise ValueError("strikes e barreiras devem ser positivos")


def _validar_quanto_pontos(quanto: Quanto | None) -> None:
    if quanto is not None and quanto.cambio_fixo is None:
        raise ValueError("payoff em pontos no Quanto exige cambio_fixo")


# ------------------------------------------------------------------------------------- fórmulas fechadas

def black_scholes(spot, strike, prazo: float, taxa, dividendos, vol, tipo: str = "call") -> Numero:
    """Black-Scholes vetorizado (mesma fórmula do `black_scholes_call`; put pela forma direta).

    Com vol zero, devolve o valor intrínseco do forward descontado, como o pricer original.
    """
    s, k, r, q, v = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (spot, strike, taxa, dividendos, vol)))
    sinal = 1.0 if tipo == "call" else -1.0
    t = float(prazo)
    desc_r, desc_q = np.exp(-r * t), np.exp(-q * t)
    com_vol = v > 0
    v_seguro = np.where(com_vol, v, 1.0)
    d1 = (np.log(s / k) + (r - q + 0.5 * v_seguro**2) * t) / (v_seguro * sqrt(t))
    d2 = d1 - v_seguro * sqrt(t)
    preco = sinal * (s * desc_q * norm.cdf(sinal * d1) - k * desc_r * norm.cdf(sinal * d2))
    intrinseco = desc_r * np.maximum(sinal * (s * np.exp((r - q) * t) - k), 0.0)
    return np.where(com_vol, preco, intrinseco)


def digital_cash(spot, strike, prazo: float, taxa, dividendos, vol, tipo: str = "call") -> Numero:
    """Digital cash-or-nothing que paga 1: e^{-rT} N(±d2) (mesma fórmula do `analytical._bs_digital`)."""
    s, k, r, q, v = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (spot, strike, taxa, dividendos, vol)))
    sinal = 1.0 if tipo == "call" else -1.0
    t = float(prazo)
    com_vol = v > 0
    v_seguro = np.where(com_vol, v, 1.0)
    d2 = (np.log(s / k) + (r - q - 0.5 * v_seguro**2) * t) / (v_seguro * sqrt(t))
    prob = np.where(com_vol, norm.cdf(sinal * d2), (sinal * (s * np.exp((r - q) * t) - k) > 0).astype(float))
    return np.exp(-r * t) * prob


# ------------------------------------------------------------------------------------------ europeus

@dataclass(frozen=True)
class Vanilla(Composicao):
    """Call ou put europeia sobre `ativo`. No Quanto, paga `cambio_fixo * max(±(S_T - K), 0)` em BRL."""

    ativo: str
    strike: float
    vencimento: float
    tipo: str = "call"
    quanto: Quanto | None = None

    def __post_init__(self):
        if self.tipo not in _TIPOS:
            raise ValueError("tipo deve ser 'call' ou 'put'")
        _validar_comum(self.ativo, self.vencimento, self.strike)
        _validar_quanto_pontos(self.quanto)

    def valor(self, estado: EstadoMercado) -> Numero:
        prazo = self.vencimento - estado.tempo
        if prazo <= 0:
            fator = 1.0 if self.quanto is None else self.quanto.cambio_fixo
            return saida(fator * self._intrinseco(estado.ativo(self.ativo).spot), estado)
        p = _parametros(estado, self.ativo, self.strike, prazo, self.quanto)
        return saida(p.fator * black_scholes(p.spot, self.strike, prazo, p.taxa, p.dividendos, p.vol, self.tipo),
                     estado)

    def _intrinseco(self, nivel) -> np.ndarray:
        sinal = 1.0 if self.tipo == "call" else -1.0
        return np.maximum(sinal * (np.asarray(nivel, dtype=float) - self.strike), 0.0)

    def payoff(self, finais: Mapping[str, np.ndarray]) -> np.ndarray:
        fator = 1.0 if self.quanto is None else self.quanto.cambio_fixo
        return fator * self._intrinseco(finais[self.ativo])

    def subjacentes(self) -> tuple[str, ...]:
        return (self.ativo,)

    def fluxos(self) -> tuple[Fluxo, ...]:
        return (Fluxo(self.vencimento, None, f"payoff da {self.tipo} {self.ativo} K={self.strike:g}"),)

    def observacoes(self) -> tuple[Observacao, ...]:
        return (Observacao(self.vencimento, self.vencimento, "vencimento", self.strike, self.ativo),)


@dataclass(frozen=True)
class CallSpread(Composicao):
    """Call comprada em `strike_baixo` e vendida em `strike_alto`: paga min(max(S_T - K1, 0), K2 - K1).

    Cada perna usa a vol no próprio strike. É o pacote de opções do COE com participação e cap.
    """

    ativo: str
    strike_baixo: float
    strike_alto: float
    vencimento: float
    quanto: Quanto | None = None

    def __post_init__(self):
        _validar_comum(self.ativo, self.vencimento, self.strike_baixo, self.strike_alto)
        if self.strike_alto <= self.strike_baixo:
            raise ValueError("strike_alto deve ser maior que strike_baixo")
        _validar_quanto_pontos(self.quanto)

    def _opcoes(self) -> tuple[Vanilla, Vanilla]:
        return (Vanilla(self.ativo, self.strike_baixo, self.vencimento, "call", self.quanto),
                Vanilla(self.ativo, self.strike_alto, self.vencimento, "call", self.quanto))

    def valor(self, estado: EstadoMercado) -> Numero:
        comprada, vendida = self._opcoes()
        return saida(np.asarray(comprada.valor(estado)) - np.asarray(vendida.valor(estado)), estado)

    def payoff(self, finais: Mapping[str, np.ndarray]) -> np.ndarray:
        comprada, vendida = self._opcoes()
        return comprada.payoff(finais) - vendida.payoff(finais)

    def subjacentes(self) -> tuple[str, ...]:
        return (self.ativo,)

    def fluxos(self) -> tuple[Fluxo, ...]:
        return (Fluxo(self.vencimento, None,
                      f"payoff do call spread {self.ativo} {self.strike_baixo:g}/{self.strike_alto:g}"),)

    def observacoes(self) -> tuple[Observacao, ...]:
        return (Observacao(self.vencimento, self.vencimento, "vencimento", self.strike_baixo, self.ativo),)


@dataclass(frozen=True)
class Digital(Composicao):
    """Digital europeia cash-or-nothing: paga `pagamento` (na moeda de liquidação) se S_T > K (call) ou S_T < K
    (put)."""

    ativo: str
    strike: float
    vencimento: float
    tipo: str = "call"
    pagamento: float = 1.0
    quanto: Quanto | None = None

    def __post_init__(self):
        if self.tipo not in _TIPOS:
            raise ValueError("tipo deve ser 'call' ou 'put'")
        _validar_comum(self.ativo, self.vencimento, self.strike)
        if not np.isfinite(self.pagamento):
            raise ValueError("pagamento deve ser finito")

    def valor(self, estado: EstadoMercado) -> Numero:
        prazo = self.vencimento - estado.tempo
        if prazo <= 0:
            return saida(self.payoff({self.ativo: estado.ativo(self.ativo).spot}), estado)
        p = _parametros(estado, self.ativo, self.strike, prazo, self.quanto)
        return saida(self.pagamento * digital_cash(p.spot, self.strike, prazo, p.taxa, p.dividendos, p.vol,
                                                   self.tipo), estado)

    def payoff(self, finais: Mapping[str, np.ndarray]) -> np.ndarray:
        nivel = np.asarray(finais[self.ativo], dtype=float)
        acerto = nivel > self.strike if self.tipo == "call" else nivel < self.strike
        return self.pagamento * acerto.astype(float)

    def subjacentes(self) -> tuple[str, ...]:
        return (self.ativo,)

    def fluxos(self) -> tuple[Fluxo, ...]:
        return (Fluxo(self.vencimento, None, f"digital {self.tipo} {self.ativo} K={self.strike:g}"),)

    def observacoes(self) -> tuple[Observacao, ...]:
        return (Observacao(self.vencimento, self.vencimento, "vencimento", self.strike, self.ativo),)


# ------------------------------------------------------------------------------- dependentes de trajetória

@dataclass(frozen=True, kw_only=True)
class _MonteCarlo(Composicao):
    """Parâmetros de simulação comuns. A mesma semente dá os mesmos números aleatórios em todo cenário."""

    caminhos: int = 200_000
    passos: int = 252
    semente: int = 42
    monitoramento: str = "brownian_bridge"

    def _validar_mc(self) -> None:
        if self.caminhos < 2 or self.passos < 1:
            raise ValueError("caminhos >= 2 e passos >= 1")
        if self.monitoramento not in _MONITORAMENTOS:
            raise ValueError("monitoramento deve ser 'discrete' ou 'brownian_bridge'")

    def _kwargs_mc(self) -> dict:
        return dict(paths=self.caminhos, steps=self.passos, seed=self.semente, monitoring=self.monitoramento)

    def _prazo(self, estado: EstadoMercado) -> float:
        prazo = self.vencimento - estado.tempo
        if prazo <= 0:
            raise ValueError("produto dependente de trajetória vencido: o valor depende do histórico")
        return prazo

    def valor(self, estado: EstadoMercado) -> Numero:
        return por_cenario(lambda e: self.resultado(e).price, estado)


@dataclass(frozen=True, kw_only=True)
class BarreiraSimples(_MonteCarlo):
    """Barreira simples (DOC, UOC, DIC, UIC, DOP, UOP, DIP, UIP) com rebate pago no vencimento.

    Adapta `price_single_barrier`. O `rebate` está em pontos do ativo, como no pricer; no Quanto vira BRL pelo
    câmbio fixo.
    """

    ativo: str
    strike: float
    barreira: float
    vencimento: float
    tipo: str = "call"
    direcao: str = "down"
    knock: str = "out"
    rebate: float = 0.0
    quanto: Quanto | None = None

    def __post_init__(self):
        if self.tipo not in _TIPOS or self.direcao not in _DIRECOES or self.knock not in _KNOCKS:
            raise ValueError("tipo call/put, direcao down/up e knock out/in")
        _validar_comum(self.ativo, self.vencimento, self.strike, self.barreira)
        if not np.isfinite(self.rebate) or self.rebate < 0:
            raise ValueError("rebate deve ser não negativo")
        _validar_quanto_pontos(self.quanto)
        self._validar_mc()

    def resultado(self, estado: EstadoMercado) -> PriceResult:
        """O `PriceResult` do pricer (preço, erro-padrão, probabilidade), na moeda de liquidação."""
        prazo = self._prazo(estado)
        p = _parametros(estado, self.ativo, self.strike, prazo, self.quanto)
        res = price_single_barrier(
            _market_data(p), BarrierContract(self.strike, self.barreira, prazo),
            option_type=self.tipo, direction=self.direcao, knock=self.knock, rebate=self.rebate, **self._kwargs_mc(),
        )
        return _escalar(res, p.fator)

    def subjacentes(self) -> tuple[str, ...]:
        return (self.ativo,)

    def fluxos(self) -> tuple[Fluxo, ...]:
        return (Fluxo(self.vencimento, None,
                      f"payoff da barreira {self.direcao}-and-{self.knock} {self.tipo} {self.ativo}"),)

    def observacoes(self) -> tuple[Observacao, ...]:
        return (Observacao(0.0, self.vencimento, "barreira_continua", self.barreira, self.ativo),
                Observacao(self.vencimento, self.vencimento, "vencimento", self.strike, self.ativo))


@dataclass(frozen=True, kw_only=True)
class OneTouch(_MonteCarlo):
    """Paga `pagamento` no vencimento se o ativo tocar a barreira (`direcao` down ou up) até lá.

    Adapta `price_one_touch`. O `pagamento` está na moeda de liquidação (BRL no Quanto, sem conversão).
    """

    ativo: str
    barreira: float
    vencimento: float
    direcao: str
    pagamento: float = 1.0
    quanto: Quanto | None = None

    def __post_init__(self):
        if self.direcao not in _DIRECOES:
            raise ValueError("direcao deve ser 'down' ou 'up'")
        _validar_comum(self.ativo, self.vencimento, self.barreira)
        if not np.isfinite(self.pagamento) or self.pagamento <= 0:
            raise ValueError("pagamento deve ser positivo")
        self._validar_mc()

    def _pricer(self):
        return price_one_touch

    def resultado(self, estado: EstadoMercado) -> PriceResult:
        prazo = self._prazo(estado)
        p = _parametros(estado, self.ativo, self.barreira, prazo, self.quanto)
        return self._pricer()(
            _market_data(p), BarrierContract(self.barreira, self.barreira, prazo),
            direction=self.direcao, payout=self.pagamento, **self._kwargs_mc(),
        )

    def subjacentes(self) -> tuple[str, ...]:
        return (self.ativo,)

    def fluxos(self) -> tuple[Fluxo, ...]:
        return (Fluxo(self.vencimento, None, f"{type(self).__name__} {self.direcao} {self.ativo}"),)

    def observacoes(self) -> tuple[Observacao, ...]:
        return (Observacao(0.0, self.vencimento, "toque_continuo", self.barreira, self.ativo),)


@dataclass(frozen=True, kw_only=True)
class NoTouch(OneTouch):
    """Paga `pagamento` no vencimento se o ativo NUNCA tocar a barreira. Adapta `price_no_touch`."""

    def _pricer(self):
        return price_no_touch
