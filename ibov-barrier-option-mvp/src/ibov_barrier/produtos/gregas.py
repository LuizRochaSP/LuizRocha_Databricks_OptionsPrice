"""Gregas genéricas por diferenças finitas sobre a interface `Produto` (etapas A1 e A2 do plano de COE).

Servem para qualquer produto, inclusive os de Monte Carlo: como cada adaptador reusa a mesma semente, os
cenários chocados usam os mesmos números aleatórios e a diferença finita não carrega o ruído da simulação.

| Grega | Choque | Unidade |
| :--- | :--- | :--- |
| delta, gamma | spot × (1 ± `choque_spot`), diferença central | por ponto do ativo |
| vega | vol ± `choque_vol` (aditivo), diferença central | por 1 ponto percentual de vol |
| dv01 | curva da moeda ± `choque_bp` (`Curva.deslocar`), diferença central | por +1 bp |
| cs01 | spread do emissor ± `choque_bp`, diferença central (só se o produto depende do spread) | por +1 bp |
| theta | tempo + `dt` (anos) | variação do valor em `dt` |

O choque de spot e o de vol vão num único estado vetorial de 2 cenários: o produto de fórmula fechada se avalia
uma vez só. `dv01_por_vertice` desloca um vértice da curva DI por vez (`bump_key`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ibov_barrier.produtos.base import Produto
from ibov_barrier.produtos.mercado import EstadoMercado


@dataclass(frozen=True)
class Gregas:
    valor: float
    delta: dict[str, float] = field(default_factory=dict)
    gamma: dict[str, float] = field(default_factory=dict)
    vega: dict[str, float] = field(default_factory=dict)
    dv01: dict[str, float] = field(default_factory=dict)
    theta: float = 0.0
    cs01: float | None = None   # None quando o produto não depende do spread do emissor

    def como_dict(self) -> dict[str, float]:
        """Achatado para relatório: {"valor", "delta[IBOV]", "dv01[BRL]", "cs01", "theta", ...}."""
        saida = {"valor": self.valor}
        for nome in ("delta", "gamma", "vega", "dv01"):
            saida.update({f"{nome}[{k}]": v for k, v in getattr(self, nome).items()})
        if self.cs01 is not None:
            saida["cs01"] = self.cs01
        saida["theta"] = self.theta
        return saida


def _central(produto: Produto, estado_alto: EstadoMercado, estado_baixo: EstadoMercado, choque: float) -> float:
    return (float(produto.valor(estado_alto)) - float(produto.valor(estado_baixo))) / (2.0 * choque)


def gregas(produto: Produto, estado: EstadoMercado, *, choque_spot: float = 0.01, choque_vol: float = 0.01,
           choque_bp: float = 1.0, dt: float = 1.0 / 252.0) -> Gregas:
    """Gregas de `produto` no estado escalar `estado`, para cada subjacente e cada curva do estado."""
    if estado.vetorial:
        raise ValueError("gregas pedem um estado escalar (sem vetores de cenários)")
    if choque_spot <= 0 or choque_vol <= 0 or choque_bp <= 0 or dt <= 0:
        raise ValueError("choques e dt devem ser positivos")
    base = float(produto.valor(estado))
    pares = np.array([-1.0, 1.0])
    delta, gamma, vega = {}, {}, {}
    for ativo in produto.subjacentes():
        if ativo not in estado.ativos:
            continue
        mercado = estado.ativo(ativo)
        h = choque_spot * float(mercado.spot)
        baixo, alto = np.asarray(produto.valor(estado.com_spot(ativo, float(mercado.spot) + h * pares)))
        delta[ativo] = float((alto - baixo) / (2.0 * h))
        gamma[ativo] = float((alto - 2.0 * base + baixo) / h**2)
        baixo, alto = np.asarray(produto.valor(
            estado.com_choque_vol(ativo, float(mercado.choque_vol) + choque_vol * pares)))
        vega[ativo] = float((alto - baixo) / (2.0 * choque_vol) * 0.01)
    dv01 = {
        moeda: _central(produto, estado.com_curva(moeda, curva.deslocar(choque_bp)),
                        estado.com_curva(moeda, curva.deslocar(-choque_bp)), choque_bp)
        for moeda, curva in estado.curvas.items()
    }
    cs01 = None
    if getattr(produto, "depende_do_spread", False):
        spread = float(estado.spread_emissor_bp)
        cs01 = _central(produto, estado.com_spread(spread + choque_bp), estado.com_spread(spread - choque_bp),
                        choque_bp)
    theta = float(produto.valor(estado.avancar(dt))) - base
    return Gregas(valor=base, delta=delta, gamma=gamma, vega=vega, dv01=dv01, theta=theta, cs01=cs01)


def dv01_por_vertice(produto: Produto, estado: EstadoMercado, moeda: str = "BRL", *,
                     choque_bp: float = 1.0) -> dict[float, float]:
    """DV01 (por +1 bp) de cada vértice da curva da moeda, deslocando um vértice por vez.

    Exige uma curva com `vertices` e `deslocar_vertice` (`CurvaDI`). A soma dos vértices bate com o DV01 paralelo.
    """
    if estado.vetorial:
        raise ValueError("DV01 por vértice pede um estado escalar")
    curva = estado.curva(moeda)
    if not hasattr(curva, "deslocar_vertice"):
        raise TypeError(f"a curva em {moeda} não tem vértices (use CurvaDI)")
    return {
        du: _central(produto, estado.com_curva(moeda, curva.deslocar_vertice(du, choque_bp)),
                     estado.com_curva(moeda, curva.deslocar_vertice(du, -choque_bp)), choque_bp)
        for du in curva.vertices
    }
