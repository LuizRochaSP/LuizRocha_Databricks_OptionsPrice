"""Gregas genéricas por diferenças finitas sobre a interface `Produto` (etapa A1 do plano de COE).

Servem para qualquer produto, inclusive os de Monte Carlo: como cada adaptador reusa a mesma semente, os
cenários chocados usam os mesmos números aleatórios e a diferença finita não carrega o ruído da simulação.

| Grega | Choque | Unidade |
| :--- | :--- | :--- |
| delta, gamma | spot × (1 ± `choque_spot`), diferença central | por ponto do ativo |
| vega | vol ± `choque_vol` (aditivo), diferença central | por 1 ponto percentual de vol |
| dv01 | curva da moeda ± `choque_bp` (`Curva.deslocar`), diferença central | por +1 bp |
| theta | tempo + `dt` (anos) | variação do valor em `dt` |

O choque de spot e o de vol vão num único estado vetorial de 2 cenários: o produto de fórmula fechada se avalia
uma vez só.
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

    def como_dict(self) -> dict[str, float]:
        """Achatado para relatório: {"valor", "delta[IBOV]", "dv01[BRL]", "theta", ...}."""
        saida = {"valor": self.valor}
        for nome in ("delta", "gamma", "vega", "dv01"):
            saida.update({f"{nome}[{k}]": v for k, v in getattr(self, nome).items()})
        saida["theta"] = self.theta
        return saida


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
    dv01 = {}
    for moeda, curva in estado.curvas.items():
        alto = float(produto.valor(estado.com_curva(moeda, curva.deslocar(choque_bp))))
        baixo = float(produto.valor(estado.com_curva(moeda, curva.deslocar(-choque_bp))))
        dv01[moeda] = (alto - baixo) / (2.0 * choque_bp)
    theta = float(produto.valor(estado.avancar(dt))) - base
    return Gregas(valor=base, delta=delta, gamma=gamma, vega=vega, dv01=dv01, theta=theta)
