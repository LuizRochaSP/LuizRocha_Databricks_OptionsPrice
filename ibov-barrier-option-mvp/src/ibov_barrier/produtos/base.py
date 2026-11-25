"""Interface comum de produtos (etapa A1 do plano de COE).

Todo produto (opção, título, COE, carteira) responde a quatro perguntas:

- `valor(estado)`: o valor justo no `EstadoMercado`. Escalar se o estado é escalar; vetor de n valores se o
  estado traz n cenários.
- `fluxos()`: os pagamentos do produto (`Fluxo`), com valor conhecido ou contingente (`valor=None`).
- `observacoes()`: as datas ou janelas em que o produto observa o mercado (`Observacao`).
- `subjacentes()`: os ativos de que o valor depende.

VaR, simulação de hedge, solver e Desk só conversam com essa interface. Produtos europeus também expõem
`payoff(finais)`, o valor pago no vencimento dados os níveis finais dos ativos.

Soma, subtração e multiplicação por número formam uma `Carteira`:

    cs = Vanilla("IBOV", 100, 1.0) - Vanilla("IBOV", 120, 1.0)
    coe = titulo + 0.8 * cs
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ibov_barrier.produtos.mercado import EstadoMercado, Numero


@dataclass(frozen=True)
class Fluxo:
    """Um pagamento no `tempo` (anos). `valor=None` quando depende do mercado (payoff contingente)."""

    tempo: float
    valor: float | None
    descricao: str
    moeda: str | None = None


@dataclass(frozen=True)
class Observacao:
    """Uma data (`inicio == fim`) ou janela de observação do mercado.

    `tipo`: "vencimento", "barreira_continua", "toque_continuo" etc. `nivel`: strike ou barreira observados.
    """

    inicio: float
    fim: float
    tipo: str
    nivel: float | None = None
    ativo: str | None = None


@runtime_checkable
class Produto(Protocol):
    def valor(self, estado: EstadoMercado) -> Numero: ...

    def fluxos(self) -> tuple[Fluxo, ...]: ...

    def observacoes(self) -> tuple[Observacao, ...]: ...

    def subjacentes(self) -> tuple[str, ...]: ...


def saida(valor, estado: EstadoMercado) -> Numero:
    """Formato de saída de `valor`: float no estado escalar, vetor de n posições no estado vetorial."""
    if estado.vetorial:
        return np.array(np.broadcast_to(np.asarray(valor, dtype=float), (estado.n_cenarios,)))
    return float(np.asarray(valor, dtype=float).reshape(-1)[0]) if np.ndim(valor) else float(valor)


def por_cenario(funcao, estado: EstadoMercado) -> Numero:
    """Avalia um produto sem fórmula vetorial (Monte Carlo) um cenário por vez."""
    if not estado.vetorial:
        return float(funcao(estado))
    return np.array([float(funcao(c)) for c in estado.cenarios()])


def _pernas(produto) -> tuple[tuple[float, Produto], ...]:
    if isinstance(produto, Carteira):
        return produto.pernas
    if not isinstance(produto, Produto):
        raise TypeError(f"{type(produto).__name__} não é um Produto")
    return ((1.0, produto),)


class Composicao:
    """Aritmética de produtos: `a + b`, `a - b`, `k * a` e `-a` formam uma `Carteira`."""

    # Faz o numpy devolver a operação ao Python (np.float64(2) * produto chama __rmul__).
    __array_ufunc__ = None

    def __add__(self, outro):
        if isinstance(outro, (int, float)) and outro == 0:
            return Carteira(_pernas(self))
        return Carteira(_pernas(self) + _pernas(outro))

    def __radd__(self, outro):
        # sum() começa em 0.
        if isinstance(outro, (int, float)) and outro == 0:
            return Carteira(_pernas(self))
        return Carteira(_pernas(outro) + _pernas(self))

    def __sub__(self, outro):
        return self + (-1.0) * outro

    def __mul__(self, peso):
        if not isinstance(peso, (int, float, np.floating, np.integer)):
            return NotImplemented
        return Carteira(tuple((float(peso) * w, p) for w, p in _pernas(self)))

    __rmul__ = __mul__

    def __neg__(self):
        return (-1.0) * self


@dataclass(frozen=True)
class Carteira(Composicao):
    """Soma ponderada de produtos: `pernas = ((peso, produto), ...)`."""

    pernas: tuple[tuple[float, Produto], ...]

    def __post_init__(self):
        pernas = tuple((float(w), p) for w, p in self.pernas)
        if not pernas:
            raise ValueError("carteira vazia")
        for _, p in pernas:
            if not isinstance(p, Produto):
                raise TypeError(f"{type(p).__name__} não é um Produto")
        object.__setattr__(self, "pernas", pernas)

    def valor(self, estado: EstadoMercado) -> Numero:
        total = sum(w * np.asarray(p.valor(estado), dtype=float) for w, p in self.pernas)
        return saida(total, estado)

    def payoff(self, finais: Mapping[str, np.ndarray]) -> np.ndarray:
        """Payoff no vencimento; exige que todas as pernas sejam europeias (tenham `payoff`)."""
        for _, p in self.pernas:
            if not hasattr(p, "payoff"):
                raise TypeError(f"{type(p).__name__} não tem payoff europeu")
        return sum(w * np.asarray(p.payoff(finais), dtype=float) for w, p in self.pernas)

    @property
    def vencimento(self) -> float:
        return max(float(getattr(p, "vencimento", 0.0)) for _, p in self.pernas)

    @property
    def depende_do_spread(self) -> bool:
        """True se alguma perna depende do spread do emissor (o título): as gregas calculam o CS01."""
        return any(getattr(p, "depende_do_spread", False) for _, p in self.pernas)

    def subjacentes(self) -> tuple[str, ...]:
        vistos: dict[str, None] = {}
        for _, p in self.pernas:
            vistos.update(dict.fromkeys(p.subjacentes()))
        return tuple(vistos)

    def fluxos(self) -> tuple[Fluxo, ...]:
        saida_fluxos = []
        for w, p in self.pernas:
            for f in p.fluxos():
                valor = None if f.valor is None else w * f.valor
                saida_fluxos.append(Fluxo(f.tempo, valor, f.descricao, f.moeda))
        return tuple(sorted(saida_fluxos, key=lambda f: f.tempo))

    def observacoes(self) -> tuple[Observacao, ...]:
        todas = {o for _, p in self.pernas for o in p.observacoes()}
        return tuple(sorted(todas, key=lambda o: (o.fim, o.inicio, o.tipo)))
