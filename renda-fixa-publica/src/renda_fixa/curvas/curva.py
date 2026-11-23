"""Curvas zero em taxa efetiva anual (% a.a.), prazo em dias úteis, base 252.

Toda curva expõe ``taxa(du)``, ``fator_desconto(du)``, ``taxa_termo(du1, du2)`` e
``deslocada(choque)``. Assim títulos, derivativos e risco funcionam com qualquer uma.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..calendario import DU_ANO

Choque = float | Callable[[float], float]  # bps constantes ou função du -> bps


class Curva(ABC):
    nome: str = ""

    @abstractmethod
    def taxa(self, du: float) -> float:
        """Taxa zero efetiva em % a.a. para o prazo ``du``."""

    def fator_desconto(self, du: float) -> float:
        if du <= 0:
            return 1.0
        return (1.0 + self.taxa(du) / 100.0) ** (-du / DU_ANO)

    def taxa_termo(self, du1: float, du2: float) -> float:
        """Taxa a termo efetiva (% a.a.) entre ``du1`` e ``du2``."""
        if du2 <= du1:
            raise ValueError("du2 deve ser maior que du1.")
        razao = self.fator_desconto(du1) / self.fator_desconto(du2)
        return (razao ** (DU_ANO / (du2 - du1)) - 1.0) * 100.0

    def taxas(self, dus) -> np.ndarray:
        return np.array([self.taxa(float(d)) for d in np.atleast_1d(dus)])

    def deslocada(self, choque: Choque) -> "Curva":
        """Nova curva com a taxa zero somada de ``choque`` bps (constante ou função do prazo)."""
        return CurvaDeslocada(self, choque)


class CurvaZero(Curva):
    """Curva definida por vértices (DU, taxa % a.a.).

    Interpolação:
    - ``flat_forward`` (padrão do mercado brasileiro): log do fator de desconto linear em DU,
      ou seja, taxa a termo constante entre vértices;
    - ``linear``: taxa linear em DU.
    Fora dos vértices a taxa é mantida constante (primeiro e último vértice).
    """

    def __init__(self, dus, taxas, interpolacao: str = "flat_forward", nome: str = ""):
        dus = np.asarray(dus, dtype=float)
        taxas = np.asarray(taxas, dtype=float)
        if dus.shape != taxas.shape or dus.size == 0:
            raise ValueError("dus e taxas precisam ter o mesmo tamanho (>0).")
        ordem = np.argsort(dus)
        self.dus = dus[ordem]
        self.taxas_vertices = taxas[ordem]
        if np.any(np.diff(self.dus) <= 0):
            raise ValueError("Vértices repetidos.")
        if interpolacao not in ("flat_forward", "linear"):
            raise ValueError("interpolacao deve ser 'flat_forward' ou 'linear'.")
        self.interpolacao = interpolacao
        self.nome = nome
        self._logdf = -(self.dus / DU_ANO) * np.log1p(self.taxas_vertices / 100.0)

    def taxa(self, du: float) -> float:
        du = float(du)
        if du <= self.dus[0]:
            return float(self.taxas_vertices[0])
        if du >= self.dus[-1]:
            return float(self.taxas_vertices[-1])
        if self.interpolacao == "linear":
            return float(np.interp(du, self.dus, self.taxas_vertices))
        logdf = float(np.interp(du, self.dus, self._logdf))
        return (np.exp(-logdf * DU_ANO / du) - 1.0) * 100.0

    def com_vertice(self, du: float, taxa: float) -> "CurvaZero":
        """Cópia com um vértice adicionado ou substituído."""
        m = self.dus != du
        return CurvaZero(np.append(self.dus[m], du), np.append(self.taxas_vertices[m], taxa), self.interpolacao, self.nome)

    def __repr__(self) -> str:
        return f"CurvaZero({self.nome or '-'}, {len(self.dus)} vértices, {self.interpolacao})"


class CurvaDeslocada(Curva):
    def __init__(self, base: Curva, choque: Choque):
        self.base = base
        self.choque = choque
        self.nome = getattr(base, "nome", "")

    def taxa(self, du: float) -> float:
        bps = self.choque(du) if callable(self.choque) else float(self.choque)
        return self.base.taxa(du) + bps / 100.0


class CurvaPlana(Curva):
    """Taxa única para todos os prazos (útil em testes e para curvas de spread)."""

    def __init__(self, taxa_pct: float, nome: str = ""):
        self._t = float(taxa_pct)
        self.nome = nome

    def taxa(self, du: float) -> float:
        return self._t
