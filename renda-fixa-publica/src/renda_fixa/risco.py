"""Risco de carteiras de renda fixa: posições, DV01, key-rate, cenários, carry, hedge, VaR e PCA.

As curvas viajam num dicionário ``{"PRE": curva_pre, "IPCA": curva_real, "IGPM": curva_igpm}``. Cada posição
sabe se reprecificar dada uma data e as curvas (reavaliação completa, sem aproximação
por duration), então qualquer choque é só uma curva deslocada.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .calendario import dias_uteis, soma_dias_uteis
from .curvas.bootstrap import z_spread
from .curvas.curva import Curva
from .derivativos.futuros import DAP, DI1
from .derivativos.swaps import SwapDIIPCA, SwapDIPre
from .titulos.base import TituloPublico

Curvas = Mapping[str, Curva]
CURVA_DO_INDEXADOR = {"PRE": "PRE", "IPCA": "IPCA", "IGPM": "IGPM", "SELIC": None}


class Posicao(ABC):
    descricao: str = ""

    @abstractmethod
    def vp(self, ref: date, curvas: Curvas) -> float:
        """Valor presente (R$) na data, dadas as curvas."""


@dataclass
class PosicaoTitulo(Posicao):
    """Quantidade de títulos, precificados pela curva do indexador + z-spread.

    LFT: o risco de juros é desprezível (pós-fixado); o valor usa a taxa de ágio fixa.
    """

    titulo: TituloPublico
    quantidade: float
    vna: float | None = None
    spread_bps: float = 0.0
    taxa_lft: float | None = None

    def __post_init__(self):
        self.descricao = f"{self.quantidade:g} x {self.titulo}"

    @classmethod
    def a_mercado(cls, titulo, quantidade, pu, ref, curvas: Curvas, vna=None):
        """Posição calibrada para valer exatamente o PU de mercado (z-spread sobre a curva)."""
        nome = CURVA_DO_INDEXADOR[titulo.indexador]
        if nome is None:
            return cls(titulo, quantidade, vna, taxa_lft=titulo.taxa(pu, ref, vna, casas=None))
        return cls(titulo, quantidade, vna, spread_bps=z_spread(titulo, pu, curvas[nome], ref, vna))

    def vp(self, ref, curvas):
        nome = CURVA_DO_INDEXADOR[self.titulo.indexador]
        if nome is None:
            return self.quantidade * self.titulo.pu_bruto(self.taxa_lft or 0.0, ref, self.vna)
        curva = curvas[nome].deslocada(self.spread_bps) if self.spread_bps else curvas[nome]
        return self.quantidade * self.titulo.pu_curva(curva, ref, self.vna)


@dataclass
class PosicaoDI1(Posicao):
    """Quantidade > 0 = comprado em PU (vendido em taxa). Valor = q x (PU pela curva - PU de entrada)."""

    contrato: DI1
    quantidade: float
    pu_entrada: float

    def __post_init__(self):
        self.descricao = f"{self.quantidade:g} x {self.contrato}"

    def vp(self, ref, curvas):
        return self.quantidade * (self.contrato.pu_curva(curvas["PRE"], ref) - self.pu_entrada)


@dataclass
class PosicaoDAP(Posicao):
    contrato: DAP
    quantidade: float
    pu_entrada: float
    indice_ipca: float

    def __post_init__(self):
        self.descricao = f"{self.quantidade:g} x {self.contrato}"

    def vp(self, ref, curvas):
        pontos = self.contrato.pu_curva(curvas["IPCA"], ref) - self.pu_entrada
        return self.quantidade * pontos * DAP.valor_ponto(self.indice_ipca)


@dataclass
class PosicaoSwap(Posicao):
    """Swap avaliado na data ``ref_base`` com o CDI realizado até ali (``fator_cdi``).

    Para datas posteriores (carry), o CDI do intervalo é projetado pela curva pré.
    """

    swap: SwapDIPre | SwapDIIPCA
    ref_base: date
    fator_cdi: float = 1.0
    vna_ref: float | None = None

    def __post_init__(self):
        self.descricao = f"{type(self.swap).__name__} {self.swap.nocional:,.0f} até {self.swap.vencimento}"

    def vp(self, ref, curvas):
        fator = self.fator_cdi
        if ref > self.ref_base:
            fator /= curvas["PRE"].fator_desconto(dias_uteis(self.ref_base, ref))
        if isinstance(self.swap, SwapDIPre):
            return self.swap.avaliar(ref, curvas["PRE"], fator).mtm
        return self.swap.avaliar(ref, curvas["PRE"], curvas["IPCA"], self.vna_ref, fator).mtm


def _bump(curvas: Curvas, nome: str, choque) -> dict:
    out = dict(curvas)
    out[nome] = curvas[nome].deslocada(choque)
    return out


def chapeu(vertices: Sequence[float], i: int) -> Callable[[float], float]:
    """Função triangular com 1 bp no vértice i e 0 nos vizinhos (key-rate)."""
    v = list(vertices)

    def f(du: float) -> float:
        x = v[i]
        if du == x:
            return 1.0
        if du < x:
            if i == 0:
                return 1.0
            a = v[i - 1]
            return max(0.0, (du - a) / (x - a)) if du > a else 0.0
        if i == len(v) - 1:
            return 1.0
        b = v[i + 1]
        return max(0.0, (b - du) / (b - x)) if du < b else 0.0

    return f


@dataclass
class Carteira:
    posicoes: list[Posicao] = field(default_factory=list)

    def adicionar(self, p: Posicao) -> "Carteira":
        self.posicoes.append(p)
        return self

    def valor(self, ref: date, curvas: Curvas) -> float:
        return float(sum(p.vp(ref, curvas) for p in self.posicoes))

    def valores(self, ref: date, curvas: Curvas) -> pd.DataFrame:
        return pd.DataFrame([{"posicao": p.descricao, "vp": p.vp(ref, curvas)} for p in self.posicoes])

    # ----- sensibilidades -----
    def dv01(self, ref: date, curvas: Curvas, nome: str = "PRE") -> float:
        """Variação do valor para +1 bp paralelo na curva ``nome``."""
        return self.valor(ref, _bump(curvas, nome, 1.0)) - self.valor(ref, curvas)

    def key_rate_dv01(self, ref: date, curvas: Curvas, nome: str, vertices_du: Sequence[float]) -> pd.Series:
        """DV01 por vértice (choques triangulares de 1 bp). A soma ~ DV01 paralelo."""
        base = self.valor(ref, curvas)
        out = {int(v): self.valor(ref, _bump(curvas, nome, chapeu(vertices_du, i))) - base for i, v in enumerate(vertices_du)}
        return pd.Series(out, name=f"key_rate_dv01_{nome}")

    def dv01_por_posicao(self, ref: date, curvas: Curvas) -> pd.DataFrame:
        linhas = []
        for p in self.posicoes:
            v0 = p.vp(ref, curvas)
            linha = {"posicao": p.descricao, "vp": v0}
            for nome in curvas:
                linha[f"dv01_{nome}"] = p.vp(ref, _bump(curvas, nome, 1.0)) - v0
            linhas.append(linha)
        return pd.DataFrame(linhas)

    # ----- cenários -----
    def cenarios(self, ref: date, curvas: Curvas, cenarios: Mapping[str, Mapping[str, object]]) -> pd.Series:
        """P&L de cada cenário: {nome_cenario: {nome_curva: choque (bps ou função du->bps)}}."""
        base = self.valor(ref, curvas)
        out = {}
        for nome_cen, choques in cenarios.items():
            c = dict(curvas)
            for nome_curva, choque in choques.items():
                c[nome_curva] = curvas[nome_curva].deslocada(choque)
            out[nome_cen] = self.valor(ref, c) - base
        return pd.Series(out, name="pnl")

    # ----- carry e roll-down -----
    def carry_rolldown(self, ref: date, curvas: Curvas, horizonte_du: int = 21) -> float:
        """Variação de valor em ``horizonte_du`` dias úteis com as curvas paradas (em prazo).

        Títulos IPCA ficam em termos reais (VNA constante); a ponta DI dos swaps acumula o CDI
        projetado pela curva pré.
        """
        futuro = soma_dias_uteis(ref, horizonte_du)
        return self.valor(futuro, curvas) - self.valor(ref, curvas)

    # ----- VaR histórico -----
    def var_historico(
        self,
        ref: date,
        curvas: Curvas,
        choques: pd.DataFrame,
        nome: str = "PRE",
        nivel: float = 0.99,
    ) -> dict:
        """VaR e Expected Shortfall por reavaliação completa.

        ``choques``: uma linha por dia histórico, colunas = vértices em DU, valores = variação
        da taxa em bps. O choque entre vértices é interpolado linearmente.
        """
        base = self.valor(ref, curvas)
        vertices = np.array([float(c) for c in choques.columns])
        pnls = []
        for _, linha in choques.iterrows():
            bps = linha.values.astype(float)
            f = lambda du, b=bps: float(np.interp(du, vertices, b))
            pnls.append(self.valor(ref, _bump(curvas, nome, f)) - base)
        pnls = np.array(pnls)
        q = np.quantile(pnls, 1 - nivel)
        return {
            "var": -float(q),
            "es": -float(pnls[pnls <= q].mean()),
            "nivel": nivel,
            "n_cenarios": len(pnls),
            "pnls": pnls,
        }


def cenarios_padrao(pivo_du: float = 504.0) -> dict:
    """Cenários de stress usuais na curva pré e na real."""
    inclinacao = lambda du: 50.0 * (du - pivo_du) / pivo_du if du < 2 * pivo_du else 50.0
    borboleta = lambda du: 25.0 if abs(du - pivo_du * 2) < pivo_du else -25.0
    return {
        "Pré +100 bps": {"PRE": 100.0},
        "Pré -100 bps": {"PRE": -100.0},
        "Pré inclinação +50": {"PRE": inclinacao},
        "Pré borboleta": {"PRE": borboleta},
        "Real +50 bps": {"IPCA": 50.0},
        "Real -50 bps": {"IPCA": -50.0},
        "Pré +100 e real +50": {"PRE": 100.0, "IPCA": 50.0},
    }


def contratos_hedge_di1(dv01_carteira: float, contrato: DI1, taxa: float, ref: date) -> float:
    """Contratos de DI1 (em PU: + comprado) que zeram o DV01 pré da carteira.

    Comprado em PU ganha quando a taxa cai; o DV01 de +1 contrato é -dv01(contrato).
    """
    return dv01_carteira / contrato.dv01(taxa, ref)


@dataclass(frozen=True)
class ResultadoPCA:
    autovalores: np.ndarray
    variancia_explicada: np.ndarray
    cargas: pd.DataFrame  # linhas = vértices, colunas = PC1, PC2...


def pca_curva(taxas_historicas: pd.DataFrame, n_componentes: int = 3) -> ResultadoPCA:
    """PCA das variações diárias das taxas (colunas = vértices). PC1 ~ nível, PC2 ~ inclinação, PC3 ~ curvatura."""
    dx = taxas_historicas.sort_index().diff().dropna()
    cov = np.cov(dx.values, rowvar=False)
    autoval, autovet = np.linalg.eigh(cov)
    ordem = np.argsort(autoval)[::-1]
    autoval, autovet = autoval[ordem], autovet[:, ordem]
    # sinal: PC1 positivo na média
    for k in range(autovet.shape[1]):
        if autovet[:, k].sum() < 0:
            autovet[:, k] *= -1
    n = min(n_componentes, len(autoval))
    cargas = pd.DataFrame(autovet[:, :n], index=dx.columns, columns=[f"PC{i + 1}" for i in range(n)])
    return ResultadoPCA(autoval[:n], (autoval / autoval.sum())[:n], cargas)
