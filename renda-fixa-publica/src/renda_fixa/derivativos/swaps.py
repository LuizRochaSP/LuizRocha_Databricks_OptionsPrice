"""Swaps bullet (pagamento único no vencimento), padrão do mercado de balcão brasileiro.

Ponta DI: N x fator CDI acumulado (até a data de referência, realizado) x fator projetado
(da data de referência ao vencimento, pela curva pré), com ``pct_cdi`` e spread aditivo
em % a.a. (CDI + s). O valor presente desconta pela curva pré.

Ponta pré: N x (1 + K)^(DU_total/252), descontada pela curva pré.

Ponta IPCA: N x (VNA_ref / VNA_inicio) x (1 + cupom)^(DU_total/252), descontada pela curva
de juro real (cupom de IPCA). A razão de VNAs é a inflação acumulada (projetada na data).

Convenção de sinal: ``mtm`` é sempre do ponto de vista de quem é ATIVO (recebe) na ponta
indicada por ``ativo``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..calendario import DU_ANO, dias_uteis
from ..curvas.curva import Curva


def fator_projetado_cdi(curva_pre: Curva, du_restante: int, pct_cdi: float = 100.0, spread_aa: float = 0.0) -> float:
    """Fator projetado do CDI pela curva (taxas a termo diárias), a ``pct_cdi``% + spread."""
    if du_restante <= 0:
        return 1.0
    if pct_cdi == 100.0:
        fator = 1.0 / curva_pre.fator_desconto(du_restante)
    else:
        dfs = np.array([curva_pre.fator_desconto(d) for d in range(0, du_restante + 1)])
        diarias = dfs[:-1] / dfs[1:] - 1.0
        fator = float(np.prod(1.0 + diarias * pct_cdi / 100.0))
    return fator * (1.0 + spread_aa / 100.0) ** (du_restante / DU_ANO)


@dataclass(frozen=True)
class ResultadoSwap:
    vp_ativo: float
    vp_passivo: float
    mtm: float
    dv01: float  # variação do MtM para +1 bp na curva pré (paralelo)
    du_restante: int


class _SwapBase:
    def __init__(
        self,
        nocional: float,
        inicio: date,
        vencimento: date,
        ativo: str,
        pct_cdi: float = 100.0,
        spread_cdi_aa: float = 0.0,
    ):
        self.nocional = float(nocional)
        self.inicio = inicio
        self.vencimento = vencimento
        self.ativo = ativo
        self.pct_cdi = pct_cdi
        self.spread_cdi_aa = spread_cdi_aa
        self.du_total = dias_uteis(inicio, vencimento)

    def vp_di(self, ref: date, curva_pre: Curva, fator_cdi_acumulado: float = 1.0) -> float:
        n = dias_uteis(ref, self.vencimento)
        fator_proj = fator_projetado_cdi(curva_pre, n, self.pct_cdi, self.spread_cdi_aa)
        return self.nocional * fator_cdi_acumulado * fator_proj * curva_pre.fator_desconto(n)


class SwapDIPre(_SwapBase):
    """Swap DI x Pré. ``ativo``: 'pre' (recebe pré, paga DI) ou 'di' (recebe DI, paga pré)."""

    def __init__(self, nocional, inicio, vencimento, taxa_pre, ativo="pre", pct_cdi=100.0, spread_cdi_aa=0.0):
        super().__init__(nocional, inicio, vencimento, ativo, pct_cdi, spread_cdi_aa)
        if ativo not in ("pre", "di"):
            raise ValueError("ativo deve ser 'pre' ou 'di'.")
        self.taxa_pre = taxa_pre

    def valor_futuro_pre(self) -> float:
        return self.nocional * (1.0 + self.taxa_pre / 100.0) ** (self.du_total / DU_ANO)

    def vp_pre(self, ref: date, curva_pre: Curva) -> float:
        return self.valor_futuro_pre() * curva_pre.fator_desconto(dias_uteis(ref, self.vencimento))

    def _mtm(self, ref, curva_pre, fator_cdi_acumulado):
        pre = self.vp_pre(ref, curva_pre)
        di = self.vp_di(ref, curva_pre, fator_cdi_acumulado)
        return (pre, di) if self.ativo == "pre" else (di, pre)

    def avaliar(self, ref: date, curva_pre: Curva, fator_cdi_acumulado: float = 1.0) -> ResultadoSwap:
        a, p = self._mtm(ref, curva_pre, fator_cdi_acumulado)
        a2, p2 = self._mtm(ref, curva_pre.deslocada(1.0), fator_cdi_acumulado)
        return ResultadoSwap(a, p, a - p, (a2 - p2) - (a - p), dias_uteis(ref, self.vencimento))

    @staticmethod
    def taxa_par(curva_pre: Curva, inicio: date, vencimento: date, pct_cdi: float = 100.0, spread_cdi_aa: float = 0.0) -> float:
        """Taxa pré que zera o swap no início (% a.a.)."""
        n = dias_uteis(inicio, vencimento)
        fator = fator_projetado_cdi(curva_pre, n, pct_cdi, spread_cdi_aa)
        return (fator ** (DU_ANO / n) - 1.0) * 100.0


class SwapDIIPCA(_SwapBase):
    """Swap DI x IPCA. ``ativo``: 'ipca' (recebe IPCA + cupom) ou 'di'."""

    def __init__(self, nocional, inicio, vencimento, cupom_real, vna_inicio, ativo="ipca", pct_cdi=100.0, spread_cdi_aa=0.0):
        super().__init__(nocional, inicio, vencimento, ativo, pct_cdi, spread_cdi_aa)
        if ativo not in ("ipca", "di"):
            raise ValueError("ativo deve ser 'ipca' ou 'di'.")
        self.cupom_real = cupom_real
        self.vna_inicio = vna_inicio

    def vp_ipca(self, ref: date, curva_real: Curva, vna_ref: float) -> float:
        n = dias_uteis(ref, self.vencimento)
        vf_real = self.nocional * (vna_ref / self.vna_inicio) * (1.0 + self.cupom_real / 100.0) ** (self.du_total / DU_ANO)
        return vf_real * curva_real.fator_desconto(n)

    def _mtm(self, ref, curva_pre, curva_real, vna_ref, fator_cdi_acumulado):
        ipca = self.vp_ipca(ref, curva_real, vna_ref)
        di = self.vp_di(ref, curva_pre, fator_cdi_acumulado)
        return (ipca, di) if self.ativo == "ipca" else (di, ipca)

    def avaliar(self, ref: date, curva_pre: Curva, curva_real: Curva, vna_ref: float, fator_cdi_acumulado: float = 1.0) -> ResultadoSwap:
        a, p = self._mtm(ref, curva_pre, curva_real, vna_ref, fator_cdi_acumulado)
        a2, p2 = self._mtm(ref, curva_pre.deslocada(1.0), curva_real, vna_ref, fator_cdi_acumulado)
        return ResultadoSwap(a, p, a - p, (a2 - p2) - (a - p), dias_uteis(ref, self.vencimento))

    def dv01_real(self, ref: date, curva_pre: Curva, curva_real: Curva, vna_ref: float, fator_cdi_acumulado: float = 1.0) -> float:
        """Variação do MtM para +1 bp na curva de juro real."""
        a, p = self._mtm(ref, curva_pre, curva_real, vna_ref, fator_cdi_acumulado)
        a2, p2 = self._mtm(ref, curva_pre, curva_real.deslocada(1.0), vna_ref, fator_cdi_acumulado)
        return (a2 - p2) - (a - p)

    @staticmethod
    def cupom_par(curva_pre: Curva, curva_real: Curva, inicio: date, vencimento: date) -> float:
        """Cupom real que zera o swap DI x IPCA no início, com 100% do CDI (% a.a.).

        No início, VP DI = N e VP IPCA = N x (1+c)^(n/252) x DF_real(n); logo c é a própria
        taxa zero real do vencimento.
        """
        return curva_real.taxa(dias_uteis(inicio, vencimento))
