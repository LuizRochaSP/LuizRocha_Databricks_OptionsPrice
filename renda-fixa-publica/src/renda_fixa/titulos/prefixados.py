"""Títulos prefixados: LTN (Tesouro Prefixado) e NTN-F (Tesouro Prefixado com Juros Semestrais)."""
from __future__ import annotations

from datetime import date

from ..arredondamento import truncar
from ..calendario import DU_ANO
from .base import TituloComCupomSemestral, TituloPublico

# Cupom semestral da NTN-F: 1000 x ((1,10)^(1/2) - 1) = R$ 48,80885 (5 casas, Tesouro).
CUPOM_NTNF_RS = 48.80885


class LTN(TituloPublico):
    """Prefixado sem cupom, valor de face R$ 1.000."""

    nome = "LTN"
    indexador = "PRE"

    def pu(self, taxa: float, ref: date, vna: float | None = None) -> float:
        fl = self.fluxos(ref)
        if not fl:
            return 0.0
        return truncar(1000.0 / (1.0 + taxa / 100.0) ** (fl[-1].du / DU_ANO), 6)


class NTNF(TituloComCupomSemestral):
    """Prefixado com cupom de 10% a.a. pago em 1/jan e 1/jul."""

    nome = "NTN-F"
    indexador = "PRE"

    def cupom(self) -> float:
        return CUPOM_NTNF_RS / 10.0  # em % de R$ 1.000

    def pu(self, taxa: float, ref: date, vna: float | None = None) -> float:
        t = 1.0 + taxa / 100.0
        fl = self.fluxos(ref)
        total = sum(CUPOM_NTNF_RS / t ** (f.du / DU_ANO) for f in fl)
        if fl:
            total += 1000.0 / t ** (fl[-1].du / DU_ANO)
        return truncar(total, 6)
