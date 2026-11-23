"""Títulos com VNA: NTN-B (Tesouro IPCA+ com Juros Semestrais), NTN-B Principal (Tesouro IPCA+) e LFT (Tesouro Selic)."""
from __future__ import annotations

from datetime import date

from .base import TituloComCupomSemestral, TituloPublico, pu_por_cotacao

# Cupom semestral da NTN-B: (1,06)^(1/2) - 1 = 2,956301% (6 casas em %).
CUPOM_NTNB_PCT = 2.956301


class NTNB(TituloComCupomSemestral):
    """IPCA + cupom de 6% a.a. semestral; cupons no mês do vencimento e 6 meses depois, dia 15."""

    nome = "NTN-B"
    indexador = "IPCA"
    usa_vna = True
    valor_base_fixo = None

    def cupom(self) -> float:
        return CUPOM_NTNB_PCT

    def pu(self, taxa: float, ref: date, vna: float | None = None) -> float:
        return pu_por_cotacao(self._base(vna), self.cotacao_bruta(taxa, ref))


class NTNBPrincipal(TituloPublico):
    """IPCA sem cupom (Tesouro IPCA+)."""

    nome = "NTN-B Principal"
    indexador = "IPCA"
    usa_vna = True
    valor_base_fixo = None

    def pu(self, taxa: float, ref: date, vna: float | None = None) -> float:
        return pu_por_cotacao(self._base(vna), self.cotacao_bruta(taxa, ref))


class LFT(TituloPublico):
    """Pós-fixado na Selic; a taxa negociada é o ágio (taxa > 0, PU < VNA) ou deságio sobre a Selic."""

    nome = "LFT"
    indexador = "SELIC"
    usa_vna = True
    valor_base_fixo = None

    def pu(self, taxa: float, ref: date, vna: float | None = None) -> float:
        return pu_por_cotacao(self._base(vna), self.cotacao_bruta(taxa, ref))
