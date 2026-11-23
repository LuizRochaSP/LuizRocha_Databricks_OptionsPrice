"""Títulos com VNA: NTN-B (Tesouro IPCA+ com Juros Semestrais), NTN-B Principal (Tesouro IPCA+),
NTN-C (Tesouro IGPM+ com Juros Semestrais) e LFT (Tesouro Selic)."""
from __future__ import annotations

from datetime import date

from .base import TituloComCupomSemestral, TituloPublico, pu_por_cotacao

# Cupom semestral da NTN-B: (1,06)^(1/2) - 1 = 2,956301% (6 casas em %).
CUPOM_NTNB_PCT = 2.956301
# NTN-C: cupom de 6% a.a. (2,956301% no semestre), exceto a de 01/01/2031, que paga 12% a.a.:
# (1,12)^(1/2) - 1 = 5,830052%. Conferido no arquivo da ANBIMA e no Tesouro Direto.
CUPOM_NTNC_6_PCT = 2.956301
CUPOM_NTNC_12_PCT = 5.830052
NTNC_CUPOM_12 = {date(2031, 1, 1)}


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


class NTNC(TituloComCupomSemestral):
    """IGP-M + cupom semestral; cupons em 1º/jan e 1º/jul. VNA-base R$ 1.000 em 01/07/2000, corrigido pelo IGP-M.

    Convenção igual à da NTN-B: cotação truncada na 4ª casa e PU = VNA x cotação truncado na 6ª.
    ``cupom_aa``: cupom anual em % (6 ou 12). Sem ele, usa 12% para a NTN-C 2031 e 6% para as demais.
    """

    nome = "NTN-C"
    indexador = "IGPM"
    usa_vna = True
    valor_base_fixo = None

    def __init__(self, vencimento: date, cupom_aa: float | None = None):
        super().__init__(vencimento)
        if cupom_aa is None:
            cupom_aa = 12.0 if vencimento in NTNC_CUPOM_12 else 6.0
        if cupom_aa not in (6.0, 12.0):
            raise ValueError("A NTN-C paga 6% ou 12% a.a.")
        self.cupom_aa = cupom_aa

    def cupom(self) -> float:
        return CUPOM_NTNC_12_PCT if self.cupom_aa == 12.0 else CUPOM_NTNC_6_PCT

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
