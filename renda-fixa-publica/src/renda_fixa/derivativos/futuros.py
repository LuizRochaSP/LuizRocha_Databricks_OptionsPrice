"""Futuros de juros da B3: DI1 (DI de um dia) e DAP (cupom de IPCA).

DI1:
- vencimento no primeiro dia útil do mês do contrato (códigos F, G, H, J, K, M, N, Q, U, V, X, Z);
- PU = 100.000 / (1 + taxa)^(DU/252), em pontos (1 ponto = R$ 1);
- comprar taxa = vender PU; o DV01 é em R$ por contrato;
- ajuste diário: (PA_t - PA_{t-1} x (1 + CDI_{t-1})) x quantidade, com o sinal da posição em PU.

DAP:
- vencimento no dia 15 do mês do contrato (próximo dia útil);
- PU = 100.000 / (1 + taxa_real)^(DU/252) pontos; valor do ponto = R$ 0,00025 x índice do IPCA
  (número-índice do IBGE), por isso o valor financeiro acompanha a inflação.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..arredondamento import arredondar
from ..calendario import DU_ANO, dias_uteis, proximo_dia_util
from ..curvas.curva import Curva

PU_FACE = 100_000.0
CODIGOS_MES = "FGHJKMNQUVXZ"
VALOR_PONTO_DAP = 0.00025


def vencimento_di1(codigo: str) -> date:
    """'F27' ou 'DI1F27' -> primeiro dia útil de janeiro de 2027."""
    c = codigo.strip().upper().replace("DI1", "")
    mes = CODIGOS_MES.index(c[0]) + 1
    ano = 2000 + int(c[1:3])
    return proximo_dia_util(date(ano, mes, 1))


def vencimento_dap(codigo: str) -> date:
    """'K35' ou 'DAPK35' -> dia 15 de maio de 2035 (ou próximo dia útil)."""
    c = codigo.strip().upper().replace("DAP", "")
    mes = CODIGOS_MES.index(c[0]) + 1
    ano = 2000 + int(c[1:3])
    return proximo_dia_util(date(ano, mes, 15))


def codigo_contrato(vencimento: date) -> str:
    return f"{CODIGOS_MES[vencimento.month - 1]}{vencimento.year % 100:02d}"


class _FuturoTaxa:
    prefixo = ""

    def __init__(self, vencimento: date):
        self.vencimento = vencimento

    def __repr__(self) -> str:
        return f"{self.prefixo}{codigo_contrato(self.vencimento)}"

    def du(self, ref: date) -> int:
        return dias_uteis(ref, self.vencimento)

    def pu(self, taxa: float, ref: date) -> float:
        """PU em pontos, arredondado na 2ª casa (como o ajuste da B3)."""
        return arredondar(self.pu_bruto(taxa, ref), 2)

    def pu_bruto(self, taxa: float, ref: date) -> float:
        return PU_FACE / (1.0 + taxa / 100.0) ** (self.du(ref) / DU_ANO)

    def taxa(self, pu: float, ref: date) -> float:
        n = self.du(ref)
        return ((PU_FACE / pu) ** (DU_ANO / n) - 1.0) * 100.0

    def pu_curva(self, curva: Curva, ref: date) -> float:
        return PU_FACE * curva.fator_desconto(self.du(ref))

    def dv01_pontos(self, taxa: float, ref: date) -> float:
        """Variação do PU (pontos) para +1 bp, positiva (o PU cai quando a taxa sobe)."""
        return self.pu_bruto(taxa, ref) - self.pu_bruto(taxa + 0.01, ref)


class DI1(_FuturoTaxa):
    prefixo = "DI1"

    @classmethod
    def de_codigo(cls, codigo: str) -> "DI1":
        return cls(vencimento_di1(codigo))

    def dv01(self, taxa: float, ref: date) -> float:
        """R$ por contrato para +1 bp (1 ponto = R$ 1)."""
        return self.dv01_pontos(taxa, ref)


class DAP(_FuturoTaxa):
    prefixo = "DAP"

    @classmethod
    def de_codigo(cls, codigo: str) -> "DAP":
        return cls(vencimento_dap(codigo))

    @staticmethod
    def valor_ponto(indice_ipca: float) -> float:
        return VALOR_PONTO_DAP * indice_ipca

    def dv01(self, taxa: float, ref: date, indice_ipca: float) -> float:
        """R$ por contrato para +1 bp na taxa real."""
        return self.dv01_pontos(taxa, ref) * self.valor_ponto(indice_ipca)


@dataclass(frozen=True)
class AjusteDiario:
    pa_anterior_corrigido: float
    ajuste: float  # R$ (positivo = recebe)


def ajuste_diario_di1(
    pa_hoje: float,
    pa_anterior: float,
    cdi_anterior_pct_aa: float,
    quantidade: int,
    posicao: str = "comprado_taxa",
) -> AjusteDiario:
    """Ajuste diário do DI1 entre dois pregões consecutivos.

    ``cdi_anterior_pct_aa``: CDI do dia anterior em % a.a. (base 252).
    ``posicao``: 'comprado_taxa' (vendido em PU) ou 'vendido_taxa' (comprado em PU).
    """
    fator = (1.0 + cdi_anterior_pct_aa / 100.0) ** (1.0 / DU_ANO)
    pa_corr = arredondar(pa_anterior * fator, 2)
    sinal_pu = -1.0 if posicao == "comprado_taxa" else 1.0
    return AjusteDiario(pa_corr, arredondar((pa_hoje - pa_corr) * quantidade * sinal_pu, 2))


def taxa_fra(taxa_curta: float, du_curto: int, taxa_longa: float, du_longo: int) -> float:
    """Taxa a termo (% a.a.) entre dois vértices: o FRA de DI implícito em dois DI1."""
    if du_longo <= du_curto:
        raise ValueError("du_longo deve ser maior que du_curto.")
    f_l = (1 + taxa_longa / 100.0) ** (du_longo / DU_ANO)
    f_c = (1 + taxa_curta / 100.0) ** (du_curto / DU_ANO)
    return ((f_l / f_c) ** (DU_ANO / (du_longo - du_curto)) - 1.0) * 100.0


def curva_di1(contratos: list[tuple[str | date, float]], ref: date, interpolacao: str = "flat_forward"):
    """Curva pré a partir de taxas de ajuste de DI1: lista de (código ou vencimento, taxa % a.a.)."""
    from ..curvas.curva import CurvaZero

    dus, taxas = [], []
    for c, tx in contratos:
        venc = vencimento_di1(c) if isinstance(c, str) else c
        n = dias_uteis(ref, venc)
        if n > 0:
            dus.append(n)
            taxas.append(tx)
    return CurvaZero(dus, taxas, interpolacao, nome="DI1")
