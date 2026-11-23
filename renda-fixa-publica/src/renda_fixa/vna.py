"""Valor Nominal Atualizado (VNA) da LFT e da NTN-B.

LFT (Tesouro Selic):
    base R$ 1.000 em 01/07/2000; a cada dia útil d, VNA(d) = VNA(d-1) x (1 + Selic(d-1)),
    com Selic(d-1) sendo a taxa diária do dia útil anterior (série SGS 11 do BCB, em % a.d.).

NTN-B (Tesouro IPCA+):
    base R$ 1.000 em 15/07/2000; no dia 15 de cada mês o VNA é corrigido pelo IPCA do mês
    anterior. Entre dois dias 15 usa-se o VNA projetado com a projeção de IPCA da ANBIMA:
    VNA_proj = VNA(15 anterior) x (1 + IPCA_proj)^(x / y),
    x = dias corridos do dia 15 anterior até a data, y = dias corridos entre os dois dias 15.

Na prática o VNA do dia vem do próprio arquivo da ANBIMA (``dados.anbima.vna_implicito``),
e estas funções servem para projetar, atualizar e conferir.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .arredondamento import truncar
from .calendario import dias_corridos, eh_dia_util, soma_meses


def atualizar_vna_lft(vna: float, selic_diaria_pct: float) -> float:
    """Avança o VNA da LFT um dia útil com a taxa Selic diária (% a.d.) do dia anterior.

    O Tesouro acumula o fator desde 2000 e só trunca o VNA no fim; partindo de um VNA já
    truncado, o resultado pode diferir do oficial em R$ 0,000001 (1 na 6ª casa).
    """
    return truncar(vna * (1.0 + selic_diaria_pct / 100.0), 6)


def vna_lft_serie(vna_inicial: float, data_inicial: date, selic_diaria: pd.Series, data_final: date) -> pd.Series:
    """VNA da LFT dia a dia, de ``data_inicial`` (VNA conhecido) até ``data_final``.

    ``selic_diaria``: série indexada por data (date) com a taxa diária em % a.d. (SGS 11).
    """
    s = selic_diaria.copy()
    s.index = pd.to_datetime(s.index).date
    s = s.sort_index()
    vna = vna_inicial
    out = {data_inicial: vna}
    for d, taxa in s.items():
        if d < data_inicial or d >= data_final or not eh_dia_util(d):
            continue
        vna = atualizar_vna_lft(vna, float(taxa))
        out[_proximo_util(d)] = vna
    return pd.Series(out).sort_index()


def _proximo_util(d: date) -> date:
    from .calendario import soma_dias_uteis

    return soma_dias_uteis(d, 1)


def aniversario_anterior(ref: date) -> date:
    """Último dia 15 em ou antes de ``ref``."""
    return date(ref.year, ref.month, 15) if ref.day >= 15 else soma_meses(date(ref.year, ref.month, 15), -1)


def vna_ntnb_projetado(vna_aniversario: float, ipca_projetado_pct: float, ref: date) -> float:
    """VNA projetado da NTN-B em ``ref`` a partir do VNA do último dia 15 e da projeção mensal de IPCA (%)."""
    a0 = aniversario_anterior(ref)
    a1 = soma_meses(a0, 1)
    expo = dias_corridos(a0, ref) / dias_corridos(a0, a1)
    return truncar(vna_aniversario * (1.0 + ipca_projetado_pct / 100.0) ** expo, 6)


def ipca_projetado_implicito(vna_projetado: float, vna_aniversario: float, ref: date) -> float:
    """Projeção mensal de IPCA (%) implícita num VNA projetado (inversa de ``vna_ntnb_projetado``)."""
    a0 = aniversario_anterior(ref)
    a1 = soma_meses(a0, 1)
    x, y = dias_corridos(a0, ref), dias_corridos(a0, a1)
    if x == 0:
        raise ValueError("Em dia 15 o VNA não depende da projeção.")
    return ((vna_projetado / vna_aniversario) ** (y / x) - 1.0) * 100.0


def atualizar_vna_ntnb(vna_aniversario: float, ipca_mes_pct: float) -> float:
    """VNA do próximo dia 15 a partir do VNA do dia 15 anterior e do IPCA fechado do mês (%)."""
    return truncar(vna_aniversario * (1.0 + ipca_mes_pct / 100.0), 6)
