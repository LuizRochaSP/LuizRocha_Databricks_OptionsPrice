"""Módulo de precificação de NTN-B (Nota do Tesouro Nacional, série B).

A NTN-B é um título público indexado à inflação (IPCA) com pagamento de cupons semestrais.
O preço unitário (PU) é calculado descontando-se os fluxos de caixa pela taxa indicativa
da ANBIMA para o prazo correspondente (ETTJ IPCA).
"""
from datetime import date
from typing import List, Tuple

import numpy as np

# Importa a função de dias úteis já existente no projeto para manter consistência
from ..di_curve.conventions import business_days


def gerar_fluxos_caixa(
    vna: float,
    cupom_anual: float,
    data_referencia: date,
    data_vencimento: date,
) -> List[Tuple[float, float]]:
    """Gera os fluxos de caixa da NTN-B (cupons semestrais + principal no vencimento).
    Retorna uma lista de tuplas (tempo_em_anos, valor_do_fluxo).
    """
    fluxos = []
    du_total = business_days(data_referencia, data_vencimento)
    anos_total = du_total / 252.0
    num_cupons = int(np.ceil(anos_total * 2))

    for i in range(1, num_cupons + 1):
        # Aproximação: cada cupom ocorre a cada 126 dias úteis (meio ano)
        du_cupom = min(int(i * 126), du_total)
        anos_cupom = du_cupom / 252.0
        # Valor do cupom (metade da taxa anual) sobre o VNA
        valor_cupom = vna * (cupom_anual / 100.0) / 2.0

        if i == num_cupons:
            # Último fluxo: cupom + principal
            fluxos.append((anos_cupom, valor_cupom + vna))
        else:
            fluxos.append((anos_cupom, valor_cupom))

    return fluxos


def ntnb_price(
    vna: float,
    taxa_anbima: float,
    cupom_anual: float,
    data_referencia: date,
    data_vencimento: date,
) -> float:
    """
    Calcula o Preço Unitário (PU) de uma NTN-B.

    Args:
        vna: Valor Nominal Atualizado (IPCA acumulado).
        taxa_anbima: Taxa indicativa da ANBIMA (ETTJ IPCA) em % a.a.
        cupom_anual: Taxa de juros do cupom em % a.a. (geralmente 6.0%).
        data_referencia: Data de referência do cálculo.
        data_vencimento: Data de vencimento do título.

    Returns:
        Preço unitário (PU) do título.
    """
    taxa_dec = taxa_anbima / 100.0
    fluxos = gerar_fluxos_caixa(vna, cupom_anual, data_referencia, data_vencimento)

    pu = 0.0
    for t_anos, valor in fluxos:
        vp = valor / ((1 + taxa_dec) ** t_anos)
        pu += vp

    return pu


def ntnb_duration(
    vna: float,
    taxa_anbima: float,
    cupom_anual: float,
    data_referencia: date,
    data_vencimento: date,
) -> float:
    """Calcula a Duration de Macaulay (em anos) da NTN-B."""
    taxa_dec = taxa_anbima / 100.0
    fluxos = gerar_fluxos_caixa(vna, cupom_anual, data_referencia, data_vencimento)
    pu = ntnb_price(vna, taxa_anbima, cupom_anual, data_referencia, data_vencimento)

    duration = 0.0
    for t_anos, valor in fluxos:
        vp = valor / ((1 + taxa_dec) ** t_anos)
        duration += t_anos * vp

    return duration / pu if pu > 0 else 0.0
