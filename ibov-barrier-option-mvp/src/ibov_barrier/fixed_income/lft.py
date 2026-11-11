"""Módulo de precificação de LFT (Letra Financeira do Tesouro).

A LFT é um título público pós-fixado que rende a taxa Selic (ou Selic + spread).
O preço unitário (PU) é calculado capitalizando o VNA (Valor Nominal Atualizado) 
pela taxa Selic projetada para o período.
"""
from datetime import date
from ..di_curve.conventions import business_days
import numpy as np

def lft_price(
    vna: float,
    taxa_selic_anual: float,
    spread_anual: float,
    data_referencia: date,
    data_vencimento: date,
) -> float:
    """
    Calcula o Preço Unitário (PU) de uma LFT.
    
    Args:
        vna: Valor Nominal Atualizado na data de referência.
        taxa_selic_anual: Projeção da taxa Selic anual (% a.a.).
        spread_anual: Spread do título sobre a Selic (% a.a.). Geralmente 0.0 ou 0.1.
        data_referencia: Data de referência do cálculo.
        data_vencimento: Data de vencimento do título.
    
    Returns:
        Preço unitário (PU) do título.
    """
    du_total = business_days(data_referencia, data_vencimento)
    anos_total = du_total / 252.0
    
    # Taxa efetiva do título = (1 + Selic) * (1 + Spread) - 1
    taxa_efetiva = ((1 + (taxa_selic_anual / 100.0)) * (1 + (spread_anual / 100.0))) - 1
    
    # O PU é o VNA capitalizado pela taxa efetiva
    pu = vna * ((1 + taxa_efetiva) ** anos_total)
    return pu