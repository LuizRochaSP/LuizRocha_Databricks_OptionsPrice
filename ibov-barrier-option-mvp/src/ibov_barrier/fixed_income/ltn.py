"""Módulo de precificação de LTN (Letra do Tesouro Nacional).

A LTN é um título público prefixado, com pagamento único no vencimento (cupom zero).
O preço unitário (PU) é calculado descontando-se o valor nominal (R$ 1.000,00) 
pela taxa de juros prefixada (ETTJ Pré) no prazo correspondente.
"""
from datetime import date
from ..di_curve.conventions import business_days
from ..di_curve.interpolators import FlatForwardInterpolator
import numpy as np

def ltn_price(
    taxa_anbima: float,
    data_referencia: date,
    data_vencimento: date,
    valor_nominal: float = 1000.0,
) -> float:
    """
    Calcula o Preço Unitário (PU) de uma LTN.
    
    Args:
        taxa_anbima: Taxa prefixada da ANBIMA (ETTJ Pré) em % a.a.
        data_referencia: Data de referência do cálculo.
        data_vencimento: Data de vencimento do título.
        valor_nominal: Valor de face da LTN (padrão R$ 1.000,00).
    
    Returns:
        Preço unitário (PU) do título.
    """
    taxa_dec = taxa_anbima / 100.0
    du_total = business_days(data_referencia, data_vencimento)
    anos_total = du_total / 252.0
    
    # Fórmula de cupom zero (desconto composto)
    pu = valor_nominal / ((1 + taxa_dec) ** anos_total)
    return pu

def ltn_duration(
    taxa_anbima: float,
    data_referencia: date,
    data_vencimento: date,
) -> float:
    """A Duration de Macaulay para um título de cupom zero é exatamente o seu prazo (em anos)."""
    du_total = business_days(data_referencia, data_vencimento)
    return du_total / 252.0