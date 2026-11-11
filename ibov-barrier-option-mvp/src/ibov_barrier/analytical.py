"""Benchmark Analítico: Fórmula de Reiner-Rubinstein.

Permite validar o motor de Monte Carlo comparando com a solução analítica exata
para opções com barreira (monitoramento contínuo).
"""
import numpy as np
from scipy.stats import norm

def _bs_call(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float) -> float:
    """Função auxiliar para o preço de uma Call Europeia (Black-Scholes)."""
    if maturity <= 0:
        return max(spot - strike, 0.0)
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    return spot * np.exp(-dividend_yield * maturity) * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)

def price_down_and_out_call_analytical(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """"
    Preço de uma Call Down-and-Out (sem rebate) via Reiner-Rubinstein.
    Assume monitoramento contínuo da barreira e volatilidade constante.
    """
    if spot <= barrier:
        return 0.0  # Já bateu na barreira antes de começar

    # Parâmetro lambda da fórmula
    lam = (rate - dividend_yield + (volatility ** 2) / 2.0) / (volatility ** 2)
    
    # Termo 1: Preço da Call Vanilla
    vanilla = _bs_call(spot, strike, maturity, rate, dividend_yield, volatility)
    
    # Termo 2: Preço da Call "refletida" na barreira
    reflected_spot = (barrier ** 2) / spot
    reflected_call = _bs_call(reflected_spot, strike, maturity, rate, dividend_yield, volatility)
    
    # Fórmula de Reiner-Rubinstein
    price = vanilla - ((spot / barrier) ** (2 * lam)) * reflected_call
    return float(max(price, 0.0))
