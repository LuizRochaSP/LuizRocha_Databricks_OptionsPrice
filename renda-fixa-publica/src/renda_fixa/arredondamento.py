"""Truncamento e arredondamento decimais, como nas metodologias do Tesouro e da ANBIMA.

Os cálculos usam float, mas o corte final passa por ``Decimal`` a partir da
representação mais curta do float (``repr``). Assim 955.038091 não vira 955.038090
por causa de 955.0380909999... na representação binária.
"""
from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal


def _q(casas: int) -> Decimal:
    return Decimal(1).scaleb(-casas)


def truncar(x: float, casas: int) -> float:
    """Trunca x em ``casas`` decimais (em direção a zero)."""
    return float(Decimal(repr(float(x))).quantize(_q(casas), rounding=ROUND_DOWN))


def arredondar(x: float, casas: int) -> float:
    """Arredonda x em ``casas`` decimais (meio para cima, longe de zero)."""
    return float(Decimal(repr(float(x))).quantize(_q(casas), rounding=ROUND_HALF_UP))
