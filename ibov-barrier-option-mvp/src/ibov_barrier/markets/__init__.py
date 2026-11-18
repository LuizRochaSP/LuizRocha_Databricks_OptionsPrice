"""Camada multi-mercado: um adaptador por índice, núcleo comum a partir da superfície.

Ver ARQUITETURA_MULTIATIVOS.md na raiz do projeto.
"""
from datetime import time

from ..market_date import SAO_PAULO
from .base import (CALL, CHAIN_COLUMNS, PUT, CleaningReport, ForwardPoint, MarketAdapter, MarketSnapshot,
                   MarketSpec, TermStructure, validate_chain)
from .cleaning import SPX_CLEANING, CleaningConfig
from .spx import SPX_SPEC, SPXAdapter
from .vol import VolSurface

# Metadados do IBOV (o adaptador IBOV, que reaproveita o notebook 01, entra na próxima etapa).
IBOV_SPEC = MarketSpec(
    code="IBOV", name="Ibovespa", currency="BRL", exchange="B3", timezone=SAO_PAULO,
    close_time=time(18, 0), exercise="european", multiplier=1.0, day_count="BUS/252",
    rate_index="DI1", roots=("IBOV",),
)

MARKETS = {spec.code: spec for spec in (IBOV_SPEC, SPX_SPEC)}

__all__ = [
    "CALL", "PUT", "CHAIN_COLUMNS", "CleaningConfig", "CleaningReport", "ForwardPoint", "IBOV_SPEC",
    "MARKETS", "MarketAdapter", "MarketSnapshot", "MarketSpec", "SPX_CLEANING", "SPX_SPEC", "SPXAdapter",
    "TermStructure", "VolSurface", "validate_chain",
]
