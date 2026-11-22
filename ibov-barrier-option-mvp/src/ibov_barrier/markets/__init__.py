"""Camada multi-mercado: um adaptador por índice, núcleo comum a partir da superfície.

Ver ARQUITETURA_MULTIATIVOS.md na raiz do projeto.
"""
from .base import (CALL, CHAIN_COLUMNS, PUT, CleaningReport, ForwardPoint, MarketAdapter, MarketSnapshot,
                   MarketSpec, TermStructure, validate_chain)
from .cleaning import SPX_CLEANING, CleaningConfig
from .spx import SPX_SPEC, SPXAdapter
from .ibov import IBOV_SPEC, IBOVAdapter, IbovFilters
from .vol import VolSurface

MARKETS = {spec.code: spec for spec in (IBOV_SPEC, SPX_SPEC)}

__all__ = [
    "CALL", "PUT", "CHAIN_COLUMNS", "CleaningConfig", "CleaningReport", "ForwardPoint", "IBOV_SPEC",
    "IBOVAdapter", "IbovFilters",
    "MARKETS", "MarketAdapter", "MarketSnapshot", "MarketSpec", "SPX_CLEANING", "SPX_SPEC", "SPXAdapter",
    "TermStructure", "VolSurface", "validate_chain",
]
