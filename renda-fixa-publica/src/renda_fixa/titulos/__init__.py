"""Títulos públicos federais."""
from __future__ import annotations

from datetime import date

from .base import Analiticos, Fluxo, TituloPublico
from .indexados import LFT, NTNB, NTNBPrincipal
from .prefixados import LTN, NTNF

_CLASSES = {
    "LTN": LTN,
    "NTN-F": NTNF,
    "NTN-B": NTNB,
    "NTN-B PRINCIPAL": NTNBPrincipal,
    "NTN-B P": NTNBPrincipal,
    "LFT": LFT,
}


def criar_titulo(nome: str, vencimento: date) -> TituloPublico:
    """Cria o título pelo nome usado pela ANBIMA (LTN, NTN-F, NTN-B, LFT, NTN-B Principal)."""
    chave = nome.strip().upper()
    if chave not in _CLASSES:
        raise ValueError(f"Título não suportado: {nome!r}. Suportados: {sorted(set(_CLASSES))}")
    return _CLASSES[chave](vencimento)


__all__ = [
    "Analiticos",
    "Fluxo",
    "TituloPublico",
    "LTN",
    "NTNF",
    "NTNB",
    "NTNBPrincipal",
    "LFT",
    "criar_titulo",
]
