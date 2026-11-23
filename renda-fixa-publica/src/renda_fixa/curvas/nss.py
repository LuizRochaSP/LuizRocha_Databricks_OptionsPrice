"""Nelson-Siegel-Svensson, a forma paramétrica usada pela ANBIMA nas ETTJs.

y(t) = b1 + b2 * h1 + b3 * (h1 - e^(-l1 t)) + b4 * (h2 - e^(-l2 t)),
h_i = (1 - e^(-l_i t)) / (l_i t), t em anos (DU/252), y em % a.a. (taxa efetiva).

O ajuste pode ser feito em taxas zero (``ajustar_nss_taxas``) ou diretamente nos
preços dos títulos (``ajustar_nss_precos``), que é o que a ANBIMA faz.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from ..calendario import DU_ANO
from .curva import Curva


@dataclass(frozen=True)
class ParametrosNSS:
    b1: float
    b2: float
    b3: float
    b4: float
    l1: float
    l2: float

    def como_array(self) -> np.ndarray:
        return np.array([self.b1, self.b2, self.b3, self.b4, self.l1, self.l2])


def _nss(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    b1, b2, b3, b4, l1, l2 = p
    t = np.maximum(np.asarray(t, dtype=float), 1e-8)
    e1, e2 = np.exp(-l1 * t), np.exp(-l2 * t)
    h1 = (1 - e1) / (l1 * t)
    h2 = (1 - e2) / (l2 * t)
    return b1 + b2 * h1 + b3 * (h1 - e1) + b4 * (h2 - e2)


class CurvaNSS(Curva):
    def __init__(self, parametros: ParametrosNSS, nome: str = ""):
        self.parametros = parametros
        self.nome = nome
        self._p = parametros.como_array()

    def taxa(self, du: float) -> float:
        return float(_nss(np.array([du / DU_ANO]), self._p)[0])

    def __repr__(self) -> str:
        p = self.parametros
        return f"CurvaNSS({self.nome or '-'}, b=({p.b1:.4f}, {p.b2:.4f}, {p.b3:.4f}, {p.b4:.4f}), l=({p.l1:.4f}, {p.l2:.4f}))"


def _chutes(taxas: np.ndarray) -> list[np.ndarray]:
    longo, curto = float(taxas[-1]), float(taxas[0])
    out = []
    for l1 in (0.3, 0.8, 1.5, 3.0):
        for l2 in (0.05, 0.15, 0.4):
            out.append(np.array([longo, curto - longo, 0.0, 0.0, l1, l2]))
    return out


_LIM_INF = [-50, -100, -200, -200, 0.01, 0.01]
_LIM_SUP = [100, 100, 200, 200, 10.0, 10.0]


def ajustar_nss_taxas(dus: Sequence[float], taxas: Sequence[float], pesos: Sequence[float] | None = None, nome: str = "") -> CurvaNSS:
    """Ajuste por mínimos quadrados das taxas zero (% a.a.)."""
    from scipy.optimize import least_squares

    t = np.asarray(dus, dtype=float) / DU_ANO
    y = np.asarray(taxas, dtype=float)
    w = np.sqrt(np.asarray(pesos, dtype=float)) if pesos is not None else np.ones_like(y)
    melhor = None
    for x0 in _chutes(y):
        r = least_squares(lambda p: (_nss(t, p) - y) * w, x0, bounds=(_LIM_INF, _LIM_SUP))
        if melhor is None or r.cost < melhor.cost:
            melhor = r
    return CurvaNSS(ParametrosNSS(*melhor.x), nome)


def ajustar_nss_precos(
    titulos: Sequence,
    pus: Sequence[float],
    ref: date,
    vnas: Sequence[float | None] | None = None,
    nome: str = "",
) -> CurvaNSS:
    """Ajusta NSS minimizando o erro de preço relativo, ponderado por 1/duration (padrão ANBIMA)."""
    from scipy.optimize import least_squares

    vnas = list(vnas) if vnas is not None else [None] * len(titulos)
    pus = np.asarray(pus, dtype=float)
    # chute pelas taxas de cada título
    taxas = np.array([t.taxa(p, ref, v, casas=None) for t, p, v in zip(titulos, pus, vnas)])
    durs = np.array([t.analiticos(tx, ref, v).duration_macaulay for t, tx, v in zip(titulos, taxas, vnas)])
    fluxos = [t.fluxos(ref) for t in titulos]
    bases = np.array([t._base(v) for t, v in zip(titulos, vnas)])

    def residuos(p):
        out = np.empty(len(titulos))
        for i, fl in enumerate(fluxos):
            tt = np.array([f.du for f in fl]) / DU_ANO
            vals = np.array([f.valor for f in fl])
            y = _nss(tt, p) / 100.0
            preco = bases[i] * np.sum(vals * (1 + y) ** (-tt)) / 100.0
            out[i] = (preco - pus[i]) / pus[i] / max(durs[i], 1e-3)
        return out

    ordem = np.argsort(durs)
    melhor = None
    for x0 in _chutes(taxas[ordem]):
        r = least_squares(residuos, x0, bounds=(_LIM_INF, _LIM_SUP))
        if melhor is None or r.cost < melhor.cost:
            melhor = r
    return CurvaNSS(ParametrosNSS(*melhor.x), nome)
