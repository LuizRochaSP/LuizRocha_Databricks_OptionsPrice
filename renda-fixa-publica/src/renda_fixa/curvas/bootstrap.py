"""Bootstrap de curvas zero a partir de títulos públicos e análises derivadas.

- ``bootstrap_titulos``: ordena os títulos por vencimento e, um a um, acha a taxa zero do
  vencimento que faz o preço pela curva (flat-forward) igualar o PU de mercado.
  LTN + NTN-F -> curva prefixada; NTN-B (+ Principal) -> curva de juro real (IPCA).
- ``inflacao_implicita``: breakeven (1 + pré) / (1 + real) - 1 por prazo.
- ``z_spread``: spread constante (bps) sobre uma curva que reprecifica o título.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from .curva import Curva, CurvaZero


def _resolver(f, lo: float, hi: float, tol: float = 1e-12, iteracoes: int = 200) -> float:
    """Raiz de f em [lo, hi]: Brent (scipy) e, sem scipy, bisseção."""
    try:
        from scipy.optimize import brentq

        return float(brentq(f, lo, hi, xtol=tol, maxiter=iteracoes))
    except ImportError:
        pass
    flo = f(lo)
    for _ in range(iteracoes):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def bootstrap_titulos(
    titulos: Sequence,
    pus: Sequence[float],
    ref: date,
    vnas: Sequence[float | None] | None = None,
    nome: str = "",
) -> CurvaZero:
    """Curva zero flat-forward cujos vértices são os vencimentos dos títulos.

    Títulos sem cupom fixam o vértice diretamente; com cupom, os cupons antes do vencimento
    são descontados pela curva já construída (interpolada ou extrapolada de forma constante).
    """
    vnas = list(vnas) if vnas is not None else [None] * len(titulos)
    # No mesmo vencimento, o título sem cupom (menos fluxos) define o vértice; os demais
    # ficam fora do bootstrap e aparecem como resíduo (ex.: LTN e NTN-F jan/29).
    itens = sorted(zip(titulos, pus, vnas), key=lambda x: (x[0].vencimento, len(x[0].fluxos(ref))))
    curva: CurvaZero | None = None
    for titulo, pu, vna in itens:
        fl = titulo.fluxos(ref)
        if not fl:
            continue
        du_venc = fl[-1].du
        if curva is not None and du_venc in curva.dus:
            continue  # mesmo vencimento já usado (ex.: NTN-B e NTN-B Principal)

        def erro(taxa_pct, titulo=titulo, pu=pu, vna=vna, du_venc=du_venc):
            c = CurvaZero([du_venc], [taxa_pct], nome=nome) if curva is None else curva.com_vertice(du_venc, taxa_pct)
            return titulo.pu_curva(c, ref, vna) - pu

        taxa = _resolver(erro, -20.0, 200.0)
        curva = CurvaZero([du_venc], [taxa], nome=nome) if curva is None else curva.com_vertice(du_venc, taxa)
    if curva is None:
        raise ValueError("Nenhum título com fluxo após a data de referência.")
    return curva


def inflacao_implicita(curva_pre: Curva, curva_real: Curva, dus: Sequence[float]) -> np.ndarray:
    """Inflação implícita (% a.a.) por prazo: (1 + pré) / (1 + real) - 1."""
    pre = curva_pre.taxas(dus) / 100.0
    real = curva_real.taxas(dus) / 100.0
    return ((1 + pre) / (1 + real) - 1) * 100.0


def z_spread(titulo, pu: float, curva: Curva, ref: date, vna: float | None = None) -> float:
    """Spread constante (bps) somado à curva zero que reprecifica o título."""
    return _resolver(lambda s: titulo.pu_curva(curva.deslocada(s), ref, vna) - pu, -5000.0, 5000.0, tol=1e-8)


@dataclass(frozen=True)
class ErroAjuste:
    titulo: str
    vencimento: date
    pu_mercado: float
    pu_curva: float
    erro_pu: float
    erro_bps: float


def erros_de_ajuste(curva: Curva, titulos: Sequence, pus: Sequence[float], ref: date, vnas=None) -> list[ErroAjuste]:
    """Erro de cada título ao ser precificado pela curva (em R$ e em bps de taxa)."""
    vnas = list(vnas) if vnas is not None else [None] * len(titulos)
    out = []
    for t, pu, v in zip(titulos, pus, vnas):
        pc = t.pu_curva(curva, ref, v)
        dv01 = t.analiticos(t.taxa(pu, ref, v, casas=None), ref, v).dv01
        out.append(ErroAjuste(t.nome, t.vencimento, pu, pc, pc - pu, (pc - pu) / dv01 if dv01 else float("nan")))
    return out
