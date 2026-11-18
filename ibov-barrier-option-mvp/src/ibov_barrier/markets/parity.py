"""Forward, fator de desconto, spot e dividend yield implícitos pela paridade put-call.

Para opções europeias de mesmo strike e vencimento:  C - P = DF·(F - K).
Com vários strikes, C - P é uma reta em K: intercepto DF·F e inclinação -DF. Usamos os strikes
mais próximos do ATM (menor |C - P|), onde as cotações são mais líquidas e o erro relativo é menor.

Se uma curva de juros externa (SOFR/Treasuries) fornecer DF, só F é estimado (média de K + (C-P)/DF),
o que é mais estável em vencimentos curtos, onde -ln(DF)/T amplifica o ruído de cotação.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .base import CALL, PUT, ForwardPoint, TermStructure


def _pairs(expiry_quotes: pd.DataFrame) -> pd.DataFrame:
    """Strikes com call e put cotadas (bid > 0 e ask >= bid) no mesmo vencimento/liquidação."""
    q = expiry_quotes.loc[(expiry_quotes["bid"] > 0) & (expiry_quotes["ask"] >= expiry_quotes["bid"])]
    mid = q.assign(mid=(q["bid"] + q["ask"]) / 2.0)
    calls = mid.loc[mid["option_type"].eq(CALL)].groupby("strike")["mid"].mean()
    puts = mid.loc[mid["option_type"].eq(PUT)].groupby("strike")["mid"].mean()
    pairs = pd.concat({"call": calls, "put": puts}, axis=1, join="inner").reset_index()
    pairs["c_minus_p"] = pairs["call"] - pairs["put"]
    return pairs


def implied_forward(expiry_quotes: pd.DataFrame, T: float, expiration, *, max_pairs: int = 10,
                    discount_factor: float | None = None) -> ForwardPoint:
    """Forward (e DF, se não informado) de um único vencimento."""
    if T <= 0:
        raise ValueError("Prazo deve ser positivo.")
    pairs = _pairs(expiry_quotes)
    pairs = pairs.reindex(pairs["c_minus_p"].abs().sort_values().index).head(max_pairs)
    K, y = pairs["strike"].to_numpy(float), pairs["c_minus_p"].to_numpy(float)

    if discount_factor is not None:
        if len(K) < 1:
            raise ValueError(f"Vencimento {expiration}: nenhum par call/put cotado.")
        df = float(discount_factor)
        estimates = K + y / df
        forward = float(np.mean(estimates))
        resid = float(np.std(estimates, ddof=1)) if len(K) > 1 else 0.0
    else:
        if len(K) < 3 or np.ptp(K) == 0:
            raise ValueError(f"Vencimento {expiration}: menos de 3 strikes com call e put cotadas.")
        slope, intercept = np.polyfit(K, y, 1)
        df = float(-slope)
        if not (0.5 < df < 1.1):
            raise ValueError(f"Vencimento {expiration}: fator de desconto implícito absurdo ({df:.6f}).")
        forward = float(intercept / df)
        fitted = intercept + slope * K
        resid = float(np.std(y - fitted, ddof=2)) if len(K) > 2 else 0.0
    if not forward > 0:
        raise ValueError(f"Vencimento {expiration}: forward implícito não positivo ({forward}).")
    return ForwardPoint(expiration, float(T), forward, df, int(len(K)), resid)


def implied_forwards(quotes: pd.DataFrame, *, max_pairs: int = 10, min_T: float = 0.0,
                     discount_curve: Callable[[float], float] | None = None,
                     min_T_rate: float = 0.25,
                     report_warnings: list | None = None) -> list[ForwardPoint]:
    """Um ForwardPoint por (vencimento, liquidação). `quotes` precisa da coluna T (anos).

    Vencimentos sem pares suficientes são descartados (com aviso), não derrubam o pipeline.
    `discount_curve(T) -> r contínuo`, se informado, fixa DF = exp(-r·T) em todos os prazos.

    Sem curva externa, a taxa vem da regressão só para T >= `min_T_rate` (~3 meses): em prazos curtos,
    r = -ln(DF)/T amplifica o ruído de cotação (nos dados reais do SPX de 17/11/2026 saíram taxas entre
    -21% e +26% em vencimentos de 1-3 semanas). Nesses prazos usa-se a taxa do vencimento confiável mais
    curto e estima-se só o forward.
    """
    groups = [((e, T), g) for (e, T), g in quotes.groupby(["expiration", "T"], sort=True) if T > min_T]

    def estimate(expiration, T, group, df):
        try:
            return implied_forward(group, T, expiration, max_pairs=max_pairs, discount_factor=df)
        except ValueError as exc:
            if report_warnings is not None:
                report_warnings.append(str(exc))
            return None

    if discount_curve is not None:
        points = [estimate(e, T, g, float(np.exp(-discount_curve(T) * T))) for (e, T), g in groups]
    else:
        long_pts = {(e, T): estimate(e, T, g, None) for (e, T), g in groups if T >= min_T_rate}
        reliable = sorted((p for p in long_pts.values() if p is not None), key=lambda p: p.T)
        points = []
        for (e, T), g in groups:
            if T >= min_T_rate:
                points.append(long_pts[(e, T)])
            elif reliable:
                points.append(estimate(e, T, g, float(np.exp(-reliable[0].rate * T))))
            else:
                points.append(estimate(e, T, g, None))
    points = [p for p in points if p is not None]
    if not points:
        raise ValueError("Nenhum vencimento com forward implícito válido.")
    return points


def implied_spot(forwards: list[ForwardPoint], dividend_yield: float = 0.0) -> float:
    """Spot pelo vencimento mais curto: S = F·DF·exp(q·T). Erro ~ q·T (centésimos de % em poucos dias)."""
    p = min(forwards, key=lambda f: f.T)
    return float(p.forward * p.discount_factor * np.exp(dividend_yield * p.T))


def term_structure(forwards: list[ForwardPoint], spot: float) -> TermStructure:
    """r(T) = -ln(DF)/T e q(T) = r(T) - ln(F/S)/T por vencimento (um ponto por prazo)."""
    by_T = {}
    for p in forwards:
        by_T.setdefault(p.T, p)  # AM/PM no mesmo instante não acontecem; mantém o primeiro
    pts = sorted(by_T.values(), key=lambda f: f.T)
    T = np.array([p.T for p in pts])
    r = np.array([p.rate for p in pts])
    q = r - np.log(np.array([p.forward for p in pts]) / spot) / T
    return TermStructure(T, r, q)
