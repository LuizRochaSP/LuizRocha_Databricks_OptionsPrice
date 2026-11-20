"""Forward, fator de desconto, spot e dividend yield implícitos pela paridade put-call.

Para opções europeias de mesmo strike e vencimento:  C - P = DF·(F - K).
Com vários strikes, C - P é uma reta em K: intercepto DF·F e inclinação -DF. Usamos os strikes
mais próximos do ATM (menor |C - P|), onde as cotações são mais líquidas e o erro relativo é menor.

Se uma curva de juros externa (SOFR/Treasuries) fornecer DF, só F é estimado (média de K + (C-P)/DF),
o que é mais estável em vencimentos curtos, onde -ln(DF)/T amplifica o ruído de cotação.
"""
from __future__ import annotations

from dataclasses import replace
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


def implied_discount_factor(expiry_quotes: pd.DataFrame, expiration, *, band: float = 0.10,
                            min_pairs: int = 5) -> tuple[float, int]:
    """DF pela regressão de C - P em K com todos os pares em |ln(K/K_atm)| <= `band`.

    K_atm é o strike com menor |C - P| (~ forward). Com ~60 strikes espalhados por ±10%, a inclinação
    (-DF) fica muito mais bem medida que com os 10 pares colados no ATM (faixa de ~50 pontos).
    """
    pairs = _pairs(expiry_quotes)
    if pairs.empty:
        raise ValueError(f"Vencimento {expiration}: nenhum par call/put cotado.")
    center = float(pairs.loc[pairs["c_minus_p"].abs().idxmin(), "strike"])
    pairs = pairs.loc[np.abs(np.log(pairs["strike"] / center)) <= band]
    if len(pairs) < min_pairs or np.ptp(pairs["strike"]) == 0:
        raise ValueError(f"Vencimento {expiration}: menos de {min_pairs} pares na faixa de ±{band:.0%}.")
    slope, _ = np.polyfit(pairs["strike"].to_numpy(float), pairs["c_minus_p"].to_numpy(float), 1)
    df = float(-slope)
    if not (0.5 < df < 1.1):
        raise ValueError(f"Vencimento {expiration}: fator de desconto implícito absurdo ({df:.6f}).")
    return df, int(len(pairs))


def fit_rate_curve(T, r, *, z: float = 4.0, mad_floor: float = 0.0005,
                   max_iter: int = 5) -> tuple[Callable[[float], float], np.ndarray]:
    """Curva r(T) = a + b·T robusta: reajusta descartando taxas a mais de `z` desvios robustos (MAD).

    Devolve (r_curve, usados). Fora do intervalo de prazos ajustados a taxa fica constante (sem extrapolar).
    Com menos de 3 taxas, a curva é a mediana (constante). `mad_floor` (5 bps) evita descartar tudo
    quando as taxas são quase idênticas.
    """
    T, r = np.asarray(T, float), np.asarray(r, float)

    def line(mask):
        if mask.sum() >= 3 and np.ptp(T[mask]) > 0:
            b, a = np.polyfit(T[mask], r[mask], 1)
            return float(a), float(b)
        return float(np.median(r[mask])), 0.0

    # Ponto de partida robusto (Theil-Sen): um outlier não puxa a reta inicial.
    i, j = np.triu_indices(len(T), 1)
    ok = T[j] != T[i]
    if len(T) >= 3 and ok.any():
        b = float(np.median((r[j] - r[i])[ok] / (T[j] - T[i])[ok]))
        a = float(np.median(r - b * T))
    else:
        a, b = float(np.median(r)), 0.0
    used = np.ones(len(T), bool)
    for _ in range(max_iter):
        resid = r - (a + b * T)
        scale = max(1.4826 * float(np.median(np.abs(resid))), mad_floor)
        new_used = np.abs(resid) <= z * scale
        if new_used.sum() < min(3, len(T)):
            break
        used = new_used
        a_new, b_new = line(used)
        if np.isclose(a_new, a, rtol=0, atol=1e-12) and np.isclose(b_new, b, rtol=0, atol=1e-12):
            break
        a, b = a_new, b_new
    lo, hi = float(T[used].min()), float(T[used].max())

    def r_curve(t: float) -> float:
        return a + b * min(max(float(t), lo), hi)

    return r_curve, used


def implied_forwards(quotes: pd.DataFrame, *, max_pairs: int = 10, min_T: float = 0.0,
                     discount_curve: Callable[[float], float] | None = None,
                     min_T_rate: float = 0.25,
                     rate_method: str = "smooth",
                     rate_band: float = 0.10,
                     max_residual_ratio: float = 10.0,
                     residual_floor: float = 1.0,
                     report_warnings: list | None = None) -> list[ForwardPoint]:
    """Um ForwardPoint por (vencimento, liquidação). `quotes` precisa da coluna T (anos).

    Vencimentos sem pares suficientes são descartados (com aviso), não derrubam o pipeline.
    `discount_curve(T) -> r contínuo`, se informado, fixa DF = exp(-r·T) em todos os prazos.

    Sem curva externa, `rate_method` escolhe como a taxa é medida:
    - "smooth" (padrão): DF por regressão ampla (±`rate_band` em ln K) em cada vencimento com
      T >= `min_T_rate`; uma curva r(T) = a + b·T robusta é ajustada a essas taxas (outliers ficam fora,
      com aviso) e aplicada a TODOS os vencimentos; o forward vem dos pares perto do ATM com esse DF.
      Base: 78 pregões de 08-11/2026, em que a regressão ampla foi estável e a de 10 pares tinha
      saltos de centenas de bps num vencimento isolado.
    - "atm" (anterior): regressão com os `max_pairs` pares mais perto do ATM em cada T >= `min_T_rate`;
      prazos menores usam a taxa do vencimento confiável mais curto. Abaixo de ~3 meses,
      r = -ln(DF)/T amplifica o ruído de cotação (em 17/11/2026 saíram taxas entre -21% e +26%).

    Em ambos, um vencimento cujo resíduo passe de max(`residual_floor`, `max_residual_ratio` x mediana
    do dia) é descartado com aviso (cotações incoerentes: forward não confiável).
    """
    groups = [((e, T), g) for (e, T), g in quotes.groupby(["expiration", "T"], sort=True) if T > min_T]

    def warn(msg):
        if report_warnings is not None:
            report_warnings.append(msg)

    def estimate(expiration, T, group, df, **extra):
        try:
            p = implied_forward(group, T, expiration, max_pairs=max_pairs, discount_factor=df)
        except ValueError as exc:
            warn(str(exc))
            return None
        return replace(p, **extra) if extra else p

    if discount_curve is not None:
        points = [estimate(e, T, g, float(np.exp(-discount_curve(T) * T))) for (e, T), g in groups]
    elif rate_method == "smooth":
        fits = {}
        for (e, T), g in groups:
            if T >= min_T_rate:
                try:
                    df, _ = implied_discount_factor(g, e, band=rate_band)
                    fits[(e, T)] = float(-np.log(df) / T)
                except ValueError as exc:
                    warn(str(exc))
        if not fits:
            warn("Nenhum vencimento longo com taxa pela regressão ampla: usando o método 'atm'.")
            return implied_forwards(quotes, max_pairs=max_pairs, min_T=min_T, min_T_rate=min_T_rate,
                                    rate_method="atm", max_residual_ratio=max_residual_ratio,
                                    residual_floor=residual_floor, report_warnings=report_warnings)
        keys = list(fits)
        r_curve, used = fit_rate_curve([k[1] for k in keys], [fits[k] for k in keys])
        in_fit = dict(zip(keys, used))
        for (e, T), ok in in_fit.items():
            if not ok:
                warn(f"Vencimento {e}: taxa implícita {fits[(e, T)]:.3%} fora da curva r(T) "
                     f"({r_curve(T):.3%}); ignorada no ajuste.")
        points = [estimate(e, T, g, float(np.exp(-r_curve(T) * T)),
                           fit_rate=fits.get((e, T)), in_fit=bool(in_fit.get((e, T), False)))
                  for (e, T), g in groups]
    elif rate_method == "atm":
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
    else:
        raise ValueError(f"rate_method desconhecido: {rate_method!r} (use 'smooth' ou 'atm').")

    points = [p for p in points if p is not None]
    if not points:
        raise ValueError("Nenhum vencimento com forward implícito válido.")
    return _drop_incoherent(points, max_residual_ratio, residual_floor, warn)


def _drop_incoherent(points, ratio: float, floor: float, warn) -> list[ForwardPoint]:
    """Descarta vencimentos com resíduo muito acima da mediana do dia (cotações incoerentes)."""
    if len(points) < 3:
        return points
    limit = max(floor, ratio * float(np.median([p.residual_std for p in points])))
    kept = []
    for p in points:
        if p.residual_std > limit:
            warn(f"Vencimento {p.expiration}: resíduo da paridade {p.residual_std:.3f} acima do limite "
                 f"{limit:.3f}; vencimento descartado.")
        else:
            kept.append(p)
    return kept or points


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
