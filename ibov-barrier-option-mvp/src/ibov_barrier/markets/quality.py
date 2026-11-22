"""Qualidade do dia: alerta quando a superfície sai com muito menos pontos que o normal.

Nos 78 pregões de 03/08 a 19/11/2026, a superfície do SPX teve ~8.700 pontos na mediana. Entre 16/10 e
30/10 caiu para 4.200-5.600 sem mudança relevante de volatilidade (provável problema de cotação no
fornecedor), e em dias de queda forte (05/11, 19/11) também perdeu pontos. Nesses dias o forward, r e q
ficam menos confiáveis, então o dia é marcado, não descartado: quem consome decide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COVERAGE_WINDOW = 20      # pregões anteriores usados como referência
COVERAGE_THRESHOLD = 0.70  # alerta abaixo de 70% da mediana de referência
COVERAGE_MIN_HISTORY = 5   # sem pelo menos 5 dias anteriores, não há referência


def low_coverage_flags(points: pd.Series, *, window: int = COVERAGE_WINDOW,
                       threshold: float = COVERAGE_THRESHOLD,
                       min_history: int = COVERAGE_MIN_HISTORY) -> pd.DataFrame:
    """Para uma série de pontos por dia (em ordem cronológica), devolve referência, razão e alerta.

    A referência é a mediana dos `window` dias ANTERIORES que não foram marcados. Assim uma sequência de
    dias ruins (ex.: 19/10 a 30/10/2026) não rebaixa a referência dos dias seguintes: com a mediana
    contaminada, 05/11/2026 (5.385 pontos) passava como 82% em vez de ~62%.
    """
    points = pd.Series(points, dtype=float)
    clean: list[float] = []
    refs, flags = [], []
    for value in points:
        hist = clean[-window:]
        ref = float(np.median(hist)) if len(hist) >= min_history else np.nan
        low = bool(np.isfinite(ref) and value < threshold * ref)
        refs.append(ref)
        flags.append(low)
        if not low and np.isfinite(value):
            clean.append(float(value))
    ref = pd.Series(refs, index=points.index)
    return pd.DataFrame({"referência": ref, "razão": points / ref, "baixa cobertura": flags},
                        index=points.index)


def coverage_warning(points_today: int, history: pd.Series, flagged: pd.Series | None = None, *,
                     window: int = COVERAGE_WINDOW, threshold: float = COVERAGE_THRESHOLD,
                     min_history: int = COVERAGE_MIN_HISTORY) -> str | None:
    """Mensagem de alerta para o dia corrente, dado o histórico de pontos dos dias anteriores.

    `flagged` (mesmo índice de `history`) marca dias de baixa cobertura, que saem da referência.
    """
    history = pd.Series(history, dtype=float)
    if flagged is not None:
        history = history.loc[~pd.Series(flagged, index=history.index).fillna(False).astype(bool)]
    history = history.dropna().tail(window)
    if len(history) < min_history:
        return None
    ref = float(history.median())
    if points_today < threshold * ref:
        return (f"Baixa cobertura: {points_today:,} pontos, {points_today / ref:.0%} da mediana dos "
                f"últimos {len(history)} pregões ({ref:,.0f}). Forward, r e q deste dia são menos confiáveis.")
    return None
