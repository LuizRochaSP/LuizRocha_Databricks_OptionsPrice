"""Qualidade do dia: alerta quando a superfície sai com muito menos pontos que o normal.

Nos 78 pregões de 03/08 a 19/11/2026, a superfície do SPX teve ~8.700 pontos na mediana. Entre 16/10 e
30/10 caiu para 4.200-5.600 sem mudança relevante de volatilidade (provável problema de cotação no
fornecedor), e em dias de queda forte (05/11, 19/11) também perdeu pontos. Nesses dias o forward, r e q
ficam menos confiáveis, então o dia é marcado, não descartado: quem consome decide.
"""
from __future__ import annotations

import pandas as pd

COVERAGE_WINDOW = 20      # pregões anteriores usados como referência
COVERAGE_THRESHOLD = 0.70  # alerta abaixo de 70% da mediana de referência
COVERAGE_MIN_HISTORY = 5   # sem pelo menos 5 dias anteriores, não há referência


def low_coverage_flags(points: pd.Series, *, window: int = COVERAGE_WINDOW,
                       threshold: float = COVERAGE_THRESHOLD,
                       min_history: int = COVERAGE_MIN_HISTORY) -> pd.DataFrame:
    """Para uma série de pontos por dia (em ordem cronológica), devolve referência, razão e alerta.

    A referência é a mediana dos `window` dias ANTERIORES (o próprio dia não entra), então um dia ruim
    não puxa a própria referência para baixo.
    """
    points = pd.Series(points, dtype=float)
    ref = points.shift(1).rolling(window, min_periods=min_history).median()
    ratio = points / ref
    return pd.DataFrame({"referência": ref, "razão": ratio, "baixa cobertura": ratio < threshold},
                        index=points.index)


def coverage_warning(points_today: int, history: pd.Series, *, window: int = COVERAGE_WINDOW,
                     threshold: float = COVERAGE_THRESHOLD,
                     min_history: int = COVERAGE_MIN_HISTORY) -> str | None:
    """Mensagem de alerta para o dia corrente, dado o histórico de pontos dos dias anteriores."""
    history = pd.Series(history, dtype=float).dropna().tail(window)
    if len(history) < min_history:
        return None
    ref = float(history.median())
    if points_today < threshold * ref:
        return (f"Baixa cobertura: {points_today:,} pontos, {points_today / ref:.0%} da mediana dos "
                f"últimos {len(history)} pregões ({ref:,.0f}). Forward, r e q deste dia são menos confiáveis.")
    return None
