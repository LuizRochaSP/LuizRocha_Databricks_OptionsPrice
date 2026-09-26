"""Validação da curva contra ANBIMA e testes de consistência."""
from __future__ import annotations

import numpy as np

from .curve import DiscountCurve


def compare_with_anbima(
    curve: DiscountCurve, anbima_df, tolerance_bp: float = 5.0
) -> dict:
    """Compara taxa da curva com ETTJ da ANBIMA nos vértices comuns."""
    errors_bp = []
    for _, row in anbima_df.iterrows():
        du = float(row["vertice_du"])
        taxa = float(row["taxa"])
        if du > curve.tenors_bd.max():
            continue
        model = curve.rate(du)
        errors_bp.append((model - taxa) * 1e4)
    errors_bp = np.array(errors_bp)
    if len(errors_bp) == 0:
        return {
            "n_points": 0,
            "mae_bp": float("nan"),
            "rmse_bp": float("nan"),
            "max_abs_bp": float("nan"),
            "within_tolerance": False,
        }
    return {
        "n_points": int(len(errors_bp)),
        "mae_bp": float(np.abs(errors_bp).mean()),
        "rmse_bp": float(np.sqrt((errors_bp ** 2).mean())),
        "max_abs_bp": float(np.abs(errors_bp).max()),
        "within_tolerance": bool(np.abs(errors_bp).max() <= tolerance_bp),
    }


def check_monotonic_df(curve: DiscountCurve, n_points: int = 200) -> bool:
    """DF deve ser decrescente em T."""
    tenors = np.linspace(curve.tenors_bd.min(), curve.tenors_bd.max(), n_points)
    dfs = np.array([curve.df(t) for t in tenors])
    return bool(np.all(np.diff(dfs) < 0))


def check_positive_forwards(curve: DiscountCurve, n_points: int = 200) -> bool:
    """Forwards devem ser positivos em todos os segmentos."""
    tenors = np.linspace(curve.tenors_bd.min(), curve.tenors_bd.max(), n_points)
    for i in range(len(tenors) - 1):
        try:
            f = curve.forward_rate(tenors[i], tenors[i + 1])
            if f <= 0:
                return False
        except ValueError:
            return False
    return True