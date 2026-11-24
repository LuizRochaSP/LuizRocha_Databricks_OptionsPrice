"""Contrato de referência do S&P 500 (call down-and-out), precificado como o notebook 01 faz com o IBOV.

- Vol no strike: interpolação linear em (ln(K/F), T) sobre os pontos da superfície e vizinho mais próximo
  fora do casco convexo (a regra do notebook 01).
- r(T) e q(T): lineares em T entre os vencimentos da paridade put-call (o forward reproduz o da paridade).
- Monte Carlo com ponte browniana nos parâmetros do 01 (200 mil caminhos, 252 passos, semente 42) e
  Reiner-Rubinstein como referência analítica.

Os dicionários devolvidos têm as mesmas chaves do RESULTS do 01 (``pricing_result`` e
``simulation_diagnostics``), para o dashboard, a validação e o notebook 03 lerem os dois ativos igual.

``salvar_cadeia_do_dia`` grava a cadeia bruta no formato do notebook 08 (``brutos/<data>/<raiz>.parquet`` e
``meta.json`` por último), para o notebook 09 achar o D-1 em disco.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from ..analytical import price_down_and_out_call_analytical
from ..pricing import BarrierContract, MarketData, price_down_and_out_call

MC_PADRAO = {"paths": 200_000, "steps": 252, "seed": 42, "monitoring": "brownian_bridge"}
CONTRATO_PADRAO = {"strike_pct": 0.95, "barrier_pct": 0.85, "maturity": 0.5}
FORA_DO_CASCO = "nearest — alvo fora do casco convexo"


def vol_no_ponto(options: pd.DataFrame, strike: float, maturity: float, forward: float) -> tuple[float, str]:
    """Vol implícita em (ln(K/F), T): linear; fora do casco convexo, vizinho mais próximo."""
    sd = options.dropna(subset=["log_moneyness", "T", "implied_vol"]).drop_duplicates(["T", "log_moneyness"])
    pts = sd[["log_moneyness", "T"]].to_numpy(dtype=float)
    vals = sd["implied_vol"].to_numpy(dtype=float)
    alvo = np.array([[math.log(strike / forward), maturity]])
    v = griddata(pts, vals, alvo, method="linear")[0]
    if np.isfinite(v):
        return float(v), "linear"
    return float(griddata(pts, vals, alvo, method="nearest")[0]), FORA_DO_CASCO


def curvas_no_prazo(forwards: pd.DataFrame, maturity: float) -> tuple[float, float]:
    """r(T) e q(T) contínuas, lineares em T entre os vencimentos (planas fora deles)."""
    f = forwards.dropna(subset=["T", "r", "q"]).sort_values("T")
    if f.empty:
        raise ValueError("forwards sem T, r e q")
    return float(np.interp(maturity, f["T"], f["r"])), float(np.interp(maturity, f["T"], f["q"]))


def precificar_referencia(spot: float, options: pd.DataFrame, forwards: pd.DataFrame, *,
                          strike_pct: float = CONTRATO_PADRAO["strike_pct"],
                          barrier_pct: float = CONTRATO_PADRAO["barrier_pct"],
                          maturity: float = CONTRATO_PADRAO["maturity"],
                          paths: int = MC_PADRAO["paths"], steps: int = MC_PADRAO["steps"],
                          seed: int = MC_PADRAO["seed"], monitoring: str = MC_PADRAO["monitoring"]) -> dict:
    """Call down-and-out com strike e barreira em % do spot. Devolve pricing_result, simulation_diagnostics e alvo."""
    if not 0 < barrier_pct < 1:
        raise ValueError("a barreira down-and-out precisa ficar abaixo do spot (0 < barreira < 100%)")
    if strike_pct <= 0 or maturity <= 0:
        raise ValueError("strike e prazo precisam ser positivos")
    strike, barrier = spot * strike_pct, spot * barrier_pct
    rate, q = curvas_no_prazo(forwards, maturity)
    forward = spot * math.exp((rate - q) * maturity)
    vol, metodo = vol_no_ponto(options, strike, maturity, forward)

    market = MarketData(spot=float(spot), rate=rate, dividend_yield=q, volatility=vol)
    contract = BarrierContract(strike=strike, barrier=barrier, maturity=maturity)
    res = price_down_and_out_call(market, contract, paths=paths, steps=steps, seed=seed, monitoring=monitoring)
    analitico = price_down_and_out_call_analytical(spot=market.spot, strike=strike, barrier=barrier,
                                                   maturity=maturity, rate=rate, dividend_yield=q, volatility=vol)
    desconto = res.vanilla_price - res.price
    pricing_result = {
        "contract": {"strike": strike, "barrier": barrier, "maturity": maturity},
        "market": {"spot": market.spot, "rate": rate, "dividend_yield": q, "volatility": vol, "forward": forward,
                   "interpolation_method": metodo},
        "vanilla_price": float(res.vanilla_price),
        "barrier_price": float(res.price),
        "analytical_price": float(analitico),
        "analytical_error_pct": float(abs(res.price - analitico) / analitico) if analitico > 0 else 0.0,
        "barrier_discount_abs": float(desconto),
        "barrier_discount_pct": float(desconto / res.vanilla_price) if res.vanilla_price > 0 else 0.0,
        "knock_out_probability": float(res.knock_out_probability),
        "standard_error": float(res.standard_error),
        "ci_low_95": float(res.ci_low),
        "ci_high_95": float(res.ci_high),
    }
    simulation = {"paths": int(paths), "steps": int(steps), "seed": int(seed), "monitoring": monitoring}
    alvo = {"strike_pct": strike_pct, "barrier_pct": barrier_pct, "target_strike": strike, "target_barrier": barrier,
            "target_maturity": maturity, "target_r": rate, "target_q": q, "target_iv": vol,
            "target_forward": forward, "interpolation_method": metodo}
    return {"pricing_result": pricing_result, "simulation_diagnostics": simulation, "alvo": alvo}


def salvar_cadeia_do_dia(pasta_brutos: Path | str, market_date: date, chains: Mapping[str, pd.DataFrame],
                         spot_fechamento: float, sofr: float | None) -> str:
    """Grava ``<pasta>/<data>/<raiz>.parquet`` e ``meta.json`` (por último: marca o dia completo).

    Mesmo formato do notebook 08. Se o dia já está em disco, não regrava.
    """
    folder = Path(pasta_brutos) / market_date.isoformat()
    meta_path = folder / "meta.json"
    if meta_path.exists():
        return "em disco"
    folder.mkdir(parents=True, exist_ok=True)
    linhas = {}
    for root, raw in chains.items():
        raw.to_parquet(folder / f"{root}.parquet", index=False)
        linhas[root] = int(len(raw))
    meta = {"data": market_date.isoformat(), "spot": float(spot_fechamento), "linhas": linhas, "sofr": sofr,
            "origem": "notebook 07"}
    meta_path.write_text(json.dumps(meta))
    return "gravado"
