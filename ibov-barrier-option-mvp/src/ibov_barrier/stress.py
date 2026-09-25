"""Stress-test engine for barrier-option scenarios.

Reuses the existing Monte-Carlo pricing engine to reprice the option under
shocked market parameters and compute position P&L in index points.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pricing import (
    BarrierContract,
    MarketData,
    PriceResult,
    price_down_and_out_call,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationParams:
    """Parameters from the base execution to replay."""

    paths: int
    steps: int
    seed: int
    monitoring: str


@dataclass(frozen=True)
class ScenarioInput:
    """One stress scenario read from the Excel template.

    Shocks are applied to the **base** market data:

    * ``spot_shock_pct`` — relative spot shock, e.g. -0.10 = -10 %.
    * ``vol_shock_abs`` — absolute volatility shock in annual terms,
      e.g. +0.05 = +5 percentage points.
    * ``rate_shock_bp`` — absolute rate shock in basis points,
      e.g. +100 = +100 bp.
    """

    name: str
    spot_shock_pct: float
    vol_shock_abs: float
    rate_shock_bp: float
    side: str  # "COMPRADO" or "VENDIDO"
    qtd_pontos: float


@dataclass
class ScenarioResult:
    """Result of one stress scenario."""

    name: str
    base_price: float
    stressed_price: float
    pnl: float
    side: str
    qtd_pontos: float
    knock_out: bool
    spot_stressed: float
    volatility_stressed: float
    rate_stressed: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Provenance verification
# ---------------------------------------------------------------------------

def verify_provenance(
    results_path: Path,
    validation: dict,
    risk: dict | None = None,
) -> dict:
    """Verify the chain of trust RESULTS → VALIDATION → RISK.

    Returns a dict with the validated ``metadata``, ``pricing_result`` and
    ``simulation`` on success.  Raises ``RuntimeError`` on any mismatch.
    """
    import pandas as pd

    # --- VALIDATION checks ---
    if validation.get("schema_version") != 1:
        raise RuntimeError("VALIDATION schema_version incompatível.")
    if validation.get("fail_count") != 0:
        raise RuntimeError("VALIDATION contém FAIL ≠ 0.")
    if not validation.get("approved_for_risk"):
        raise RuntimeError("VALIDATION não aprovou o resultado para risco.")
    if validation.get("decision") not in {"APROVADO", "APROVADO COM RESSALVAS"}:
        raise RuntimeError(f"VALIDATION decisão inválida: {validation.get('decision')}")

    # --- RESULTS file reference ---
    results_file = validation.get("results_file")
    if not results_file:
        raise RuntimeError("VALIDATION não referencia o RESULTS.")

    if results_path.name != results_file:
        raise RuntimeError(
            f"RESULTS ({results_path.name}) ≠ VALIDATION ({results_file})."
        )

    # --- SHA-256 ---
    actual_sha = hashlib.sha256(results_path.read_bytes()).hexdigest()
    expected_sha = validation.get("results_sha256")
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"SHA-256 do RESULTS diverge: {actual_sha} ≠ {expected_sha}."
        )

    # --- RESULTS schema ---
    bundle = pd.read_csv(results_path)

    def _obj(name: str) -> dict:
        rows = bundle.loc[bundle["object_name"].eq(name), "payload_json"]
        if len(rows) != 1:
            raise RuntimeError(f"Objeto '{name}' ausente ou duplicado no RESULTS.")
        return json.loads(rows.iloc[0])

    metadata = _obj("metadata")
    if metadata.get("schema_version") != 2:
        raise RuntimeError("RESULTS schema_version ≠ 2.")

    pricing_result = _obj("pricing_result")
    simulation = _obj("simulation_diagnostics")

    # --- Cross-check VALIDATION ↔ RESULTS market_date ---
    if validation.get("market_date") != metadata.get("market_date"):
        raise RuntimeError(
            f"market_date diverge: VALIDATION={validation.get('market_date')} "
            f"vs RESULTS={metadata.get('market_date')}."
        )

    # --- Optional RISK cross-check ---
    if risk is not None:
        if risk.get("schema_version") != 1:
            raise RuntimeError("RISK schema_version incompatível.")
        if risk.get("results_file") != results_path.name:
            raise RuntimeError("RISK referencia RESULTS diferente.")
        if risk.get("results_sha256") != actual_sha:
            raise RuntimeError("RISK SHA-256 do RESULTS diverge.")
        if risk.get("validation_decision") != validation.get("decision"):
            raise RuntimeError("RISK refere decisão de validação diferente.")

    return {
        "metadata": metadata,
        "pricing_result": pricing_result,
        "simulation": simulation,
        "results_sha256": actual_sha,
    }


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------

def _side_multiplier(side: str) -> float:
    """Return +1 for COMPRADO, -1 for VENDIDO."""
    side = side.strip().upper()
    if side == "COMPRADO":
        return 1.0
    if side == "VENDIDO":
        return -1.0
    raise ValueError(f"Lado da posição inválido: {side!r} (use COMPRADO ou VENDIDO).")


def run_scenario(
    base_market: MarketData,
    base_contract: BarrierContract,
    sim: SimulationParams,
    scenario: ScenarioInput,
    *,
    normals: np.ndarray | None = None,
    uniforms: np.ndarray | None = None,
) -> ScenarioResult:
    """Reprice the option under shocked parameters and compute P&L.

    Common random numbers (same ``normals``/``uniforms``) are used when
    provided so that the base and stressed prices share simulation noise,
    reducing variance in the P&L estimate.
    """
    # --- Validate scenario inputs ---
    errors: list[str] = []

    try:
        side_mult = _side_multiplier(scenario.side)
    except ValueError as exc:
        errors.append(str(exc))
        side_mult = 0.0

    if not np.isfinite(scenario.spot_shock_pct):
        errors.append(f"spot_shock_pct não é finito: {scenario.spot_shock_pct}")
    if not np.isfinite(scenario.vol_shock_abs):
        errors.append(f"vol_shock_abs não é finito: {scenario.vol_shock_abs}")
    if not np.isfinite(scenario.rate_shock_bp):
        errors.append(f"rate_shock_bp não é finito: {scenario.rate_shock_bp}")
    if not np.isfinite(scenario.qtd_pontos) or scenario.qtd_pontos == 0:
        errors.append(f"qtd_pontos inválido: {scenario.qtd_pontos}")

    if errors:
        return ScenarioResult(
            name=scenario.name,
            base_price=0.0,
            stressed_price=0.0,
            pnl=0.0,
            side=scenario.side,
            qtd_pontos=scenario.qtd_pontos,
            knock_out=False,
            spot_stressed=0.0,
            volatility_stressed=0.0,
            rate_stressed=0.0,
            error="; ".join(errors),
        )

    # --- Apply shocks ---
    spot_shocked = base_market.spot * (1.0 + scenario.spot_shock_pct)
    vol_shocked = base_market.volatility + scenario.vol_shock_abs
    rate_shocked = base_market.rate + scenario.rate_shock_bp * 1e-4

    # --- Validate shocked domain ---
    domain_errors: list[str] = []
    if spot_shocked <= 0:
        domain_errors.append(f"spot estressado ≤ 0: {spot_shocked}")
    if vol_shocked < 0:
        domain_errors.append(f"volatility estressada < 0: {vol_shocked}")
    if rate_shocked < -0.99:
        domain_errors.append(f"rate estressado < -99%: {rate_shocked}")

    # --- Knock-out check: if spot at or below barrier, option is worthless ---
    knock_out = spot_shocked <= base_contract.barrier

    if knock_out:
        stressed_price = 0.0
    elif domain_errors:
        return ScenarioResult(
            name=scenario.name,
            base_price=0.0,
            stressed_price=0.0,
            pnl=0.0,
            side=scenario.side,
            qtd_pontos=scenario.qtd_pontos,
            knock_out=False,
            spot_stressed=spot_shocked,
            volatility_stressed=vol_shocked,
            rate_stressed=rate_shocked,
            error="; ".join(domain_errors),
        )
    else:
        shocked_market = MarketData(
            spot=spot_shocked,
            rate=rate_shocked,
            dividend_yield=base_market.dividend_yield,
            volatility=vol_shocked,
        )
        result: PriceResult = price_down_and_out_call(
            shocked_market,
            base_contract,
            paths=sim.paths,
            steps=sim.steps,
            seed=sim.seed,
            monitoring=sim.monitoring,
            normals=normals,
            uniforms=uniforms,
        )
        stressed_price = result.price

    # --- Base price (recomputed with same RNs if available) ---
    base_result = price_down_and_out_call(
        base_market,
        base_contract,
        paths=sim.paths,
        steps=sim.steps,
        seed=sim.seed,
        monitoring=sim.monitoring,
        normals=normals,
        uniforms=uniforms,
    )
    base_price = base_result.price

    # --- P&L in index points ---
    # COMPRADO:  P&L = (stressed - base) * QtdPontos
    # VENDIDO:   P&L = (base - stressed) * QtdPontos
    pnl = side_mult * (stressed_price - base_price) * scenario.qtd_pontos

    return ScenarioResult(
        name=scenario.name,
        base_price=base_price,
        stressed_price=stressed_price,
        pnl=pnl,
        side=scenario.side,
        qtd_pontos=scenario.qtd_pontos,
        knock_out=knock_out,
        spot_stressed=spot_shocked,
        volatility_stressed=vol_shocked,
        rate_stressed=rate_shocked,
    )


def run_all_scenarios(
    base_market: MarketData,
    base_contract: BarrierContract,
    sim: SimulationParams,
    scenarios: list[ScenarioInput],
) -> list[ScenarioResult]:
    """Run all scenarios with common random numbers for comparability."""
    rng = np.random.default_rng(sim.seed)
    normals = rng.standard_normal((sim.paths, sim.steps))
    uniforms = rng.random((sim.paths, sim.steps))

    results: list[ScenarioResult] = []
    for sc in scenarios:
        results.append(
            run_scenario(
                base_market,
                base_contract,
                sim,
                sc,
                normals=normals,
                uniforms=uniforms,
            )
        )
    return results
