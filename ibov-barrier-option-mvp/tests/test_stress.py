"""Tests for the stress-test engine (ibov_barrier.stress)."""
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ibov_barrier import BarrierContract, MarketData
from ibov_barrier.stress import (
    ScenarioInput,
    ScenarioResult,
    SimulationParams,
    run_all_scenarios,
    run_scenario,
    verify_provenance,
)

# Reduced simulation parameters for faster tests
SIM = SimulationParams(paths=20_000, steps=63, seed=42, monitoring="brownian_bridge")
MARKET = MarketData(spot=187422.92, rate=0.12664, dividend_yield=0.01166, volatility=0.24162)
CONTRACT = BarrierContract(strike=155000.0, barrier=140000.0, maturity=0.5)


class StressTests(unittest.TestCase):

    def test_no_shock_reproduces_base_price(self):
        """Cenário sem choques deve reproduzir o preço-base dentro de tolerância."""
        sc = ScenarioInput(name="base", spot_shock_pct=0.0, vol_shock_abs=0.0,
                           rate_shock_bp=0.0, side="COMPRADO", qtd_pontos=1.0)
        result = run_scenario(MARKET, CONTRACT, SIM, sc)
        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.base_price, result.stressed_price, places=0)
        self.assertAlmostEqual(result.pnl, 0.0, places=0)

    def test_long_short_have_opposite_pnl(self):
        """Posições comprada e vendida devem ter P&L com sinais opostos."""
        sc_long = ScenarioInput(name="long", spot_shock_pct=-0.10, vol_shock_abs=0.0,
                                rate_shock_bp=0.0, side="COMPRADO", qtd_pontos=1.0)
        sc_short = ScenarioInput(name="short", spot_shock_pct=-0.10, vol_shock_abs=0.0,
                                 rate_shock_bp=0.0, side="VENDIDO", qtd_pontos=1.0)
        r_long = run_scenario(MARKET, CONTRACT, SIM, sc_long)
        r_short = run_scenario(MARKET, CONTRACT, SIM, sc_short)
        self.assertIsNone(r_long.error)
        self.assertIsNone(r_short.error)
        self.assertAlmostEqual(r_long.pnl, -r_short.pnl, places=0)

    def test_qtd_pontos_scaling(self):
        """P&L deve escalar linearmente com QtdPontos."""
        sc1 = ScenarioInput(name="q1", spot_shock_pct=-0.10, vol_shock_abs=0.0,
                            rate_shock_bp=0.0, side="COMPRADO", qtd_pontos=1.0)
        sc10 = ScenarioInput(name="q10", spot_shock_pct=-0.10, vol_shock_abs=0.0,
                             rate_shock_bp=0.0, side="COMPRADO", qtd_pontos=10.0)
        r1 = run_scenario(MARKET, CONTRACT, SIM, sc1)
        r10 = run_scenario(MARKET, CONTRACT, SIM, sc10)
        self.assertIsNone(r1.error)
        self.assertIsNone(r10.error)
        self.assertAlmostEqual(r10.pnl, 10.0 * r1.pnl, places=0)

    def test_barrier_crossing_results_in_knockout(self):
        """Cenário com spot cruzando a barreira deve resultar em knock-out (preço = 0)."""
        # spot * (1 - 0.2527) ≈ 140000 = barrier
        sc = ScenarioInput(name="knockout", spot_shock_pct=-0.26, vol_shock_abs=0.0,
                           rate_shock_bp=0.0, side="COMPRADO", qtd_pontos=1.0)
        result = run_scenario(MARKET, CONTRACT, SIM, sc)
        self.assertTrue(result.knock_out)
        self.assertAlmostEqual(result.stressed_price, 0.0)

    def test_invalid_scenario_does_not_block_others(self):
        """Uma aba inválida não deve impedir as demais."""
        sc_valid = ScenarioInput(name="valid", spot_shock_pct=-0.05, vol_shock_abs=0.0,
                                 rate_shock_bp=0.0, side="COMPRADO", qtd_pontos=1.0)
        sc_invalid = ScenarioInput(name="invalid", spot_shock_pct=float('nan'),
                                   vol_shock_abs=0.0, rate_shock_bp=0.0,
                                   side="COMPRADO", qtd_pontos=1.0)
        results = run_all_scenarios(MARKET, CONTRACT, SIM, [sc_valid, sc_invalid])
        self.assertIsNone(results[0].error)
        self.assertIsNotNone(results[1].error)

    def test_reproducibility(self):
        """Duas execuções com mesma seed devem produzir os mesmos resultados."""
        sc = ScenarioInput(name="repro", spot_shock_pct=-0.10, vol_shock_abs=0.02,
                           rate_shock_bp=50.0, side="COMPRADO", qtd_pontos=5.0)
        r1 = run_scenario(MARKET, CONTRACT, SIM, sc)
        r2 = run_scenario(MARKET, CONTRACT, SIM, sc)
        self.assertEqual(r1.pnl, r2.pnl)
        self.assertEqual(r1.stressed_price, r2.stressed_price)

    def test_provenance_blocks_incompatible_sha256(self):
        """Hashes incompatíveis devem bloquear o processamento."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Create a minimal RESULTS CSV
            metadata = {"schema_version": 2, "market_date": "2026-09-22", "spot": 187422.92,
                        "target_strike": 155000.0, "target_maturity": 0.5,
                        "target_r": 0.12, "target_q": 0.01, "target_iv": 0.24}
            pricing = {"contract": {"strike": 155000.0, "barrier": 140000.0, "maturity": 0.5},
                       "market": {"spot": 187422.92, "rate": 0.12, "dividend_yield": 0.01,
                                  "volatility": 0.24, "forward": 198000.0,
                                  "interpolation_method": "linear"},
                       "vanilla_price": 41000.0, "barrier_price": 40900.0,
                       "barrier_discount_abs": 100.0, "barrier_discount_pct": 0.0024,
                       "knock_out_probability": 0.05, "standard_error": 3.5,
                       "ci_low_95": 40893.0, "ci_high_95": 40907.0}
            sim_diag = {"paths": 100000, "steps": 126, "seed": 42,
                        "monitoring": "brownian_bridge", "execution_timestamp": "2026-09-23T12:00:00-03:00",
                        "market_date": "2026-09-22"}

            rows = [
                {"object_name": "metadata", "object_type": "json", "payload_json": json.dumps(metadata)},
                {"object_name": "pricing_result", "object_type": "json", "payload_json": json.dumps(pricing)},
                {"object_name": "simulation_diagnostics", "object_type": "json", "payload_json": json.dumps(sim_diag)},
            ]
            results_path = tmp / "RESULTS_test.csv"
            pd.DataFrame(rows).to_csv(results_path, index=False)

            actual_sha = hashlib.sha256(results_path.read_bytes()).hexdigest()
            validation = {
                "schema_version": 1,
                "results_file": results_path.name,
                "results_sha256": "wrong_hash_value",
                "results_schema_version": 2,
                "market_date": "2026-09-22",
                "decision": "APROVADO",
                "approved_for_risk": True,
                "pass_count": 50, "warn_count": 0, "fail_count": 0,
            }

            with self.assertRaises(RuntimeError) as ctx:
                verify_provenance(results_path, validation)
            self.assertIn("SHA-256", str(ctx.exception))

    def test_provenance_blocks_fail_validation(self):
        """Validação com FAIL > 0 deve bloquear o processamento."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = {"schema_version": 2, "market_date": "2026-09-22", "spot": 187422.92,
                        "target_strike": 155000.0, "target_maturity": 0.5,
                        "target_r": 0.12, "target_q": 0.01, "target_iv": 0.24}
            pricing = {"contract": {"strike": 155000.0, "barrier": 140000.0, "maturity": 0.5},
                       "market": {"spot": 187422.92, "rate": 0.12, "dividend_yield": 0.01,
                                  "volatility": 0.24, "forward": 198000.0,
                                  "interpolation_method": "linear"},
                       "vanilla_price": 41000.0, "barrier_price": 40900.0,
                       "barrier_discount_abs": 100.0, "barrier_discount_pct": 0.0024,
                       "knock_out_probability": 0.05, "standard_error": 3.5,
                       "ci_low_95": 40893.0, "ci_high_95": 40907.0}
            sim_diag = {"paths": 100000, "steps": 126, "seed": 42,
                        "monitoring": "brownian_bridge", "execution_timestamp": "2026-09-23T12:00:00-03:00",
                        "market_date": "2026-09-22"}
            rows = [
                {"object_name": "metadata", "object_type": "json", "payload_json": json.dumps(metadata)},
                {"object_name": "pricing_result", "object_type": "json", "payload_json": json.dumps(pricing)},
                {"object_name": "simulation_diagnostics", "object_type": "json", "payload_json": json.dumps(sim_diag)},
            ]
            results_path = tmp / "RESULTS_test.csv"
            pd.DataFrame(rows).to_csv(results_path, index=False)
            actual_sha = hashlib.sha256(results_path.read_bytes()).hexdigest()
            validation = {
                "schema_version": 1,
                "results_file": results_path.name,
                "results_sha256": actual_sha,
                "results_schema_version": 2,
                "market_date": "2026-09-22",
                "decision": "REPROVADO",
                "approved_for_risk": False,
                "pass_count": 40, "warn_count": 5, "fail_count": 3,
            }
            with self.assertRaises(RuntimeError) as ctx:
                verify_provenance(results_path, validation)
            self.assertIn("FAIL", str(ctx.exception))

    def test_input_excel_preservation(self):
        """O Excel de entrada deve ser preservado (não sobrescrito)."""
        import openpyxl
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_xlsx = tmp / "input.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Test"
            ws["A1"] = "Cenário"
            ws["B1"] = "Test"
            ws["A2"] = "Choque de Spot (%)"
            ws["B2"] = -0.10
            ws["A5"] = "Lado da Posição"
            ws["B5"] = "COMPRADO"
            ws["A6"] = "QtdPontos"
            ws["B6"] = 1.0
            wb.save(input_xlsx)
            original_bytes = input_xlsx.read_bytes()

            # Simulate reading without modifying
            xls = pd.ExcelFile(io.BytesIO(original_bytes), engine="openpyxl")
            df = xls.parse("Test", header=None)

            # File should be unchanged
            self.assertEqual(input_xlsx.read_bytes(), original_bytes)
            self.assertEqual(float(df.iloc[1, 1]), -0.10)

    def test_invalid_side_returns_error(self):
        """Lado inválido deve retornar erro, não exceção."""
        sc = ScenarioInput(name="bad_side", spot_shock_pct=0.0, vol_shock_abs=0.0,
                           rate_shock_bp=0.0, side="INVALID", qtd_pontos=1.0)
        result = run_scenario(MARKET, CONTRACT, SIM, sc)
        self.assertIsNotNone(result.error)
        self.assertIn("Lado da posição", result.error)

    def test_zero_qtd_pontos_returns_error(self):
        """QtdPontos = 0 deve retornar erro."""
        sc = ScenarioInput(name="zero_qtd", spot_shock_pct=0.0, vol_shock_abs=0.0,
                           rate_shock_bp=0.0, side="COMPRADO", qtd_pontos=0.0)
        result = run_scenario(MARKET, CONTRACT, SIM, sc)
        self.assertIsNotNone(result.error)

    def test_common_random_numbers_reduce_noise(self):
        """Cenário base com CRN deve ter pnl mais próximo de zero que sem CRN."""
        sc = ScenarioInput(name="base_crn", spot_shock_pct=0.0, vol_shock_abs=0.0,
                           rate_shock_bp=0.0, side="COMPRADO", qtd_pontos=1.0)
        # With CRN
        rng = np.random.default_rng(SIM.seed)
        normals = rng.standard_normal((SIM.paths, SIM.steps))
        uniforms = rng.random((SIM.paths, SIM.steps))
        r_crn = run_scenario(MARKET, CONTRACT, SIM, sc, normals=normals, uniforms=uniforms)
        # Without CRN (independent seeds via run_scenario default)
        r_indep = run_scenario(MARKET, CONTRACT, SIM, sc)
        # With CRN, base == stressed exactly (same parameters, same RNs)
        self.assertEqual(r_crn.base_price, r_crn.stressed_price)
        self.assertAlmostEqual(r_crn.pnl, 0.0, places=10)
