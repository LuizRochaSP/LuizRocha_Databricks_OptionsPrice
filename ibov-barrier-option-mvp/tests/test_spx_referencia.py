"""Precificação de referência do SPX (notebook 07), cadeia do dia em disco e validação (notebook 11)."""
import importlib.util
import json
import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
if HAS_SCIPY:
    from scipy.interpolate import griddata

    from ibov_barrier.analytical import price_down_and_out_call_analytical
    from ibov_barrier.markets.referencia import (FORA_DO_CASCO, curvas_no_prazo, precificar_referencia,
                                                 salvar_cadeia_do_dia, vol_no_ponto)
    from ibov_barrier.markets.validacao_spx import decisao, evidencia, ordenar, validar_spx
    from ibov_barrier.results_bundle import read_bundle, write_bundle

SPOT = 6952.50
# escala do arquivo real (45 vencimentos, ~7.700 pontos): 24 vencimentos x 230 strikes
T_VENC = list(np.round(np.linspace(0.05, 1.6, 24), 6))
R = list(np.interp(T_VENC, [0.05, 0.25, 0.5, 1.0, 1.6], [0.0469, 0.0469, 0.0476, 0.0490, 0.0507]))
Q = list(np.interp(T_VENC, [0.05, 0.5, 1.6], [0.012, 0.0095, 0.0110]))
MC = dict(paths=20_000, steps=60, seed=42, monitoring="brownian_bridge")


def mercado():
    rows = []
    for t, r, q in zip(T_VENC, R, Q):
        f = SPOT * math.exp((r - q) * t)
        for k in np.linspace(-0.45, 0.2, 230):
            rows.append({"expiration": f"E{t}", "T": t, "strike": f * math.exp(k), "option_type": "PUT" if k < 0 else "CALL",
                         "market_price": 1.0, "forward": f, "log_moneyness": k,
                         "implied_vol": 0.19 - 0.30 * k + 0.25 * k * k + 0.01 * t})
    options = pd.DataFrame(rows)
    forwards = pd.DataFrame({"vencimento": [f"E{t}" for t in T_VENC], "T": T_VENC, "r": R, "q": Q,
                             "forward": [SPOT * math.exp((r - q) * t) for t, r, q in zip(T_VENC, R, Q)]})
    return options, forwards


def bundle(gap_bps=1.5, **kw):
    """RESULTS como o 07 grava, relido do disco (o 11 lê assim)."""
    options, forwards = mercado()
    ref = precificar_referencia(SPOT, options, forwards, **MC, **kw)
    arb = pd.DataFrame({"expiration": [f"E{T_VENC[5]}"], "option_type": ["CALL"], "antes": [100], "removidos": [8]})
    objetos = {
        "metadata": {"schema_version": 1, "market_date": date(2026, 11, 23), "spot": SPOT, "sofr": 0.0397,
                     "avisos": [], "cobertura": None, "faixa q": [0.0, 0.03], "T minimo taxa": 0.25,
                     "spot_fonte": "implícito (paridade, vencimento mais curto)", "spot_gap_bps": gap_bps,
                     **ref["alvo"]},
        "forwards": forwards, "options": options, "arb": arb,
        "pricing_result": ref["pricing_result"], "simulation_diagnostics": ref["simulation_diagnostics"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        return read_bundle(write_bundle(Path(tmp) / "RESULTS_spx_superficie_20261124_101500.csv", objetos))


@unittest.skipUnless(HAS_SCIPY, "scipy ausente")
class ReferenciaTests(unittest.TestCase):
    def test_vol_regra_do_notebook_01(self):
        options, _ = mercado()
        sd = options.drop_duplicates(["T", "log_moneyness"])
        pts, vals = sd[["log_moneyness", "T"]].to_numpy(), sd["implied_vol"].to_numpy()
        f = SPOT * 1.02
        v, metodo = vol_no_ponto(options, SPOT * 0.95, 0.7, f)
        self.assertEqual(metodo, "linear")
        self.assertAlmostEqual(v, griddata(pts, vals, np.array([[math.log(0.95 * SPOT / f), 0.7]]))[0], places=12)
        v, metodo = vol_no_ponto(options, SPOT * 0.95, 3.0, f)  # além do último vencimento
        self.assertEqual(metodo, FORA_DO_CASCO)
        self.assertAlmostEqual(v, griddata(pts, vals, np.array([[math.log(0.95 * SPOT / f), 3.0]]),
                                           method="nearest")[0], places=12)

    def test_curvas_lineares_em_T(self):
        _, forwards = mercado()
        r, q = curvas_no_prazo(forwards, 0.75)
        self.assertAlmostEqual(r, np.interp(0.75, T_VENC, R), places=14)
        self.assertAlmostEqual(q, np.interp(0.75, T_VENC, Q), places=14)

    def test_referencia_tem_as_chaves_do_01(self):
        options, forwards = mercado()
        ref = precificar_referencia(SPOT, options, forwards, **MC)
        p, a = ref["pricing_result"], ref["alvo"]
        self.assertEqual(set(p), {"contract", "market", "vanilla_price", "barrier_price", "analytical_price",
                                  "analytical_error_pct", "barrier_discount_abs", "barrier_discount_pct",
                                  "knock_out_probability", "standard_error", "ci_low_95", "ci_high_95"})
        self.assertAlmostEqual(p["contract"]["strike"], SPOT * 0.95)
        self.assertAlmostEqual(p["contract"]["barrier"], SPOT * 0.85)
        self.assertEqual(ref["simulation_diagnostics"], {**MC})
        rr = price_down_and_out_call_analytical(SPOT, SPOT * 0.95, SPOT * 0.85, 0.5, a["target_r"], a["target_q"],
                                                a["target_iv"])
        self.assertAlmostEqual(p["analytical_price"], rr, places=8)
        self.assertLess(p["analytical_error_pct"], 0.02)
        self.assertLessEqual(p["barrier_price"], p["vanilla_price"])
        self.assertAlmostEqual(p["market"]["forward"], SPOT * math.exp((a["target_r"] - a["target_q"]) * 0.5))
        with self.assertRaises(ValueError):
            precificar_referencia(SPOT, options, forwards, barrier_pct=1.05, **MC)

    def test_cadeia_do_dia_no_formato_do_08(self):
        chains = {"SPX": pd.DataFrame({"strike": [7000.0]}), "SPXW": pd.DataFrame({"strike": [7000.0, 7100.0]})}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(salvar_cadeia_do_dia(tmp, date(2026, 11, 23), chains, 6910.39, 0.0397), "gravado")
            pasta = Path(tmp) / "2026-11-23"
            meta = json.loads((pasta / "meta.json").read_text())
            self.assertEqual((meta["spot"], meta["linhas"], meta["sofr"]), (6910.39, {"SPX": 1, "SPXW": 2}, 0.0397))
            self.assertEqual(len(pd.read_parquet(pasta / "SPXW.parquet")), 2)
            self.assertEqual(salvar_cadeia_do_dia(tmp, date(2026, 11, 23), chains, 1.0, None), "em disco")


@unittest.skipUnless(HAS_SCIPY, "scipy ausente")
class ValidacaoSpxTests(unittest.TestCase):
    def test_resultado_bom_passa(self):
        checks = validar_spx(bundle(), d1_esperado=date(2026, 11, 23))
        dec, n_pass, n_warn, n_fail = decisao(checks)
        self.assertEqual(n_fail, 0, checks.loc[checks["status"] != "PASS"].to_string())
        self.assertIn(dec, ("APROVADO", "APROVADO COM RESSALVAS"))
        self.assertEqual(checks.loc[checks["controle"].eq("Replay determinístico do Monte Carlo"), "status"].iloc[0],
                         "PASS")

    def test_avisos_viram_ressalvas(self):
        checks = validar_spx(bundle(gap_bps=61.0), d1_esperado=date(2026, 11, 24))
        st = dict(zip(checks["controle"], checks["status"]))
        self.assertEqual(st["Spot: fechamento x paridade"], "WARN")
        self.assertEqual(st["Data de mercado"], "WARN")
        self.assertEqual(decisao(checks)[0], "APROVADO COM RESSALVAS")

    def test_preco_adulterado_reprova(self):
        b = bundle()
        b["pricing_result"]["barrier_price"] *= 1.001
        st = dict(zip(*validar_spx(b)[["controle", "status"]].T.values))
        self.assertEqual(st["Replay determinístico do Monte Carlo"], "FAIL")
        b = bundle()
        b["pricing_result"]["market"]["volatility"] += 0.01
        checks = validar_spx(b, replay=False)
        self.assertEqual(checks.loc[checks["controle"].eq("Vol no strike reproduzida"), "status"].iloc[0], "FAIL")
        self.assertEqual(decisao(checks)[0], "REPROVADO")

    def test_arquivo_antigo_sem_precificacao_reprova(self):
        b = bundle()
        del b["pricing_result"]
        checks = validar_spx(b)
        self.assertEqual(len(checks), 1)
        self.assertEqual(decisao(checks)[0], "REPROVADO")

    def test_evidencia_no_formato_do_02(self):
        b = bundle(gap_bps=61.0)
        checks = ordenar(validar_spx(b, replay=False))
        self.assertEqual(checks["status"].iloc[0], "WARN")  # FAIL, WARN, PASS
        ev = evidencia(checks, results_file="RESULTS_spx_superficie_20261124_101500.csv", results_sha256="ab",
                       metadata=b["metadata"], validation_timestamp="2026-11-24T10:20:00-03:00")
        for k in ("schema_version", "results_file", "results_sha256", "market_date", "decision", "approved_for_risk",
                  "warn_count", "fail_count", "warnings"):
            self.assertIn(k, ev)
        self.assertEqual((ev["schema_version"], ev["fail_count"], ev["approved_for_risk"]), (1, 0, True))
        self.assertEqual(ev["market_date"], "2026-11-23")
        self.assertEqual(len(ev["warnings"]), ev["warn_count"])
        json.dumps(ev)  # serializável


if __name__ == "__main__":
    unittest.main()
