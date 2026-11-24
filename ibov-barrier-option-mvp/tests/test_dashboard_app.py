"""O app inteiro (dashboard/app.py) no AppTest, com uma pasta de resultados sintética no lugar do Workspace.

Cobre a troca de instrumento com um clique em cada sentido (no Streamlit 1.38 do Databricks Apps o combo
perdia o clique) e a página do S&P 500 com precificação, validação e risco.
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
HAS_ST = all(importlib.util.find_spec(m) is not None for m in ("streamlit", "plotly", "scipy"))
BASE = "/Workspace/Users/teste/ibov-barrier-results"


def _ibov_bundle(path: Path):
    spot, k, h, t, r, q, vol = 166376.43, 155000.0, 140000.0, 0.5, 0.133, 0.0095, 0.21
    iv = pd.DataFrame([{"T": tt, "tenor_bd": round(tt * 252), "log_moneyness": lm, "implied_vol": 0.2 - 0.2 * lm,
                        "maturity_date": f"2026-{m:02d}-15", "option_type": "CALL", "strike": spot * math.exp(lm)}
                       for m, tt in ((12, 0.08), (1, 0.17), (2, 0.25)) for lm in np.linspace(-0.2, 0.1, 7)])
    curve = pd.DataFrame({"tenor_bd": [21.0, 63.0, 126.0, 252.0], "r_cont": [0.128, 0.130, 0.133, 0.137],
                          "q_cont": [0.02, 0.012, 0.0095, 0.009]})
    pricing = {"contract": {"strike": k, "barrier": h, "maturity": t},
               "market": {"spot": spot, "rate": r, "dividend_yield": q, "volatility": vol,
                          "forward": spot * math.exp((r - q) * t)},
               "vanilla_price": 22953.5, "barrier_price": 22639.4, "analytical_price": 22641.5,
               "analytical_error_pct": 0.0001, "barrier_discount_pct": 0.0137, "knock_out_probability": 0.163,
               "standard_error": 5.3, "ci_low_95": 22629.0, "ci_high_95": 22649.9}
    meta = {"schema_version": 2, "market_date": "2026-11-19", "spot": spot, "target_strike": k, "target_iv": vol}
    sim = {"paths": 1000, "steps": 20, "seed": 42, "monitoring": "brownian_bridge"}
    rows = [{"object_name": n, "object_type": "json", "payload_json": json.dumps(v)}
            for n, v in (("metadata", meta), ("pricing_result", pricing), ("simulation_diagnostics", sim))]
    rows += [{"object_name": n, "object_type": "dataframe", "payload_json": f.to_json(orient="table")}
             for n, f in (("curve", curve), ("iv_data", iv), ("surface_data", iv))]
    pd.DataFrame(rows).to_csv(path, index=False)


def _spx_bundle(path: Path):
    from ibov_barrier.markets.referencia import precificar_referencia
    from ibov_barrier.results_bundle import write_bundle
    spot, ts = 6952.5, [0.1, 0.3, 0.6, 1.1]
    rows = []
    for t in ts:
        f = spot * math.exp(0.038 * t)
        for lm in np.linspace(-0.4, 0.2, 25):
            rows.append({"expiration": f"2027-{int(t * 10) + 1:02d}-15", "T": t, "strike": f * math.exp(lm),
                         "option_type": "PUT" if lm < 0 else "CALL", "market_price": 1.0, "forward": f,
                         "log_moneyness": lm, "implied_vol": 0.2 - 0.3 * lm})
    options = pd.DataFrame(rows)
    forwards = pd.DataFrame({"vencimento": options.groupby("T")["expiration"].first().values, "T": ts,
                             "forward": [spot * math.exp(0.038 * t) for t in ts], "r": 0.048, "q": 0.010})
    ref = precificar_referencia(spot, options, forwards, paths=2000, steps=20, seed=42, monitoring="brownian_bridge")
    write_bundle(path, {
        "metadata": {"schema_version": 1, "market_date": date(2026, 11, 20), "spot": spot, "sofr": 0.0397,
                     "vencimentos": len(ts), "pontos": len(options), "avisos": [], "cobertura": None,
                     "faixa q": [0.0, 0.03], "fonte": "teste", "execution_timestamp": "2026-11-24T10:15:00-03:00",
                     "spot_fonte": "implícito (paridade, vencimento mais curto)", "spot_fechamento": 6910.39,
                     "spot_paridade": spot, "spot_gap_bps": 60.9, **ref["alvo"]},
        "forwards": forwards, "options": options,
        "pricing_result": ref["pricing_result"], "simulation_diagnostics": ref["simulation_diagnostics"],
    })


def _risk(path: Path, price: float):
    path.write_text(json.dumps({"metrics": {"price": price, "delta": 0.75, "gamma": -0.0007, "vega_1pct": 11.08,
                                            "var": 216.97, "expected_shortfall": 252.36,
                                            "methodology": "delta-gamma Monte Carlo with lognormal spot shocks"},
                                "validation_decision": "APROVADO COM RESSALVAS", "validation_warn_count": 2,
                                "limitations": []}), encoding="utf-8")


@unittest.skipUnless(HAS_ST, "streamlit/plotly/scipy ausentes")
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        pasta = Path(cls.tmp.name)
        _ibov_bundle(pasta / "RESULTS_superficieVol_20261124_132739.csv")
        (pasta / "VALIDATION_superficieVol_20261124_132739.json").write_text("{}", encoding="utf-8")
        _risk(pasta / "RISK_superficieVol_20261124_132739.json", 22639.4)
        _spx_bundle(pasta / "RESULTS_spx_superficie_20261124_101500.csv")
        (pasta / "VALIDATION_spx_superficie_20261124_101500.json").write_text(json.dumps(
            {"decision": "APROVADO COM RESSALVAS", "pass_count": 19, "warn_count": 2, "fail_count": 0,
             "checks": [{"bloco": "Precificação", "controle": "Replay", "status": "PASS"}]}), encoding="utf-8")
        _risk(pasta / "RISK_spx_superficie_20261124_101500.json", 714.29)

        class _WS:
            def list(self, base):
                return [SimpleNamespace(path=f"{base}/{n}") for n in sorted(os.listdir(pasta))]

            def download(self, p):
                return open(pasta / p.split("/")[-1], "rb")

        class _Client:
            def __init__(self, *a, **k):
                self.workspace = _WS()
                self.jobs = SimpleNamespace(list=lambda **k: iter(()))

        # databricks-sdk não precisa estar instalado: o app só usa o WorkspaceClient
        cls._mods = {m: sys.modules.get(m) for m in ("databricks", "databricks.sdk")}
        pkg, sdk = types.ModuleType("databricks"), types.ModuleType("databricks.sdk")
        sdk.WorkspaceClient, pkg.sdk = _Client, sdk
        sys.modules["databricks"], sys.modules["databricks.sdk"] = pkg, sdk
        os.environ["RESULTS_BASE_PATH"] = BASE
        cls._cwd = os.getcwd()
        os.chdir(DASHBOARD)
        sys.path.insert(0, str(DASHBOARD))

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        sys.path.remove(str(DASHBOARD))
        for m, mod in cls._mods.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod
        os.environ.pop("RESULTS_BASE_PATH", None)
        cls.tmp.cleanup()

    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=300).run()

    def test_troca_de_instrumento_com_um_clique(self):
        at = self.app()
        self.assertFalse(at.exception)
        for alvo, aba in [("option_barrier_spx", "Superfície e Smiles"), ("option_barrier", "Superfície e Volatilidade"),
                          ("option_barrier_spx", "Superfície e Smiles"), ("option_barrier", "Superfície e Volatilidade")]:
            at.sidebar.selectbox(key="instrumento").select(alvo).run()
            self.assertFalse(at.exception)
            self.assertEqual(at.sidebar.selectbox(key="instrumento").value, alvo)
            self.assertEqual(at.tabs[1].label, aba)

    def test_pagina_do_spx_com_preco_validacao_e_risco(self):
        at = self.app()
        at.sidebar.selectbox(key="instrumento").select("option_barrier_spx").run()
        self.assertFalse(at.exception)
        self.assertEqual([t.label for t in at.tabs][:5], ["Resumo do Gestor", "Superfície e Smiles", "Taxa e Dividendos",
                                                          "Motor de Precificação", "Controle e Ingestão"])
        metricas = {m.label: m.value for m in at.metric}
        self.assertEqual(metricas["Spot SPX"], "6.952,50")
        self.assertIn("Preço Down-and-Out", metricas)
        self.assertEqual(metricas["VaR 99% (1d)"], "216,97")
        self.assertEqual(metricas["Decisão"], "APROVADO COM RESSALVAS")
        legendas = " ".join(c.value for c in at.caption)
        self.assertIn("Monte Carlo delta-gamma com choques lognormais no spot", legendas)  # item 5
        self.assertIn("executado em 24/11/2026 10:15:00", legendas)  # item 1
        self.assertIn("+60,9 bps", legendas)
        self.assertEqual(at.sidebar.selectbox(key="spx_file").options, ["24/11/2026 10:15:00 · completo"])  # item 2
        self.assertFalse(at.error)

    def test_ibov_rotulo_e_historico(self):
        at = self.app()
        self.assertFalse(at.exception)
        rotulos = [s.options for s in at.sidebar.selectbox if s.label == "Arquivo RESULTS"][0]
        self.assertEqual(rotulos, ["24/11/2026 13:27:39 · completo"])
        metricas = {m.label: m.value for m in at.metric}
        self.assertEqual(metricas["Spot IBOV"], "166.376,43")
        self.assertEqual(metricas["Preço (MC)"], "22.639,40")


if __name__ == "__main__":
    unittest.main()
