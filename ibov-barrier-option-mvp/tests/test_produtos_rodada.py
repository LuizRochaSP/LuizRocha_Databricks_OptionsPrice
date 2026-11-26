"""Testes do leitor da rodada aprovada e do estado de mercado do IBOV (etapa A5 do plano).

- A rodada: a validação aprovada mais recente, com as exigências do notebook 03 (schema, FAIL = 0, decisão,
  SHA-256 do RESULTS, data de mercado igual nos dois arquivos); evidências não aprovadas são puladas, e falhas
  de integridade param.
- O estado: r(T) flat-forward nos vértices da curva DI, q(T) linear em DU, vol linear em (ln(K/F), T) com vizinho
  mais próximo fora do casco (a regra do 01, conferida contra o `griddata`), superfície sticky strike.
- De ponta a ponta: o call spread do COE sobre o estado bate com Black-Scholes escrito no teste, com a vol de
  cada strike lida da superfície.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HAS_DEPS = all(importlib.util.find_spec(m) is not None for m in ("scipy", "exchange_calendars"))

if HAS_DEPS:
    from scipy.interpolate import griddata
    from scipy.stats import norm

    from ibov_barrier.coe import COE, Participacao
    from ibov_barrier.produtos import gregas
    from ibov_barrier.produtos.rodada import (
        SuperficieVol,
        conferir_referencia,
        estado_ibov,
        ultima_rodada_aprovada,
    )
    from ibov_barrier.results_bundle import write_bundle

SPOT = 160_000.0
TENORES = np.array([21.0, 63.0, 126.0, 252.0, 504.0])
R_CONT = np.array([0.1420, 0.1400, 0.1370, 0.1330, 0.1300])
Q_CONT = np.array([0.030, 0.025, 0.022, 0.020, 0.019])
T_SUP = [0.1, 0.25, 0.5, 1.0]
K_SUP = np.linspace(-0.35, 0.25, 13)


def _r_flat_forward(du: float) -> float:
    """r contínua com r·DU linear entre vértices e o forward do último segmento depois dele (conta à mão)."""
    i = 0 if du <= TENORES[0] else len(TENORES) - 2 if du >= TENORES[-1] else int(np.searchsorted(TENORES, du) - 1)
    a, b = TENORES[i], TENORES[i + 1]
    ra, rb = R_CONT[i] * a, R_CONT[i + 1] * b
    return (ra + (rb - ra) * (du - a) / (b - a)) / du


def _q(du: float) -> float:
    return float(np.interp(du, TENORES, Q_CONT))


def _superficie() -> pd.DataFrame:
    linhas = []
    for t in T_SUP:
        f = SPOT * math.exp((_r_flat_forward(t * 252) - _q(t * 252)) * t)
        for k in K_SUP:
            linhas.append({"T": t, "log_moneyness": k, "strike": f * math.exp(k),
                           "implied_vol": 0.21 - 0.20 * k + 0.10 * k * k + 0.01 * t})
    return pd.DataFrame(linhas)


def _vol_01(strike: float, t: float) -> float:
    """A regra do notebook 01: griddata linear em (ln(K/F), T) e nearest fora do casco."""
    sup = _superficie()
    pts, vals = sup[["log_moneyness", "T"]].to_numpy(), sup["implied_vol"].to_numpy()
    k = math.log(strike / (SPOT * math.exp((_r_flat_forward(t * 252) - _q(t * 252)) * t)))
    v = griddata(pts, vals, np.array([[k, t]]), method="linear")[0]
    return float(v if np.isfinite(v) else griddata(pts, vals, np.array([[k, t]]), method="nearest")[0])


def _metadata(market_date="2026-11-24", schema=2) -> dict:
    t, k = 0.5, 0.95 * SPOT
    return {"schema_version": schema, "market_date": market_date, "spot": SPOT, "target_strike": k,
            "target_maturity": t, "target_r": _r_flat_forward(126.0), "target_q": _q(126.0),
            "target_iv": _vol_01(k, t)}


def _curva() -> pd.DataFrame:
    return pd.DataFrame({"tenor_bd": TENORES, "r_cont": R_CONT, "q_cont": Q_CONT})


def _gravar_rodada(pasta: Path, carimbo: str, *, decisao="APROVADO", fail=0, schema=2, data_validacao=None,
                   market_date="2026-11-24", sha=None) -> Path:
    results = pasta / f"RESULTS_superficieVol_{carimbo}.csv"
    write_bundle(results, {"metadata": _metadata(market_date, schema), "curve": _curva(),
                           "surface_data": _superficie()})
    validacao = {"schema_version": 1, "decision": decisao, "approved_for_risk": fail == 0, "fail_count": fail,
                 "warn_count": 2, "results_file": results.name, "market_date": data_validacao or market_date,
                 "results_sha256": sha or hashlib.sha256(results.read_bytes()).hexdigest(),
                 "warnings": [{"check": "liquidez", "status": "WARN"}]}
    caminho = pasta / f"VALIDATION_superficieVol_{carimbo}.json"
    caminho.write_text(json.dumps(validacao), encoding="utf-8")
    return caminho


def _bs_call(s, k, t, r, q, vol) -> float:
    d1 = (math.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d1 - vol * math.sqrt(t))


@unittest.skipUnless(HAS_DEPS, "scipy ou exchange_calendars ausente")
class TestRodadaAprovada(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_mais_recente_aprovada(self):
        _gravar_rodada(self.pasta, "20261124_190000")
        _gravar_rodada(self.pasta, "20261125_190000", decisao="APROVADO COM RESSALVAS")
        rodada = ultima_rodada_aprovada(self.pasta)
        self.assertEqual(rodada.validacao_path.name, "VALIDATION_superficieVol_20261125_190000.json")
        self.assertEqual(rodada.results_path.name, "RESULTS_superficieVol_20261125_190000.csv")
        self.assertEqual(rodada.decisao, "APROVADO COM RESSALVAS")
        self.assertEqual(rodada.warn_count, 2)
        self.assertEqual(len(rodada.ressalvas), 1)
        self.assertEqual(rodada.data_mercado.isoformat(), "2026-11-24")
        self.assertEqual(rodada.sha256, hashlib.sha256(rodada.results_path.read_bytes()).hexdigest())
        self.assertEqual(set(rodada.proveniencia()), {"validation_file", "results_file", "results_sha256",
                                                      "market_date", "validation_decision",
                                                      "validation_warn_count"})

    def test_evidencia_nao_aprovada_e_pulada(self):
        _gravar_rodada(self.pasta, "20261124_190000")
        _gravar_rodada(self.pasta, "20261125_190000", decisao="REPROVADO", fail=1)
        rodada = ultima_rodada_aprovada(self.pasta)
        self.assertEqual(rodada.validacao_path.name, "VALIDATION_superficieVol_20261124_190000.json")
        self.assertEqual(len(rodada.ignoradas), 1)
        self.assertIn("20261125_190000", rodada.ignoradas[0])

    def test_outro_prefixo_nao_entra(self):
        _gravar_rodada(self.pasta, "20261124_190000")
        (self.pasta / "VALIDATION_spx_superficie_20261126_100000.json").write_text("{}", encoding="utf-8")
        self.assertEqual(ultima_rodada_aprovada(self.pasta).validacao_path.name,
                         "VALIDATION_superficieVol_20261124_190000.json")

    def test_backfill_por_arquivo(self):
        _gravar_rodada(self.pasta, "20261124_190000")
        _gravar_rodada(self.pasta, "20261125_190000", decisao="REPROVADO", fail=1)
        rodada = ultima_rodada_aprovada(self.pasta, arquivo_validacao="VALIDATION_superficieVol_20261124_190000.json")
        self.assertEqual(rodada.results_path.name, "RESULTS_superficieVol_20261124_190000.csv")
        with self.assertRaisesRegex(RuntimeError, "não está aprovada"):
            ultima_rodada_aprovada(self.pasta, arquivo_validacao="VALIDATION_superficieVol_20261125_190000.json")

    def test_falhas_de_integridade_param(self):
        casos = {
            "SHA-256": dict(sha="0" * 64),
            "Data de mercado divergente": dict(data_validacao="2026-11-21"),
            "schema_version": dict(schema=1),
        }
        for mensagem, campos in casos.items():
            with self.subTest(mensagem), tempfile.TemporaryDirectory() as tmp:
                _gravar_rodada(Path(tmp), "20261124_190000")
                _gravar_rodada(Path(tmp), "20261125_190000", **campos)
                with self.assertRaisesRegex(RuntimeError, mensagem):
                    ultima_rodada_aprovada(tmp)

    def test_results_ausente_e_pasta_vazia(self):
        _gravar_rodada(self.pasta, "20261124_190000").parent.joinpath(
            "RESULTS_superficieVol_20261124_190000.csv").unlink()
        with self.assertRaisesRegex(RuntimeError, "não encontrado"):
            ultima_rodada_aprovada(self.pasta)
        with tempfile.TemporaryDirectory() as vazia, self.assertRaisesRegex(RuntimeError, "Nenhuma validação"):
            ultima_rodada_aprovada(vazia)


@unittest.skipUnless(HAS_DEPS, "scipy ou exchange_calendars ausente")
class TestEstadoIbov(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        _gravar_rodada(Path(cls._tmp.name), "20261124_190000")
        cls.rodada = ultima_rodada_aprovada(cls._tmp.name)
        cls.e = estado_ibov(cls.rodada, spread_emissor_bp=120.0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_curva_flat_forward_da_rodada(self):
        curva = self.e.curva("BRL")
        for du, r in zip(TENORES, R_CONT):
            self.assertAlmostEqual(curva.taxa_em(du / 252), r, places=12)
        for du in (40.0, 100.0, 300.0, 756.0):  # entre vértices e depois do último
            self.assertAlmostEqual(curva.taxa_em(du / 252), _r_flat_forward(du), places=12)
        self.assertEqual(curva.vertices, tuple(TENORES))

    def test_dividendos_lineares_em_du(self):
        for du in (10.0, 100.0, 400.0, 900.0):
            self.assertAlmostEqual(self.e.ativo("IBOV").dividendos_em(du / 252), _q(du), places=14)

    def test_vol_pela_regra_do_notebook_01(self):
        superficie = self.e.ativo("IBOV").vol
        self.assertIsInstance(superficie, SuperficieVol)
        for strike, t, regra in ((150_000.0, 0.3, "linear"), (180_000.0, 0.8, "linear"),
                                 (90_000.0, 0.5, "vizinho mais próximo"), (165_000.0, 1.5, "vizinho mais próximo")):
            with self.subTest(strike=strike, t=t):
                vol, obtida = superficie.com_regra(strike, t)
                self.assertAlmostEqual(vol, _vol_01(strike, t), places=12)
                self.assertEqual(obtida, regra)
        self.assertEqual((superficie.prazo_minimo, superficie.prazo_maximo), (0.1, 1.0))

    def test_sticky_strike(self):
        chocado = self.e.com_spot("IBOV", 1.05 * SPOT)
        self.assertEqual(chocado.ativo("IBOV").vol_em(160_000.0, 0.5), self.e.ativo("IBOV").vol_em(160_000.0, 0.5))

    def test_conferencia_com_o_01(self):
        linhas = {l["grandeza"]: l for l in conferir_referencia(self.e, self.rodada.metadata)}
        self.assertEqual(set(linhas), {"r(T)", "q(T)", "vol no strike"})
        for linha in linhas.values():
            self.assertAlmostEqual(linha["diferenca"], 0.0, places=12)

    def test_spread_e_data(self):
        self.assertEqual(float(self.e.spread_emissor_bp), 120.0)
        self.assertEqual(self.e.data.isoformat(), "2026-11-24")

    def test_call_spread_do_coe_com_a_vol_de_cada_strike(self):
        ts = Participacao(ativo="IBOV", nivel_inicial=SPOT, vencimento=0.75, nominal=1_000_000.0,
                          participacao=1.0, cap=0.2)
        coe = COE(ts)
        t = 0.75
        r, q = _r_flat_forward(t * 252), _q(t * 252)
        k1, k2 = SPOT, 1.2 * SPOT
        esperado = ts.quantidade * (_bs_call(SPOT, k1, t, r, q, _vol_01(k1, t))
                                    - _bs_call(SPOT, k2, t, r, q, _vol_01(k2, t)))
        self.assertAlmostEqual(float(coe.opcoes().valor(self.e)), esperado, delta=1e-6)
        d = coe.decompor(self.e, custo_distribuicao=0.03)
        self.assertAlmostEqual(d.opcoes, esperado, delta=1e-6)
        g = gregas(coe, self.e)
        self.assertIsNotNone(g.cs01)
        self.assertGreater(g.delta["IBOV"], 0.0)

    def test_objeto_ausente(self):
        objetos = dict(self.rodada.objetos)
        del objetos["surface_data"]
        with self.assertRaisesRegex(ValueError, "surface_data"):
            estado_ibov(objetos)

    def test_superficie_pequena_e_recusada(self):
        with self.assertRaises(ValueError):
            SuperficieVol(spot=SPOT, curva=self.e.curva("BRL"), dividendos=lambda t: 0.0,
                          pontos=np.array([[0.0, 0.5], [0.1, 0.5]]), vols=np.array([0.2, 0.2]))


if __name__ == "__main__":
    unittest.main()
