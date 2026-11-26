"""Testes da estruturação do notebook 12 (etapa A5 do plano) e da estrutura do notebook.

- `estruturar`: a decomposição é a do A3, o solver é o do A4, as gregas são do COE final (o resolvido, quando há)
  e somam as das pernas; o DV01 por vértice soma o paralelo; vols e avisos (prazo além da superfície e da curva,
  vizinho mais próximo, margem inviável).
- `texto` e `como_dict`: o relatório abre e fecha como os dos outros notebooks, e o JSON é serializável.
- O notebook 12: compila, instala as bibliotecas na primeira célula (com o exchange-calendars do projeto) e
  reinicia o Python, no ambiente 6.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from math import log
from pathlib import Path

from tests.test_produtos_rodada import HAS_DEPS, SPOT, _gravar_rodada

if HAS_DEPS:
    from ibov_barrier.coe import COE, Participacao, estruturar, percentual_cdi_de_spread
    from ibov_barrier.produtos import CurvaPlana, EstadoMercado, MercadoAtivo, estado_ibov, ultima_rodada_aprovada

NOTEBOOK = Path(__file__).resolve().parents[1] / "databricks" / "rotinas" / "12_coe_estruturacao.py"
CUSTOS = dict(custo_distribuicao=0.03, custo_operacional=0.005, reservas=0.005)


def _ts(**campos):
    base = dict(ativo="IBOV", nivel_inicial=SPOT, vencimento=0.75, nominal=1_000_000.0, participacao=1.0, cap=0.2)
    base.update(campos)
    return Participacao(**base)


@unittest.skipUnless(HAS_DEPS, "scipy ou exchange_calendars ausente")
class TestEstruturar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        _gravar_rodada(Path(cls._tmp.name), "20261124_190000")
        cls.rodada = ultima_rodada_aprovada(cls._tmp.name)
        cls.e = estado_ibov(cls.rodada, spread_emissor_bp=100.0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_sem_solver(self):
        ts = _ts()
        est = estruturar(ts, self.e, **CUSTOS)
        self.assertIsNone(est.solucao)
        self.assertIs(est.termsheet_final, ts)
        self.assertEqual(est.decomposicao, COE(ts).decompor(self.e, **CUSTOS))
        self.assertEqual(est.decomposicao_final, est.decomposicao)
        self.assertEqual(set(est.gregas), {"COE", "Título", "Opções"})
        self.assertAlmostEqual(est.gregas["COE"].valor, float(COE(ts).valor(self.e)), places=6)

    def test_gregas_do_coe_somam_as_das_pernas(self):
        est = estruturar(_ts(), self.e, **CUSTOS)
        coe, titulo, opcoes = (est.gregas[p] for p in ("COE", "Título", "Opções"))
        self.assertAlmostEqual(coe.delta["IBOV"], opcoes.delta["IBOV"], places=9)
        self.assertAlmostEqual(coe.vega["IBOV"], opcoes.vega["IBOV"], places=6)
        self.assertAlmostEqual(coe.dv01["BRL"], titulo.dv01["BRL"] + opcoes.dv01["BRL"], places=6)
        self.assertAlmostEqual(coe.cs01, titulo.cs01, places=6)
        self.assertAlmostEqual(coe.theta, titulo.theta + opcoes.theta, places=6)
        self.assertAlmostEqual(sum(est.dv01_vertices.values()), coe.dv01["BRL"], places=6)

    def test_com_solver_as_gregas_sao_do_resolvido(self):
        est = estruturar(_ts(), self.e, parametro="participacao", margem_alvo=0.02, **CUSTOS)
        self.assertTrue(est.solucao.viavel)
        self.assertEqual(est.termsheet_final, est.solucao.termsheet)
        self.assertAlmostEqual(est.decomposicao_final.margem, 20_000.0, delta=1e-4)
        self.assertAlmostEqual(est.gregas["COE"].valor, float(COE(est.termsheet_final).valor(self.e)), places=6)
        self.assertNotAlmostEqual(est.termsheet_final.participacao, 1.0, places=3)

    def test_inviavel_avisa_e_fica_no_digitado(self):
        est = estruturar(_ts(), self.e, parametro="cap", margem_alvo=0.30, **CUSTOS)
        self.assertEqual(est.solucao.situacao, "inviavel")
        self.assertEqual(est.termsheet_final, _ts())
        self.assertTrue(any(a.startswith("Solver: Margem de 30% inviável") for a in est.avisos))

    def test_mercado_taxas_e_vols(self):
        ts = _ts()
        est = estruturar(ts, self.e, **CUSTOS)
        curva = self.e.curva("BRL")
        self.assertAlmostEqual(est.taxa_pre, curva.fator_desconto(0.75) ** (-1 / 0.75) - 1.0, places=12)
        self.assertAlmostEqual(est.percentual_cdi, percentual_cdi_de_spread(curva, 100.0, 0.75), places=9)
        superficie = self.e.ativo("IBOV").vol
        self.assertEqual([v.rotulo for v in est.vols], ["strike", "cap"])
        for v, nivel in zip(est.vols, (SPOT, 1.2 * SPOT)):
            self.assertAlmostEqual(v.nivel, nivel)
            self.assertAlmostEqual(v.vol, superficie(nivel, 0.75), places=12)
            self.assertEqual(v.regra, "linear")
        self.assertEqual(est.avisos, ())

    def test_avisos_de_prazo_e_de_vizinho(self):
        alem = estruturar(_ts(vencimento=1.5), self.e, **CUSTOS)
        self.assertTrue(any("último vencimento da superfície" in a for a in alem.avisos))
        self.assertFalse(any(a.startswith("Vol do") for a in alem.avisos))  # já explicado pelo prazo
        curva = estruturar(_ts(vencimento=600 / 252), self.e, **CUSTOS)
        self.assertTrue(any("último vértice DI (504 DU)" in a for a in curva.avisos))
        fora = estruturar(_ts(strike=0.6, cap=None, vencimento=0.5), self.e, **CUSTOS)
        self.assertEqual(fora.vols[0].regra, "vizinho mais próximo")
        self.assertTrue(any(a.startswith("Vol do strike (60%)") for a in fora.avisos))
        com_rodada = estruturar(_ts(), self.e, avisos=["rodada defasada"], **CUSTOS)
        self.assertEqual(com_rodada.avisos[0], "rodada defasada")

    def test_parametro_e_margem_juntos(self):
        with self.assertRaises(ValueError):
            estruturar(_ts(), self.e, parametro="cap")
        with self.assertRaises(ValueError):
            estruturar(_ts(), self.e, margem_alvo=0.02)

    def test_texto(self):
        est = estruturar(_ts(), self.e, parametro="cap", margem_alvo=0.025, **CUSTOS)
        texto = est.texto(proveniencia=self.rodada.proveniencia(), agora=datetime(2026, 11, 26, 15, 30))
        linhas = texto.splitlines()
        self.assertEqual(linhas[0], "RELATÓRIO COE ESTRUTURAÇÃO | 2026-11-26 15:30 (Brasília)")
        self.assertEqual(linhas[-1], "FIM DO RELATÓRIO")
        for trecho in ("Rodada: VALIDATION_superficieVol_20261124_190000.json", "spread do emissor 100 bp",
                       "Preço de emissão", "Título (proteção)", "Margem", "Solver: cap para margem de 2,5%",
                       "Termsheet resolvido:", "Delta:", "CS01", "Theta:", "Avisos:", "(nenhum)"):
            self.assertIn(trecho, texto)
        self.assertNotIn("nan", texto.lower())

    def test_como_dict_serializavel(self):
        est = estruturar(_ts(), self.e, parametro="participacao", margem_alvo=0.02, **CUSTOS)
        dados = json.loads(json.dumps(est.como_dict(), allow_nan=False))
        self.assertEqual(dados["termsheet"]["tipo"], "Participacao")
        self.assertEqual(dados["solver"]["situacao"], "resolvido")
        self.assertAlmostEqual(dados["decomposicao_final"]["margem_pct"], 0.02, places=10)
        self.assertEqual(set(dados["gregas"]), {"COE", "Título", "Opções"})
        self.assertEqual(dados["market_date"], "2026-11-24")

    def test_mercado_plano_sem_superficie_nem_vertices(self):
        e = EstadoMercado(ativos={"IBOV": MercadoAtivo(spot=SPOT, vol=0.22, dividendos=0.03)},
                          curvas={"BRL": CurvaPlana(log(1.14))}, spread_emissor_bp=80.0)
        est = estruturar(_ts(), e, **CUSTOS)
        self.assertEqual(est.dv01_vertices, {})
        self.assertEqual({v.regra for v in est.vols}, {"plana"})
        self.assertIn("FIM DO RELATÓRIO", est.texto())


class TestNotebook12(unittest.TestCase):
    def setUp(self):
        self.fonte = NOTEBOOK.read_text(encoding="utf-8")

    def test_compila(self):
        compile(self.fonte, str(NOTEBOOK), "exec")

    def test_cabecalho_pip_e_restart(self):
        celulas = self.fonte.split("# COMMAND ----------")
        self.assertIn('environment_version = "6"', celulas[0])
        self.assertTrue(celulas[1].strip().startswith("# MAGIC %pip install"))
        for lib in ('"exchange-calendars==4.13.2"', '"lxml>=6.0"', "scipy", "numpy", "pandas"):
            self.assertIn(lib, celulas[1])
        self.assertEqual(celulas[2].strip(), "dbutils.library.restartPython()")

    def test_relatorio_e_gravacao_atomica(self):
        self.assertIn("gravar_atomico(", self.fonte)
        self.assertIn('ZoneInfo("America/Sao_Paulo")', self.fonte)
        self.assertIn("print(est.texto(", self.fonte.split("# COMMAND ----------")[-1])


if __name__ == "__main__":
    unittest.main()
