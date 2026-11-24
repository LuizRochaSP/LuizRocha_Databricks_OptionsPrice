"""Dashboard: execução completa (RESULTS -> VALIDATION -> RISK), disparo do job e números no padrão brasileiro."""
import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.append(str(DASHBOARD))  # no fim: dashboard/ibov_barrier não pode esconder src/
HAS_ST = importlib.util.find_spec("streamlit") is not None and importlib.util.find_spec("plotly") is not None

import pipeline  # noqa: E402

NOMES = [
    "RESULTS_superficieVol_20261123_081830.csv",  # só o 01 (o caso da tela)
    "RESULTS_superficieVol_20261123_080221.csv",
    "RESULTS_superficieVol_20261122_190000.csv",
    "VALIDATION_superficieVol_20261122_190000.json",
    "RISK_superficieVol_20261122_190000.json",
    "RESULTS_superficieVol_20261121_190000.csv",
    "VALIDATION_superficieVol_20261121_190000.json",  # validado, 03 não rodou
    "RESULTS_quanto_20261122_214200.csv",  # outros arquivos da pasta são ignorados
    "VALIDATION_superficieVol_20261101_000000.json",  # validação sem RESULTS
]


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.st = pipeline.status_execucoes(NOMES)

    def test_linhas_e_ordem(self):
        self.assertEqual(self.st["ts"].tolist(),
                         ["20261123_081830", "20261123_080221", "20261122_190000", "20261121_190000"])
        self.assertEqual(self.st["falta"].tolist(), ["02", "02", None, "03"])

    def test_abre_na_completa_mais_recente(self):
        self.assertEqual(self.st["ts"].iloc[pipeline.indice_padrao(self.st)], "20261122_190000")
        nenhuma = pipeline.status_execucoes(NOMES[:2])
        self.assertEqual(pipeline.indice_padrao(nenhuma), 0)

    def test_rotulos_e_motivos(self):
        self.assertEqual(pipeline.rotulo("20261122_190000", self.st), "20261122_190000 · completo")
        self.assertEqual(pipeline.rotulo("20261123_081830", self.st), "20261123_081830 · não validado")
        self.assertEqual(pipeline.rotulo("20261121_190000", self.st), "20261121_190000 · sem risco")
        self.assertIsNone(pipeline.motivo("20261122_190000", self.st))
        self.assertIn("notebook 02", pipeline.motivo("20261123_081830", self.st))
        self.assertIn("notebook 03 não rodou", pipeline.motivo("20261121_190000", self.st))

    def test_pasta_vazia(self):
        vazio = pipeline.status_execucoes([])
        self.assertEqual(len(vazio), 0)
        self.assertEqual(pipeline.indice_padrao(vazio), 0)


class FakeJobs:
    def __init__(self, runs):
        self.runs, self.disparos = runs, 0

    def list(self, name=None):
        return [NS(job_id=42)] if name == pipeline.JOB_PADRAO else []

    def list_runs(self, job_id, limit, expand_tasks):
        return self.runs[:limit]

    def get_run(self, run_id):
        return next(r for r in self.runs if r.run_id == run_id)

    def run_now(self, job_id):
        self.disparos += 1
        run = _run(99, "PENDING")
        self.runs.insert(0, run)
        return NS(response=NS(run_id=99))


def _run(run_id, ciclo, resultado=None, tarefas=()):
    return NS(run_id=run_id, start_time=1764000000000, run_page_url=None,
              state=NS(life_cycle_state=NS(value=ciclo), result_state=NS(value=resultado) if resultado else None),
              tasks=[NS(task_key=k, state=NS(life_cycle_state=NS(value=c), result_state=NS(value=r) if r else None))
                     for k, c, r in tarefas])


class PipelineTests(unittest.TestCase):
    def test_dispara_quando_nada_roda(self):
        jobs = FakeJobs([_run(7, "TERMINATED", "FAILED", [("01", "TERMINATED", "FAILED")])])
        pl = pipeline.Pipeline(NS(jobs=jobs))
        ultima = pl.ultima()
        self.assertFalse(ultima.ativo)
        self.assertEqual((ultima.resultado, ultima.tarefas), ("FAILED", [("01", "TERMINATED", "FAILED")]))
        self.assertEqual(pl.disparar(), 99)
        self.assertEqual(jobs.disparos, 1)

    def test_nao_dispara_duas_vezes(self):
        jobs = FakeJobs([_run(8, "RUNNING", tarefas=[("02_validacao", "RUNNING", None),
                                                    ("01_pricer", "TERMINATED", "SUCCESS")])])
        pl = pipeline.Pipeline(NS(jobs=jobs))
        self.assertEqual(pl.disparar(), 8)
        self.assertEqual(jobs.disparos, 0)
        prog = pl.acompanhar(8)
        self.assertTrue(prog.ativo)
        self.assertIsNone(prog.resultado)
        self.assertEqual(prog.tarefas[0], ("01_pricer", "TERMINATED", "SUCCESS"))  # em ordem de etapa

    def test_job_inexistente(self):
        self.assertIsNone(pipeline.Pipeline(NS(jobs=FakeJobs([])), nome="outro").job_id())
        self.assertEqual(pipeline.Pipeline(NS(jobs=None), job_id="5").job_id(), 5)


@unittest.skipUnless(HAS_ST, "streamlit/plotly ausentes")
class FormatoBrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import formato_br
        cls.f = formato_br

    def test_texto(self):
        br = self.f.br_texto
        self.assertEqual(br("166,376.43"), "166.376,43")
        self.assertEqual(br("21.15%"), "21,15%")
        self.assertEqual(br("0.0092%"), "0,0092%")
        self.assertEqual(br("-1,102.44"), "-1.102,44")
        self.assertEqual(br("2026-11-19"), "2026-11-19")
        self.assertEqual(br("RESULTS_superficieVol_20261123_081830.csv"), "RESULTS_superficieVol_20261123_081830.csv")
        self.assertEqual(br("1. item, 2. item"), "1. item, 2. item")  # só entre dígitos
        self.assertEqual(br(50), 50)

    def test_tabela_crua(self):
        df = pd.DataFrame({"Spot": [166376.43], "Desconto %": [0.0137], "Arquivo": ["a.csv"]})
        html = self.f.estilo_br(df).to_html()
        self.assertIn("166.376,4300", html)
        self.assertIn("0,0137", html)

    def test_grafico(self):
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(x=["a"], y=[1], name="σ=21.15%", text=["53.2%"]))
        fig.add_hline(y=1, annotation_text="Spot 166,376")
        fig.update_layout(title="Média 22,639.45")
        self.f.figura_br(fig)
        self.assertEqual(fig.layout.separators, ",.")
        self.assertEqual(fig.layout.annotations[0].text, "Spot 166.376")
        self.assertEqual(fig.layout.title.text, "Média 22.639,45")
        self.assertEqual((fig.data[0].name, fig.data[0].text[0]), ("σ=21,15%", "53,2%"))

    def test_streamlit_instalado(self):
        from streamlit.testing.v1 import AppTest
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, r"{DASHBOARD}")
            import pandas as pd
            import streamlit as st
            import formato_br
            formato_br.instalar(st)
            st.markdown("<style>.x {{ font-size: 0.85em; }}</style>", unsafe_allow_html=True)
            st.metric("Spot IBOV", f"{{166376.43:,.2f}}")
            c = st.columns(2)
            c[0].metric("Vol", f"{{0.2115:.2%}}")
            with c[1]:
                st.metric("Erro", f"{{0.000092:.4%}}")
            st.caption(f"{{10000:,}} cenários")
            st.dataframe(pd.DataFrame({{"x": [1234.5]}}))
        """)
        with tempfile.TemporaryDirectory() as tmp:
            arq = Path(tmp) / "app.py"
            arq.write_text(script, encoding="utf-8")
            at = AppTest.from_file(str(arq)).run()
        self.assertFalse(at.exception)
        self.assertEqual([m.value for m in at.metric], ["166.376,43", "21,15%", "0,0092%"])
        self.assertEqual(at.caption[0].value, "10.000 cenários")
        self.assertIn("0.85em", at.markdown[0].value)  # CSS intacto


if __name__ == "__main__":
    unittest.main()
