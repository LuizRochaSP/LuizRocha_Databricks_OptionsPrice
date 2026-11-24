"""Números do dashboard no padrão brasileiro (ponto no milhar, vírgula no decimal) e CSV para o Excel."""
import io
import unittest

import numpy as np
import pandas as pd

from renda_fixa import graficos as G
from renda_fixa.exibicao import br, csv_br, tabela, tem_vazios


def mostrado(styler) -> pd.DataFrame:
    """O texto que cada célula exibe (o que o st.dataframe mostra)."""
    ctx = styler._translate(False, False)
    linhas = [[c["display_value"] for c in r if c["type"] == "td"] for r in ctx["body"]]
    return pd.DataFrame(linhas, columns=styler.data.columns)


class TestBr(unittest.TestCase):
    def test_milhar_e_decimal(self):
        self.assertEqual(br(32353501.49), "32.353.501,49")
        self.assertEqual(br(-7189.04), "-7.189,04")
        self.assertEqual(br(20345.161065, 6), "20.345,161065")
        self.assertEqual(br(13.8469, 4), "13,8469")
        self.assertEqual(br(-7.04, 1, True), "-7,0")
        self.assertEqual(br(14.2, 1, True), "+14,2")
        self.assertEqual(br(532, 0), "532")
        self.assertEqual(br(np.nan), "—")


class TestTabela(unittest.TestCase):
    def setUp(self):
        # como a tabela de títulos: pu_nss vazio na LFT e na NTN-C, chegando como None (objeto)
        self.df = pd.DataFrame({
            "titulo": ["LFT", "NTN-B", "NTN-C"],
            "du": [66, 8449, 1028],
            "pu_anbima": [20343.696213, 4837.481628, 8112.654316],
            "pu_nss": [None, 4837.483612, None],
            "erro_pu": [0, 0, 0],
        })
        self.fmt = {"pu_anbima": "{:,.6f}", "pu_nss": "{:,.6f}"}

    def test_celulas_no_padrao_brasileiro(self):
        tela = mostrado(tabela(self.df, self.fmt))
        self.assertEqual(tela["pu_anbima"].tolist(), ["20.343,696213", "4.837,481628", "8.112,654316"])
        self.assertEqual(tela["pu_nss"].tolist(), ["—", "4.837,483612", "—"])
        self.assertEqual(tela["du"].tolist(), ["66", "8449", "1028"])  # dias úteis sem separador
        self.assertNotIn("None", tela.to_string())

    def test_colunas_completas_seguem_numericas(self):
        dados = tabela(self.df, self.fmt).data
        self.assertTrue(pd.api.types.is_float_dtype(dados["pu_anbima"]))  # ordena como número

    def test_formato_padrao_com_milhar(self):
        tela = mostrado(tabela(pd.DataFrame({"vp": [16044539.52, 1681.69], "dv01": [-2235.31, -1102.44]}), casas=2))
        self.assertEqual(tela["vp"].tolist(), ["16.044.539,52", "1.681,69"])
        self.assertEqual(tela["dv01"].tolist(), ["-2.235,31", "-1.102,44"])

    def test_csv_para_excel(self):
        self.assertTrue(tem_vazios(self.df))
        self.assertFalse(tem_vazios(self.df.drop(columns="pu_nss")))
        texto = csv_br(self.df).decode("utf-8-sig")
        self.assertIn("NTN-B;8449;4837,481628;4837,483612;0", texto)
        lido = pd.read_csv(io.StringIO(texto), sep=";", decimal=",")
        self.assertTrue(pd.api.types.is_float_dtype(lido["pu_nss"]))
        self.assertAlmostEqual(lido["pu_nss"].iloc[1], 4837.483612)
        self.assertTrue(lido["pu_nss"].iloc[[0, 2]].isna().all())


class TestGraficos(unittest.TestCase):
    def test_separadores_brasileiros(self):
        fig = G.figura([G.linha(0, "x", [1, 2], [1.5, 2.5])], "t")
        self.assertEqual(fig.layout.separators, ",.")


if __name__ == "__main__":
    unittest.main()
