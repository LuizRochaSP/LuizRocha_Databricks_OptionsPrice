import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from renda_fixa.dados.anbima import ler_mercado_secundario
from renda_fixa.dados.tesouro_direto import ler_tesouro_direto
from renda_fixa.mercado import curvas_por_taxas, historico_curvas, montar_mercado, variacoes_diarias_bps

DADOS = Path(__file__).parent / "dados"

TD_AMOSTRA = """Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;Taxa Venda Manha;PU Compra Manha;PU Venda Manha;PU Base Manha
Tesouro Prefixado;01/01/2029;19/11/2026;15,04;15,16;746,01;743,97;743,97
Tesouro Prefixado;01/01/2032;19/11/2026;15,40;15,52;480,00;477,00;477,00
Tesouro Prefixado com Juros Semestrais;01/01/2035;19/11/2026;15,31;15,43;809,13;804,36;804,36
Tesouro IPCA+ com Juros Semestrais;15/05/2035;19/11/2026;7,81;7,93;4283,66;4248,81;4248,81
Tesouro IPCA+;15/05/2029;19/11/2026;0,00;7,57;0,00;3990,12;3990,12
Tesouro IGPM+ com Juros Semestrais;01/01/2031;19/11/2026;7,63;7,75;8110,14;8076,03;8076,03
Tesouro Renda+ Aposentadoria Extra;15/12/2079;19/11/2026;7,49;7,61;235,31;224,81;224,81
Tesouro Prefixado;01/01/2029;18/11/2026;14,94;15,06;746,96;744,91;744,91
Tesouro Prefixado;01/01/2032;18/11/2026;15,28;15,40;485,38;482,54;482,54
Tesouro Prefixado com Juros Semestrais;01/01/2035;18/11/2026;15,31;15,43;809,13;804,36;804,36
Tesouro IPCA+ com Juros Semestrais;15/05/2035;18/11/2026;7,73;7,85;4301,32;4267,89;4267,89
"""


class TestTesouroDireto(unittest.TestCase):
    def test_leitura(self):
        td = ler_tesouro_direto(TD_AMOSTRA)
        self.assertEqual(set(td["titulo"]), {"LTN", "NTN-F", "NTN-B", "NTN-B Principal", "NTN-C"})
        self.assertEqual(len(td), 10)  # Renda+ fica de fora
        ltn = td[(td.titulo == "LTN") & (td.vencimento == date(2029, 1, 1)) & (td.data_referencia == date(2026, 11, 19))]
        self.assertAlmostEqual(float(ltn.taxa_indicativa.iloc[0]), 15.10)
        # taxa de compra zerada: usa a de venda
        p = td[td.titulo == "NTN-B Principal"]
        self.assertAlmostEqual(float(p.taxa_indicativa.iloc[0]), 7.57)

    def test_historico_e_variacoes(self):
        hc = historico_curvas(ler_tesouro_direto(TD_AMOSTRA), vertices=(252, 504, 1260))
        self.assertEqual(list(hc["PRE"].index), [date(2026, 11, 18), date(2026, 11, 19)])
        d = variacoes_diarias_bps(hc["PRE"])
        self.assertEqual(len(d), 1)
        self.assertGreater(d.iloc[0][504], 0)  # a curva subiu de 18 para 19/11


class TestCurvasPorTaxas(unittest.TestCase):
    def test_igual_ao_bootstrap_por_pu(self):
        df = ler_mercado_secundario((DADOS / "ms261119.txt").read_text(encoding="latin-1"))
        pre, real = curvas_por_taxas(df)
        m = montar_mercado(df, ajustar_nss=False)
        for du in (252, 504, 1260, 2520):
            self.assertAlmostEqual(pre.taxa(du), m.curva_pre.taxa(du), delta=0.002)  # só o truncamento do PU
            self.assertAlmostEqual(real.taxa(du), m.curva_real.taxa(du), delta=0.002)

    def test_lacunas_nao_viram_choque(self):
        h = pd.DataFrame({252: [10.0, 10.1, 10.5, 10.6]},
                         index=[date(2026, 10, 8), date(2026, 10, 9), date(2026, 10, 20), date(2026, 10, 21)])
        d = variacoes_diarias_bps(h)
        self.assertEqual(list(d.index), [date(2026, 10, 9), date(2026, 10, 21)])
        self.assertAlmostEqual(d.iloc[0][252], 10.0)


if __name__ == "__main__":
    unittest.main()
