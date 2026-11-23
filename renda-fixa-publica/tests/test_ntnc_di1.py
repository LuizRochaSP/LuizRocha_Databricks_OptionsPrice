"""NTN-C (IGP-M) e curva de ajustes de DI1 da B3, com os arquivos reais de 18 e 19/11/2026."""
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from renda_fixa.dados.anbima import ler_mercado_secundario, vna_implicito
from renda_fixa.dados.b3 import ajustes_di1, baixar_ajustes_b3, ler_ajustes_b3
from renda_fixa.dados.tesouro_direto import ler_tesouro_direto
from renda_fixa.derivativos import DAP, DI1
from renda_fixa.mercado import comparar_di1, curva_di1_b3, montar_mercado
from renda_fixa.risco import Carteira, PosicaoTitulo
from renda_fixa.titulos import NTNC, criar_titulo
from renda_fixa.vna import (atualizar_vna_ntnc, igpm_projetado_implicito, vna_ntnc_por_igpm,
                            vna_ntnc_projetado)

DADOS = Path(__file__).parent / "dados"
DIAS = {date(2026, 11, 18): "ms261118.txt", date(2026, 11, 19): "ms261119.txt"}


def anbima(d):
    return ler_mercado_secundario((DADOS / DIAS[d]).read_text(encoding="latin-1"))


class TestNTNC(unittest.TestCase):
    def test_pu_igual_anbima(self):
        for d in DIAS:
            df = anbima(d)
            r = df[df.titulo == "NTN-C"].iloc[0]
            vna = vna_implicito(df, "NTN-C").vna
            t = criar_titulo("NTN-C", r.vencimento)
            self.assertEqual(t.pu(r.taxa_indicativa, d, vna), r.pu)
            self.assertAlmostEqual(t.taxa(r.pu, d, vna), r.taxa_indicativa, delta=0.0005)

    def test_cupom_de_12_confirmado_pelo_tesouro_direto(self):
        # Tesouro Direto 19/11/2026, NTN-C 2031: taxa de venda 7,75% e PU de venda R$ 8.089,65.
        # Com o VNA da ANBIMA do mesmo dia, o cupom de 12% a.a. reproduz o PU com folga de 0,1%
        # (a diferença é a projeção de IGP-M da manhã); o de 6% erraria em mais de 10%.
        d = date(2026, 11, 19)
        vna = vna_implicito(anbima(d), "NTN-C").vna
        pu12 = NTNC(date(2031, 1, 1)).pu(7.75, d, vna)
        pu6 = NTNC(date(2031, 1, 1), cupom_aa=6.0).pu(7.75, d, vna)
        self.assertAlmostEqual(pu12 / 8089.65 - 1, 0.0, delta=0.001)
        self.assertGreater(abs(pu6 / 8089.65 - 1), 0.10)

    def test_fluxos_e_fabrica(self):
        t = criar_titulo("NTN-C", date(2031, 1, 1))
        self.assertEqual(t.cupom_aa, 12.0)
        fl = t.fluxos(date(2026, 11, 19))
        self.assertEqual([f.data for f in fl][:3], [date(2027, 1, 1), date(2027, 7, 1), date(2028, 1, 1)])
        self.assertEqual(len(fl), 9)
        self.assertAlmostEqual(fl[-1].valor, 105.830052)
        self.assertEqual(NTNC(date(2021, 1, 1)).cupom_aa, 6.0)
        with self.assertRaises(ValueError):
            NTNC(date(2031, 1, 1), cupom_aa=8.0)

    def test_vna_igpm(self):
        igpm = pd.Series([1.0, 2.0, -0.5], index=[date(2000, 7, 1), date(2000, 8, 1), date(2000, 9, 1)])
        self.assertEqual(vna_ntnc_por_igpm(igpm, date(2000, 7, 1)), 1000.0)
        self.assertAlmostEqual(vna_ntnc_por_igpm(igpm, date(2000, 10, 1)), 1000 * 1.01 * 1.02 * 0.995, places=6)
        with self.assertRaises(ValueError):
            vna_ntnc_por_igpm(igpm, date(2000, 11, 1))  # falta outubro
        self.assertAlmostEqual(atualizar_vna_ntnc(6000.0, 1.0), 6060.0)

    def test_pro_rata_em_dias_corridos(self):
        # 19/11: 18 dias corridos desde 01/11, num mês de 30 dias
        v = vna_ntnc_projetado(6772.77, 0.9, date(2026, 11, 19))
        self.assertAlmostEqual(v, 6772.77 * 1.009 ** (18 / 30), places=5)
        self.assertAlmostEqual(igpm_projetado_implicito(v, 6772.77, date(2026, 11, 19)), 0.9, places=4)

    def test_risco(self):
        d = date(2026, 11, 19)
        m = montar_mercado(anbima(d), ajustar_nss=False)
        self.assertIn("IGPM", m.curvas)
        r = m.df[m.df.titulo == "NTN-C"].iloc[0]
        t = NTNC(r.vencimento)
        pos = PosicaoTitulo.a_mercado(t, 100, r.pu, d, m.curvas, m.vna("NTN-C"))
        cart = Carteira([pos])
        self.assertAlmostEqual(cart.valor(d, m.curvas), 100 * r.pu, delta=0.01)
        dv01 = t.analiticos(r.taxa_indicativa, d, m.vna("NTN-C")).dv01
        self.assertAlmostEqual(abs(cart.dv01(d, m.curvas, "IGPM")), 100 * dv01, delta=0.5)

    def test_tesouro_direto_le_ntnc(self):
        txt = ("Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;Taxa Venda Manha;PU Compra Manha;"
               "PU Venda Manha;PU Base Manha\n"
               "Tesouro IGPM+ com Juros Semestrais;01/01/2031;19/11/2026;7,63;7,75;8132,14;8089,65;8089,65\n")
        td = ler_tesouro_direto(txt)
        self.assertEqual(td.titulo.tolist(), ["NTN-C"])


B3_AMOSTRA = """Status do Arquivo: Final
RptDt;TckrSymb;ISIN;SgmtNm;MinPric;MaxPric;TradAvrgPric;LastPric;OscnPctg;AdjstdQt;AdjstdQtTax;RefPric;TradQty;FinInstrmQty;NtlFinVol
2026-11-19;PETR4;BRPETRACNPR6;CASH;30;31;30,5;30,7;1;;;;10;100;3050
2026-11-19;DI1F27;BRBMEFD1I4Z0;FINANCIAL;13,685;13,777;13,723;13,724;0,18;98530,93;13,724;;16377;1324443;130498652269,2
2026-11-19;DI1D;BRBMEFDI1D00;FINANCIAL;13,65;13,65;13,65;13,65;0;;;;1;1;1
2026-11-19;DAPK35;BRBMEFDAP000;FINANCIAL;7,1;7,2;7,15;7,15;0;31000,5;7,15;;1;10;1
2026-11-19;DI1F28;BRBMEFD1I4Y3;FINANCIAL;14,475;15,06;14,732;14,85;2,31;85759,1;14,828;;95840;1382866;118703816789,98
"""


class TestAjustesB3(unittest.TestCase):
    def test_parser(self):
        df = ler_ajustes_b3(B3_AMOSTRA)
        self.assertEqual(df.contrato.tolist(), ["DAPK35", "DI1F27", "DI1F28"])
        di = df[df.ativo == "DI1"].iloc[0]
        self.assertEqual(di.vencimento, date(2027, 1, 4))
        self.assertEqual(di.pu_ajuste, 98530.93)
        self.assertEqual(di.taxa_ajuste, 13.724)
        self.assertEqual(df[df.ativo == "DAP"].iloc[0].vencimento, date(2035, 5, 15))

    def test_pu_de_ajuste_de_todos_os_contratos(self):
        # 43 DI1 e 20 DAP por dia: o PU de ajuste da B3 sai da taxa de ajuste ao centavo
        for d in DIAS:
            df = baixar_ajustes_b3(d, DADOS)  # lê do cache em tests/dados
            self.assertEqual(df.ativo.value_counts().to_dict(), {"DI1": 43, "DAP": 20})
            for r in df.itertuples():
                c = DI1(r.vencimento) if r.ativo == "DI1" else DAP(r.vencimento)
                with self.subTest(dia=d, contrato=r.contrato):
                    self.assertAlmostEqual(c.pu(r.taxa_ajuste, d), r.pu_ajuste, places=2)

    def test_curva_di1_reprecifica_ajustes(self):
        d = date(2026, 11, 19)
        df = baixar_ajustes_b3(d, DADOS)
        curva = curva_di1_b3(df, d)
        for venc, tx in ajustes_di1(df):
            self.assertAlmostEqual(DI1(venc).pu_curva(curva, d), DI1(venc).pu_bruto(tx, d), places=6)

    def test_comparacao_com_titulos(self):
        for d in DIAS:
            m = montar_mercado(anbima(d), ajustar_nss=False)
            c = comparar_di1(m, baixar_ajustes_b3(d, DADOS))
            self.assertEqual(len(c), 43)
            self.assertTrue((c.erro_pu == 0).all())
            dentro = c[~c.extrapolado]
            self.assertLess(dentro.spread_bps.abs().max(), 50.0)
            self.assertTrue(c[c.vencimento > date(2037, 1, 2)].extrapolado.all())
        with self.assertRaises(ValueError):
            comparar_di1(montar_mercado(anbima(date(2026, 11, 18)), ajustar_nss=False),
                         baixar_ajustes_b3(date(2026, 11, 19), DADOS))


if __name__ == "__main__":
    unittest.main()
