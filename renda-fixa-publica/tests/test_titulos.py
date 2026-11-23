"""Títulos públicos validados contra os arquivos reais da ANBIMA de 18/11/2026 e 19/11/2026."""
import unittest
from datetime import date
from pathlib import Path

from renda_fixa.calendario import DU_ANO, dias_uteis
from renda_fixa.dados.anbima import ler_mercado_secundario, vna_implicito
from renda_fixa.titulos import LFT, LTN, NTNB, NTNF, NTNBPrincipal, criar_titulo
from renda_fixa.vna import atualizar_vna_lft

DADOS = Path(__file__).parent / "dados"


def carregar(nome):
    return ler_mercado_secundario((DADOS / nome).read_text(encoding="latin-1"))


class TestContraANBIMA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dias = {n: carregar(n) for n in ("ms261118.txt", "ms261119.txt")}

    def test_pu_de_todos_os_titulos(self):
        for nome, df in self.dias.items():
            vnas = {"LFT": vna_implicito(df, "LFT"), "NTN-B": vna_implicito(df, "NTN-B")}
            for v in vnas.values():
                self.assertTrue(v.consistente, nome)
            n = 0
            for r in df.itertuples():
                if r.titulo == "NTN-C":
                    continue
                t = criar_titulo(r.titulo, r.vencimento)
                vna = vnas[r.titulo].vna if r.titulo in vnas else None
                with self.subTest(dia=nome, titulo=r.titulo, venc=r.vencimento):
                    self.assertEqual(t.pu(r.taxa_indicativa, r.data_referencia, vna), r.pu)
                    taxa = t.taxa(r.pu, r.data_referencia, vna)
                    self.assertEqual(t.pu(taxa, r.data_referencia, vna), r.pu)
                    self.assertAlmostEqual(taxa, r.taxa_indicativa, delta=0.0005)
                n += 1
            self.assertEqual(n, 49)

    def test_vna_lft_avanca_com_selic(self):
        v18 = vna_implicito(self.dias["ms261118.txt"], "LFT").vna
        v19 = vna_implicito(self.dias["ms261119.txt"], "LFT").vna
        # Selic diária de 18/11/2026 (SGS 11): 0,050788% a.d.
        self.assertEqual(atualizar_vna_lft(v18, 0.050788), v19)


class TestFluxosEAnaliticos(unittest.TestCase):
    ref = date(2026, 11, 19)

    def test_datas_de_cupom(self):
        self.assertEqual(
            [f.data for f in NTNF(date(2029, 1, 1)).fluxos(self.ref)],
            [date(2027, 1, 1), date(2027, 7, 1), date(2028, 1, 1), date(2028, 7, 1), date(2029, 1, 1)],
        )
        datas_b = [f.data for f in NTNB(date(2028, 8, 15)).fluxos(self.ref)]
        self.assertEqual(datas_b[0], date(2027, 2, 15))
        self.assertEqual(datas_b[-1], date(2028, 8, 15))
        fl = NTNB(date(2027, 5, 15)).fluxos(self.ref)
        self.assertEqual([f.valor for f in fl], [102.956301])  # cupom de 15/11/2026 já pago
        fl = NTNB(date(2027, 5, 15)).fluxos(date(2026, 11, 13))
        self.assertEqual([f.valor for f in fl], [2.956301, 102.956301])

    def test_ltn_duration_igual_prazo(self):
        t = LTN(date(2030, 1, 1))
        a = t.analiticos(15.0, self.ref)
        self.assertAlmostEqual(a.duration_macaulay, dias_uteis(self.ref, t.vencimento) / DU_ANO, places=12)
        self.assertAlmostEqual(a.duration_modificada, a.duration_macaulay / 1.15, places=12)

    def test_dv01_aproxima_duration_modificada(self):
        for t, vna in [(NTNF(date(2035, 1, 1)), None), (NTNB(date(2045, 5, 15)), 4775.0), (LTN(date(2032, 1, 1)), None)]:
            a = t.analiticos(10.0, self.ref, vna)
            aprox = a.duration_modificada * t.pu_bruto(10.0, self.ref, vna) * 1e-4
            self.assertAlmostEqual(a.dv01 / aprox, 1.0, delta=2e-3)
            # convexidade numérica
            p0, pu, pd = (t.pu_bruto(x, self.ref, vna) for x in (10.0, 10.1, 9.9))
            conv_num = (pu + pd - 2 * p0) / (p0 * 0.001**2)
            self.assertAlmostEqual(a.convexidade / conv_num, 1.0, delta=1e-3)

    def test_ntnb_principal_e_lft(self):
        vna = 4775.721642
        p = NTNBPrincipal(date(2035, 5, 15))
        cot = 100 / 1.08 ** (dias_uteis(self.ref, date(2035, 5, 15)) / 252)
        self.assertAlmostEqual(p.pu(8.0, self.ref, vna), vna * int(cot * 1e4) / 1e6, places=5)
        lft = LFT(date(2030, 3, 1))
        self.assertLess(lft.pu(0.1, self.ref, 20000.0), 20000.0)
        self.assertGreater(lft.pu(-0.1, self.ref, 20000.0), 20000.0)
        with self.assertRaises(ValueError):
            lft.pu(0.1, self.ref)

    def test_vencido_e_fabrica(self):
        self.assertEqual(LTN(date(2026, 1, 1)).fluxos(self.ref), [])
        self.assertIsInstance(criar_titulo("ntn-b principal", date(2035, 5, 15)), NTNBPrincipal)
        with self.assertRaises(ValueError):
            criar_titulo("CDB", date(2030, 1, 1))


if __name__ == "__main__":
    unittest.main()
