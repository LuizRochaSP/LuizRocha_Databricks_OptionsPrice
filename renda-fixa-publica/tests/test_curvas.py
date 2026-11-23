import unittest
from datetime import date
from pathlib import Path

import numpy as np

from renda_fixa.curvas import (
    CurvaPlana,
    CurvaZero,
    ParametrosNSS,
    CurvaNSS,
    ajustar_nss_taxas,
    bootstrap_titulos,
    inflacao_implicita,
    z_spread,
)
from renda_fixa.dados.anbima import ler_mercado_secundario
from renda_fixa.mercado import montar_mercado, tabela_curvas, tabela_titulos
from renda_fixa.titulos import LTN, NTNF, criar_titulo

DADOS = Path(__file__).parent / "dados"
REF = date(2026, 11, 19)


class TestCurvaZero(unittest.TestCase):
    def test_flat_forward_termo_constante(self):
        c = CurvaZero([100, 500], [10.0, 12.0])
        f1 = c.taxa_termo(100, 200)
        f2 = c.taxa_termo(300, 500)
        self.assertAlmostEqual(f1, f2, places=10)
        self.assertAlmostEqual(c.taxa(100), 10.0)
        self.assertAlmostEqual(c.taxa(50), 10.0)  # extrapolação constante
        self.assertAlmostEqual(c.taxa(900), 12.0)
        self.assertAlmostEqual(c.fator_desconto(252), 1 / ((1.10) ** (100 / 252) * (1 + f1 / 100) ** (152 / 252)), places=12)

    def test_linear_e_deslocamento(self):
        c = CurvaZero([100, 300], [10.0, 12.0], interpolacao="linear")
        self.assertAlmostEqual(c.taxa(200), 11.0)
        self.assertAlmostEqual(c.deslocada(25).taxa(200), 11.25)
        self.assertAlmostEqual(c.deslocada(lambda du: du / 100).taxa(200), 11.02)
        with self.assertRaises(ValueError):
            CurvaZero([1, 1], [1, 2])

    def test_nss_recupera_parametros(self):
        verdade = CurvaNSS(ParametrosNSS(12.0, 2.0, -3.0, 1.5, 1.2, 0.2))
        dus = np.arange(21, 7560, 63)
        ajust = ajustar_nss_taxas(dus, verdade.taxas(dus))
        self.assertLess(np.max(np.abs(ajust.taxas(dus) - verdade.taxas(dus))), 1e-4)

    def test_inflacao_implicita(self):
        b = inflacao_implicita(CurvaPlana(15.0), CurvaPlana(7.0), [252, 504])
        np.testing.assert_allclose(b, (1.15 / 1.07 - 1) * 100)


class TestBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = ler_mercado_secundario((DADOS / "ms261119.txt").read_text(encoding="latin-1"))
        cls.m = montar_mercado(cls.df)

    def test_bootstrap_reprecifica_sinteticos(self):
        verdade = CurvaZero([200, 600, 1500], [13.0, 14.5, 15.0])
        titulos = [LTN(date(2027, 7, 1)), NTNF(date(2029, 1, 1)), NTNF(date(2033, 1, 1))]
        pus = [t.pu_curva(verdade, REF) for t in titulos]
        c = bootstrap_titulos(titulos, pus, REF)
        for t, pu in zip(titulos, pus):
            self.assertAlmostEqual(t.pu_curva(c, REF), pu, places=6)

    def test_curvas_do_dia_reprecificam_titulos(self):
        tt = tabela_titulos(self.m)
        self.assertEqual(int((tt["erro_pu"] != 0).sum()), 0)
        sub = self.df[self.df["titulo"].isin(["LTN", "NTN-F", "NTN-B"])]
        # NTN-F jan/29 divide o vencimento com a LTN jan/29, que define o vértice
        sub = sub[~((sub.titulo == "NTN-F") & (sub.vencimento == date(2029, 1, 1)))]
        for r in sub.itertuples():
            t = criar_titulo(r.titulo, r.vencimento)
            curva = self.m.curva_pre if t.indexador == "PRE" else self.m.curva_real
            with self.subTest(t=str(t)):
                self.assertAlmostEqual(t.pu_curva(curva, REF, self.m.vna(r.titulo)), r.pu, delta=0.02)
        # NSS dentro de 15 bps em todos os títulos pré e IPCA
        self.assertLess(tt["residuo_nss_bps"].abs().max(), 15.0)

    def test_tabela_curvas_e_z_spread(self):
        tc = tabela_curvas(self.m)
        self.assertTrue((tc["pre"] > 10).all() and (tc["pre"] < 20).all())
        self.assertIn("inflacao_implicita", tc)
        ltn = LTN(date(2029, 1, 1))
        pu = float(self.df[(self.df.titulo == "LTN") & (self.df.vencimento == date(2029, 1, 1))].pu.iloc[0])
        self.assertAlmostEqual(z_spread(ltn, pu, self.m.curva_pre, REF), 0.0, delta=0.05)
        self.assertAlmostEqual(z_spread(ltn, pu, self.m.curva_pre.deslocada(-20), REF), 20.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
