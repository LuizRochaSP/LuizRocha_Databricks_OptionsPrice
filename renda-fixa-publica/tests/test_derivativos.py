import unittest
from datetime import date

import pandas as pd

from renda_fixa.calendario import dias_uteis
from renda_fixa.curvas import CurvaPlana, CurvaZero
from renda_fixa.dados.bcb import fator_cdi
from renda_fixa.derivativos import (
    DAP,
    DI1,
    SwapDIIPCA,
    SwapDIPre,
    ajuste_diario_di1,
    curva_di1,
    fator_projetado_cdi,
    taxa_fra,
    vencimento_dap,
    vencimento_di1,
)

REF = date(2026, 11, 19)
CURVA = CurvaZero([30, 252, 756, 2520], [14.0, 14.7, 15.5, 15.4])


class TestFuturos(unittest.TestCase):
    def test_vencimentos(self):
        self.assertEqual(vencimento_di1("F27"), date(2027, 1, 4))  # 1/1 feriado, 2-3 fim de semana
        self.assertEqual(vencimento_di1("DI1N28"), date(2028, 7, 3))
        self.assertEqual(vencimento_dap("K35"), date(2035, 5, 15))
        self.assertEqual(str(DI1.de_codigo("F30")), "DI1F30")

    def test_pu_taxa_dv01(self):
        c = DI1.de_codigo("F29")
        n = c.du(REF)
        self.assertAlmostEqual(c.pu_bruto(15.0, REF), 100000 / 1.15 ** (n / 252))
        self.assertAlmostEqual(c.taxa(c.pu_bruto(15.0, REF), REF), 15.0, places=10)
        self.assertGreater(c.dv01(15.0, REF), 0)
        d = DAP.de_codigo("K35")
        self.assertAlmostEqual(d.dv01(7.5, REF, 7000.0), d.dv01_pontos(7.5, REF) * 0.00025 * 7000.0)

    def test_fra_igual_termo_da_curva(self):
        c1, c2 = DI1.de_codigo("F28"), DI1.de_codigo("F30")
        n1, n2 = c1.du(REF), c2.du(REF)
        f = taxa_fra(CURVA.taxa(n1), n1, CURVA.taxa(n2), n2)
        self.assertAlmostEqual(f, CURVA.taxa_termo(n1, n2), places=10)
        curva = curva_di1([("F28", CURVA.taxa(n1)), ("F30", CURVA.taxa(n2))], REF)
        self.assertAlmostEqual(curva.taxa(n2), CURVA.taxa(n2))

    def test_ajuste_diario(self):
        # PU sobe (taxa caiu): comprado em taxa perde
        a = ajuste_diario_di1(90010.0, 90000.0, 14.9, 10, "comprado_taxa")
        pa_corr = round(90000.0 * 1.149 ** (1 / 252), 2)
        self.assertAlmostEqual(a.pa_anterior_corrigido, pa_corr)
        self.assertAlmostEqual(a.ajuste, -(90010.0 - pa_corr) * 10, places=2)
        b = ajuste_diario_di1(90010.0, 90000.0, 14.9, 10, "vendido_taxa")
        self.assertAlmostEqual(b.ajuste, -a.ajuste)


class TestSwaps(unittest.TestCase):
    def test_taxa_par_zera_swap(self):
        venc = date(2029, 1, 2)
        k = SwapDIPre.taxa_par(CURVA, REF, venc)
        s = SwapDIPre(1e6, REF, venc, k, ativo="pre")
        r = s.avaliar(REF, CURVA)
        self.assertAlmostEqual(r.mtm, 0.0, places=4)
        self.assertAlmostEqual(r.vp_passivo, 1e6, places=4)  # ponta DI vale o nocional no início
        self.assertLess(r.dv01, 0)  # recebe pré perde quando a taxa sobe
        r_di = SwapDIPre(1e6, REF, venc, k, ativo="di").avaliar(REF, CURVA)
        self.assertAlmostEqual(r_di.dv01, -r.dv01, places=6)

    def test_pct_cdi_e_spread(self):
        n = 500
        self.assertAlmostEqual(fator_projetado_cdi(CURVA, n, 100.0), 1 / CURVA.fator_desconto(n), places=10)
        # caminho diário a 100% coincide com o fechado
        diario = fator_projetado_cdi(CURVA, n, 100.0000001)
        self.assertAlmostEqual(diario, 1 / CURVA.fator_desconto(n), places=6)
        self.assertLess(fator_projetado_cdi(CURVA, n, 90.0), fator_projetado_cdi(CURVA, n, 100.0))
        k100 = SwapDIPre.taxa_par(CURVA, REF, date(2028, 1, 3))
        k_sp = SwapDIPre.taxa_par(CURVA, REF, date(2028, 1, 3), spread_cdi_aa=1.0)
        self.assertAlmostEqual((1 + k_sp / 100) / (1 + k100 / 100), 1.01, places=10)

    def test_swap_em_andamento(self):
        inicio = date(2026, 1, 2)
        venc = date(2028, 1, 3)
        s = SwapDIPre(1e6, inicio, venc, 14.0, ativo="di")
        fator = 1.13  # CDI realizado
        r = s.avaliar(REF, CURVA, fator)
        n = dias_uteis(REF, venc)
        self.assertAlmostEqual(r.vp_ativo, 1e6 * fator, places=4)
        self.assertAlmostEqual(r.vp_passivo, 1e6 * 1.14 ** (dias_uteis(inicio, venc) / 252) * CURVA.fator_desconto(n), places=4)

    def test_swap_ipca(self):
        real = CurvaPlana(7.5)
        venc = date(2030, 8, 15)
        c = SwapDIIPCA.cupom_par(CURVA, real, REF, venc)
        s = SwapDIIPCA(1e6, REF, venc, c, vna_inicio=4775.0, ativo="ipca")
        r = s.avaliar(REF, CURVA, real, vna_ref=4775.0)
        self.assertAlmostEqual(r.mtm, 0.0, places=4)
        self.assertLess(s.dv01_real(REF, CURVA, real, 4775.0), 0)
        r2 = s.avaliar(REF, CURVA, real, vna_ref=4775.0 * 1.01)
        self.assertAlmostEqual(r2.vp_ativo / r.vp_ativo, 1.01, places=10)


class TestFatorCDI(unittest.TestCase):
    def test_fator(self):
        s = pd.Series([0.050788] * 3, index=[date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)])
        self.assertAlmostEqual(fator_cdi(s, date(2026, 11, 17), date(2026, 11, 19)), 1.00050788**2, places=14)
        self.assertAlmostEqual(fator_cdi(s, date(2026, 11, 17), date(2026, 11, 20), 50.0), 1.00025394**3, places=14)


if __name__ == "__main__":
    unittest.main()
