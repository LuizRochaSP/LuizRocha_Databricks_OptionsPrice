import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from renda_fixa.curvas import CurvaPlana, CurvaZero
from renda_fixa.dados.anbima import DadoIndisponivel, ler_mercado_secundario, vna_implicito
from renda_fixa.derivativos import DI1, SwapDIPre
from renda_fixa.risco import (
    Carteira,
    PosicaoDI1,
    PosicaoSwap,
    PosicaoTitulo,
    cenarios_padrao,
    contratos_hedge_di1,
    pca_curva,
)
from renda_fixa.titulos import LFT, LTN, NTNB, NTNF
from renda_fixa.vna import aniversario_anterior, ipca_projetado_implicito, vna_ntnb_projetado

REF = date(2026, 11, 19)
PRE = CurvaZero([30, 252, 756, 2520], [14.0, 14.7, 15.5, 15.4])
REAL = CurvaZero([126, 504, 2520, 7560], [5.0, 7.5, 7.9, 7.6])
CURVAS = {"PRE": PRE, "IPCA": REAL}


def carteira_exemplo():
    return Carteira(
        [
            PosicaoTitulo(LTN(date(2029, 1, 1)), 1000),
            PosicaoTitulo(NTNF(date(2033, 1, 1)), 500),
            PosicaoTitulo(NTNB(date(2035, 5, 15)), 200, vna=4775.0),
            PosicaoSwap(SwapDIPre(1e6, REF, date(2030, 1, 2), 15.0, ativo="di"), REF),
        ]
    )


class TestRisco(unittest.TestCase):
    def test_key_rate_soma_paralelo(self):
        c = carteira_exemplo()
        vertices = [21, 126, 252, 504, 756, 1260, 2520, 5040]
        kr = c.key_rate_dv01(REF, CURVAS, "PRE", vertices)
        self.assertAlmostEqual(kr.sum(), c.dv01(REF, CURVAS, "PRE"), delta=abs(c.dv01(REF, CURVAS, "PRE")) * 1e-3)
        self.assertLess(c.dv01(REF, CURVAS, "IPCA"), 0)

    def test_hedge_di1_zera_dv01(self):
        c = Carteira([PosicaoTitulo(LTN(date(2029, 1, 1)), 10000)])
        dv = c.dv01(REF, CURVAS)
        self.assertLess(dv, 0)
        contrato = DI1(date(2029, 1, 2))
        taxa = PRE.taxa(contrato.du(REF))
        q = contratos_hedge_di1(dv, contrato, taxa, REF)
        self.assertLess(q, 0)  # vendido em PU = comprado em taxa
        c.adicionar(PosicaoDI1(contrato, q, contrato.pu_curva(PRE, REF)))
        self.assertAlmostEqual(c.dv01(REF, CURVAS), 0.0, delta=abs(dv) * 1e-3)

    def test_cenarios_e_var(self):
        c = carteira_exemplo()
        cen = c.cenarios(REF, CURVAS, cenarios_padrao())
        self.assertLess(cen["Pré +100 bps"], 0)
        self.assertGreater(cen["Pré -100 bps"], 0)
        choques = pd.DataFrame([[100.0, 100.0], [-50.0, -50.0], [10.0, 10.0]], columns=[21, 7560])
        v = c.var_historico(REF, CURVAS, choques, "PRE", nivel=0.99)
        self.assertAlmostEqual(v["pnls"][0], cen["Pré +100 bps"], places=6)
        self.assertGreater(v["var"], 0)

    def test_carry_e_posicao_a_mercado(self):
        c = Carteira([PosicaoTitulo(LTN(date(2029, 1, 1)), 1)])
        carry = c.carry_rolldown(REF, CURVAS, 21)
        self.assertGreater(carry, 0)
        t = LTN(date(2029, 1, 1))
        p = PosicaoTitulo.a_mercado(t, 1, 740.0, REF, CURVAS)
        self.assertAlmostEqual(p.vp(REF, CURVAS), 740.0, places=4)
        lft = PosicaoTitulo.a_mercado(LFT(date(2030, 3, 1)), 1, 20274.601633, REF, CURVAS, vna=20334.83341)
        self.assertAlmostEqual(lft.vp(REF, CURVAS), 20274.601633, places=3)
        self.assertEqual(Carteira([lft]).dv01(REF, CURVAS), 0.0)

    def test_pca_nivel(self):
        rng = np.random.default_rng(0)
        n = 500
        nivel = np.cumsum(rng.normal(0, 10, n))
        incl = np.cumsum(rng.normal(0, 2, n))
        vert = np.array([1, 2, 5, 10])
        df = pd.DataFrame(nivel[:, None] + incl[:, None] * (vert - 5) / 5 + rng.normal(0, 0.3, (n, 4)), columns=vert)
        r = pca_curva(df)
        self.assertGreater(r.variancia_explicada[0], 0.9)
        self.assertTrue((r.cargas["PC1"] > 0).all())


class TestDadosEVNA(unittest.TestCase):
    def test_parser(self):
        df = ler_mercado_secundario((Path(__file__).parent / "dados" / "ms261119.txt").read_text(encoding="latin-1"))
        self.assertEqual(set(df["titulo"]), {"LTN", "LFT", "NTN-B", "NTN-C", "NTN-F"})
        self.assertEqual(len(df), 50)
        self.assertEqual(df["data_referencia"].iloc[0], REF)
        self.assertTrue(df["pu"].notna().all())
        with self.assertRaises(ValueError):
            ler_mercado_secundario("lixo")
        with self.assertRaises(ValueError):
            vna_implicito(df, "XYZ")
        self.assertTrue(issubclass(DadoIndisponivel, RuntimeError))

    def test_vna_projetado(self):
        self.assertEqual(aniversario_anterior(date(2026, 11, 19)), date(2026, 11, 15))
        self.assertEqual(aniversario_anterior(date(2026, 11, 3)), date(2026, 10, 15))
        v = vna_ntnb_projetado(4770.0, 0.35, date(2026, 11, 19))
        self.assertAlmostEqual(v, 4770.0 * 1.0035 ** (4 / 30), places=5)
        self.assertAlmostEqual(ipca_projetado_implicito(4770.0 * 1.0035 ** (4 / 30), 4770.0, date(2026, 11, 19)), 0.35, places=8)


if __name__ == "__main__":
    unittest.main()
