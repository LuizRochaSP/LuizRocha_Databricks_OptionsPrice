"""Curvas para os gráficos do dashboard: só onde há títulos, termo de 1 ano suave (arquivo ANBIMA de 19/11/2026)."""
import unittest
from pathlib import Path

import numpy as np

from renda_fixa.calendario import dias_uteis
from renda_fixa.dados.anbima import ler_mercado_secundario
from renda_fixa.mercado import curvas_para_grafico, montar_mercado

DADOS = Path(__file__).parent / "dados"


class TestCurvasParaGrafico(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = montar_mercado(ler_mercado_secundario((DADOS / "ms261119.txt").read_text(encoding="latin-1")))
        cls.tc = curvas_para_grafico(cls.m)
        du = lambda nomes: [dias_uteis(cls.m.ref, v) for v in cls.m.df[cls.m.df.titulo.isin(nomes)].vencimento]
        cls.pre_max = max(du(("LTN", "NTN-F")))
        cls.real_min, cls.real_max = min(du(("NTN-B",))), max(du(("NTN-B",)))

    def test_prazos_comecam_em_zero(self):
        self.assertGreater(self.tc["anos"].min(), 0.0)  # o eixo chegava a -20 anos

    def test_pre_so_ate_o_ultimo_prefixado(self):
        tc = self.tc
        for c in ("pre", "pre_nss"):
            self.assertTrue(tc.loc[tc.du > self.pre_max, c].isna().all())
            self.assertTrue(tc.loc[tc.du <= self.pre_max, c].notna().all())
        # a NSS pré caía a 14% em 35 anos por extrapolação; dentro da faixa fica perto dos títulos
        self.assertGreater(tc["pre_nss"].min(), 13.0)

    def test_real_e_inflacao_so_entre_as_ntnb(self):
        tc = self.tc
        fora = (tc.du < self.real_min) | (tc.du > self.real_max)
        self.assertTrue(tc.loc[fora, ["real", "real_nss"]].isna().all().all())
        # a NSS real ia a -5% em 1 mês (antes da primeira NTN-B)
        self.assertGreater(tc["real_nss"].min(), 2.0)
        # a implícita começa na primeira NTN-B (perto de 10% em 6 meses é dado: IPCA corrente alto)
        self.assertGreaterEqual(tc.loc[tc["inflacao_implicita"].notna(), "du"].min(), self.real_min)
        self.assertTrue(tc.loc[tc.du > self.pre_max, "inflacao_implicita"].isna().all())

    def test_termo_nss_suave_e_dentro_dos_prefixados(self):
        tc = self.tc
        ok = tc["termo_pre_1a_nss"].notna()
        self.assertTrue((tc.loc[ok, "du"] + 252 <= self.pre_max).all())
        self.assertEqual(ok.sum(), tc["termo_pre_1a"].notna().sum())
        # suavidade pela curvatura (segunda diferença): a inclinação pode ser grande, o serrilhado não
        curv_nss = np.abs(np.diff(tc.loc[ok, "termo_pre_1a_nss"], 2)).max()
        curv_boot = np.abs(np.diff(tc.loc[ok, "termo_pre_1a"], 2)).max()
        self.assertLess(curv_nss, 0.02)
        self.assertGreater(curv_boot, curv_nss * 5)  # o bootstrap é serrilhado

    def test_mesmos_valores_da_tabela_dentro_da_faixa(self):
        from renda_fixa.mercado import tabela_curvas
        ref = tabela_curvas(self.m, vertices=(252, 504, 1260))
        for du, linha in ref.set_index("du").iterrows():
            row = self.tc[self.tc.du == du].iloc[0]
            self.assertAlmostEqual(row["pre"], linha["pre"])
            self.assertAlmostEqual(row["real"], linha["real"])


if __name__ == "__main__":
    unittest.main()
