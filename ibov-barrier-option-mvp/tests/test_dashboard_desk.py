"""Structuring Desk para IBOV e S&P 500: dados de mercado do dia (mercado.py) e a aba (desk.py) no AppTest."""
import importlib.util
import math
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.append(str(DASHBOARD))  # no fim: dashboard/ibov_barrier não pode esconder src/
HAS_ST = all(importlib.util.find_spec(m) is not None for m in ("streamlit", "plotly", "scipy"))

if importlib.util.find_spec("scipy") is not None:
    from scipy.interpolate import griddata

    import mercado as mk

# ---- dados sintéticos (os mesmos no teste e no script do AppTest) ----
DADOS = textwrap.dedent("""
    import numpy as np
    import pandas as pd

    def superficie(spot, curva_t, r_t, q_t):
        rows = []
        for t in curva_t:
            f = spot * np.exp((np.interp(t, curva_t, r_t) - np.interp(t, curva_t, q_t)) * t)
            for k in np.linspace(-0.35, 0.25, 13):
                rows.append({"T": t, "log_moneyness": k, "strike": f * np.exp(k), "expiration": f"E{t:.3f}",
                             "implied_vol": 0.20 - 0.25 * k + 0.02 * k * k + 0.01 * t})
        return pd.DataFrame(rows)

    IBOV_META = {"market_date": "2026-11-19", "spot": 166376.43}
    IBOV_CURVA = pd.DataFrame({"tenor_bd": [21.0, 63.0, 126.0, 189.0],
                               "r_cont": [0.128, 0.130, 0.133, 0.1345], "q_cont": [0.020, 0.012, 0.0095, 0.0093]})
    IBOV_SUP = superficie(166376.43, [0.08, 0.17, 0.25, 0.32], [0.128, 0.129, 0.130, 0.131], [0.02, 0.015, 0.012, 0.011])

    SPX_T = [0.05, 0.25, 0.5, 1.0, 1.5]
    SPX_R = [0.0469, 0.0469, 0.0476, 0.0490, 0.0507]
    SPX_Q = [-0.10, -0.013, -0.0019, 0.0040, 0.0072]
    SPX = {"metadata": {"market_date": "2026-11-20", "spot": 6910.39},
           "forwards": pd.DataFrame({"T": SPX_T, "r": SPX_R, "q": SPX_Q}),
           "options": superficie(6910.39, SPX_T, SPX_R, SPX_Q)}
    QUANTO_REF = {"X0": 5.8889, "r_brl": 0.1293, "vol_fx": 0.1154, "rho": -0.24, "data": "2026-11-20",
                  "janela_rho": "completo"}
""")
NS = {}
exec(DADOS, NS)


@unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "scipy ausente")
class MercadoTests(unittest.TestCase):
    def ibov(self):
        return mk.mercado_ibov(NS["IBOV_META"], NS["IBOV_CURVA"], NS["IBOV_SUP"])

    def test_ibov_flat_forward(self):
        m, c = self.ibov(), NS["IBOV_CURVA"]
        for du, r in zip(c["tenor_bd"], c["r_cont"]):  # nos vértices: a taxa do vértice
            self.assertAlmostEqual(m.r(du / 252), r, places=12)
        # entre vértices: r·DU linear (forward constante), como o FlatForwardInterpolator
        rt = np.interp(100.0, c["tenor_bd"], c["r_cont"] * c["tenor_bd"])
        self.assertAlmostEqual(m.r(100 / 252), rt / 100, places=12)
        # depois do último vértice: segue o forward do último segmento (não fica plana)
        f = (0.1345 * 189 - 0.133 * 126) / 63
        self.assertAlmostEqual(m.r(252 / 252), (0.1345 * 189 + f * 63) / 252, places=12)
        self.assertAlmostEqual(m.q(100 / 252), float(np.interp(100.0, c["tenor_bd"], c["q_cont"])), places=12)

    def test_vol_regra_do_notebook_01(self):
        m, sup = self.ibov(), NS["IBOV_SUP"]
        pts, vals = sup[["log_moneyness", "T"]].to_numpy(), sup["implied_vol"].to_numpy()
        for strike, t in ((160_000.0, 0.2), (150_000.0, 0.5)):  # dentro e fora do casco convexo
            k = math.log(strike / m.forward(t))
            lin = griddata(pts, vals, np.array([[k, t]]), method="linear")[0]
            esperado = lin if np.isfinite(lin) else griddata(pts, vals, np.array([[k, t]]), method="nearest")[0]
            vol, regra = m.vol(strike, t)
            self.assertAlmostEqual(vol, esperado, places=12)
            self.assertEqual(regra, "linear" if np.isfinite(lin) else "vizinho mais próximo")

    def test_spx_forward_da_paridade(self):
        m, b = mk.mercado_spx(NS["SPX"]), NS["SPX"]
        for t, r, q in zip(NS["SPX_T"], NS["SPX_R"], NS["SPX_Q"]):
            self.assertAlmostEqual(m.forward(t), 6910.39 * math.exp((r - q) * t), places=8)
        self.assertEqual(m.data, "2026-11-20")
        self.assertIsNone(mk.mercado_spx(None))
        self.assertIsNone(mk.mercado_spx({"metadata": b["metadata"]}))

    def test_padroes_e_reserva(self):
        reserva = {"spot": 7000.0, "rate": 0.045, "q": 0.012, "vol": 0.18}
        self.assertEqual(mk.padroes(None, reserva), dict(reserva, origem=None))
        m = mk.mercado_spx(NS["SPX"])
        p = mk.padroes(m, reserva)
        self.assertEqual((p["spot"], p["origem"]), (6910.39, "2026-11-20"))
        self.assertAlmostEqual(p["rate"], 0.0476)
        self.assertAlmostEqual(p["vol"], m.vol(m.forward(0.5), 0.5)[0])

    def test_referencia_quanto(self):
        meta = {"X0": 5.8889, "r_brl": 0.1293, "vol_fx": 0.1154, "rho": -0.24, "market_date": "2026-11-20"}
        ref = mk.referencia_quanto({"metadata": meta})
        self.assertEqual((ref["X0"], ref["rho"], ref["data"]), (5.8889, -0.24, "2026-11-20"))
        self.assertIsNone(mk.referencia_quanto({"metadata": {**meta, "rho": None}}))
        self.assertIsNone(mk.referencia_quanto(None))


def _num(texto: str) -> float:
    """'US$ 1.234,56' -> 1234.56 (o script instala o formato_br, como o app)."""
    return float(texto.split()[-1].replace(".", "").replace(",", "."))


def _br(texto: str) -> str:
    """O mesmo texto no padrão brasileiro que o app mostra (ponto no milhar, vírgula no decimal)."""
    import formato_br
    return formato_br.br_texto(texto)


@unittest.skipUnless(HAS_ST, "streamlit/plotly/scipy ausentes")
class DeskAppTests(unittest.TestCase):
    """A aba de verdade (desk.render) no AppTest, com os dados sintéticos acima."""

    def app(self, ativo: str, com_mercado: bool = True):
        from streamlit.testing.v1 import AppTest
        script = DADOS + textwrap.dedent(f"""
            import sys
            sys.path.insert(0, r"{DASHBOARD}")
            import streamlit as st
            import formato_br
            formato_br.instalar(st)  # como o app: números no padrão brasileiro
            import desk
            import mercado as mk
            if "{ativo}" == "ibov":
                m = mk.mercado_ibov(IBOV_META, IBOV_CURVA, IBOV_SUP) if {com_mercado} else None
                desk.render(desk.IBOV, m)
            else:
                desk.render(desk.SPX, mk.mercado_spx(SPX) if {com_mercado} else None, QUANTO_REF)
        """)
        # arquivo em UTF-8 + from_file: no Streamlit 1.38 (o do Databricks Apps), o from_string grava o script na
        # codificação do sistema e não lê os acentos de volta no Windows
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        arq = Path(tmp.name) / "desk_app.py"
        arq.write_text(script, encoding="utf-8")
        return AppTest.from_file(str(arq), default_timeout=300).run()

    @staticmethod
    def calcular(at):
        next(b for b in at.button if b.label == "Calcular Preço").click().run()
        return {m.label: m.value for m in at.metric}

    def test_ibov_com_parametros_antigos_reproduz_o_motor(self):
        from ibov_barrier.pricing import BarrierContract, MarketData
        from ibov_barrier.pricing_extensions import price_single_barrier

        at = self.app("ibov")
        self.assertFalse(at.exception)
        tag = "desk_ibov_2026-11-19"
        self.assertAlmostEqual(at.number_input(key=f"{tag}_spot").value, 166376.43)
        at.number_input(key=f"{tag}_spot").set_value(172000.0)
        at.slider(key=f"{tag}_vol").set_value(20.0)
        at.slider(key=f"{tag}_rate").set_value(12.0)
        at.slider(key=f"{tag}_q").set_value(2.0)
        metrics = self.calcular(at.run())
        esperado = price_single_barrier(MarketData(172000.0, 0.12, 0.02, 0.20),
                                        BarrierContract(172000.0 * 0.9, 172000.0 * 0.8, 0.5), option_type="call",
                                        direction="down", knock="out", rebate=0.0, paths=50_000, steps=126, seed=42)
        self.assertEqual(metrics["Preço da Estrutura"], _br(f"R$ {esperado.price:,.2f}"))
        self.assertEqual(metrics["Preço Vanilla"], _br(f"R$ {esperado.vanilla_price:,.2f}"))

    def test_spx_aceita_o_spot_do_indice(self):
        at = self.app("spx", com_mercado=False)
        self.assertFalse(at.exception)
        spot = at.number_input(key="desk_spx_reserva_spot")
        self.assertEqual((spot.label, spot.value), ("Spot Inicial (pontos)", 7000.0))
        spot.set_value(6910.39).run()
        self.assertTrue(at.toggle(key="desk_spx_mercado").disabled)  # sem arquivo: modo mercado desligado
        metrics = self.calcular(at)
        self.assertTrue(metrics["Preço da Estrutura"].startswith("US$ "))
        self.assertFalse([e for e in at.error if "Ops" in e.value])

    def test_spx_dados_de_mercado(self):
        at = self.app("spx")
        at.toggle(key="desk_spx_mercado").set_value(True).run()
        self.assertFalse(at.exception)
        self.assertEqual([n.label for n in at.number_input if n.label.startswith("Spot")], [])  # spot do arquivo
        m = mk.mercado_spx(NS["SPX"])
        vol, _ = m.vol(6910.39 * 0.9, 0.5)  # strike padrão 90%, prazo padrão 6 meses (126/252)
        texto = " ".join(x.value for x in at.markdown)
        self.assertIn(_br(f"**{vol:.2%}**"), texto)
        self.assertIn(_br(f"**{m.r(0.5):.3%}**"), texto)

    def test_spx_quanto_bate_com_a_formula_fechada(self):
        from ibov_barrier.pricing_quanto import QuantoMarket, price_quanto_vanilla, quanto_market_data
        from ibov_barrier.pricing import BarrierContract
        from ibov_barrier.pricing_extensions import price_single_barrier

        at = self.app("spx")
        at.toggle(key="desk_spx_mercado").set_value(True)
        at.radio(key="desk_spx_liquidacao").set_value("R$ (Quanto, câmbio fixo)")
        metrics = self.calcular(at.run())
        self.assertFalse(at.exception)
        m, ref, T = mk.mercado_spx(NS["SPX"]), NS["QUANTO_REF"], 0.5
        K, H = 6910.39 * 0.9, 6910.39 * 0.8
        qm = QuantoMarket(spot=6910.39, rate_usd=m.r(T), dividend_yield=m.q(T), vol_asset=m.vol(K, T)[0],
                          fx_spot=ref["X0"], rate_brl=ref["r_brl"], vol_fx=ref["vol_fx"], correlation=ref["rho"])
        # vanilla do Desk em R$ = fórmula fechada do Quanto (Black no forward ajustado, desconto em r BRL)
        self.assertAlmostEqual(_num(metrics["Preço Vanilla"]), price_quanto_vanilla(qm, K, T), places=2)
        res = price_single_barrier(quanto_market_data(qm), BarrierContract(K, H, T), option_type="call",
                                   direction="down", knock="out", rebate=0.0, paths=50_000, steps=126, seed=42)
        self.assertEqual(metrics["Preço da Estrutura"], _br(f"R$ {res.price * ref['X0']:,.2f}"))
        self.assertTrue(any("Ajuste Quanto" in c.value for c in at.caption))

    def test_quanto_payout_em_reais(self):
        # One-Touch Down com payout de R$ 100: o preço em R$ não depende do câmbio fixado
        at = self.app("spx")
        at.radio(key="desk_spx_liquidacao").set_value("R$ (Quanto, câmbio fixo)")
        at.selectbox(key="desk_estrategia").select("One-Touch Down").run()
        precos = []
        for x0 in (5.0, 6.0):
            at.number_input(key="desk_quanto_2026-11-20_x0").set_value(x0).run()
            metrics = self.calcular(at)
            precos.append(_num(metrics["Preço da Estrutura"]))
            self.assertLess(_num(metrics["Benchmark Analítico"]), 100.0)
        self.assertAlmostEqual(precos[0], precos[1], places=2)


if __name__ == "__main__":
    unittest.main()
