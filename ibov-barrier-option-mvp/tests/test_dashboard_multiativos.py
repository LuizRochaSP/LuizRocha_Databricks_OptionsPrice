"""Contrato entre os notebooks 07/09 (gravação) e as abas multiativos do dashboard (leitura e desenho)."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ibov_barrier.results_bundle import read_bundle, results_filename, write_bundle

HAS_PLOTLY = importlib.util.find_spec("plotly") is not None
DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
TS = datetime(2026, 11, 22, 21, 42, tzinfo=ZoneInfo("America/Sao_Paulo"))


def spx_objects():
    rng = np.random.default_rng(1)
    exps = [date(2026, 12, 18), date(2027, 2, 19), date(2027, 5, 21), date(2027, 11, 19), date(2028, 12, 15)]
    rows = []
    for e in exps:
        t = (e - date(2026, 11, 20)).days / 365
        f = 6910 * np.exp(0.045 * t)
        for k in np.linspace(-0.3, 0.15, 12):
            for opt in ("C", "P"):
                rows.append({"expiration": e, "T": t, "strike": f * np.exp(k), "option_type": opt,
                             "market_price": 10.0, "forward": f, "log_moneyness": k,
                             "implied_vol": 0.20 - 0.3 * k + rng.normal(0, 0.002)})
    options = pd.DataFrame(rows)
    forwards = pd.DataFrame({"vencimento": exps, "T": options.groupby("expiration")["T"].first().to_numpy(),
                             "forward": 7000.0, "r": 0.045, "r do vencimento": [np.nan, np.nan, 0.046, 0.044, 0.045],
                             "no ajuste": [False, False, True, True, True], "q": 0.002})
    arb = pd.DataFrame({"expiration": exps * 2, "option_type": ["C"] * 5 + ["P"] * 5,
                        "antes": 100, "depois": 90, "removidos": 10, "% removido": 0.1,
                        "T": list(forwards["T"]) * 2})
    return {
        "metadata": {"schema_version": 1, "kind": "spx_superficie", "execution_timestamp": TS,
                     "market_date": date(2026, 11, 20), "spot": 6910.39, "sofr": 0.0365, "vencimentos": 5,
                     "pontos": len(options), "avisos": ["exemplo"], "cobertura": None, "faixa q": [0.0, 0.03],
                     "fonte": "teste"},
        "summary": pd.Series({"vol ATM 1m": "25.00%", "vol ATM 12m": "21.00%", "r 1a": "4.500%", "q 1a": "0.200%",
                              "pontos": np.int64(len(options))}),
        "forwards": forwards, "options": options, "funnel": pd.DataFrame({"etapa": ["a"], "quantidade": [1]}),
        "arb": arb,
    }


def quanto_objects(with_lv=True):
    products = [("quanto", "call"), ("quanto", "put"), ("compo", "call"), ("compo", "put"),
                ("converted", "call"), ("converted", "put")]
    prices = pd.DataFrame([{"produto": p, "tipo": t, "strike": 6910.39, "fechada (BRL)": 3000.0,
                            "com barreira MC (BRL)": 2900.0, "erro padrão": 1.2, "prob. knock-out": 0.3,
                            "com barreira analítica (BRL)": 2899.0 if t == "call" else np.nan}
                           for p, t in products])
    smile = pd.DataFrame([{"produto": p, "tipo": t, "vol em K": 7.5377, "vol local": 7.0164,
                           "erro padrão vol local": 0.014, "prob. KO vol local": 0.3165,
                           "vanilla vol local": 7.6, "vanilla fechada vol em K": 7.7,
                           "vol local - vol em K (BRL)": -212.1} for p, t in products])
    rhos, vols = np.linspace(-0.9, 0.9, 5), np.linspace(0.05, 0.3, 4)
    curves = pd.concat([pd.DataFrame({"parâmetro": par, "x": xs, "produto": name, "call (% nocional)": 7 + xs})
                        for name in ("Quanto", "Compo", "Convertida") for par, xs in (("rho", rhos), ("vol_fx", vols))],
                       ignore_index=True)
    est = pd.DataFrame({"vol_asset": [0.15], "vol_fx": [0.115], "correlation": [-0.24], "observations": [106]},
                       index=pd.Index(["semanal"], name="janela"))
    objects = {
        "metadata": {"schema_version": 1, "kind": "quanto", "execution_timestamp": TS,
                     "market_date": date(2026, 11, 20), "expiry": date(2027, 5, 21), "T": 0.4979, "dias": 182,
                     "S0": 6910.39, "X0": 5.8889, "notional": 40694.6, "K": 6910.39, "H": 5873.83,
                     "strike_pct": 1.0, "barrier_pct": 0.85, "forward": 7082.84, "r_usd": 0.0476, "q": -0.0019,
                     "vol_k": 0.2349, "vol_h": 0.3064, "r_brl": 0.1293, "vol_fx": 0.1154, "rho": -0.2406,
                     "janela_rho": "completo", "quanto_adj_bps": 65.3, "compo_vol": 0.2355, "fx_forward": 6.1333,
                     "paths": 200000, "steps": 125, "seed": 42, "chain_source": "disco", "ptax_source": "SGS",
                     "close_source": "cache", "lv_error": None, "lv_clipped_share": 0.069 if with_lv else None,
                     "widgets": {"prazo_dias": "182"}},
        "spx_inputs": pd.Series({"spot": 6910.39, "no ajuste r(T)": True, "interpolação K": "linear"}),
        "brl": {"fonte": "DI1", "data curva": date(2026, 11, 19), "DU": 122, "r_brl": 0.1293},
        "fx_estimates": est,
        "rho_band": {"usado": -0.2406, "52 semanas": -0.3464},
        "rolling_corr": pd.DataFrame({"correlação 26 semanas": [-0.3, -0.26]},
                                     index=pd.Index(pd.to_datetime(["2026-11-13", "2026-11-20"]), name="data")),
        "prices": prices,
        "sens_table": pd.DataFrame({"Compo": [0.1007], "Convertida": [0.0782], "Quanto": [0.0770]},
                                   index=pd.Index(["usado (-0.24)"], name="rho")),
        "sens_curves": curves,
        "smile_barrier": smile,
    }
    if with_lv:
        kf = np.linspace(0.7, 1.15, 10)
        objects.update({
            "lv_smile": pd.DataFrame({"K/F": kf, "implícita": 0.2 + 0.3 * (1 - kf), "local T/2": 0.2 + 0.6 * (1 - kf)}),
            "lv_clip_table": pd.DataFrame({"período": ["até 2 semanas", "T/2 a T"], "75%-85%": [0.352, 0.0],
                                           "105%-125%": [0.36, 0.065]}),
            "lv_clip_reason": pd.Series({"abaixo do piso": 0.044, "acima do teto": 0.008, "denominador": 0.001}),
            "lv_fit": pd.DataFrame({"dias": [7, 13], "RMSE vol": [0.0072, 0.0012]}),
        })
    return objects


class FakeColumn:
    def __init__(self, log):
        self.log = log

    def metric(self, label, value, delta=None):
        self.log.append(("metric", label, value))


class FakeExpander:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeStreamlit:
    """Registra as chamadas; a seleção de arquivo devolve sempre o primeiro."""

    def __init__(self):
        self.log = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.append((name, args[0] if args else None))
            if name == "selectbox":
                return kwargs["options"][0]
            if name == "columns":
                return [FakeColumn(self.log) for _ in range(args[0])]
            if name == "expander":
                return FakeExpander()
            if name == "dataframe":
                obj = args[0]
                if hasattr(obj, "to_html"):           # Styler: força a formatação a rodar
                    obj.to_html()
            return None
        return call

    def charts(self):
        return [entry for entry in self.log if entry[0] == "plotly_chart"]


class BundleRoundTripTests(unittest.TestCase):
    def test_filename_and_roundtrip(self):
        self.assertEqual(results_filename("quanto", TS), "RESULTS_quanto_20261122_214200.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path = write_bundle(Path(tmp) / "x.csv", quanto_objects())
            back = read_bundle(path)
        self.assertEqual(back["metadata"]["market_date"], "2026-11-20")
        self.assertIsNone(back["metadata"]["lv_error"])
        self.assertEqual(back["brl"]["data curva"], "2026-11-19")
        self.assertEqual(list(back["fx_estimates"].columns[:2]), ["janela", "vol_asset"])
        pd.testing.assert_series_equal(back["prices"]["com barreira MC (BRL)"],
                                       quanto_objects()["prices"]["com barreira MC (BRL)"])
        self.assertTrue(np.isnan(back["prices"].loc[1, "com barreira analítica (BRL)"]))
        self.assertEqual(back["spx_inputs"]["no ajuste r(T)"], True)

    def test_nan_in_json_becomes_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            back = read_bundle(write_bundle(Path(tmp) / "x.csv", {"m": {"a": float("nan"), "b": np.float64(1.5)}}))
        self.assertEqual(back["m"], {"a": None, "b": 1.5})

    def test_sorted_frame_with_shuffled_index(self):
        # tabela "arb" do 07: reset_index + sort_values deixa o índice fora de ordem; antes isso gravava
        # dois campos "index" e a leitura quebrava com "Index data must be 1-dimensional"
        arb = pd.DataFrame({"expiration": ["2027-01-15", "2026-12-18"], "antes": [10, 20], "T": [0.15, 0.07]})
        arb = arb.sort_values("T")
        dated = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.to_datetime(["2026-11-19", "2026-11-20"]))
        with tempfile.TemporaryDirectory() as tmp:
            back = read_bundle(write_bundle(Path(tmp) / "x.csv", {"arb": arb, "dated": dated}))
        self.assertEqual(list(back["arb"].columns), ["expiration", "antes", "T"])
        self.assertEqual(back["arb"]["antes"].tolist(), [20, 10])
        self.assertEqual(list(back["dated"].columns), ["index", "v"])

    def test_reads_files_written_before_the_fix(self):
        # payload como o do RESULTS_spx_superficie_20261123_201211.csv: "index" duplicado no schema
        legacy = json.dumps({
            "schema": {"fields": [{"name": "index", "type": "integer"}, {"name": "index", "type": "integer"},
                                  {"name": "expiration", "type": "string"}, {"name": "antes", "type": "integer"}],
                       "primaryKey": ["index"], "pandas_version": "1.4.0"},
            "data": [{"level_0": 0, "index": 1, "expiration": "2026-12-18", "antes": 20},
                     {"level_0": 1, "index": 0, "expiration": "2027-01-15", "antes": 10}]})
        frame = pd.DataFrame([{"object_name": "arb", "object_type": "dataframe", "payload_json": legacy}])
        with self.assertRaises(ValueError):
            pd.read_json(io.StringIO(legacy), orient="table")
        back = read_bundle(frame)
        self.assertEqual(list(back["arb"].columns), ["expiration", "antes"])
        self.assertEqual(back["arb"]["antes"].tolist(), [20, 10])
        if HAS_PLOTLY:  # o leitor do dashboard (cópia própria) segue a mesma regra
            sys.path.insert(0, str(DASHBOARD))
            import multiativos
            pd.testing.assert_frame_equal(multiativos.parse_bundle(frame)["arb"], back["arb"])


@unittest.skipUnless(HAS_PLOTLY, "plotly ausente")
class DashboardTabsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(DASHBOARD))
        import multiativos
        cls.m = multiativos

    def _bundle(self, objects):
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.read_csv(write_bundle(Path(tmp) / "x.csv", objects))
        return self.m.parse_bundle(frame)

    def _render(self, fn, objects, files=("/r/RESULTS_x_20261122_214200.csv",)):
        st = FakeStreamlit()
        bundle = self._bundle(objects) if objects else None
        fn(st, list(files), lambda _p: bundle)
        return st

    def test_spx_tab(self):
        st = self._render(self.m.render_spx, spx_objects())
        self.assertEqual(len(st.charts()), 4)
        metrics = {e[1]: e[2] for e in st.log if e[0] == "metric"}
        self.assertEqual(metrics["Spot SPX"], "6,910.39")
        self.assertEqual(metrics["Vol ATM 12m"], "21.00%")
        for _, fig in st.charts():
            self.assertEqual(fig.layout.paper_bgcolor, self.m.BG)

    def test_quanto_tab(self):
        st = self._render(self.m.render_quanto, quanto_objects())
        self.assertEqual(len(st.charts()), 2)
        metrics = {e[1]: e[2] for e in st.log if e[0] == "metric"}
        self.assertEqual(metrics["Ajuste Quanto"], "+65 bps")
        self.assertEqual(metrics["Correlação usada"], "-0.24 (completo)")
        view = self.m.prices_view(self._bundle(quanto_objects())["prices"], 40694.6)
        self.assertAlmostEqual(view.loc[0, "com barreira MC (% nocional)"], 2900.0 / 40694.6)
        self.assertEqual(view.loc[4, "produto"], "Convertida")

    def test_local_vol_tab(self):
        st = self._render(self.m.render_local_vol, quanto_objects())
        self.assertEqual(len(st.charts()), 3)
        metrics = {e[1]: e[2] for e in st.log if e[0] == "metric"}
        self.assertEqual(metrics["Quanto DOC call, vol local"], "7.0164%")
        self.assertEqual(metrics["Vol local cortada (barreira a 125%)"], "6.9%")

    def test_local_vol_tab_without_lv(self):
        objects = quanto_objects(with_lv=False)
        objects["metadata"]["lv_error"] = "ValueError: poucas fatias"
        st = self._render(self.m.render_local_vol, objects)
        self.assertIn(("error", "Vol local indisponível nesta execução: ValueError: poucas fatias"), st.log)
        self.assertEqual(st.charts(), [])

    def test_empty_folder_shows_instructions(self):
        for fn in (self.m.render_spx, self.m.render_quanto, self.m.render_local_vol):
            st = self._render(fn, None, files=())
            self.assertTrue(any(e[0] == "info" for e in st.log))
            self.assertEqual(st.charts(), [])

    def test_captions_hide_workspace_user(self):
        objects = quanto_objects()
        objects["metadata"]["chain_source"] = "disco (/Workspace/Users/alguem@exemplo.com/spx_dados/brutos/2026-11-20)"
        st = self._render(self.m.render_quanto, objects)
        captions = " ".join(str(e[1]) for e in st.log if e[0] == "caption")
        self.assertIn("disco (~/spx_dados/brutos/2026-11-20)", captions)
        self.assertNotIn("@", captions)
        self.assertEqual(self.m.short_path("/Users/a.b@c.com/x/y"), "~/x/y")
        self.assertEqual(self.m.short_path("cache"), "cache")

    def test_prazo_metric_not_truncated(self):
        st = self._render(self.m.render_quanto, quanto_objects())
        metrics = {e[1]: e[2] for e in st.log if e[0] == "metric"}
        self.assertEqual(metrics["Prazo (vence 2027-05-21)"], "182 dias")

    def test_prices_without_closed_form_are_nan(self):
        view = self.m.prices_view(self._bundle(quanto_objects())["prices"], 40694.6)
        col = "com barreira analítica (BRL)"
        self.assertTrue(pd.api.types.is_float_dtype(view[col]))
        self.assertTrue(view.loc[view["tipo"] == "put", col].isna().all())
        self.assertAlmostEqual(view.loc[0, col], 2899.0)

    def test_empty_cells_show_dash_not_none(self):
        # o st.dataframe ignora o na_rep do Styler: vazio tem de chegar como texto "—"
        view = self.m.prices_view(self._bundle(quanto_objects())["prices"], 40694.6)
        shown = self.m.sem_vazios(view, {"com barreira analítica (BRL)": "{:,.2f}"}).data
        col = shown["com barreira analítica (BRL)"]
        self.assertEqual(col[view["tipo"] == "put"].unique().tolist(), ["—"])
        self.assertEqual(col.iloc[0], "2.899,00")  # padrão brasileiro
        self.assertFalse(shown.isna().any().any())
        fwd = self.m.sem_vazios(spx_objects()["forwards"]).data
        self.assertEqual(fwd["r do vencimento"].iloc[0], "—")
        self.assertTrue(pd.api.types.is_float_dtype(fwd["forward"]))  # sem vazios: segue numérica

    def test_q_axis_ignores_short_tenor_noise(self):
        fwd = spx_objects()["forwards"].copy()
        fwd.loc[0, "T"], fwd.loc[0, "q"] = 0.02, -0.30  # q de -30% no primeiro vencimento (tela 7)
        fig = self.m.fig_spx_rates(fwd, 0.0365)
        lo, hi = fig.layout.yaxis2.range
        self.assertGreater(lo, -2.0)
        self.assertLess(hi, 5.0)
        self.assertTrue(any("fora da escala" in a.text for a in fig.layout.annotations))
        clean = self.m.fig_spx_rates(spx_objects()["forwards"], 0.0365)
        self.assertFalse(any("fora da escala" in (a.text or "") for a in clean.layout.annotations))

    def test_clip_heatmap_text_contrast(self):
        fig = self.m.fig_lv_clip(quanto_objects()["lv_clip_table"])
        colors = {a.text: a.font.color for a in fig.layout.annotations if a.text.endswith("%")}
        self.assertEqual(colors["35.2%"], "#0B0B0B")  # célula clara: texto escuro
        self.assertEqual(colors["0.0%"], "white")

    def test_file_label(self):
        self.assertEqual(self.m.file_label("/a/RESULTS_quanto_20261122_214200.csv", self.m.QUANTO_PREFIX),
                         "22/11/2026 21:42:00")


if __name__ == "__main__":
    unittest.main()
