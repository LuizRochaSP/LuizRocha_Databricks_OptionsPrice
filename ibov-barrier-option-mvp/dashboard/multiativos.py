"""Abas multiativos do dashboard: SPX, Quanto/Compo em BRL e Vol local.

Lê os arquivos que os notebooks 07 (`RESULTS_spx_superficie_*.csv`) e 09 (`RESULTS_quanto_*.csv`) gravam em
`ibov-barrier-results`, no mesmo formato do notebook 01 (uma linha por objeto: object_name, object_type,
payload_json). As funções `fig_*` só montam figuras (testáveis sem Streamlit); as `render_*` desenham a aba.
"""
from __future__ import annotations

import json
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SPX_PREFIX = "RESULTS_spx_superficie_"
QUANTO_PREFIX = "RESULTS_quanto_"

# Padrão gráfico do projeto: fundo preto, texto branco, rampa quente, marcador e traço por série.
BG, GRID, MUTED = "#0B0B0B", "#2A2A2A", "#8C8C8C"
WARM = ["#FFF3B0", "#FFC233", "#FF8A1F", "#F0432F", "#D6246E"]
SYMBOLS = ["circle", "square", "triangle-up", "diamond", "triangle-down"]
DASHES = ["solid", "dash", "dashdot", "dot", "longdashdot"]
PLOT_CONFIG = {"displaylogo": False, "displayModeBar": True, "scrollZoom": True,
               "toImageButtonOptions": {"format": "png", "scale": 2}}
PRODUCT_LABEL = {"quanto": "Quanto", "compo": "Compo", "converted": "Convertida"}


# ---------------------------------------------------------------- leitura

def parse_bundle(frame: pd.DataFrame) -> dict:
    """{nome: dict | DataFrame} a partir do CSV já carregado."""
    out = {}
    for row in frame.itertuples(index=False):
        if row.object_type == "dataframe":
            out[row.object_name] = pd.read_json(StringIO(row.payload_json), orient="table")
        else:
            out[row.object_name] = json.loads(row.payload_json)
    return out


def file_label(path: str, prefix: str) -> str:
    """RESULTS_quanto_20261122_214200.csv -> '22/11/2026 21:42:00'."""
    stamp = path.split("/")[-1].replace(prefix, "").replace(".csv", "")
    try:
        return pd.to_datetime(stamp, format="%Y%m%d_%H%M%S").strftime("%d/%m/%Y %H:%M:%S")
    except ValueError:
        return stamp


# ---------------------------------------------------------------- estilo

def style(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, showline=True,
                tickfont=dict(color="white"), title_font=dict(color="white"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="white", size=12),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(color="white", size=16)),
        legend=dict(bgcolor="#141414", bordercolor="#3A3A3A", borderwidth=1, font=dict(color="white")),
        hoverlabel=dict(bgcolor="#141414", bordercolor="#3A3A3A", font=dict(color="white")),
        height=height, margin=dict(t=80, l=60, r=20, b=50))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_annotations(font=dict(color="white"))
    return fig


def _line(i: int, width: float = 2.0) -> dict:
    return dict(color=WARM[i % len(WARM)], dash=DASHES[i % len(DASHES)], width=width)


def _marker(i: int, size: int = 6) -> dict:
    return dict(symbol=SYMBOLS[i % len(SYMBOLS)], size=size, color=WARM[i % len(WARM)])


# ---------------------------------------------------------------- SPX

def reference_expiries(options: pd.DataFrame, targets=None) -> dict:
    """Vencimento mais próximo de cada prazo de referência (1m, 3m, 6m, 12m, 24m)."""
    targets = targets or {"1m": 1 / 12, "3m": 0.25, "6m": 0.5, "12m": 1.0, "24m": 2.0}
    by_exp = options.groupby("expiration")["T"].first().sort_values()
    chosen = {}
    for label, t in targets.items():
        exp = (by_exp - t).abs().idxmin()
        if exp not in chosen.values():
            chosen[label] = exp
    return chosen


def fig_spx_smiles(options: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, (label, exp) in enumerate(reference_expiries(options).items()):
        s = options.loc[options["expiration"].eq(exp)].sort_values("log_moneyness")
        s = s.loc[s["log_moneyness"].between(-0.4, 0.2)]
        fig.add_trace(go.Scatter(
            x=s["log_moneyness"], y=s["implied_vol"] * 100, mode="lines+markers",
            name=f"{label} ({str(exp)[:10]})", line=_line(i), marker=_marker(i, 5),
            customdata=s["strike"], hovertemplate="K %{customdata:,.0f}<br>ln(K/F) %{x:+.3f}<br>"
                                                  "vol %{y:.2f}%<extra>%{fullData.name}</extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, dash="dot"))
    fig.update_xaxes(title="ln(K/F)")
    fig.update_yaxes(title="vol implícita (%)")
    return style(fig, "SPX: smiles nos prazos de referência", height=480)


def atm_term_structure(options: pd.DataFrame) -> pd.DataFrame:
    near = options.assign(dist=options["log_moneyness"].abs()).sort_values("dist")
    return near.groupby("expiration", as_index=False).first().sort_values("T")[["expiration", "T", "implied_vol"]]


def fig_spx_atm(options: pd.DataFrame) -> go.Figure:
    atm = atm_term_structure(options)
    fig = go.Figure(go.Scatter(
        x=atm["T"], y=atm["implied_vol"] * 100, mode="lines+markers", name="vol ATM", line=_line(2),
        marker=_marker(2), customdata=atm["expiration"].astype(str).str[:10],
        hovertemplate="%{customdata}<br>T = %{x:.3f}<br>vol ATM %{y:.2f}%<extra></extra>"))
    fig.update_xaxes(title="prazo (anos)")
    fig.update_yaxes(title="vol ATM (%)")
    return style(fig, "SPX: estrutura a termo da vol ATM")


def fig_spx_rates(forwards: pd.DataFrame, sofr: float | None, q_range=(0.0, 0.03)) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("r(T) da paridade", "q(T) implícito"),
                        horizontal_spacing=0.08)
    hover = "%{customdata}<br>T = %{x:.3f}<br>%{y:.3f}% a.a.<extra>%{fullData.name}</extra>"
    venc = forwards["vencimento"].astype(str).str[:10]
    fig.add_trace(go.Scatter(x=forwards["T"], y=forwards["r"] * 100, mode="lines+markers", name="r (curva)",
                             line=_line(1), marker=_marker(1, 5), customdata=venc, hovertemplate=hover), 1, 1)
    if "r do vencimento" in forwards:
        own = forwards["r do vencimento"].notna()
        fig.add_trace(go.Scatter(x=forwards.loc[own, "T"], y=forwards.loc[own, "r do vencimento"] * 100,
                                 mode="markers", name="r do vencimento", marker=_marker(3, 7),
                                 customdata=venc[own], hovertemplate=hover), 1, 1)
    if sofr is not None:
        fig.add_hline(y=sofr * 100, line=dict(color=MUTED, dash="dot"), annotation_text=f"SOFR {sofr:.2%}",
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=forwards["T"], y=forwards["q"] * 100, mode="lines+markers", name="q implícito",
                             line=_line(2), marker=_marker(2, 5), customdata=venc, hovertemplate=hover), 1, 2)
    fig.add_hrect(y0=q_range[0] * 100, y1=q_range[1] * 100, fillcolor=WARM[2], opacity=0.10, line_width=0,
                  row=1, col=2)
    fig.update_xaxes(title="prazo (anos)")
    fig.update_yaxes(title="% a.a.", row=1, col=1)
    return style(fig, "SPX: taxa e dividendos implícitos por vencimento", height=460)


def fig_spx_arb(arb: pd.DataFrame) -> go.Figure:
    frame = arb.groupby("expiration", as_index=False).agg(T=("T", "first"), antes=("antes", "sum"),
                                                          removidos=("removidos", "sum")).sort_values("T")
    frame["pct"] = frame["removidos"] / frame["antes"] * 100
    fig = go.Figure(go.Bar(x=frame["expiration"].astype(str).str[:10], y=frame["pct"], marker_color=WARM[3],
                           customdata=frame[["removidos", "antes"]],
                           hovertemplate="%{x}<br>%{y:.1f}% (%{customdata[0]} de %{customdata[1]})<extra></extra>"))
    fig.update_xaxes(title="vencimento")
    fig.update_yaxes(title="% removido")
    return style(fig, "SPX: cotações removidas por arbitragem vertical/butterfly")


# ---------------------------------------------------------------- Quanto / Compo

def prices_view(prices: pd.DataFrame, notional: float) -> pd.DataFrame:
    """Tabela de preços com BRL e % do nocional lado a lado."""
    view = prices.copy()
    view["produto"] = view["produto"].map(PRODUCT_LABEL).fillna(view["produto"])
    for col in [c for c in prices.columns if c.endswith("(BRL)")]:
        view[col.replace("(BRL)", "(% nocional)")] = view[col] / notional
    return view


def fig_rolling_corr(rolling: pd.DataFrame, rho: float) -> go.Figure:
    col = [c for c in rolling.columns if c != "data"][0]
    fig = go.Figure(go.Scatter(x=pd.to_datetime(rolling["data"]), y=rolling[col], mode="lines",
                               name="correlação 26 semanas", line=_line(2)))
    fig.add_hline(y=rho, line=dict(color=WARM[0], dash="dash"), annotation_text=f"usada {rho:+.2f}")
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dot"))
    fig.update_yaxes(title="correlação SPX x USDBRL", range=[-1, 1])
    return style(fig, "Correlação móvel SPX x USDBRL (retornos semanais)")


def fig_sensitivity(curves: pd.DataFrame, rho: float, vol_fx: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Correlação SPX x USDBRL", "Vol do USDBRL"))
    for i, (name, grp) in enumerate(curves.groupby("produto", sort=False)):
        for col, param, scale in ((1, "rho", 1.0), (2, "vol_fx", 100.0)):
            g = grp.loc[grp["parâmetro"].eq(param)].sort_values("x")
            fig.add_trace(go.Scatter(x=g["x"] * scale, y=g["call (% nocional)"], mode="lines+markers", name=name,
                                     line=_line(i + 1), marker=_marker(i + 1, 5), legendgroup=name,
                                     showlegend=col == 1), 1, col)
    fig.add_vline(x=rho, line=dict(color=MUTED, dash="dot"), row=1, col=1)
    fig.add_vline(x=vol_fx * 100, line=dict(color=MUTED, dash="dot"), row=1, col=2)
    fig.update_xaxes(title="rho", row=1, col=1)
    fig.update_xaxes(title="vol USDBRL (%)", row=1, col=2)
    fig.update_yaxes(title="call (% do nocional)", row=1, col=1)
    return style(fig, "Call em BRL: sensibilidade aos parâmetros do câmbio", height=460)


# ---------------------------------------------------------------- Vol local

def fig_lv_smile(lv_smile: pd.DataFrame, barrier_over_forward: float | None) -> go.Figure:
    fig = go.Figure()
    series = [c for c in lv_smile.columns if c != "K/F"]
    for i, col in enumerate(series):
        fig.add_trace(go.Scatter(x=lv_smile["K/F"] * 100, y=lv_smile[col] * 100, mode="lines+markers",
                                 name=col, line=_line(i + 1), marker=_marker(i + 1, 5)))
    if barrier_over_forward:
        fig.add_vline(x=barrier_over_forward * 100, line=dict(color=MUTED, dash="dot"),
                      annotation_text="barreira")
    fig.update_xaxes(title="strike / forward (%)")
    fig.update_yaxes(title="vol (%)")
    return style(fig, "Smile no vencimento: implícita x vol local")


def fig_lv_clip(clip_table: pd.DataFrame) -> go.Figure:
    frame = clip_table.set_index("período")
    fig = go.Figure(go.Heatmap(
        z=frame.to_numpy() * 100, x=list(frame.columns), y=list(frame.index),
        colorscale=[[0, BG], [0.5, WARM[2]], [1, WARM[4]]], zmin=0, zmax=100,
        text=[[f"{v:.1%}" for v in row] for row in frame.to_numpy()], texttemplate="%{text}",
        colorbar=dict(title="% cortado", tickfont=dict(color="white"))))
    fig.update_yaxes(autorange="reversed", title="período do prazo")
    fig.update_xaxes(title="faixa strike / forward")
    return style(fig, "Vol local cortada pelos limites, por período e faixa de strike", height=380)


def smile_effect(smile: pd.DataFrame) -> pd.DataFrame:
    """Preço com barreira: vol constante em K x vol local, em % do nocional."""
    view = smile.copy()
    view["produto"] = view["produto"].map(PRODUCT_LABEL).fillna(view["produto"])
    return view


def fig_smile_effect(smile: pd.DataFrame) -> go.Figure:
    view = smile_effect(smile)
    labels = view["produto"] + " " + view["tipo"]
    fig = go.Figure()
    for i, col in enumerate(c for c in ("vol em K", "vol local") if c in view):
        fig.add_trace(go.Bar(x=labels, y=view[col], name=col, marker_color=WARM[[1, 3][i]]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="% do nocional")
    return style(fig, "Preço com barreira: vol constante em K x vol local")


# ---------------------------------------------------------------- abas

def _chart(st, fig):
    # theme=None: o Streamlit não sobrescreve o tema da figura.
    st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOT_CONFIG)


def _pick(st, files, prefix, key):
    return st.selectbox("Execução", options=files, format_func=lambda p: file_label(p, prefix), key=key)


def render_spx(st, files, load):
    st.header("S&P 500 (SPX): superfície D-1")
    if not files:
        st.info("Nenhum resultado do SPX ainda. Rode o notebook `07_spx_superficie_d1` "
                "(ele grava `RESULTS_spx_superficie_*.csv`).")
        return
    b = load(_pick(st, files, SPX_PREFIX, "spx_file"))
    if b is None:
        return
    meta = b["metadata"]
    c = st.columns(4)
    c[0].metric("Data de mercado", str(meta.get("market_date", "—")))
    c[1].metric("Spot SPX", f"{meta.get('spot', 0):,.2f}")
    c[2].metric("Vencimentos", meta.get("vencimentos", "—"))
    c[3].metric("Pontos na superfície", f"{meta.get('pontos', 0):,}")
    summary = b.get("summary", {})
    c = st.columns(4)
    c[0].metric("Vol ATM 1m", summary.get("vol ATM 1m", "—"))
    c[1].metric("Vol ATM 12m", summary.get("vol ATM 12m", "—"))
    c[2].metric("r 1 ano", summary.get("r 1a", "—"))
    c[3].metric("q 1 ano", summary.get("q 1a", "—"))
    if meta.get("cobertura"):
        st.warning(f"Cobertura: {meta['cobertura']}")
    for w in meta.get("avisos") or []:
        st.caption(f"Aviso do pipeline: {w}")

    options = b["options"]
    _chart(st, fig_spx_smiles(options))
    _chart(st, fig_spx_atm(options))
    _chart(st, fig_spx_rates(b["forwards"], meta.get("sofr"), tuple(meta.get("faixa q") or (0.0, 0.03))))
    if "arb" in b:
        _chart(st, fig_spx_arb(b["arb"]))
    with st.expander("Forwards, r e q por vencimento"):
        st.dataframe(b["forwards"], use_container_width=True, hide_index=True)
    if "funnel" in b:
        with st.expander("Funil de limpeza"):
            st.dataframe(b["funnel"], use_container_width=True, hide_index=True)
    st.caption(f"Fonte: {meta.get('fonte', '—')} | executado em {meta.get('execution_timestamp', '—')}")


def _quanto_header(st, meta):
    c = st.columns(4)
    c[0].metric("Data de mercado", str(meta.get("market_date", "—")))
    c[1].metric("Spot SPX", f"{meta.get('S0', 0):,.2f}")
    c[2].metric("PTAX (X0)", f"{meta.get('X0', 0):.4f}")
    c[3].metric("Nocional S0·X0 (BRL)", f"{meta.get('notional', 0):,.2f}")


def render_quanto(st, files, load):
    st.header("Quanto / Compo / Convertida em BRL sobre o SPX")
    if not files:
        st.info("Nenhum resultado Quanto ainda. Rode o notebook `09_quanto_compo_spx` "
                "(ele grava `RESULTS_quanto_*.csv`).")
        return
    b = load(_pick(st, files, QUANTO_PREFIX, "quanto_file"))
    if b is None:
        return
    meta = b["metadata"]
    _quanto_header(st, meta)
    c = st.columns(4)
    c[0].metric("Prazo", f"{meta.get('dias', '—')} dias ({str(meta.get('expiry', ''))[:10]})")
    c[1].metric("Strike / Barreira", f"{meta.get('strike_pct', 0):.0%} / "
                                     + (f"{meta['barrier_pct']:.0%}" if meta.get("barrier_pct") else "—"))
    c[2].metric("Vol em K", f"{meta.get('vol_k', 0):.2%}")
    c[3].metric("Vol na barreira", f"{meta['vol_h']:.2%}" if meta.get("vol_h") is not None else "—")
    c = st.columns(4)
    c[0].metric("r USD", f"{meta.get('r_usd', 0):.3%}")
    c[1].metric("q", f"{meta.get('q', 0):.3%}")
    c[2].metric("r BRL", f"{meta.get('r_brl', 0):.3%}")
    c[3].metric("Ajuste Quanto", f"{meta.get('quanto_adj_bps', 0):+.0f} bps")
    c = st.columns(4)
    c[0].metric("Vol USDBRL", f"{meta.get('vol_fx', 0):.2%}")
    c[1].metric("Correlação usada", f"{meta.get('rho', 0):+.2f} ({meta.get('janela_rho', '—')})")
    c[2].metric("Vol composta (Compo)", f"{meta.get('compo_vol', 0):.2%}")
    c[3].metric("Câmbio a termo", f"{meta.get('fx_forward', 0):.4f}")

    st.subheader("Preços")
    view = prices_view(b["prices"], meta["notional"])
    fmt = {c: "{:,.2f}" for c in view.columns if c.endswith("(BRL)")}
    fmt.update({c: "{:.3%}" for c in view.columns if c.endswith("(% nocional)")})
    fmt.update({"strike": "{:,.2f}", "erro padrão": "{:.4f}", "prob. knock-out": "{:.2%}"})
    st.dataframe(view.style.format({k: v for k, v in fmt.items() if k in view}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    st.caption(f"Monte Carlo: {meta.get('paths', 0):,} caminhos, {meta.get('steps', '—')} passos, semente "
               f"{meta.get('seed', '—')}. Preço com barreira com vol constante em K; com smile, veja a aba Vol local.")

    st.subheader("Câmbio e correlação")
    if "fx_estimates" in b:
        st.dataframe(b["fx_estimates"].style.format({"vol_asset": "{:.2%}", "vol_fx": "{:.2%}",
                                                     "correlation": "{:+.3f}"}),
                     use_container_width=True, hide_index=True)
    if "rolling_corr" in b:
        _chart(st, fig_rolling_corr(b["rolling_corr"], meta["rho"]))
    if "sens_curves" in b:
        _chart(st, fig_sensitivity(b["sens_curves"], meta["rho"], meta["vol_fx"]))
    if "sens_table" in b:
        with st.expander("Call (% do nocional) por janela de correlação"):
            st.dataframe(b["sens_table"].style.format({c: "{:.4%}" for c in ("Compo", "Convertida", "Quanto")
                                                       if c in b["sens_table"]}),
                         use_container_width=True, hide_index=True)
    st.caption(f"PTAX: {meta.get('ptax_source', '—')} | SPX histórico: {meta.get('close_source', '—')} | "
               f"cadeia: {meta.get('chain_source', '—')}")


def render_local_vol(st, files, load):
    st.header("Vol local de Dupire: efeito do smile na barreira")
    if not files:
        st.info("A vol local vem do notebook `09_quanto_compo_spx`. Rode-o para gerar `RESULTS_quanto_*.csv`.")
        return
    b = load(_pick(st, files, QUANTO_PREFIX, "lv_file"))
    if b is None:
        return
    meta = b["metadata"]
    if meta.get("lv_error"):
        st.error(f"Vol local indisponível nesta execução: {meta['lv_error']}")
        return
    if "smile_barrier" not in b:
        st.info("Esta execução não tem barreira (widget barreira_pct vazio): não há comparação com vol local.")
        return
    smile = b["smile_barrier"]
    row = smile.loc[smile["produto"].eq("quanto") & smile["tipo"].eq("call")]
    c = st.columns(4)
    if len(row) and "vol local" in smile:
        r = row.iloc[0]
        c[0].metric("Quanto DOC call, vol em K", f"{r['vol em K']:.4f}%")
        c[1].metric("Quanto DOC call, vol local", f"{r['vol local']:.4f}%",
                    f"{r['vol local'] - r['vol em K']:+.4f} p.p.")
        if "prob. KO vol local" in smile:
            c[2].metric("P(knock-out) vol local", f"{r['prob. KO vol local']:.2%}")
    if meta.get("lv_clipped_share") is not None:
        c[3].metric("Vol local cortada (barreira a 125%)", f"{meta['lv_clipped_share']:.1%}")

    _chart(st, fig_smile_effect(smile))
    with st.expander("Tabela completa (% do nocional; última coluna em BRL)"):
        st.dataframe(smile_effect(smile), use_container_width=True, hide_index=True)
    if "lv_smile" in b:
        h = meta.get("barrier_pct")
        _chart(st, fig_lv_smile(b["lv_smile"], h * meta["S0"] / meta["forward"] if h else None))
    if "lv_clip_table" in b:
        _chart(st, fig_lv_clip(b["lv_clip_table"]))
        reason = b.get("lv_clip_reason") or {}
        if reason:
            st.caption("Motivo do corte (média): " + ", ".join(f"{k} {v:.1%}" for k, v in reason.items()))
    if "lv_fit" in b:
        with st.expander("Ajuste SVI por vencimento"):
            st.dataframe(b["lv_fit"], use_container_width=True, hide_index=True)
    st.caption("A Convertida (vanilla USD vezes X0) é a checagem da calibração: com vol local ela tem que ficar "
               "perto da fórmula fechada. Na Quanto e na Compo a diferença é efeito do smile com o câmbio.")
