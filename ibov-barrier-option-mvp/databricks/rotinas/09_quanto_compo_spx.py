# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # SPX pago em BRL: Quanto, Compo e Convertida (Fase 6)
# MAGIC
# MAGIC Notebook do motor multi-ativos (no `main` desde 23/11/2026). Não altera nada que o dashboard de produção lê.
# MAGIC
# MAGIC | Entrada | Fonte |
# MAGIC | :--- | :--- |
# MAGIC | spot, r_USD, q, vol do SPX | superfície do adaptador SPX (cadeia em disco do notebook 08 ou ThetaData) |
# MAGIC | r_BRL | curva DI1 do SPRD da B3 (notebook 00); alternativa: curva PRE da ANBIMA (`pyettj`); ou valor manual |
# MAGIC | câmbio USDBRL | PTAX venda do Banco Central (SGS 1; Olinda se o SGS falhar; cache em disco) |
# MAGIC | vol do câmbio e correlação | históricas, retornos semanais SPX x PTAX |
# MAGIC
# MAGIC Produtos (strike em % do spot do SPX; barreira knock-out observada no SPX):
# MAGIC - **Quanto:** `X_0 * max(S_T - K, 0)` em BRL, câmbio fixado hoje.
# MAGIC - **Compo:** `max(S_T X_T - K X_0, 0)` em BRL, strike em reais.
# MAGIC - **Convertida:** `X_T * max(S_T - K, 0)`, opção em USD convertida no vencimento (referência).
# MAGIC
# MAGIC A seção 4 usa vol constante no strike (fórmulas fechadas e MC). A seção 4b refaz as barreiras com vol
# MAGIC local de Dupire calibrada no smile do dia, que é o preço de referência para produtos com barreira.

# COMMAND ----------

# MAGIC %pip install --upgrade "grpcio>=1.80.0" thetadata python-dotenv "exchange-calendars==4.13.2" "lxml>=6.0" pyettj requests plotly scipy

# COMMAND ----------

# Obrigatório: o cluster já vem com grpcio 1.76 carregado; a thetadata 1.0.18 exige >= 1.80.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Ambiente: aponta o Python para o `src` deste Git folder

# COMMAND ----------

import os
import sys
import importlib

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
if not notebook_path.startswith("/Workspace"):
    notebook_path = "/Workspace" + notebook_path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
SRC = os.path.join(project_dir, "src")
if not os.path.isfile(os.path.join(SRC, "ibov_barrier", "pricing_quanto.py")):
    raise RuntimeError(f"{SRC} não tem pricing_quanto.py: faça Pull do main no Git folder.")

# Remove qualquer ibov_barrier carregado de outro Git folder (ex.: o de produção, que está no main).
for name in [m for m in sys.modules if m == "ibov_barrier" or m.startswith("ibov_barrier.")]:
    del sys.modules[name]
while SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
importlib.invalidate_caches()

import ibov_barrier
print("Código de:", ibov_barrier.__file__)

# COMMAND ----------

import json
from datetime import date, datetime, timedelta
from dataclasses import replace
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ibov_barrier.analytical import price_down_and_out_call_analytical
from ibov_barrier.local_vol import LocalVolSurface
from ibov_barrier.markets.fx_inputs import (brl_rate_from_sprd, continuous_from_effective, ettj_rate,
                                            fetch_ptax, index_closes, value_on_or_before)
from ibov_barrier.markets.spx import SPXAdapter, ThetaDataSource
from ibov_barrier.markets.vol import VolSurface
from ibov_barrier.pricing_quanto import (QuantoMarket, historical_fx_inputs, price_compo_vanilla,
                                         price_converted_vanilla, price_quanto_vanilla, price_two_factor_mc,
                                         quanto_market_data)

user_home = "/".join(notebook_path.split("/")[:4])          # /Workspace/Users/<usuário>
dbutils.widgets.text("data_mercado", "", "Data SPX (vazio = mais recente em disco)")
dbutils.widgets.text("prazo_dias", "182", "Prazo alvo (dias corridos)")
dbutils.widgets.text("strike_pct", "100", "Strike (% do spot)")
dbutils.widgets.text("barreira_pct", "85", "Barreira down-and-out (% do spot; vazio = sem)")
dbutils.widgets.text("anos_historico", "2", "Histórico para vol/correlação (anos)")
dbutils.widgets.text("caminhos", "200000", "Caminhos do Monte Carlo")
dbutils.widgets.dropdown("janela_rho", "completo", ["completo", "52 semanas", "manual"],
                         "Correlação e vol do câmbio: janela")
dbutils.widgets.text("rho_manual", "", "rho manual (com janela = manual)")
dbutils.widgets.text("r_brl_manual", "", "r BRL manual (% a.a. 252; vazio = curva)")
dbutils.widgets.text("pasta_spx", f"{user_home}/spx_dados", "Pasta SPX (notebook 08)")
dbutils.widgets.dropdown("fonte_spot", "paridade", ["paridade", "fornecedor"],
                         "Spot: paridade (padrão do 07) ou fechamento")
dbutils.widgets.text("pasta_b3", f"{user_home}/LuizRocha_Databricks_OptionsPrice/ibov-barrier-option-mvp/data",
                     "Pasta B3 (notebook 00)")

W = {k: dbutils.widgets.get(k).strip() for k in ("data_mercado", "prazo_dias", "strike_pct", "barreira_pct",
                                                 "janela_rho", "rho_manual",
                                                  "anos_historico", "caminhos", "r_brl_manual", "pasta_spx",
                                                  "pasta_b3", "fonte_spot")}
TARGET_T = float(W["prazo_dias"]) / 365.0
STRIKE_PCT = float(W["strike_pct"]) / 100.0
BARRIER_PCT = float(W["barreira_pct"]) / 100.0 if W["barreira_pct"] else None
HIST_YEARS = float(W["anos_historico"])
PATHS = int(W["caminhos"])
SPX_DIR, B3_DIR = Path(W["pasta_spx"]), Path(W["pasta_b3"])
print(W)

# Tema dos gráficos (padrão do projeto): fundo preto, textos brancos, rampa quente, marcador e traço por série.
BG, GRID, MUTED = "#0B0B0B", "#2A2A2A", "#8C8C8C"
WARM = ["#FFF3B0", "#FFC233", "#FF8A1F", "#F0432F", "#D6246E"]
SYMBOLS = ["circle", "square", "triangle-up", "diamond", "triangle-down"]
DASHES = ["solid", "dash", "dashdot", "dot", "longdashdot"]
AXIS = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, showline=True,
            tickfont=dict(color="white"), title_font=dict(color="white"))

# Template explícito: o Databricks troca o template padrão do Plotly pelo tema do workspace, por isso
# cada figura leva o tema do projeto e é exibida como HTML próprio (displayHTML), fora do renderizador dele.
DARK = go.layout.Template(layout=go.Layout(
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="white"), colorway=WARM,
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, tickfont=dict(color="white")),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, tickfont=dict(color="white"))))




def render(fig, filename):
    """Gráfico interativo no tema escuro, sem a inversão de cores do Databricks.

    Com o workspace em modo escuro, o Databricks inverte as cores de toda saída displayHTML (o preto #0B0B0B
    virava cinza claro #F4F4F4). A classe `nodarkmode` é o escape oficial: o HTML fica com as cores originais.
    """
    config = {"displaylogo": False, "displayModeBar": True, "scrollZoom": True,
              "toImageButtonOptions": {"format": "png", "scale": 2, "filename": filename}}
    fig.update_layout(template=DARK, paper_bgcolor=BG, plot_bgcolor=BG)
    html = fig.to_html(full_html=False, include_plotlyjs="cdn", config=config)
    displayHTML(f'<div class="nodarkmode" style="background:{BG};padding:4px">{html}</div>')


def style_fig(fig, title, height=440):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="white", size=12),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(color="white", size=16)),
        legend=dict(bgcolor="#141414", bordercolor="#3A3A3A", borderwidth=1, font=dict(color="white")),
        hoverlabel=dict(bgcolor="#141414", bordercolor="#3A3A3A", font=dict(color="white")),
        height=height, margin=dict(t=80, l=60, r=20, b=50))
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    fig.update_annotations(font=dict(color="white"))
    return fig


def show(fig, name):
    """Exibe com a barra de ferramentas (zoom, pan, reset, salvar PNG)."""
    render(fig, f"quanto_{name}_{market_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. SPX: spot, r_USD, q e vol no prazo do contrato

# COMMAND ----------

_source = None
def theta():
    global _source
    if _source is None:
        _source = ThetaDataSource()
    return _source


raw_dir = SPX_DIR / "brutos"
on_disk = sorted(p.parent.name for p in raw_dir.glob("*/meta.json")) if raw_dir.exists() else []
if W["data_mercado"]:
    market_date = date.fromisoformat(W["data_mercado"])
elif on_disk:
    market_date = date.fromisoformat(on_disk[-1])
else:
    market_date = SPXAdapter(lambda r, d: None).market_date()

day_dir = raw_dir / market_date.isoformat()
if (day_dir / "meta.json").exists():
    meta = json.loads((day_dir / "meta.json").read_text())
    # spot como no notebook 07: implícito pela paridade (padrão) ou fechamento oficial gravado no meta.json
    adapter = SPXAdapter(lambda root, _d: pd.read_parquet(day_dir / f"{root}.parquet"),
                         spot_source=(lambda _d: meta["spot"]) if W["fonte_spot"] == "fornecedor" else None)
    chain_source = f"disco ({day_dir})"
else:
    adapter = SPXAdapter(theta().option_chain,
                         spot_source=theta().index_close if W["fonte_spot"] == "fornecedor" else None)
    chain_source = "ThetaData"
snap = adapter.load_snapshot(market_date)
print(f"Data SPX: {market_date} | cadeia: {chain_source} | spot {snap.spot:,.2f} | "
      f"{len(snap.forwards)} vencimentos | {len(snap.options):,} pontos")

# Vencimento listado mais próximo do prazo alvo: r e q vêm direto do forward desse vencimento.
fwd = min((p for p in snap.forwards if p.T >= 7 / 365), key=lambda p: abs(p.T - TARGET_T))
T, expiry = fwd.T, fwd.expiration
S0 = snap.spot
r_usd = fwd.rate
q = r_usd - log(fwd.forward / S0) / T
K = STRIKE_PCT * S0
H = BARRIER_PCT * S0 if BARRIER_PCT else None
surface = VolSurface.from_options(snap.options)
vol_k, vol_k_how = surface.vol(log(K / fwd.forward), T)
vol_h, vol_h_how = surface.vol(log(H / fwd.forward), T) if H else (np.nan, "-")

spx_inputs = pd.Series({
    "vencimento": expiry, "dias": round(T * 365), "T": T, "spot": S0, "forward": fwd.forward,
    "r_USD": r_usd, "q": q, "K": K, "vol em K": vol_k, "interpolação K": vol_k_how,
    "barreira": H, "vol na barreira": vol_h, "interpolação barreira": vol_h_how,
    "no ajuste r(T)": fwd.in_fit, "avisos do dia": len(snap.report.warnings),
}, name="SPX")
display(spx_inputs.to_frame())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. r_BRL no prazo do contrato

# COMMAND ----------

def ettj_from_pyettj(ref_date):
    """Curva PRE da ANBIMA via pyettj. A assinatura muda entre versões, então tenta as conhecidas."""
    import pyettj
    tries = [lambda: pyettj.get_ettj(ref_date.isoformat(), curva="PRE"),
             lambda: pyettj.get_ettj(ref_date.strftime("%d/%m/%Y"), curva="PRE"),
             lambda: __import__("pyettj.ettj", fromlist=["get_ettj"]).get_ettj(ref_date.strftime("%d/%m/%Y"))]
    errors = []
    for fn in tries:
        try:
            frame = fn()
            if frame is not None and len(frame):
                return frame
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError("pyettj sem curva: " + " | ".join(errors))


rate_attempts = []
brl = None
if W["r_brl_manual"]:
    from ibov_barrier.di_curve import business_days
    eff = float(W["r_brl_manual"].replace(",", ".")) / 100.0
    du = business_days(market_date, expiry)
    brl = {"fonte": "manual (widget)", "data curva": market_date, "DU": du, "taxa efetiva 252": eff,
           "r_brl": continuous_from_effective(eff, du, T)}
if brl is None:
    try:
        brl = brl_rate_from_sprd(B3_DIR, market_date, expiry, T)
    except Exception as exc:
        rate_attempts.append(f"SPRD: {type(exc).__name__}: {str(exc)[:200]}")
if brl is None:
    from ibov_barrier.di_curve import business_days
    for k in range(6):
        d = market_date - timedelta(days=k)
        try:
            frame = ettj_from_pyettj(d)
            du = business_days(d, expiry)
            eff = ettj_rate(frame, du)
            brl = {"fonte": "ANBIMA PRE (pyettj)", "data curva": d, "DU": du, "taxa efetiva 252": eff,
                   "r_brl": continuous_from_effective(eff, du, T)}
            break
        except Exception as exc:
            rate_attempts.append(f"ANBIMA {d}: {type(exc).__name__}: {str(exc)[:200]}")
for msg in rate_attempts:
    print("tentativa:", msg)
if brl is None:
    raise RuntimeError("Sem taxa BRL: rode o notebook 00 (SPRD) ou preencha o widget r_brl_manual.")
r_brl = brl["r_brl"]
display(pd.Series(brl, name="BRL").to_frame())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Câmbio (PTAX), vol do câmbio e correlação SPX x USDBRL

# COMMAND ----------

hist_start = market_date - timedelta(days=int(HIST_YEARS * 365.25) + 14)
# PTAX: SGS (blocos anuais, com novas tentativas) -> Olinda -> cache em disco da última execução.
ptax_path = SPX_DIR / "ptax.parquet"
try:
    ptax, ptax_source = fetch_ptax(hist_start, market_date)
    SPX_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"data": [d.isoformat() for d in ptax.index], "ptax": ptax.to_numpy()})         .to_parquet(ptax_path, index=False)
except Exception as exc:
    if not ptax_path.exists():
        raise
    print("BCB indisponível, usando cache:", str(exc)[:300])
    cached_ptax = pd.read_parquet(ptax_path)
    ptax = pd.Series(cached_ptax["ptax"].to_numpy(), index=pd.to_datetime(cached_ptax["data"]).dt.date)
    ptax_source = f"cache em disco (até {ptax.index[-1]})"
ptax_date, X0 = value_on_or_before(ptax, market_date)
print(f"PTAX: {len(ptax)} dias de {ptax.index[0]} a {ptax.index[-1]} | X_0 = {X0:.4f} ({ptax_date}) "
      f"| fonte: {ptax_source}")

# Fechamentos do SPX: cache em disco, completado pela ThetaData quando faltar período.
closes_path = SPX_DIR / "spx_fechamentos.parquet"
spx_close = pd.Series(dtype=float)
if closes_path.exists():
    cached = pd.read_parquet(closes_path)
    spx_close = pd.Series(cached["spx"].to_numpy(), index=pd.to_datetime(cached["data"]).dt.date)
covered = len(spx_close) and spx_close.index[0] <= hist_start + timedelta(days=10) \
    and spx_close.index[-1] >= market_date
close_source = "cache em disco"
if not covered:
    try:
        frames = []
        cursor = hist_start
        while cursor <= market_date:           # blocos de 1 ano para não estourar limites do plano
            stop = min(market_date, cursor + timedelta(days=365))
            frames.append(theta().client.index_history_eod("SPX", cursor, stop))
            cursor = stop + timedelta(days=1)
        spx_close = index_closes(pd.concat(frames, ignore_index=True))
        SPX_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"data": [d.isoformat() for d in spx_close.index], "spx": spx_close.to_numpy()}) \
            .to_parquet(closes_path, index=False)
        close_source = "ThetaData index_history_eod"
    except Exception as exc:
        print("ThetaData sem histórico do índice:", type(exc).__name__, str(exc)[:200])
        metas = sorted(raw_dir.glob("*/meta.json"))
        spx_close = pd.Series({date.fromisoformat(p.parent.name): json.loads(p.read_text())["spot"] for p in metas})
        close_source = f"meta.json do notebook 08 ({len(spx_close)} dias; curto para correlação)"
print(f"SPX: {len(spx_close)} fechamentos de {spx_close.index[0]} a {spx_close.index[-1]} | fonte: {close_source}")

est = {
    "semanal": historical_fx_inputs(spx_close, ptax),
    "semanal, 52 semanas": historical_fx_inputs(spx_close, ptax, window=52),
    "diário": historical_fx_inputs(spx_close, ptax, frequency=None),
}
est_table = pd.DataFrame(est).T
display(est_table)
# Política: janela escolhida no widget; vol do câmbio da mesma janela (manual usa a do histórico completo).
policy = W["janela_rho"]
if policy == "manual":
    if not W["rho_manual"]:
        raise ValueError("janela_rho = manual exige o widget rho_manual.")
    rho = float(W["rho_manual"].replace(",", "."))
    vol_fx = est["semanal"]["vol_fx"]
else:
    chosen = est["semanal" if policy == "completo" else "semanal, 52 semanas"]
    vol_fx, rho = chosen["vol_fx"], chosen["correlation"]
if not -1 < rho < 1:
    raise ValueError(f"rho fora de (-1, 1): {rho}")
print(f"Usado ({policy}): vol USDBRL {vol_fx:.2%} | rho {rho:+.2f} "
      f"(diário {est['diário']['correlation']:+.2f}: a diferença é o efeito dos fechamentos 17h ET x 13h BRT)")

# Correlação móvel de 26 semanas.
wk = pd.concat({"s": spx_close, "x": ptax}, axis=1).dropna()
wk.index = pd.to_datetime(wk.index)
wk = np.log(wk.resample("W-FRI").last().dropna()).diff().dropna()
roll = wk["s"].rolling(26).corr(wk["x"]).dropna()
fig = go.Figure()
fig.add_trace(go.Scatter(x=roll.index, y=roll, mode="lines", name="correlação 26 semanas",
                         line=dict(color=WARM[2], width=2)))
fig.add_hline(y=rho, line=dict(color=WARM[0], dash="dash"), annotation_text=f"usada {rho:+.2f}")
fig.add_hline(y=0, line=dict(color=MUTED, dash="dot"))
fig.update_yaxes(title="correlação SPX x USDBRL", range=[-1, 1])
show(style_fig(fig, "Correlação móvel SPX x USDBRL (retornos semanais)"), "correlacao")

# Faixa de rho para o relatório: estimativas por janela e extremos da móvel de 26 semanas.
rho_band = {"usado": rho, "completo": est["semanal"]["correlation"],
            "52 semanas": est["semanal, 52 semanas"]["correlation"], "diário": est["diário"]["correlation"],
            "26 semanas mín": float(roll.min()), "26 semanas máx": float(roll.max()),
            "26 semanas última": float(roll.iloc[-1])}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Preços em BRL (por ponto do índice e em % do nocional S_0 X_0)

# COMMAND ----------

mkt = QuantoMarket(spot=S0, rate_usd=r_usd, dividend_yield=q, vol_asset=vol_k, fx_spot=X0, rate_brl=r_brl,
                   vol_fx=vol_fx, correlation=rho)
steps = max(50, int(round(T * 252)))
notional = S0 * X0
md_q = quanto_market_data(mkt)

rows = []
for product in ("quanto", "compo", "converted"):
    strike = K * X0 if product == "compo" else K
    for option_type in ("call", "put"):
        closed = {"quanto": lambda: price_quanto_vanilla(mkt, K, T, option_type=option_type),
                  "compo": lambda: price_compo_vanilla(mkt, K * X0, T, option_type=option_type),
                  "converted": lambda: price_converted_vanilla(mkt, K, T, option_type=option_type)}[product]()
        row = {"produto": product, "tipo": option_type, "strike": strike, "fechada (BRL)": closed}
        if H:
            mc = price_two_factor_mc(mkt, strike, T, product=product, option_type=option_type, barrier=H,
                                     paths=PATHS, steps=steps, seed=42)
            row.update({"com barreira MC (BRL)": mc.price, "erro padrão": mc.standard_error,
                        "prob. knock-out": mc.knock_out_probability})
            if option_type == "call" and product in ("quanto", "converted"):
                # Referência analítica contínua (Reiner-Rubinstein) pela troca de MarketData.
                if product == "quanto":
                    ref = X0 * price_down_and_out_call_analytical(S0, K, H, T, md_q.rate, md_q.dividend_yield,
                                                                  md_q.volatility)
                else:
                    ref = X0 * price_down_and_out_call_analytical(S0, K, H, T, r_usd, q, vol_k)
                row["com barreira analítica (BRL)"] = ref
        rows.append(row)
prices = pd.DataFrame(rows)
for c in [c for c in prices.columns if "(BRL)" in c]:
    prices[c.replace("(BRL)", "(% nocional)")] = (prices[c] / notional * 100).map(
        lambda v: "" if pd.isna(v) else f"{v:.3f}%")
display(prices)

quanto_adj_bps = mkt.quanto_adjustment * 1e4
print(f"Ajuste de drift do Quanto: {quanto_adj_bps:+.0f} bps a.a. | vol composta (Compo): {mkt.compo_volatility:.2%} "
      f"| forward do câmbio no vencimento: {mkt.fx_forward(T):.4f}")
if H:
    print(f"Vol em K {vol_k:.2%} x vol na barreira {vol_h:.2%}: o preço com barreira usa vol constante em K.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4b. Efeito do smile na barreira
# MAGIC
# MAGIC Preço com barreira com vol constante em K contra vol local de Dupire calibrada na superfície do dia
# MAGIC (SVI por vencimento). A vol local reprecifica o smile inteiro e é a referência para barreiras.
# MAGIC Vol constante na barreira ou no ponto médio não é limite de nada: com vol maior a call vale mais, mesmo
# MAGIC com mais knock-out, então essa "faixa" aponta para o lado errado e foi retirada.
# MAGIC
# MAGIC Checagem da calibração: na Convertida (vanilla USD vezes X_0), "vanilla vol local" tem que ficar perto da
# MAGIC fechada com vol em K. Na Quanto e na Compo a diferença é efeito do smile com o câmbio, não erro.

# COMMAND ----------

lv_error = None
lv = None
try:
    lv = LocalVolSurface.from_options(snap.options, max_maturity=max(T * 1.6, T + 0.25))
    lv_clip = []
    # Região que o MC visita: da barreira (menos uma folga) até 25% acima do forward.
    lv_k_range = (log((H or 0.6 * S0) / fwd.forward) - 0.1, log(1.25))
    for i in range(steps):
        lv_clip.append(lv.local_vol_row((i + 0.5) * T / steps, k_range=lv_k_range)[1])
    print(f"SVI: {len(lv.fit_report)} vencimentos | RMSE vol médio {lv.fit_report['RMSE vol'].mean():.4f} | "
          f"vol local cortada pelos limites entre barreira e 125% (média no prazo) {np.mean(lv_clip):.1%}")

    # Onde e quando a vol local é cortada: faixas de strike/forward x períodos do prazo.
    f_edges = [(H / S0 if H else 0.6) - 0.10, (H / S0 if H else 0.6), 0.95, 1.05, 1.25]
    f_labels = [f"{a:.0%}-{b:.0%}" for a, b in zip(f_edges[:-1], f_edges[1:])]
    k_edges = np.log(np.array(f_edges))
    step_times = (np.arange(steps) + 0.5) * T / steps
    clip_long = lv.clip_breakdown(step_times, k_edges)
    bounds = [0, 14 / 365, T / 4, T / 2, T + 1e-9]
    period_labels = ["até 2 semanas", "2 semanas a T/4", "T/4 a T/2", "T/2 a T"]
    clip_long["período"] = pd.cut(clip_long["t"], bounds, labels=period_labels, right=False)
    clip_long["faixa K/F"] = pd.Categorical(
        clip_long["k_de"].map(dict(zip(k_edges[:-1], f_labels))), categories=f_labels)
    clip_table = clip_long.pivot_table(index="período", columns="faixa K/F", values="cortado", aggfunc="mean",
                                       observed=False)
    clip_reason = clip_long[["abaixo do piso", "acima do teto", "denominador"]].mean()
    print("Fração cortada por período e faixa de K/F (média dos passos):")
    display(clip_table.style.format("{:.1%}"))
    print("Motivo (média geral):", ", ".join(f"{k} {v:.1%}" for k, v in clip_reason.items()))
    display(lv.fit_report)
except Exception as exc:
    lv_error = f"{type(exc).__name__}: {exc}"
    print("Vol local indisponível:", lv_error)

smile_rows = []
if H:
    flat_vols = {"vol em K": vol_k}
    for product in ("quanto", "compo", "converted"):
        strike = K * X0 if product == "compo" else K
        for option_type in ("call", "put"):
            row = {"produto": product, "tipo": option_type}
            for label, v in flat_vols.items():
                r = price_two_factor_mc(replace(mkt, vol_asset=float(v)), strike, T, product=product,
                                        option_type=option_type, barrier=H, paths=PATHS, steps=steps, seed=42)
                row[label] = r.price / notional * 100
            if lv is not None:
                r = price_two_factor_mc(mkt, strike, T, product=product, option_type=option_type, barrier=H,
                                        paths=PATHS, steps=steps, seed=42, local_vol=lv)
                row["vol local"] = r.price / notional * 100
                row["erro padrão vol local"] = r.standard_error / notional * 100
                row["prob. KO vol local"] = r.knock_out_probability
                row["vanilla vol local"] = r.vanilla_price / notional * 100
            row["vanilla fechada vol em K"] = prices.set_index(["produto", "tipo"]).loc[
                (product, option_type), "fechada (BRL)"] / notional * 100
            smile_rows.append(row)
    smile = pd.DataFrame(smile_rows).set_index(["produto", "tipo"])
    if "vol local" in smile:
        smile["vol local - vol em K (BRL)"] = (smile["vol local"] - smile["vol em K"]) / 100 * notional
    display(smile)

    if lv is not None:
        k_plot = np.linspace(log(0.70), log(1.15), 60)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.exp(k_plot) * 100, y=lv.implied_vol(k_plot, T) * 100, mode="lines+markers",
                                 name="implícita (SVI)", line=dict(color=WARM[1], width=2),
                                 marker=dict(symbol="circle", size=5)))
        fig.add_trace(go.Scatter(x=np.exp(k_plot) * 100, y=lv.local_vol(k_plot, 0.5 * T) * 100,
                                 mode="lines+markers", name="local em T/2", line=dict(color=WARM[3], dash="dash",
                                                                                      width=2),
                                 marker=dict(symbol="diamond", size=5)))
        fig.add_vline(x=BARRIER_PCT * 100 * S0 / fwd.forward, line=dict(color=MUTED, dash="dot"),
                      annotation_text="barreira")
        fig.update_xaxes(title="strike / forward (%)")
        fig.update_yaxes(title="vol (%)")
        show(style_fig(fig, f"Smile no vencimento {expiry}: implícita x local"), "smile_local")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sensibilidade à correlação e à vol do câmbio (calls, fórmula fechada)

# COMMAND ----------

rhos = np.linspace(-0.9, 0.9, 37)
vols = np.linspace(0.05, 0.30, 26)
series = [("Quanto", lambda m: price_quanto_vanilla(m, K, T)),
          ("Compo", lambda m: price_compo_vanilla(m, K * X0, T)),
          ("Convertida", lambda m: price_converted_vanilla(m, K, T))]
fig = make_subplots(rows=1, cols=2, subplot_titles=("Correlação SPX x USDBRL", "Vol do USDBRL"))
sens_rows = []
for i, (name, fn) in enumerate(series):
    by_rho = [fn(replace(mkt, correlation=float(x))) / notional * 100 for x in rhos]
    by_vol = [fn(replace(mkt, vol_fx=float(v))) / notional * 100 for v in vols]
    style = dict(color=WARM[[1, 3, 0][i]], dash=DASHES[i], width=2)
    fig.add_trace(go.Scatter(x=rhos, y=by_rho, mode="lines+markers", name=name, line=style,
                             marker=dict(symbol=SYMBOLS[i], size=5), legendgroup=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=vols * 100, y=by_vol, mode="lines+markers", name=name, line=style,
                             marker=dict(symbol=SYMBOLS[i], size=5), legendgroup=name, showlegend=False),
                  row=1, col=2)
    for label, x in rho_band.items():
        sens_rows.append({"produto": name, "rho": f"{label} ({x:+.2f})",
                          "call (% nocional)": fn(replace(mkt, correlation=float(x))) / notional})
fig.add_vline(x=rho, line=dict(color=MUTED, dash="dot"), row=1, col=1)
fig.add_vline(x=vol_fx * 100, line=dict(color=MUTED, dash="dot"), row=1, col=2)
fig.update_xaxes(title="rho", row=1, col=1)
fig.update_xaxes(title="vol USDBRL (%)", row=1, col=2)
fig.update_yaxes(title="call (% do nocional)", row=1, col=1)
show(style_fig(fig, "Call em BRL: sensibilidade aos parâmetros do câmbio", height=460), "sensibilidade")
sens = pd.DataFrame(sens_rows).pivot(index="rho", columns="produto", values="call (% nocional)")
sens = sens.loc[[f"{k} ({v:+.2f})" for k, v in rho_band.items()]]
display(sens)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5b. Resultado para o dashboard
# MAGIC
# MAGIC Grava `RESULTS_quanto_<data e hora>.csv` na pasta `ibov-barrier-results` (a mesma do notebook 01): insumos,
# MAGIC câmbio e correlação, preços, efeito do smile (vol local), cortes da vol local e curvas de sensibilidade.

# COMMAND ----------

from zoneinfo import ZoneInfo

from ibov_barrier.results_bundle import SCHEMA_VERSION, results_filename, write_bundle

run_ts = datetime.now(ZoneInfo("America/Sao_Paulo"))
results_dir = Path(user_home) / "ibov-barrier-results"
has_lv = lv is not None and "clip_table" in globals()

# Curvas de sensibilidade (fórmula fechada, mesmas grades do gráfico da seção 5).
sens_curves = pd.concat([
    pd.DataFrame({"parâmetro": "rho", "x": rhos, "produto": name,
                  "call (% nocional)": [fn(replace(mkt, correlation=float(x))) / notional * 100 for x in rhos]})
    for name, fn in series] + [
    pd.DataFrame({"parâmetro": "vol_fx", "x": vols, "produto": name,
                  "call (% nocional)": [fn(replace(mkt, vol_fx=float(v))) / notional * 100 for v in vols]})
    for name, fn in series], ignore_index=True)

objects = {
    "metadata": {
        "schema_version": SCHEMA_VERSION, "kind": "quanto", "execution_timestamp": run_ts,
        "market_date": market_date, "expiry": fwd.expiration, "T": T, "dias": round(T * 365),
        "S0": S0, "X0": X0, "notional": notional, "K": K, "H": H, "strike_pct": STRIKE_PCT,
        "barrier_pct": BARRIER_PCT, "forward": fwd.forward, "r_usd": r_usd, "q": q, "vol_k": vol_k,
        "vol_h": vol_h, "r_brl": r_brl, "vol_fx": vol_fx, "rho": rho, "janela_rho": policy,
        "quanto_adj_bps": quanto_adj_bps, "compo_vol": mkt.compo_volatility, "fx_forward": mkt.fx_forward(T),
        "paths": PATHS, "steps": steps, "seed": 42, "chain_source": chain_source, "ptax_source": ptax_source,
        "close_source": close_source, "lv_error": lv_error,
        "lv_clipped_share": float(np.mean(lv_clip)) if has_lv else None,
        "widgets": W,
    },
    "spx_inputs": spx_inputs,
    "brl": brl,
    "fx_estimates": est_table.rename_axis("janela"),
    "rho_band": rho_band,
    "rolling_corr": roll.rename("correlação 26 semanas").rename_axis("data").to_frame(),
    "prices": prices[[c for c in prices.columns if "% nocional" not in c]],
    "sens_table": sens,
    "sens_curves": sens_curves,
}
if H and "smile" in globals():
    objects["smile_barrier"] = smile.reset_index()
if has_lv:
    k_plot = np.linspace(log(0.70), log(1.15), 60)
    objects["lv_smile"] = pd.DataFrame({"K/F": np.exp(k_plot), "implícita": lv.implied_vol(k_plot, T),
                                        "local T/4": lv.local_vol(k_plot, 0.25 * T),
                                        "local T/2": lv.local_vol(k_plot, 0.5 * T),
                                        "local T": lv.local_vol(k_plot, T)})
    objects["lv_clip_table"] = clip_table.rename_axis(index="período", columns=None).reset_index()
    objects["lv_clip_reason"] = clip_reason
    objects["lv_fit"] = lv.fit_report
quanto_result_path = write_bundle(results_dir / results_filename("quanto", run_ts), objects)
print(f"Resultado para o dashboard: {quanto_result_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Relatório único para enviar
# MAGIC
# MAGIC Copie o texto inteiro, de `RELATÓRIO QUANTO/COMPO` até `FIM DO RELATÓRIO`.

# COMMAND ----------

import traceback

_g = globals()


def _df(frame, fmt=None):
    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 250,
                           "display.float_format", (fmt or "{:,.6g}").format):
        print(frame.to_string())


def _block(title, fn):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    try:
        fn()
    except Exception as exc:
        print(f"[não disponível] {type(exc).__name__}: {exc}")
        print("".join(traceback.format_exc().splitlines(keepends=True)[-4:]))


print(f"RELATÓRIO QUANTO/COMPO | SPX {_g.get('market_date')} | gerado em "
      f"{pd.Timestamp.now(tz='America/Sao_Paulo'):%Y-%m-%d %H:%M} (Brasília)")
print("widgets:", W)
_block("1. SPX", lambda: (print("cadeia:", chain_source), _df(spx_inputs.to_frame())))
_block("2. r_BRL", lambda: ([print("tentativa:", m) for m in rate_attempts], _df(pd.Series(brl).to_frame())))
_block("3. Câmbio e correlação", lambda: (
    print(f"PTAX X_0 = {X0:.4f} ({ptax_date}, {ptax_source}) | SPX histórico: {close_source}"),
    _df(est_table), print(f"usado ({policy}): vol USDBRL {vol_fx:.4f} | rho {rho:+.4f}"),
    print("correlação 26 semanas: min {:+.2f} | med {:+.2f} | max {:+.2f} | última {:+.2f}".format(
        roll.min(), roll.median(), roll.max(), roll.iloc[-1]))))
_block("4. Preços", lambda: (
    print(f"nocional S_0 X_0 = {notional:,.2f} BRL | passos MC {steps} | caminhos {PATHS:,}"),
    print(f"ajuste Quanto {quanto_adj_bps:+.1f} bps | vol composta {mkt.compo_volatility:.4f} | "
          f"forward câmbio {mkt.fx_forward(T):.4f}"),
    _df(prices.set_index(["produto", "tipo"]))))
_block("4b. Smile na barreira (% do nocional; última coluna em BRL)", lambda: (
    print("vol local:", "ok" if lv is not None else lv_error),
    lv is not None and print(f"SVI {len(lv.fit_report)} vencimentos (RMSE <= 3%) | vol local cortada entre barreira e 125%: "
          f"{np.mean(lv_clip):.1%}"),
    lv is not None and print("corte por período x faixa K/F:"),
    lv is not None and _df(clip_table, "{:.1%}"),
    lv is not None and print("motivo:", ", ".join(f"{k} {v:.1%}" for k, v in clip_reason.items())),
    lv is not None and print(f"SVI: RMSE máximo {lv.fit_report['RMSE vol'].max():.4f} | vencimentos (1 a cada 4):"),
    lv is not None and _df(lv.fit_report[["dias", "pontos", "RMSE vol", "vol ATM SVI"]].iloc[::4]),
    _df(smile, "{:,.4f}")))
_block("5. Sensibilidade ao rho (call, % do nocional)", lambda: _df(sens, "{:.4%}"))
print("\nFIM DO RELATÓRIO")
