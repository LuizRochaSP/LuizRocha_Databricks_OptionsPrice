# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # SPX: fotografia D-1 e superfície de volatilidade (Fase 3)
# MAGIC
# MAGIC Notebook do motor multi-ativos (no `main` desde 23/11/2026). Não altera nada que o dashboard de produção lê.
# MAGIC
# MAGIC Fluxo (`SPXAdapter.load_snapshot`):
# MAGIC
# MAGIC 1. cadeia EOD de SPX e SPXW na ThetaData (plano FREE, D-1);
# MAGIC 2. limpeza de cotações;
# MAGIC 3. paridade put-call: forward e fator de desconto por vencimento, `C − P = DF·(F − K)`;
# MAGIC 4. spot implícito pela paridade (padrão) ou fechamento oficial do índice; `r(T) = −ln(DF)/T` e
# MAGIC    `q(T) = r(T) − ln(F/S)/T`;
# MAGIC 5. limpeza contra o forward (limites de arbitragem, só OTM, moneyness, vertical/butterfly);
# MAGIC 6. volatilidade implícita e limpeza do smile.
# MAGIC
# MAGIC Depois: precificação da call down-and-out de referência (seção 6a, mesmas regras do notebook 01), o
# MAGIC RESULTS para o dashboard (6b) e a cadeia do dia em `spx_dados/brutos/` (6c, lida pelo notebook 09).
# MAGIC No job `opcoes-pipeline-spx`, o 11 valida o RESULTS e o 03 (`ativo=spx`) calcula gregas e VaR.
# MAGIC
# MAGIC **Spot.** O EOD das opções sai por volta de 17:15 ET, depois do fechamento do índice (16:00). Em 79 pregões
# MAGIC (ago-nov/2026) o spot implícito pela paridade ficou a +1,6 bps do fechamento na mediana (desvio 14 bps),
# MAGIC mas chegou a +61 bps em 20/11/2026. Com o spot da paridade o q de 1 ano fica estável (desvio 0,06% contra
# MAGIC 0,16% com o fechamento). Por isso o padrão é `fonte_spot = paridade`; o fechamento fica no RESULTS.
# MAGIC
# MAGIC Pré-requisitos: Python 3.12+ e o Secret `quantita/thetadata_api_key` (a chave nunca aparece aqui).
# MAGIC
# MAGIC Para validar a estabilidade, rodar em vários dias seguidos e comparar o resumo final.

# COMMAND ----------

# MAGIC %pip install --upgrade "grpcio>=1.80.0" thetadata python-dotenv "exchange-calendars==4.13.2" "lxml>=6.0" plotly scipy

# COMMAND ----------

# Obrigatório: o cluster já vem com grpcio 1.76 carregado; a thetadata 1.0.18 exige >= 1.80.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ambiente: aponta o Python para o `src` deste Git folder

# COMMAND ----------

import os
import sys
import importlib

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
# .../ibov-barrier-option-mvp/databricks/rotinas/07_spx_superficie_d1 -> .../ibov-barrier-option-mvp/src
if not notebook_path.startswith("/Workspace"):
    notebook_path = "/Workspace" + notebook_path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
SRC = os.path.join(project_dir, "src")
if not os.path.isdir(os.path.join(SRC, "ibov_barrier", "markets")):
    raise RuntimeError(f"{SRC} não tem ibov_barrier/markets: faça Pull do main no Git folder.")

# Remove qualquer ibov_barrier carregado de outro Git folder (ex.: o de produção, que está no main).
for name in [m for m in sys.modules if m == "ibov_barrier" or m.startswith("ibov_barrier.")]:
    del sys.modules[name]
while SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
importlib.invalidate_caches()

import grpc
import ibov_barrier
print("grpcio:", grpc.__version__)
print("Código de:", ibov_barrier.__file__)

# COMMAND ----------

from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ibov_barrier.markets.base import CleaningReport
from ibov_barrier.markets.cleaning import clean_against_forwards, clean_quotes
from ibov_barrier.markets.parity import implied_spot
from ibov_barrier.markets.referencia import CONTRATO_PADRAO, MC_PADRAO, precificar_referencia, salvar_cadeia_do_dia
from ibov_barrier.markets.spx import SPXAdapter, ThetaDataSource

# Prazo a partir do qual cada vencimento mede a própria taxa e entra no ajuste da curva r(T).
MIN_T_RATE = 0.25
# Faixa considerada razoável para o dividend yield implícito do SPX (alerta, não filtro).
Q_RANGE = (0.0, 0.03)

dbutils.widgets.text("data_mercado", "", "Data (AAAA-MM-DD; vazio = D-1)")
dbutils.widgets.dropdown("fonte_spot", "paridade", ["paridade", "fornecedor"], "Spot: paridade ou fechamento")
dbutils.widgets.text("ref_strike_pct", f"{CONTRATO_PADRAO['strike_pct'] * 100:g}", "Referência: strike (% do spot)")
dbutils.widgets.text("ref_barreira_pct", f"{CONTRATO_PADRAO['barrier_pct'] * 100:g}", "Referência: barreira (% do spot)")
dbutils.widgets.text("ref_prazo_anos", f"{CONTRATO_PADRAO['maturity']:g}", "Referência: prazo (anos)")

# Tema dos gráficos (padrão do projeto): fundo preto, textos brancos, curvas em tons quentes.
# Os vencimentos são ordenados (1m -> 24m), então a paleta é uma rampa quente do claro ao escuro,
# com marcador e traço diferentes por série: tons quentes ficam próximos entre si, e o formato
# garante a leitura (inclusive para daltonismo).
# Plotly: cada gráfico tem barra com zoom (caixa ou roda do mouse; duplo clique volta), pan,
# autoescala e download em PNG (2x, nome com a data). Passar o mouse mostra os valores.
BG, GRID, MUTED = "#0B0B0B", "#2A2A2A", "#8C8C8C"
WARM = ["#FFF3B0", "#FFC233", "#FF8A1F", "#F0432F", "#D6246E"]   # amarelo -> âmbar -> laranja -> vermelho -> magenta
SYMBOLS = ["circle", "square", "triangle-up", "diamond", "triangle-down"]
DASHES = ["solid", "dash", "dashdot", "dot", "longdashdot"]
AXIS = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, showline=True,
            tickfont=dict(color="white"), title_font=dict(color="white"))
print("plotly:", plotly.__version__)

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


def style_fig(fig, title, height=460):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="white", size=12),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(color="white", size=16)),
        legend=dict(bgcolor="#141414", bordercolor="#3A3A3A", borderwidth=1, font=dict(color="white"),
                    itemsizing="constant"),
        hoverlabel=dict(bgcolor="#141414", bordercolor="#3A3A3A", font=dict(color="white")),
        height=height, margin=dict(t=80, l=60, r=20, b=50),
    )
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    fig.update_annotations(font=dict(color="white"))
    return fig


def show(fig, name):
    """Exibe com a barra de ferramentas (zoom, pan, reset, salvar PNG)."""
    render(fig, f"spx_{name}_{market_date}")


def sparse_sizes(n, size=7, points=12):
    """Marcador visível em ~12 pontos da curva (o resto fica com tamanho 0, mas o hover continua)."""
    every = max(1, n // points)
    return [size if i % every == 0 else 0 for i in range(n)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fotografia D-1

# COMMAND ----------

source = ThetaDataSource()

# Guarda a resposta bruta: os diagnósticos abaixo reaproveitam a cadeia sem chamar a API de novo.
_raw_cache = {}
def fetch_chain_cached(root, market_date):
    key = (root, market_date)
    if key not in _raw_cache:
        _raw_cache[key] = source.option_chain(root, market_date)
    return _raw_cache[key]

FONTE_SPOT = dbutils.widgets.get("fonte_spot")
# paridade: spot_source=None faz o adaptador usar o spot implícito do vencimento mais curto
adapter = SPXAdapter(fetch_chain_cached, spot_source=source.index_close if FONTE_SPOT == "fornecedor" else None)

typed = dbutils.widgets.get("data_mercado").strip()
market_date = date.fromisoformat(typed) if typed else adapter.market_date()
print("Data de mercado:", market_date, "| chave:", source.key_source)

snap = adapter.load_snapshot(market_date)
print(f"Spot SPX: {snap.spot:,.2f} ({snap.spot_source})")

# Diagnóstico do spot: fechamento oficial (16:00 ET) x implícito pela paridade (cotações de ~17:15 ET)
spot_fechamento = float(source.index_close(market_date))
spot_paridade = float(implied_spot(snap.forwards, adapter.assumed_dividend_yield))
spot_gap_bps = (spot_paridade / spot_fechamento - 1) * 1e4
print(f"Fechamento oficial: {spot_fechamento:,.2f} | paridade: {spot_paridade:,.2f} | diferença: {spot_gap_bps:+.1f} bps"
      + ("  <- acima de 25 bps: fechamento e opções em horários diferentes" if abs(spot_gap_bps) > 25 else ""))
print(f"Vencimentos com forward: {len(snap.forwards)}")
print(f"Pontos na superfície: {len(snap.options):,}")

# Alerta de baixa cobertura: compara com o histórico do notebook 08, se existir.
from ibov_barrier.markets.quality import coverage_warning
coverage_msg = None
try:
    _hist = pd.read_parquet("/".join(notebook_path.split("/")[:4]) + "/spx_dados/resumo_diario.parquet")
    _hist = _hist.loc[pd.to_datetime(_hist["data"]).dt.date < market_date].sort_values("data")
    coverage_msg = coverage_warning(len(snap.options), _hist["pontos"], _hist.get("baixa cobertura"))
    print(coverage_msg or f"Cobertura normal (referência: {min(len(_hist), 20)} pregões do notebook 08).")
except Exception as exc:
    print("Sem histórico para checar cobertura (rode o notebook 08):", type(exc).__name__)

# COMMAND ----------

display(snap.report.to_frame())
if snap.report.warnings:
    print("Avisos:")
    for w in snap.report.warnings:
        print(" -", w)
else:
    print("Sem avisos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Forwards e estrutura a termo
# MAGIC
# MAGIC Cada vencimento com T >= 3 meses mede a própria taxa por uma regressão ampla (pares em ±10% do ATM).
# MAGIC Uma reta r(T) = a + b·T robusta é ajustada a essas taxas (as que ficam fora da curva são ignoradas,
# MAGIC com aviso) e aplicada a **todos** os vencimentos; o forward vem dos pares perto do ATM com esse DF.
# MAGIC Abaixo de 3 meses o ruído de cotação explode em `−ln(DF)/T`, então esses prazos só usam a curva.

# COMMAND ----------

forwards = pd.DataFrame([{
    "vencimento": p.expiration,
    "T": p.T,
    "dias": round(p.T * 365),
    "forward": p.forward,
    "DF": p.discount_factor,
    "r": p.rate,
    "pares": p.n_pairs,
    "resíduo": p.residual_std,
    "r do vencimento": p.fit_rate,
    "no ajuste": p.in_fit,
} for p in snap.forwards]).sort_values("T").reset_index(drop=True)
forwards["q"] = forwards["r"] - np.log(forwards["forward"] / snap.spot) / forwards["T"]
forwards["F/S - 1"] = forwards["forward"] / snap.spot - 1
forwards["origem de r"] = np.select(
    [forwards["no ajuste"], forwards["r do vencimento"].notna()],
    ["curva (no ajuste)", "curva (fora do ajuste)"], "curva (prazo curto)")
forwards["alerta q"] = ~forwards["q"].between(*Q_RANGE)

display(forwards.style.format({
    "T": "{:.4f}", "forward": "{:,.2f}", "DF": "{:.6f}", "r": "{:.3%}", "q": "{:.3%}",
    "F/S - 1": "{:+.3%}", "resíduo": "{:.3f}", "r do vencimento": "{:.3%}",
}, na_rep="-"))

n_alert = int(forwards["alerta q"].sum())
print(f"Vencimentos com q fora de [{Q_RANGE[0]:.0%}, {Q_RANGE[1]:.0%}]: {n_alert} de {len(forwards)}")

# COMMAND ----------

try:
    sofr = source.sofr(market_date)
except Exception as exc:  # a SOFR de D-1 sai às 08:00 ET de D+0
    sofr = None
    print("SOFR indisponível:", exc)

own = forwards["no ajuste"]
out_fit = forwards["r do vencimento"].notna() & ~own
venc = forwards["vencimento"].astype(str)
hover = "%{customdata}<br>T = %{x:.3f}<br>%{y:.3f}% a.a.<extra>%{fullData.name}</extra>"

fig = make_subplots(rows=1, cols=2, subplot_titles=("r(T)", "q(T) implícito"), horizontal_spacing=0.08)
fig.add_trace(go.Scatter(x=forwards["T"], y=forwards["r"] * 100, mode="lines",
                         name="r(T) curva suavizada", line=dict(color=WARM[2], width=2),
                         customdata=venc, hovertemplate=hover), row=1, col=1)
fig.add_trace(go.Scatter(x=forwards.loc[own, "T"], y=forwards.loc[own, "r do vencimento"] * 100, mode="markers",
                         name="taxa por vencimento (no ajuste)", marker=dict(symbol="circle", size=7, color=WARM[1]),
                         customdata=venc[own], hovertemplate=hover), row=1, col=1)
if out_fit.any():
    fig.add_trace(go.Scatter(x=forwards.loc[out_fit, "T"], y=forwards.loc[out_fit, "r do vencimento"] * 100,
                             mode="markers", name="fora do ajuste",
                             marker=dict(symbol="x", size=9, color=WARM[3]),
                             customdata=venc[out_fit], hovertemplate=hover), row=1, col=1)
if sofr is not None:
    fig.add_trace(go.Scatter(x=[0, forwards["T"].max()], y=[sofr * 100] * 2, mode="lines",
                             name=f"SOFR contínua {sofr:.3%}", line=dict(color=WARM[0], dash="dash", width=1.5),
                             hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=forwards["T"], y=forwards["q"] * 100, mode="lines+markers", name="q implícito",
                         line=dict(color=WARM[1], width=2), marker=dict(size=7),
                         customdata=venc, hovertemplate=hover), row=1, col=2)
fig.add_hrect(y0=Q_RANGE[0] * 100, y1=Q_RANGE[1] * 100, fillcolor=WARM[2], opacity=0.12, line_width=0,
              row=1, col=2, annotation_text="faixa razoável", annotation_position="top left")
fig.update_xaxes(title_text="T (anos)")
fig.update_yaxes(title_text="% a.a.")
show(style_fig(fig, f"SPX - estrutura a termo implícita em {market_date}", height=480), "r_q")

if sofr is not None:
    long_r = forwards.loc[forwards["T"] >= MIN_T_RATE, ["vencimento", "dias", "r", "r do vencimento", "no ajuste"]]
    long_r = long_r.assign(**{"curva - SOFR (bps)": (long_r["r"] - sofr) * 1e4,
                              "vencimento - SOFR (bps)": (long_r["r do vencimento"] - sofr) * 1e4})
    display(long_r.style.format({"r": "{:.3%}", "r do vencimento": "{:.3%}", "curva - SOFR (bps)": "{:+.0f}",
                                 "vencimento - SOFR (bps)": "{:+.0f}"}, na_rep="-"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cortes de arbitragem vertical/butterfly por vencimento
# MAGIC
# MAGIC Refaz a etapa contra o forward duas vezes sobre a mesma cadeia (já em cache): uma com a checagem de
# MAGIC vertical/butterfly desligada (tolerância infinita) e outra normal. A diferença é o que essa checagem
# MAGIC removeu. A contagem final precisa bater com a linha correspondente do relatório de limpeza.

# COMMAND ----------

cfg = adapter.cleaning
quotes = clean_quotes(adapter.load_chain(market_date), cfg, CleaningReport())
fwd = forwards[["vencimento", "T", "forward", "r", "q"]].rename(columns={"vencimento": "expiration"})
merged = quotes.merge(fwd, on=["expiration", "T"], how="inner")

before_arb = clean_against_forwards(merged, snap.spot, replace(cfg, arb_tolerance=np.inf), CleaningReport())
after_arb = clean_against_forwards(merged, snap.spot, cfg, CleaningReport())

expected = snap.report.to_frame().set_index("etapa").loc["Sem arbitragem de vertical/butterfly", "quantidade"]
assert len(after_arb) == expected, f"Diagnóstico divergiu do pipeline: {len(after_arb)} != {expected}"

keys = ["expiration", "option_type"]
arb = (before_arb.groupby(keys).size().rename("antes").to_frame()
       .join(after_arb.groupby(keys).size().rename("depois"), how="left").fillna(0).astype(int))
arb["removidos"] = arb["antes"] - arb["depois"]
arb["% removido"] = arb["removidos"] / arb["antes"]
arb = arb.reset_index().merge(fwd[["expiration", "T"]], on="expiration").sort_values("T")

print(f"Total removido pela checagem: {arb['removidos'].sum()} de {arb['antes'].sum()}")
display(arb.sort_values("removidos", ascending=False).head(15)
        .style.format({"T": "{:.3f}", "% removido": "{:.1%}"}))

# Onde (em moneyness) estão os pontos removidos: asas ou perto do ATM?
cols = ["expiration", "option_type", "strike"]
removed = before_arb.merge(after_arb[cols], on=cols, how="left", indicator=True)
removed = removed.loc[removed["_merge"].eq("left_only")]
bins = [-np.inf, -0.3, -0.1, -0.03, 0.03, 0.1, 0.3, np.inf]
labels = ["< -30%", "-30% a -10%", "-10% a -3%", "ATM ±3%", "3% a 10%", "10% a 30%", "> 30%"]
where = pd.DataFrame({
    "antes": pd.cut(before_arb["log_moneyness"], bins, labels=labels).value_counts(sort=False),
    "removidos": pd.cut(removed["log_moneyness"], bins, labels=labels).value_counts(sort=False),
})
where["% removido"] = where["removidos"] / where["antes"]
display(where.reset_index(names="ln(K/F)").style.format({"% removido": "{:.1%}"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Smiles em prazos de referência (1, 3, 6, 12 e 24 meses)

# COMMAND ----------

opts = snap.options
expiries = opts.groupby("expiration")["T"].first().sort_values()
targets = {"1m": 1 / 12, "3m": 0.25, "6m": 0.5, "12m": 1.0, "24m": 2.0}
chosen = {}
for label, t in targets.items():
    exp = (expiries - t).abs().idxmin()
    if exp not in chosen.values():
        chosen[label] = exp

SMILE_HOVER = "K = %{customdata:,.0f}<br>ln(K/F) = %{x:.3f}<br>vol = %{y:.2f}%<extra>%{fullData.name}</extra>"

fig = go.Figure()
for i, (label, exp) in enumerate(chosen.items()):
    s = opts.loc[opts["expiration"].eq(exp)].sort_values("log_moneyness")
    fig.add_trace(go.Scatter(
        x=s["log_moneyness"], y=s["implied_vol"] * 100, mode="lines+markers",
        name=f"{label}: {exp} (T={expiries[exp]:.2f}, {len(s)} pts)",
        line=dict(color=WARM[i], dash=DASHES[i], width=1.8),
        marker=dict(symbol=SYMBOLS[i], size=sparse_sizes(len(s)), color=WARM[i]),
        customdata=s["strike"], hovertemplate=SMILE_HOVER))
fig.add_vline(x=0, line_color=MUTED, line_width=1)
fig.update_xaxes(title_text="ln(K/F)")
fig.update_yaxes(title_text="vol implícita (%)")
show(style_fig(fig, f"SPX - smiles em {market_date}", height=560), "smiles")

# ATM por vencimento: vol do ponto mais próximo de K = F.
atm = (opts.assign(dist=opts["log_moneyness"].abs()).sort_values("dist")
       .groupby("expiration").first()[["T", "strike", "forward", "implied_vol"]].sort_values("T"))
fig = go.Figure(go.Scatter(
    x=atm["T"], y=atm["implied_vol"] * 100, mode="lines+markers", name="vol ATM",
    line=dict(color=WARM[2], width=2), marker=dict(size=7),
    customdata=atm.index.astype(str),
    hovertemplate="%{customdata}<br>T = %{x:.3f}<br>vol ATM = %{y:.2f}%<extra></extra>"))
fig.update_xaxes(title_text="T (anos)")
fig.update_yaxes(title_text="vol implícita (%)")
show(style_fig(fig, "Vol ATM por prazo", height=420), "vol_atm")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Amostra e resumo do dia
# MAGIC
# MAGIC Copie o resumo para comparar entre dias: spot, quantidade de pontos e vencimentos, vol ATM 1m/12m,
# MAGIC r e q de 1 ano devem variar pouco de um dia para o outro.

# COMMAND ----------

sample_cols = ["expiration", "T", "strike", "option_type", "bid", "ask", "market_price", "forward",
               "log_moneyness", "implied_vol"]
display(opts[sample_cols].sort_values(["T", "strike"]).head(20).style.format({
    "T": "{:.4f}", "bid": "{:.2f}", "ask": "{:.2f}", "market_price": "{:.2f}", "forward": "{:,.2f}",
    "log_moneyness": "{:+.4f}", "implied_vol": "{:.2%}",
}))

def nearest(frame, t, col):
    return float(frame.iloc[(frame["T"] - t).abs().argmin()][col])

summary = pd.Series({
    "data": str(market_date),
    "spot": round(snap.spot, 2),
    "spot usado": snap.spot_source,
    "fechamento oficial": round(spot_fechamento, 2),
    "paridade - fechamento (bps)": round(spot_gap_bps, 1),
    "vencimentos": len(snap.forwards),
    "pontos": len(opts),
    "removidos vertical/butterfly": int(arb["removidos"].sum()),
    "avisos": len(snap.report.warnings),
    "vol ATM 1m": f"{nearest(atm, 1 / 12, 'implied_vol'):.2%}",
    "vol ATM 12m": f"{nearest(atm, 1.0, 'implied_vol'):.2%}",
    "r 1a": f"{nearest(forwards, 1.0, 'r'):.3%}",
    "q 1a": f"{nearest(forwards, 1.0, 'q'):.3%}",
    "SOFR": f"{sofr:.3%}" if sofr is not None else "n/d",
    "vencimentos com q fora da faixa": n_alert,
    "cobertura": coverage_msg or "normal / sem histórico",
}, name="resumo")
display(summary.to_frame())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6a. Precificação de referência: call down-and-out sobre o SPX
# MAGIC
# MAGIC O equivalente do contrato do notebook 01 (IBOV), com strike e barreira em % do spot (padrão 95% e 85%,
# MAGIC 6 meses). Mesmas regras do 01: vol no strike por interpolação linear em ln(K/F) x prazo (vizinho mais
# MAGIC próximo fora dos dados), r(T) e q(T) da paridade no prazo, Monte Carlo com ponte browniana
# MAGIC (200 mil caminhos, 252 passos, semente 42) e Reiner-Rubinstein como referência.

# COMMAND ----------

ref = precificar_referencia(
    float(snap.spot), opts, forwards,
    strike_pct=float(dbutils.widgets.get("ref_strike_pct")) / 100,
    barrier_pct=float(dbutils.widgets.get("ref_barreira_pct")) / 100,
    maturity=float(dbutils.widgets.get("ref_prazo_anos")),
    **MC_PADRAO,
)
pricing_result, alvo = ref["pricing_result"], ref["alvo"]
display(pd.Series({
    "spot": pricing_result["market"]["spot"], "strike": alvo["target_strike"], "barreira": alvo["target_barrier"],
    "prazo (anos)": alvo["target_maturity"], "r(T)": alvo["target_r"], "q(T)": alvo["target_q"],
    "vol no strike": alvo["target_iv"], "interpolação": alvo["interpolation_method"],
    "vanilla (BS)": pricing_result["vanilla_price"], "down-and-out (MC)": pricing_result["barrier_price"],
    "Reiner-Rubinstein": pricing_result["analytical_price"],
    "erro MC x analítico": pricing_result["analytical_error_pct"],
    "P(knock-out)": pricing_result["knock_out_probability"], "erro-padrão": pricing_result["standard_error"],
}, name="referência").to_frame())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6b. Resultado para o dashboard
# MAGIC
# MAGIC Grava `RESULTS_spx_superficie_<data e hora>.csv` na pasta `ibov-barrier-results` (a mesma do notebook 01),
# MAGIC no formato que o dashboard lê: metadados, resumo do dia, forwards/r/q por vencimento, pontos da superfície,
# MAGIC funil de limpeza, cortes de arbitragem e a precificação de referência (`pricing_result` e
# MAGIC `simulation_diagnostics`, com as chaves do notebook 01).

# COMMAND ----------

from zoneinfo import ZoneInfo

from ibov_barrier.results_bundle import SCHEMA_VERSION, results_filename, write_bundle

run_ts = datetime.now(ZoneInfo("America/Sao_Paulo"))
results_dir = Path("/".join(notebook_path.split("/")[:4])) / "ibov-barrier-results"
spx_result_path = write_bundle(results_dir / results_filename("spx_superficie", run_ts), {
    "metadata": {
        "schema_version": SCHEMA_VERSION, "kind": "spx_superficie", "execution_timestamp": run_ts,
        "market_date": market_date, "spot": float(snap.spot), "sofr": sofr, "vencimentos": len(snap.forwards),
        "pontos": len(opts), "avisos": list(snap.report.warnings), "cobertura": coverage_msg,
        "faixa q": list(Q_RANGE), "T minimo taxa": MIN_T_RATE, "fonte": "ThetaData (SPXW)",
        "spot_fonte": snap.spot_source, "spot_fechamento": spot_fechamento, "spot_paridade": spot_paridade,
        "spot_gap_bps": spot_gap_bps, **alvo,
    },
    "summary": summary,
    "forwards": forwards,
    "options": opts[["expiration", "T", "strike", "option_type", "market_price", "forward", "log_moneyness",
                     "implied_vol"]],
    "funnel": snap.report.to_frame(),
    "arb": arb,
    "pricing_result": pricing_result,
    "simulation_diagnostics": {**ref["simulation_diagnostics"], "execution_timestamp": run_ts,
                               "market_date": market_date},
})
print(f"Resultado para o dashboard: {spx_result_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6c. Cadeia do dia em disco (para o notebook 09)
# MAGIC
# MAGIC O 09 (Quanto/Compo) lê a cadeia de `spx_dados/brutos/<data>/` (formato do notebook 08). Gravar aqui o dia
# MAGIC que acabou de ser baixado evita uma nova chamada à API e deixa o 09 sempre no mesmo D-1 deste notebook.

# COMMAND ----------

brutos = Path("/".join(notebook_path.split("/")[:4])) / "spx_dados" / "brutos"
cadeia = {root: raw for (root, d), raw in _raw_cache.items() if d == market_date}
print(f"Cadeia de {market_date} ({', '.join(f'{k}: {len(v):,}' for k, v in cadeia.items())}):",
      salvar_cadeia_do_dia(brutos, market_date, cadeia, spot_fechamento, sofr), "em", brutos / str(market_date))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Teste: série SPX (AM) nos vencimentos mensais
# MAGIC
# MAGIC Nas terceiras sextas-feiras existem duas séries com a mesma data: **SPX** (liquidação pela abertura, AM,
# MAGIC o contrato mensal tradicional) e **SPXW** (liquidação pelo fechamento, PM). O padrão atual fica com a
# MAGIC SPXW. Nos vencimentos mensais e longos, a série mais líquida costuma ser a SPX AM, e foi justamente
# MAGIC nessas datas que a checagem de vertical/butterfly cortou de 39% a 57% dos pontos.
# MAGIC
# MAGIC Aqui as duas escolhas rodam sobre a **mesma cadeia em cache** (sem nova chamada à API):
# MAGIC
# MAGIC - **PM (atual):** `preferred_roots=("SPXW",)`;
# MAGIC - **AM nos mensais (teste):** `preferred_roots=("SPX", "SPXW")`. Nas datas só com SPXW nada muda.
# MAGIC
# MAGIC O padrão do pipeline só muda depois de alguns dias de comparação.

# COMMAND ----------

cfg_am = replace(adapter.cleaning, preferred_roots=("SPX", "SPXW"))
adapter_am = SPXAdapter(fetch_chain_cached, spot_source=adapter.spot_source, cleaning=cfg_am)
snap_am = adapter_am.load_snapshot(market_date)

chain_all = adapter.load_chain(market_date)
monthly = sorted(set(chain_all.loc[chain_all["root"].eq("SPX"), "expiration"])
                 & set(chain_all.loc[chain_all["root"].eq("SPXW"), "expiration"]))
print(f"Datas com as duas séries (mensais): {len(monthly)}")

VARIANTS = {"PM": (adapter, snap), "AM": (adapter_am, snap_am)}


def forwards_frame(s):
    f = pd.DataFrame([{"expiration": p.expiration, "T": p.T, "forward": p.forward, "r": p.rate,
                       "pares": p.n_pairs, "resíduo": p.residual_std} for p in s.forwards])
    f["q"] = f["r"] - np.log(f["forward"] / s.spot) / f["T"]
    return f.set_index("expiration")


def arb_counts(adp, s):
    """Pontos antes/depois da checagem de vertical/butterfly, por vencimento (mesma técnica da seção 4)."""
    f = forwards_frame(s).reset_index()[["expiration", "T", "forward", "r", "q"]]
    m = clean_quotes(adp.load_chain(market_date), adp.cleaning, CleaningReport()).merge(f, on=["expiration", "T"])
    before = clean_against_forwards(m, s.spot, replace(adp.cleaning, arb_tolerance=np.inf), CleaningReport())
    after = clean_against_forwards(m, s.spot, adp.cleaning, CleaningReport())
    return pd.DataFrame({"antes": before.groupby("expiration").size(),
                         "depois": after.groupby("expiration").size()}).fillna(0).astype(int)


def atm_vol(s):
    o = s.options.assign(dist=s.options["log_moneyness"].abs()).sort_values("dist")
    return o.groupby("expiration")["implied_vol"].first()


per_variant = {}
for name, (adp, s) in VARIANTS.items():
    f = forwards_frame(s)
    a = arb_counts(adp, s)
    per_variant[name] = pd.DataFrame({
        "série": s.options.groupby("expiration")["root"].first(),
        "T": f["T"],
        "pontos": s.options.groupby("expiration").size(),
        "% cortado arb": 1 - a["depois"] / a["antes"],
        "resíduo paridade": f["resíduo"],
        "r": f["r"],
        "vol ATM": atm_vol(s),
    })

# COMMAND ----------

# Funil de limpeza lado a lado.
rep_pm, rep_am = snap.report.to_frame(), snap_am.report.to_frame()
funnel = pd.DataFrame({"etapa": rep_pm["etapa"], "PM (atual)": rep_pm["quantidade"],
                       "AM nos mensais": rep_am["quantidade"]})
funnel["diferença"] = funnel["AM nos mensais"] - funnel["PM (atual)"]
display(funnel)

# Só os vencimentos mensais: é onde as duas escolhas diferem.
cols = ["série", "T", "pontos", "% cortado arb", "resíduo paridade", "r", "vol ATM"]
cmp = per_variant["PM"][cols].join(per_variant["AM"][cols], lsuffix=" PM", rsuffix=" AM", how="outer")
cmp = cmp.loc[cmp.index.isin(monthly)].sort_values("T PM")
cmp["Δ pontos"] = cmp["pontos AM"].fillna(0) - cmp["pontos PM"].fillna(0)
cmp["Δ vol ATM (pts)"] = (cmp["vol ATM AM"] - cmp["vol ATM PM"]) * 100
display(cmp.reset_index(names="vencimento").style.format({
    "T PM": "{:.3f}", "T AM": "{:.3f}",
    "% cortado arb PM": "{:.1%}", "% cortado arb AM": "{:.1%}",
    "resíduo paridade PM": "{:.3f}", "resíduo paridade AM": "{:.3f}",
    "r PM": "{:.3%}", "r AM": "{:.3%}", "vol ATM PM": "{:.2%}", "vol ATM AM": "{:.2%}",
    "Δ pontos": "{:+.0f}", "Δ vol ATM (pts)": "{:+.2f}",
}, na_rep="-"))

# COMMAND ----------

def arb_total(s):
    rep = s.report.to_frame()
    i = rep.index[rep["etapa"].eq("Sem arbitragem de vertical/butterfly")][0]
    return int(rep.loc[i - 1, "quantidade"] - rep.loc[i, "quantidade"])


def r_minus_sofr(s):
    f = forwards_frame(s)
    gap = (f.loc[f["T"] >= MIN_T_RATE, "r"] - sofr) * 1e4 if sofr is not None else pd.Series(dtype=float)
    return gap


rows = {}
for name, (adp, s) in VARIANTS.items():
    gap = r_minus_sofr(s)
    in_monthly = s.options.loc[s.options["expiration"].isin(monthly)]
    rows[name] = {
        "vencimentos": len(s.forwards),
        "pontos": len(s.options),
        "pontos nos mensais": len(in_monthly),
        "removidos vertical/butterfly": arb_total(s),
        "avisos": len(s.report.warnings),
        "resíduo médio da paridade (mensais)": per_variant[name].loc[
            per_variant[name].index.isin(monthly), "resíduo paridade"].mean(),
        "r - SOFR mediana, T >= 3m (bps)": gap.median() if len(gap) else np.nan,
        "r - SOFR desvio-padrão (bps)": gap.std() if len(gap) else np.nan,
    }
verdict = pd.DataFrame(rows).rename(columns={"PM": "PM (atual)", "AM": "AM nos mensais"})
display(verdict.style.format("{:,.2f}"))

# COMMAND ----------

# Smiles PM x AM em até 4 vencimentos mensais (perto de 1, 3, 12 e 24 meses).
both = cmp.dropna(subset=["T PM", "T AM"])
picked = []
for t in (1 / 12, 0.25, 1.0, 2.0):
    if len(both):
        e = (both["T PM"] - t).abs().idxmin()
        if e not in picked:
            picked.append(e)

if picked:
    ncols = 2
    nrows = int(np.ceil(len(picked) / ncols))
    styles = {"PM": (WARM[1], "solid", "circle", "PM (SPXW)"), "AM": (WARM[4], "dash", "square", "AM (SPX)")}

    def smile_of(s, e):
        return s.options.loc[s.options["expiration"].eq(e)].sort_values("log_moneyness")

    titles = [f"{e} (T={both.loc[e, 'T PM']:.2f}) · PM {len(smile_of(snap, e))} / AM {len(smile_of(snap_am, e))} pts"
              for e in picked]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.14)
    for k, e in enumerate(picked):
        row, col = k // ncols + 1, k % ncols + 1
        for name, (_, s) in VARIANTS.items():
            color, dash, symbol, label = styles[name]
            sm = smile_of(s, e)
            fig.add_trace(go.Scatter(
                x=sm["log_moneyness"], y=sm["implied_vol"] * 100, mode="lines+markers", name=label,
                legendgroup=name, showlegend=(k == 0),
                line=dict(color=color, dash=dash, width=1.8),
                marker=dict(symbol=symbol, size=sparse_sizes(len(sm)), color=color),
                customdata=sm["strike"], hovertemplate=SMILE_HOVER), row=row, col=col)
        fig.add_vline(x=0, line_color=MUTED, line_width=1, row=row, col=col)
    fig.update_xaxes(title_text="ln(K/F)")
    fig.update_yaxes(title_text="vol implícita (%)")
    show(style_fig(fig, "Smiles nos vencimentos mensais: PM (SPXW) x AM (SPX)", height=430 * nrows),
         "smiles_pm_x_am")
else:
    print("Nenhum vencimento mensal presente nas duas versões.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Relatório único para enviar
# MAGIC
# MAGIC Junta em texto tudo o que as seções acima calcularam. Rode depois das outras células e copie a saída
# MAGIC inteira. Se uma seção anterior falhou, o bloco dela aparece como `[não disponível]`, com o motivo.

# COMMAND ----------

import platform
import subprocess
import traceback
from importlib import metadata as _md

_g = globals()


def _need(*names):
    missing = [n for n in names if n not in _g]
    if missing:
        raise NameError("ausente (a célula que cria isso falhou ou não rodou): " + ", ".join(missing))


def _fmt(spec):
    return {c: (lambda f: (lambda v: "-" if pd.isna(v) else f.format(v)))(f) for c, f in (spec or {}).items()}


def _df(frame, spec=None):
    frame = frame.reset_index() if not isinstance(frame.index, pd.RangeIndex) else frame
    spec = {c: f for c, f in (spec or {}).items() if c in frame.columns}
    with pd.option_context("display.max_rows", 500, "display.max_columns", None, "display.width", 300):
        print(frame.to_string(index=False, formatters=_fmt(spec)))


def _block(title, fn):
    print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)
    try:
        fn()
    except Exception as exc:
        print(f"[não disponível] {type(exc).__name__}: {exc}")
        print("".join(traceback.format_exc().splitlines(keepends=True)[-4:]))


def _r0_ambiente():
    print("python:", platform.python_version())
    for pkg in ("thetadata", "grpcio", "exchange-calendars", "lxml", "pandas", "numpy", "plotly"):
        try:
            print(f"{pkg}: {_md.version(pkg)}")
        except Exception:
            print(f"{pkg}: não instalado")
    try:
        sha = subprocess.run(["git", "-C", project_dir, "log", "-1", "--format=%h %s"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        print("commit do Git folder:", sha or "n/d")
    except Exception:
        print("commit do Git folder: n/d")


def _r1_foto():
    _need("market_date", "snap", "source")
    print("data de mercado:", market_date, "| origem da chave:", source.key_source)
    print(f"spot: {snap.spot:,.2f} ({snap.spot_source}) | vencimentos: {len(snap.forwards)} | pontos: {len(snap.options):,}")
    _df(snap.report.to_frame())
    print("avisos:", *(snap.report.warnings or ["nenhum"]), sep="\n - ")
    print("cobertura:", _g.get("coverage_msg") or "normal / sem histórico")


def _r2_forwards():
    _need("forwards", "n_alert")
    _df(forwards, {"T": "{:.4f}", "forward": "{:,.2f}", "DF": "{:.6f}", "r": "{:.3%}", "q": "{:.3%}",
                   "F/S - 1": "{:+.3%}", "resíduo": "{:.3f}", "r do vencimento": "{:.3%}"})
    print(f"\nq fora de [{Q_RANGE[0]:.0%}, {Q_RANGE[1]:.0%}]: {n_alert} de {len(forwards)}")
    print("SOFR:", f"{sofr:.3%}" if _g.get("sofr") is not None else "n/d")
    if "long_r" in _g:
        print("\nr - SOFR (T >= 3m):")
        _df(long_r, {"r": "{:.3%}", "r do vencimento": "{:.3%}", "curva - SOFR (bps)": "{:+.0f}",
                     "vencimento - SOFR (bps)": "{:+.0f}"})


def _r3_arb():
    _need("arb", "where")
    print(f"total removido: {arb['removidos'].sum()} de {arb['antes'].sum()}\n")
    _df(arb.sort_values(["T", "option_type"]), {"T": "{:.3f}", "% removido": "{:.1%}"})
    print("\npor moneyness:")
    _df(where, {"% removido": "{:.1%}"})


def _r4_smiles():
    _need("opts", "chosen", "atm")
    grid = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10]
    rows = []
    for label, exp in chosen.items():
        s = opts.loc[opts["expiration"].eq(exp)].sort_values("log_moneyness")
        row = {"prazo": label, "vencimento": exp, "T": s["T"].iloc[0], "pontos": len(s),
               "ln(K/F) min": s["log_moneyness"].min(), "ln(K/F) max": s["log_moneyness"].max()}
        for x in grid:
            inside = s["log_moneyness"].min() <= x <= s["log_moneyness"].max()
            row[f"vol@{x:+.2f}"] = np.interp(x, s["log_moneyness"], s["implied_vol"]) if inside else np.nan
        rows.append(row)
    spec = {"T": "{:.3f}", "ln(K/F) min": "{:+.3f}", "ln(K/F) max": "{:+.3f}",
            **{f"vol@{x:+.2f}": "{:.2%}" for x in grid}}
    _df(pd.DataFrame(rows), spec)
    print("\nvol ATM por vencimento:")
    _df(atm, {"T": "{:.4f}", "strike": "{:,.0f}", "forward": "{:,.2f}", "implied_vol": "{:.2%}"})


def _r5_resumo():
    _need("summary")
    _df(summary.to_frame())


def _r6_am_pm():
    _need("monthly", "funnel", "cmp", "verdict")
    print("vencimentos mensais (SPX e SPXW):", ", ".join(map(str, monthly)), "\n")
    _df(funnel)
    print()
    _df(cmp, {"T PM": "{:.3f}", "T AM": "{:.3f}", "% cortado arb PM": "{:.1%}", "% cortado arb AM": "{:.1%}",
              "resíduo paridade PM": "{:.3f}", "resíduo paridade AM": "{:.3f}", "r PM": "{:.3%}",
              "r AM": "{:.3%}", "vol ATM PM": "{:.2%}", "vol ATM AM": "{:.2%}", "Δ pontos": "{:+.0f}",
              "Δ vol ATM (pts)": "{:+.2f}"})
    print()
    _df(verdict.map(lambda v: f"{v:,.2f}") if hasattr(verdict, "map") else verdict.round(2))


def _r7_amostra():
    _need("opts", "sample_cols")
    _df(opts[sample_cols].sort_values(["T", "strike"]).head(10),
        {"T": "{:.4f}", "forward": "{:,.2f}", "log_moneyness": "{:+.4f}", "implied_vol": "{:.2%}"})


def _r8_referencia():
    _need("pricing_result", "alvo", "spot_fechamento", "spot_paridade", "spot_gap_bps", "spx_result_path")
    print(f"spot usado: {snap.spot:,.2f} ({snap.spot_source}) | fechamento oficial {spot_fechamento:,.2f} | "
          f"paridade {spot_paridade:,.2f} | diferença {spot_gap_bps:+.1f} bps")
    print(f"contrato: K {alvo['target_strike']:,.2f} ({alvo['strike_pct']:.0%}) | H {alvo['target_barrier']:,.2f} "
          f"({alvo['barrier_pct']:.0%}) | T {alvo['target_maturity']:.4f}")
    print(f"mercado: r {alvo['target_r']:.4%} | q {alvo['target_q']:.4%} | vol {alvo['target_iv']:.4%} "
          f"({alvo['interpolation_method']}) | forward {alvo['target_forward']:,.2f}")
    p = pricing_result
    print(f"vanilla {p['vanilla_price']:,.4f} | down-and-out MC {p['barrier_price']:,.4f} (EP {p['standard_error']:.4f}) | "
          f"Reiner-Rubinstein {p['analytical_price']:,.4f} | erro {p['analytical_error_pct']:.4%} | "
          f"P(KO) {p['knock_out_probability']:.4%}")
    print("RESULTS:", spx_result_path.name)


print(f"RELATÓRIO SPX D-1 | gerado em {pd.Timestamp.now(tz='America/Sao_Paulo'):%Y-%m-%d %H:%M} (Brasília)")
for _title, _fn in [
    ("0. Ambiente", _r0_ambiente),
    ("1. Fotografia D-1 e funil de limpeza", _r1_foto),
    ("2. Forwards, r, q e SOFR", _r2_forwards),
    ("3. Cortes de vertical/butterfly", _r3_arb),
    ("4. Smiles (vol interpolada em ln(K/F)) e vol ATM", _r4_smiles),
    ("5. Resumo do dia", _r5_resumo),
    ("6. Teste AM x PM nos mensais", _r6_am_pm),
    ("7. Amostra (10 linhas)", _r7_amostra),
    ("8. Spot e precificação de referência", _r8_referencia),
]:
    _block(_title, _fn)
print("\nFIM DO RELATÓRIO")
