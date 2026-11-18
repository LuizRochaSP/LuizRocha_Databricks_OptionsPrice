# Databricks notebook source
# MAGIC %md
# MAGIC # SPX: fotografia D-1 e superfície de volatilidade (Fase 3)
# MAGIC
# MAGIC Branch `feature/sp500` apenas. Este notebook **não** deve ficar no Git folder de produção (`main`).
# MAGIC
# MAGIC Fluxo (`SPXAdapter.load_snapshot`):
# MAGIC
# MAGIC 1. cadeia EOD de SPX e SPXW na ThetaData (plano FREE, D-1);
# MAGIC 2. limpeza de cotações;
# MAGIC 3. paridade put-call: forward e fator de desconto por vencimento, `C − P = DF·(F − K)`;
# MAGIC 4. spot oficial do índice; `r(T) = −ln(DF)/T` e `q(T) = r(T) − ln(F/S)/T`;
# MAGIC 5. limpeza contra o forward (limites de arbitragem, só OTM, moneyness, vertical/butterfly);
# MAGIC 6. volatilidade implícita e limpeza do smile.
# MAGIC
# MAGIC Pré-requisitos: Python 3.12+ e o Secret `quantita/thetadata_api_key` (a chave nunca aparece aqui).
# MAGIC
# MAGIC Para validar a estabilidade, rodar em vários dias seguidos e comparar o resumo final.

# COMMAND ----------

# MAGIC %pip install --upgrade "grpcio>=1.80.0" thetadata python-dotenv "exchange-calendars==4.13.2" "lxml>=6.0"

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
    raise RuntimeError(f"{SRC} não tem ibov_barrier/markets: o Git folder precisa estar na branch feature/sp500.")

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
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ibov_barrier.markets.base import CleaningReport
from ibov_barrier.markets.cleaning import clean_against_forwards, clean_quotes
from ibov_barrier.markets.spx import SPXAdapter, ThetaDataSource

# Prazo a partir do qual a taxa vem da própria regressão da paridade (igual ao padrão de implied_forwards).
MIN_T_RATE = 0.25
# Faixa considerada razoável para o dividend yield implícito do SPX (alerta, não filtro).
Q_RANGE = (0.0, 0.03)

dbutils.widgets.text("data_mercado", "", "Data (AAAA-MM-DD; vazio = D-1)")

# Tema dos gráficos: fundo preto, textos brancos, curvas em tons quentes.
# Os vencimentos são ordenados (1m -> 24m), então a paleta é uma rampa quente do claro ao escuro,
# com marcador e traço diferentes por série: tons quentes ficam próximos entre si, e o formato
# garante a leitura (inclusive para daltonismo).
BG, GRID, MUTED = "#0B0B0B", "#2A2A2A", "#8C8C8C"
WARM = ["#FFF3B0", "#FFC233", "#FF8A1F", "#F0432F", "#D6246E"]   # amarelo -> âmbar -> laranja -> vermelho -> magenta
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": "white", "axes.labelcolor": "white", "axes.titlecolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.facecolor": "#141414", "legend.edgecolor": "#3A3A3A", "legend.labelcolor": "white",
    "axes.prop_cycle": plt.cycler(color=WARM),
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "axes.titleweight": "bold", "axes.titlesize": 13,
})

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

adapter = SPXAdapter(fetch_chain_cached, spot_source=source.index_close)

typed = dbutils.widgets.get("data_mercado").strip()
market_date = date.fromisoformat(typed) if typed else adapter.market_date()
print("Data de mercado:", market_date, "| chave:", source.key_source)

snap = adapter.load_snapshot(market_date)
print(f"Spot SPX: {snap.spot:,.2f} ({snap.spot_source})")
print(f"Vencimentos com forward: {len(snap.forwards)}")
print(f"Pontos na superfície: {len(snap.options):,}")

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
# MAGIC Abaixo de 3 meses, a taxa não vem da regressão (o ruído de cotação explode em `−ln(DF)/T`):
# MAGIC usa-se a taxa do vencimento confiável mais curto, e só o forward é estimado. Esses prazos aparecem
# MAGIC como `r emprestado`.

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
} for p in snap.forwards]).sort_values("T").reset_index(drop=True)
forwards["q"] = forwards["r"] - np.log(forwards["forward"] / snap.spot) / forwards["T"]
forwards["F/S - 1"] = forwards["forward"] / snap.spot - 1
forwards["origem de r"] = np.where(forwards["T"] >= MIN_T_RATE, "regressão", "r emprestado")
forwards["alerta q"] = ~forwards["q"].between(*Q_RANGE)

display(forwards.style.format({
    "T": "{:.4f}", "forward": "{:,.2f}", "DF": "{:.6f}", "r": "{:.3%}", "q": "{:.3%}",
    "F/S - 1": "{:+.3%}", "resíduo": "{:.3f}",
}))

n_alert = int(forwards["alerta q"].sum())
print(f"Vencimentos com q fora de [{Q_RANGE[0]:.0%}, {Q_RANGE[1]:.0%}]: {n_alert} de {len(forwards)}")

# COMMAND ----------

try:
    sofr = source.sofr(market_date)
except Exception as exc:  # a SOFR de D-1 sai às 08:00 ET de D+0
    sofr = None
    print("SOFR indisponível:", exc)

fig, (ax_r, ax_q) = plt.subplots(1, 2, figsize=(13, 4.5))
own = forwards["origem de r"].eq("regressão")
ax_r.plot(forwards.loc[own, "T"], forwards.loc[own, "r"] * 100, "o-", color=WARM[2], label="r implícito (paridade)")
ax_r.plot(forwards.loc[~own, "T"], forwards.loc[~own, "r"] * 100, "o", mfc="none", color=MUTED,
          label="r emprestado (T < 3m)")
if sofr is not None:
    ax_r.axhline(sofr * 100, ls="--", color=WARM[0], label=f"SOFR contínua {sofr:.3%}")
ax_r.set(title="r(T)", xlabel="T (anos)", ylabel="% a.a.")
ax_r.legend()

ax_q.plot(forwards["T"], forwards["q"] * 100, "o-", color=WARM[1], label="q implícito")
ax_q.axhspan(Q_RANGE[0] * 100, Q_RANGE[1] * 100, color=WARM[2], alpha=0.12, label="faixa razoável")
ax_q.set(title="q(T) implícito", xlabel="T (anos)", ylabel="% a.a.")
ax_q.legend()
plt.tight_layout()
plt.show()

if sofr is not None:
    long_r = forwards.loc[own, ["vencimento", "dias", "r"]].assign(**{"r - SOFR (bps)": lambda d: (d["r"] - sofr) * 1e4})
    display(long_r.style.format({"r": "{:.3%}", "r - SOFR (bps)": "{:+.0f}"}))

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

fig, ax = plt.subplots(figsize=(11, 6))
for i, (label, exp) in enumerate(chosen.items()):
    s = opts.loc[opts["expiration"].eq(exp)].sort_values("log_moneyness")
    ax.plot(s["log_moneyness"], s["implied_vol"] * 100, color=WARM[i], ls=LINESTYLES[i], lw=1.6,
            marker=MARKERS[i], ms=4, markevery=max(1, len(s) // 12),
            label=f"{label}: {exp} (T={expiries[exp]:.2f}, {len(s)} pts)")
ax.axvline(0, color=MUTED, lw=0.8)
ax.set(title=f"SPX - smiles em {market_date}", xlabel="ln(K/F)", ylabel="vol implícita (%)")
ax.legend()
plt.show()

# ATM por vencimento: vol do ponto mais próximo de K = F.
atm = (opts.assign(dist=opts["log_moneyness"].abs()).sort_values("dist")
       .groupby("expiration").first()[["T", "strike", "forward", "implied_vol"]].sort_values("T"))
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(atm["T"], atm["implied_vol"] * 100, "o-", color=WARM[2])
ax.set(title="Vol ATM por prazo", xlabel="T (anos)", ylabel="vol implícita (%)")
plt.show()

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
}, name="resumo")
display(summary.to_frame())
