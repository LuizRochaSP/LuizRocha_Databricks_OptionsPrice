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
adapter_am = SPXAdapter(fetch_chain_cached, spot_source=source.index_close, cleaning=cfg_am)
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), squeeze=False)
    for ax, e in zip(axes.flat, picked):
        for (name, (_, s)), color, ls, mk in zip(VARIANTS.items(), (WARM[1], WARM[4]), ("-", "--"), ("o", "s")):
            sm = s.options.loc[s.options["expiration"].eq(e)].sort_values("log_moneyness")
            label = f"{name} ({sm['root'].iloc[0] if len(sm) else '-'}): {len(sm)} pts"
            ax.plot(sm["log_moneyness"], sm["implied_vol"] * 100, color=color, ls=ls, lw=1.6,
                    marker=mk, ms=4, markevery=max(1, len(sm) // 12), label=label)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set(title=f"{e} (T={both.loc[e, 'T PM']:.2f})", xlabel="ln(K/F)", ylabel="vol implícita (%)")
        ax.legend()
    for ax in list(axes.flat)[len(picked):]:
        ax.set_visible(False)
    fig.suptitle("Smiles nos vencimentos mensais: PM (SPXW) x AM (SPX)", color="white", fontweight="bold")
    plt.tight_layout()
    plt.show()
else:
    print("Nenhum vencimento mensal presente nas duas versões.")
