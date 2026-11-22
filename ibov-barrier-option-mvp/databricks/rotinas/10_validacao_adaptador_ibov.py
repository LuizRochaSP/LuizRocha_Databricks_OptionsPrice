# Databricks notebook source
# MAGIC %md
# MAGIC # Validação do adaptador IBOV contra o notebook 01 do `main` (Fase 4)
# MAGIC
# MAGIC Branch `feature/sp500` apenas. Somente leitura: não grava nada e não toca o Git folder de produção.
# MAGIC
# MAGIC Critério de aceite: para a mesma data e os mesmos ZIPs da B3, o `IBOVAdapter` reproduz o resultado que o
# MAGIC notebook 01 do `main` salvou em `ibov-barrier-results/RESULTS_superficieVol_*.csv`:
# MAGIC
# MAGIC | Item | Tolerância |
# MAGIC | :--- | :--- |
# MAGIC | quantidade no cadastro, no PR, no cruzamento, após filtros e com vol implícita | igual |
# MAGIC | spot | 1e-12 |
# MAGIC | r, q, forward e vol implícita por ticker | 1e-9 (o notebook 01 salva as tabelas com `to_json`, 10 casas) |
# MAGIC | r, q e vol do alvo, método de interpolação | 1e-12 / igual |
# MAGIC | vanilla, barreira MC (mesma semente) e analítica | 1e-8 relativo |
# MAGIC
# MAGIC Rode antes o notebook 01 no Git folder de produção para a data desejada (ou use o último resultado salvo).

# COMMAND ----------

# MAGIC %pip install "exchange-calendars==4.13.2" "lxml>=6.0" scipy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import importlib

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
if not notebook_path.startswith("/Workspace"):
    notebook_path = "/Workspace" + notebook_path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
SRC = os.path.join(project_dir, "src")
if not os.path.isfile(os.path.join(SRC, "ibov_barrier", "markets", "ibov.py")):
    raise RuntimeError(f"{SRC} não tem markets/ibov.py: o Git folder precisa estar na branch feature/sp500.")
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
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from ibov_barrier import BarrierContract, MarketData, price_down_and_out_call
from ibov_barrier.analytical import price_down_and_out_call_analytical
from ibov_barrier.markets.ibov import IBOVAdapter, IbovFilters, target_inputs

user_home = "/".join(notebook_path.split("/")[:4])
dbutils.widgets.text("pasta_b3", f"{user_home}/LuizRocha_Databricks_OptionsPrice/ibov-barrier-option-mvp/data",
                     "Pasta data do Git folder de produção")
dbutils.widgets.text("pasta_resultados", f"{user_home}/ibov-barrier-results", "Resultados do notebook 01")
dbutils.widgets.text("arquivo_resultado", "", "RESULTS_*.csv (vazio = mais recente)")
W = {k: dbutils.widgets.get(k).strip() for k in ("pasta_b3", "pasta_resultados", "arquivo_resultado")}
print(W)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Resultado salvo pelo notebook 01 do `main`

# COMMAND ----------

results_dir = Path(W["pasta_resultados"])
if W["arquivo_resultado"]:
    result_path = results_dir / W["arquivo_resultado"]
else:
    candidates = sorted(results_dir.glob("RESULTS_superficieVol_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Nenhum RESULTS_superficieVol_*.csv em {results_dir}: rode o notebook 01.")
    result_path = candidates[-1]
bundle = pd.read_csv(result_path).set_index("object_name")


def payload(name):
    row = bundle.loc[name]
    if row["object_type"] == "json":
        return json.loads(row["payload_json"])
    return pd.read_json(StringIO(row["payload_json"]), orient="table")


meta, pricing, diag = payload("metadata"), payload("pricing_result"), payload("simulation_diagnostics")
main = {name: payload(name) for name in ("catalog", "prices", "options", "filtered", "iv_data")}
market_date = date.fromisoformat(meta["market_date"])
print(f"Resultado: {result_path.name} | data de mercado {market_date} | executado em {meta['execution_timestamp']}")
print("Arquivos usados pelo main:", meta["in_zip"], meta["ir_zip"], meta["pr_zip"], meta["sprd_zip"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Adaptador sobre os mesmos ZIPs

# COMMAND ----------

filters = IbovFilters(min_trades=meta["min_trades"], min_open_interest=meta["min_open_interest"],
                      min_iv=meta["min_iv"], max_iv=meta["max_iv"])
adapter = IBOVAdapter(Path(W["pasta_b3"]), filters)
snap = adapter.load_snapshot(market_date)
last = adapter.last
display(snap.report.to_frame())
print(f"spot {snap.spot:,.2f} | {len(snap.forwards)} vencimentos | {len(snap.options)} opções com vol")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Comparação item a item

# COMMAND ----------

checks = []


def check(item, ours, theirs, tol=0.0, relative=False):
    if isinstance(ours, str) or isinstance(theirs, str):
        diff, ok = (0.0 if ours == theirs else np.nan), ours == theirs
    else:
        diff = abs(float(ours) - float(theirs))
        if relative:
            diff = diff / max(abs(float(theirs)), 1e-300)
        ok = diff <= tol
    checks.append({"item": item, "adaptador": ours, "main": theirs, "diferença": diff, "tolerância": tol,
                   "resultado": "APROVADO" if ok else "REPROVADO"})


check("arquivo IN", last["files"]["IN"].name, meta["in_zip"])
check("arquivo PR", last["files"]["PR"].name, meta["pr_zip"])
check("spot", snap.spot, meta["spot"], 1e-12)
for name, frame in (("catalog", last["catalog"]), ("prices", last["prices"]), ("options", last["options"]),
                    ("filtered", last["filtered"]), ("iv_data", snap.options)):
    check(f"linhas {name}", len(frame), len(main[name]))

cols = ["T", "r", "q", "forward", "implied_vol"]
ours = snap.options.set_index("ticker")[cols].sort_index()
theirs = main["iv_data"].set_index("ticker")[cols].sort_index()
check("tickers com vol (mesmo conjunto)", str(sorted(ours.index) == sorted(theirs.index)), "True")
joined = ours.join(theirs, rsuffix="_main", how="inner")
for c in cols:
    tol = 1e-9 * max(1.0, float(joined[c].abs().max())) if c == "forward" else 1e-9
    check(f"máx |dif| {c} por ticker", float((joined[c] - joined[f"{c}_main"]).abs().max()), 0.0, tol)

target = target_inputs(last["curves"], snap.spot, snap.options, meta["target_strike"], meta["target_maturity"])
check("r alvo", target["r"], meta["target_r"], 1e-12)
check("q alvo", target["q"], meta["target_q"], 1e-12)
check("vol alvo", target["volatility"], meta["target_iv"], 1e-12)
check("interpolação", target["interpolation"], meta["interpolation_method"])

contract = BarrierContract(strike=pricing["contract"]["strike"], barrier=pricing["contract"]["barrier"],
                           maturity=pricing["contract"]["maturity"])
market = MarketData(spot=float(snap.spot), rate=float(target["r"]), dividend_yield=float(target["q"]),
                    volatility=float(target["volatility"]))
result = price_down_and_out_call(market, contract, paths=diag["paths"], steps=diag["steps"], seed=diag["seed"],
                                 monitoring=diag["monitoring"])
analytical = price_down_and_out_call_analytical(spot=market.spot, strike=contract.strike, barrier=contract.barrier,
                                                maturity=contract.maturity, rate=market.rate,
                                                dividend_yield=market.dividend_yield,
                                                volatility=market.volatility)
check("vanilla", result.vanilla_price, pricing["vanilla_price"], 1e-8, relative=True)
check("barreira MC", result.price, pricing["barrier_price"], 1e-8, relative=True)
check("barreira analítica", analytical, pricing["analytical_price"], 1e-8, relative=True)
check("prob. knock-out", result.knock_out_probability, pricing["knock_out_probability"], 1e-12)

checks = pd.DataFrame(checks)
display(checks)
verdict = "APROVADO" if (checks["resultado"] == "APROVADO").all() else "REPROVADO"
print("VEREDITO:", verdict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Relatório único para enviar
# MAGIC
# MAGIC Copie de `RELATÓRIO VALIDAÇÃO IBOV` até `FIM DO RELATÓRIO`.

# COMMAND ----------

with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 250,
                       "display.max_colwidth", 40):
    print(f"RELATÓRIO VALIDAÇÃO IBOV | {market_date} | resultado main: {result_path.name} | "
          f"gerado em {pd.Timestamp.now(tz='America/Sao_Paulo'):%Y-%m-%d %H:%M} (Brasília)")
    print("VEREDITO:", verdict)
    print(checks.to_string(index=False))
    print("\nfiltros do adaptador:")
    print(snap.report.to_frame().to_string(index=False))
    mism = joined[(joined["implied_vol"] - joined["implied_vol_main"]).abs() > 1e-9]
    if len(mism):
        print(f"\n{len(mism)} tickers com vol diferente (até 10):")
        print(mism.head(10).to_string())
    print("\nFIM DO RELATÓRIO")
