# Databricks notebook source
# MAGIC %md
# MAGIC # Renda Fixa Pública: precificação e curvas (02)
# MAGIC
# MAGIC 1. Reprecifica todos os títulos do dia (LTN, NTN-F, NTN-B, LFT) e compara com o PU indicativo da ANBIMA.
# MAGIC 2. Constrói a curva prefixada (bootstrap de LTN + NTN-F) e a de juro real (NTN-B), com ajuste Nelson-Siegel-Svensson.
# MAGIC 3. Calcula a inflação implícita e as taxas a termo.
# MAGIC 4. Grava os resultados em CSV na pasta de dados.

# COMMAND ----------

# MAGIC %pip install requests pandas numpy scipy plotly pyarrow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import importlib
import os
import sys

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
if not notebook_path.startswith("/Workspace"):
    notebook_path = "/Workspace" + notebook_path
project_dir = os.path.dirname(os.path.dirname(notebook_path))
SRC = os.path.join(project_dir, "src")
if not os.path.isfile(os.path.join(SRC, "renda_fixa", "mercado.py")):
    raise RuntimeError(f"{SRC} não tem o pacote renda_fixa: faça Pull da branch no Git folder.")
for name in [m for m in sys.modules if m == "renda_fixa" or m.startswith("renda_fixa.")]:
    del sys.modules[name]
while SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
importlib.invalidate_caches()

# COMMAND ----------

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from renda_fixa import graficos as G
from renda_fixa.dados.anbima import baixar_mercado_secundario, ler_mercado_secundario, ultimo_disponivel
from renda_fixa.mercado import montar_mercado, tabela_curvas, tabela_titulos

user_home = "/".join(notebook_path.split("/")[:4])
dbutils.widgets.text("data_ref", "", "Data (vazio = mais recente)")
dbutils.widgets.text("pasta_dados", f"{user_home}/rf_dados", "Pasta de dados")
PASTA = Path(dbutils.widgets.get("pasta_dados").strip())
DATA = dbutils.widgets.get("data_ref").strip()

if DATA:
    df = baixar_mercado_secundario(date.fromisoformat(DATA), PASTA / "anbima")
else:
    df = ultimo_disponivel(date.today(), PASTA / "anbima")
m = montar_mercado(df)
REF = m.ref
print(f"Data de referência: {REF}")
print(f"VNA LFT {m.vna_lft}\nVNA NTN-B {m.vna_ntnb}")


def mostrar(fig, nome):
    displayHTML(G.html_figura(fig, f"rf_{nome}_{REF}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validação título a título contra a ANBIMA

# COMMAND ----------

tt = tabela_titulos(m)
n_ok = int((tt["erro_pu"] == 0).sum())
print(f"{n_ok} de {len(tt)} títulos com PU idêntico ao da ANBIMA (erro máximo {tt['erro_pu'].abs().max():.6f})")
display(tt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Curvas: pré, real e inflação implícita

# COMMAND ----------

tc = tabela_curvas(m, vertices=tuple(range(21, 9000, 21)))
anos = tc["anos"]

pts_pre = tt[tt["titulo"].isin(["LTN", "NTN-F"])]
pts_real = tt[tt["titulo"] == "NTN-B"]

fig = G.figura(
    [
        G.linha(1, "Pré (bootstrap flat-forward)", anos, tc["pre"], "lines"),
        G.linha(3, "Pré (NSS)", anos, tc["pre_nss"], "lines"),
        G.linha(0, "LTN / NTN-F (taxa ANBIMA x duration)", pts_pre["duration_mac"], pts_pre["taxa_anbima"], "markers"),
    ],
    f"Curva prefixada {REF}",
    "anos (DU/252)",
    "% a.a.",
)
mostrar(fig, "curva_pre")

fig = G.figura(
    [
        G.linha(1, "Real (bootstrap)", anos, tc["real"], "lines"),
        G.linha(3, "Real (NSS)", anos, tc["real_nss"], "lines"),
        G.linha(0, "NTN-B (taxa ANBIMA x duration)", pts_real["duration_mac"], pts_real["taxa_anbima"], "markers"),
    ],
    f"Curva de juro real (IPCA) {REF}",
    "anos (DU/252)",
    "% a.a.",
)
mostrar(fig, "curva_real")

fig = G.figura(
    [
        G.linha(2, "Inflação implícita", anos, tc["inflacao_implicita"], "lines"),
        G.linha(4, "Termo pré de 1 ano", anos, tc["termo_pre_1a"], "lines"),
    ],
    f"Inflação implícita e taxa a termo {REF}",
    "anos (DU/252)",
    "% a.a.",
)
mostrar(fig, "breakeven")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Resíduos do ajuste NSS (rico / barato)

# COMMAND ----------

res = tt.dropna(subset=["residuo_nss_bps"]).copy()
res["rotulo"] = res["titulo"] + " " + res["vencimento"].astype(str)
fig = G.figura([G.barras("Resíduo NSS (bps)", res["rotulo"], res["residuo_nss_bps"])],
               "Resíduo do NSS por título (+ = curva acima do mercado: título barato)", "", "bps")
mostrar(fig, "residuos_nss")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Gravação dos resultados

# COMMAND ----------

saida = PASTA / "resultados"
saida.mkdir(parents=True, exist_ok=True)
tt.to_csv(saida / f"RESULTS_rf_titulos_{REF}.csv", index=False)
tabela_curvas(m).to_csv(saida / f"RESULTS_rf_curvas_{REF}.csv", index=False)
print("Gravado em", saida)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Relatório

# COMMAND ----------

tcv = tabela_curvas(m)
print("RELATÓRIO RENDA FIXA", REF)
print(f"Títulos: {len(tt)} | PU idêntico à ANBIMA: {n_ok} | erro máximo R$ {tt['erro_pu'].abs().max():.6f}")
print(f"VNA LFT {m.vna_lft.vna:.6f} ({'consistente' if m.vna_lft.consistente else 'INCONSISTENTE'}) | "
      f"VNA NTN-B {m.vna_ntnb.vna:.6f} ({'consistente' if m.vna_ntnb.consistente else 'INCONSISTENTE'})")
print(f"NSS: resíduo máximo {res['residuo_nss_bps'].abs().max():.2f} bps; {m.nss_pre}; {m.nss_real}")
for _, r in tcv[tcv["du"].isin([252, 504, 1260, 2520])].iterrows():
    print(f"  {r['anos']:4.0f}a: pré {r['pre']:.4f}% | real {r['real']:.4f}% | implícita {r['inflacao_implicita']:.4f}%")
