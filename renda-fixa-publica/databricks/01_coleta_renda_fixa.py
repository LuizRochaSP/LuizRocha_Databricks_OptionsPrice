# Databricks notebook source
# MAGIC %md
# MAGIC # Renda Fixa Pública: coleta de dados (01)
# MAGIC
# MAGIC Projeto separado do pricer de opções: usa só o pacote `renda_fixa` desta pasta (`renda-fixa-publica/src`).
# MAGIC
# MAGIC | Dado | Fonte | Cache |
# MAGIC | :--- | :--- | :--- |
# MAGIC | Taxas e PUs indicativos dos títulos públicos | ANBIMA, arquivo diário `msAAMMDD.txt` | `<pasta>/anbima/` |
# MAGIC | Selic diária (VNA da LFT), CDI diário (swaps), IPCA mensal | Banco Central, SGS 11, 12 e 433 | `<pasta>/bcb/` |
# MAGIC
# MAGIC | Histórico de taxas (2004 em diante) | Tesouro Direto (Tesouro Transparente) | `<pasta>/tesouro_direto/` |
# MAGIC
# MAGIC A ANBIMA mantém públicos só os arquivos dos últimos dias úteis; rodando este notebook todo dia (job agendado)
# MAGIC o cache acumula o histórico oficial. Dia sem arquivo aparece como faltante no relatório. Nenhum dado é inventado.

# COMMAND ----------

# MAGIC %pip install requests pandas numpy scipy plotly pyarrow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Ambiente

# COMMAND ----------

import importlib
import os
import sys

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
if not notebook_path.startswith("/Workspace"):
    notebook_path = "/Workspace" + notebook_path
project_dir = os.path.dirname(os.path.dirname(notebook_path))  # .../renda-fixa-publica
SRC = os.path.join(project_dir, "src")
if not os.path.isfile(os.path.join(SRC, "renda_fixa", "mercado.py")):
    raise RuntimeError(f"{SRC} não tem o pacote renda_fixa: faça Pull da branch no Git folder.")
for name in [m for m in sys.modules if m == "renda_fixa" or m.startswith("renda_fixa.")]:
    del sys.modules[name]
while SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
importlib.invalidate_caches()

import renda_fixa

print("Código de:", renda_fixa.__file__, "versão", renda_fixa.__version__)

# COMMAND ----------

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from renda_fixa.calendario import eh_dia_util
from renda_fixa.dados import bcb
from renda_fixa.dados.anbima import DadoIndisponivel, baixar_mercado_secundario, vna_implicito

user_home = "/".join(notebook_path.split("/")[:4])  # /Workspace/Users/<usuário>
hoje = date.today()
dbutils.widgets.text("data_inicio", (hoje - timedelta(days=14)).isoformat(), "Início ANBIMA (AAAA-MM-DD)")
dbutils.widgets.text("data_fim", hoje.isoformat(), "Fim (AAAA-MM-DD)")
dbutils.widgets.text("pasta_dados", f"{user_home}/rf_dados", "Pasta de dados")

INICIO = date.fromisoformat(dbutils.widgets.get("data_inicio").strip())
FIM = date.fromisoformat(dbutils.widgets.get("data_fim").strip())
PASTA = Path(dbutils.widgets.get("pasta_dados").strip())
(PASTA / "anbima").mkdir(parents=True, exist_ok=True)
(PASTA / "bcb").mkdir(parents=True, exist_ok=True)
print(f"Período {INICIO} a {FIM}; pasta {PASTA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ANBIMA: títulos públicos, dia a dia (com cache)

# COMMAND ----------

dias, faltantes = [], []
d = INICIO
while d <= FIM:
    if eh_dia_util(d):
        try:
            dias.append(baixar_mercado_secundario(d, PASTA / "anbima"))
        except DadoIndisponivel as e:
            faltantes.append((d, str(e)))
    d += timedelta(days=1)

if not dias:
    raise RuntimeError("Nenhum arquivo ANBIMA no período.")
titulos = pd.concat(dias, ignore_index=True)
titulos.to_parquet(PASTA / "titulos_anbima.parquet", index=False)
print(f"{len(dias)} dias baixados, {len(faltantes)} dias úteis sem arquivo, {len(titulos)} linhas")
display(titulos[titulos["data_referencia"] == titulos["data_referencia"].max()])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. VNA diário implícito (LFT e NTN-B)

# COMMAND ----------

linhas = []
for df in dias:
    ref = df["data_referencia"].iloc[0]
    linha = {"data": ref}
    for t in ("LFT", "NTN-B"):
        try:
            v = vna_implicito(df, t)
            linha[f"vna_{t}"] = v.vna
            linha[f"consistente_{t}"] = v.consistente
        except ValueError:
            pass
    linhas.append(linha)
vna_diario = pd.DataFrame(linhas).sort_values("data")
vna_diario.to_parquet(PASTA / "vna_diario.parquet", index=False)
display(vna_diario.tail(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Banco Central: Selic, CDI e IPCA

# COMMAND ----------

series = {}
for nome, codigo, ini in [
    ("selic_diaria", bcb.SELIC_DIARIA, INICIO - timedelta(days=10)),
    ("cdi_diario", bcb.CDI_DIARIO, INICIO - timedelta(days=10)),
    ("ipca_mensal", bcb.IPCA_MENSAL, date(2000, 7, 1)),
]:
    s = bcb.serie_sgs(codigo, ini, FIM, cache_dir=PASTA / "bcb")
    series[nome] = s
    print(f"{nome:13s} SGS {codigo:4d}: {len(s):5d} pontos, último {s.index.max()} = {s.iloc[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Histórico do Tesouro Direto (curvas para VaR e PCA)
# MAGIC
# MAGIC O arquivo da ANBIMA só fica público por poucos dias úteis. O histórico longo vem do Tesouro Direto
# MAGIC (taxas de 2 casas, da manhã, média compra/venda): serve para variações de curva, não para validar PU.

# COMMAND ----------

from renda_fixa.dados.tesouro_direto import baixar_tesouro_direto

td = baixar_tesouro_direto(PASTA / "tesouro_direto")
print(f"Tesouro Direto: {len(td)} linhas, {td['data_referencia'].nunique()} dias, até {td['data_referencia'].max()}")
display(td[td["data_referencia"] == td["data_referencia"].max()])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Conferência: VNA da LFT avançado pela Selic

# COMMAND ----------

from renda_fixa.vna import atualizar_vna_lft

v = vna_diario.dropna(subset=["vna_LFT"]).set_index("data")["vna_LFT"]
selic = series["selic_diaria"]
conf = []
datas = list(v.index)
for a, b in zip(datas[:-1], datas[1:]):
    if a in selic.index and (b - a).days <= 4:
        conf.append({"de": a, "para": b, "vna_anbima": v[b], "vna_motor": atualizar_vna_lft(v[a], selic[a])})
conf = pd.DataFrame(conf)
if not conf.empty:
    conf["diferenca"] = (conf["vna_motor"] - conf["vna_anbima"]).round(6)
    # O Tesouro acumula o fator desde 2000 e trunca só no fim: até R$ 0,000001 é truncamento.
    conf["ok"] = conf["diferenca"].abs() <= 1.0000001e-6
    print(f"VNA LFT: {int(conf['ok'].sum())} de {len(conf)} transições dentro do truncamento (|dif| <= 0,000001)")
    display(conf[~conf["ok"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Relatório

# COMMAND ----------

print("RELATÓRIO COLETA RENDA FIXA")
print(f"Período: {INICIO} a {FIM}")
print(f"Arquivos ANBIMA: {len(dias)} | dias úteis sem arquivo: {len(faltantes)}")
for d, _ in faltantes[:15]:
    print("  faltante:", d)
print(f"Títulos no último dia: {titulos[titulos['data_referencia'] == titulos['data_referencia'].max()]['titulo'].value_counts().to_dict()}")
print(f"VNA inconsistente (LFT/NTN-B): {int((~vna_diario.get('consistente_LFT', pd.Series(dtype=bool)).fillna(True)).sum())}/"
      f"{int((~vna_diario.get('consistente_NTN-B', pd.Series(dtype=bool)).fillna(True)).sum())}")
if not conf.empty:
    print(f"VNA LFT vs Selic: {int(conf['ok'].sum())}/{len(conf)} dentro do truncamento")
print(f"Tesouro Direto: {td['data_referencia'].nunique()} dias, de {td['data_referencia'].min()} a {td['data_referencia'].max()}")
