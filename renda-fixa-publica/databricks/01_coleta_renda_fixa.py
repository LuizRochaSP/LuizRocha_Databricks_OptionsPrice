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
(PASTA / "b3").mkdir(parents=True, exist_ok=True)
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
# MAGIC ## 1b. B3: ajustes de DI1 e DAP, dia a dia (com cache)
# MAGIC
# MAGIC Arquivo público consolidado do pregão (arquivos.b3.com.br), com histórico de anos. O cache guarda só as
# MAGIC linhas de DI1 e DAP. Conferência: o PU de ajuste da B3 sai da taxa de ajuste pelo motor, ao centavo.

# COMMAND ----------

from renda_fixa.dados.b3 import baixar_ajustes_b3
from renda_fixa.derivativos import DAP, DI1

ajustes, faltantes_b3 = [], []
for df_dia in dias:
    ref = df_dia["data_referencia"].iloc[0]
    try:
        ajustes.append(baixar_ajustes_b3(ref, PASTA / "b3"))
    except DadoIndisponivel as e:
        faltantes_b3.append((ref, str(e)))
if ajustes:
    b3 = pd.concat(ajustes, ignore_index=True)
    b3.to_parquet(PASTA / "ajustes_b3.parquet", index=False)
    b3["pu_motor"] = [
        (DI1(r.vencimento) if r.ativo == "DI1" else DAP(r.vencimento)).pu(r.taxa_ajuste, r.data_referencia)
        for r in b3.itertuples()
    ]
    b3["pu_confere"] = (b3["pu_motor"] - b3["pu_ajuste"]).abs() < 0.005
    print(f"B3: {len(ajustes)} pregões, {len(b3)} ajustes; PU do motor = PU de ajuste em "
          f"{int(b3['pu_confere'].sum())} de {len(b3)}; {len(faltantes_b3)} dias sem arquivo")
    display(b3[b3["data_referencia"] == b3["data_referencia"].max()])
else:
    b3 = pd.DataFrame()
    print("B3: nenhum arquivo de pregão no período", faltantes_b3[:5])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. VNA diário implícito (LFT, NTN-B e NTN-C)

# COMMAND ----------

linhas = []
for df in dias:
    ref = df["data_referencia"].iloc[0]
    linha = {"data": ref}
    for t in ("LFT", "NTN-B", "NTN-C"):
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
# MAGIC ## 3. Banco Central: Selic, CDI, IPCA e IGP-M

# COMMAND ----------

series = {}
for nome, codigo, ini in [
    ("selic_diaria", bcb.SELIC_DIARIA, INICIO - timedelta(days=10)),
    ("cdi_diario", bcb.CDI_DIARIO, INICIO - timedelta(days=10)),
    ("ipca_mensal", bcb.IPCA_MENSAL, date(2000, 7, 1)),
    ("igpm_mensal", bcb.IGPM_MENSAL, date(2000, 7, 1)),
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
# MAGIC ## 5b. Conferência: VNA da NTN-C contra o IGP-M acumulado
# MAGIC
# MAGIC O VNA do dia 1º sai do IGP-M acumulado desde jul/2000 (SGS 189, 2 casas: diferença esperada de ~0,05%).
# MAGIC Entre dois dias 1º, a ANBIMA aplica a projeção de IGP-M pro rata em dias corridos; a projeção implícita
# MAGIC no VNA do arquivo tem de ser uma variação mensal plausível (ela muda quando saem as prévias do IGP-M).

# COMMAND ----------

from renda_fixa.vna import igpm_projetado_implicito, vna_ntnc_por_igpm

conf_ntnc = []
if "vna_NTN-C" in vna_diario:
    for _, r in vna_diario.dropna(subset=["vna_NTN-C"]).iterrows():
        dia, vna = r["data"], float(r["vna_NTN-C"])
        try:
            v1 = vna_ntnc_por_igpm(series["igpm_mensal"], date(dia.year, dia.month, 1))
        except ValueError as e:  # IGP-M do mês anterior ainda não publicado
            print(dia, e)
            continue
        proj = igpm_projetado_implicito(vna, v1, dia) if dia.day > 1 else float("nan")
        conf_ntnc.append({"data": dia, "vna_anbima": vna, "vna_dia1_igpm": v1, "igpm_projetado_pct": proj})
conf_ntnc = pd.DataFrame(conf_ntnc)
if not conf_ntnc.empty:
    conf_ntnc["plausivel"] = conf_ntnc["igpm_projetado_pct"].abs().fillna(0) < 3.0
    print(f"NTN-C: projeção de IGP-M implícita plausível (|x| < 3% a.m.) em {int(conf_ntnc['plausivel'].sum())} "
          f"de {len(conf_ntnc)} dias")
    display(conf_ntnc)

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
if not conf_ntnc.empty:
    u = conf_ntnc.iloc[-1]
    print(f"VNA NTN-C {u['data']}: {u['vna_anbima']:.6f} | projeção de IGP-M implícita {u['igpm_projetado_pct']:.3f}% a.m. "
          f"| plausível em {int(conf_ntnc['plausivel'].sum())}/{len(conf_ntnc)} dias")
if not b3.empty:
    print(f"B3: {b3['data_referencia'].nunique()} pregões | PU de ajuste conferido em {int(b3['pu_confere'].sum())}/{len(b3)}"
          f" | dias sem arquivo: {len(faltantes_b3)}")
print(f"Tesouro Direto: {td['data_referencia'].nunique()} dias, de {td['data_referencia'].min()} a {td['data_referencia'].max()}")
