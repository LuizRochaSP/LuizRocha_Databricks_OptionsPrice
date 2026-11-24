# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 11 — Validação do RESULTS do S&P 500 antes do risco
# MAGIC
# MAGIC O equivalente do notebook 02 para o SPX. Lê o `RESULTS_spx_superficie_*.csv` mais recente do notebook 07,
# MAGIC roda os controles de `ibov_barrier.markets.validacao_spx` e decide com a regra do 02:
# MAGIC
# MAGIC - **APROVADO:** nenhum `FAIL` e nenhum `WARN`;
# MAGIC - **APROVADO COM RESSALVAS:** nenhum `FAIL`, mas há `WARN`;
# MAGIC - **REPROVADO:** existe ao menos um `FAIL` (o notebook para e não grava a evidência).
# MAGIC
# MAGIC Sem `FAIL`, grava `VALIDATION_spx_superficie_<mesmo carimbo>.json`, que o notebook 03 (`ativo=spx`)
# MAGIC exige, com o SHA-256 do RESULTS (o 03 recusa um RESULTS alterado depois da validação).
# MAGIC
# MAGIC Blocos: integridade, superfície (pontos, vencimentos, faixa da vol, cobertura, cortes de arbitragem),
# MAGIC forwards e carry (avisos da paridade, q(T), r - SOFR, spot fechamento x paridade) e precificação
# MAGIC (contrato, r/q/forward/vol reproduzidos, limites do preço, MC x Reiner-Rubinstein e replay do Monte Carlo).

# COMMAND ----------

# MAGIC %pip install "exchange-calendars==4.13.2" "lxml>=6.0" scipy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import hashlib
import importlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
if not notebook_path.startswith("/Workspace"):
    notebook_path = "/Workspace" + notebook_path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
SRC = os.path.join(project_dir, "src")
# Remove qualquer ibov_barrier carregado de outro Git folder antes de apontar para o deste.
for name in [m for m in sys.modules if m == "ibov_barrier" or m.startswith("ibov_barrier.")]:
    del sys.modules[name]
while SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
importlib.invalidate_caches()

from ibov_barrier.market_date import expected_market_date
from ibov_barrier.markets.validacao_spx import decisao, evidencia, ordenar, validar_spx
from ibov_barrier.results_bundle import read_bundle

dbutils.widgets.text("arquivo", "", "RESULTS_spx_superficie_*.csv (vazio = o mais recente)")
results_root = Path("/".join(notebook_path.split("/")[:4])) / "ibov-barrier-results"
padrao = re.compile(r"^RESULTS_spx_superficie_(\d{8})_(\d{6})\.csv$")
escolhido = dbutils.widgets.get("arquivo").strip()
if escolhido:
    results_path = results_root / escolhido
else:
    candidatos = sorted(p for p in results_root.glob("RESULTS_spx_superficie_*.csv") if padrao.match(p.name))
    if not candidatos:
        raise RuntimeError("Nenhum RESULTS do SPX. Rode antes o notebook 07.")
    results_path = candidatos[-1]
if not results_path.is_file():
    raise RuntimeError(f"RESULTS não encontrado: {results_path}")
print("RESULTS:", results_path.name, "| código de:", SRC)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Controles

# COMMAND ----------

bundle = read_bundle(results_path)
d1 = expected_market_date(market="NYSE")
checks = ordenar(validar_spx(bundle, d1_esperado=d1, replay=True))
decisao_final, n_pass, n_warn, n_fail = decisao(checks)

with pd.option_context("display.max_colwidth", 200):
    display(checks)
print(f"Data de mercado: {bundle['metadata'].get('market_date')} | D-1 NYSE esperado: {d1}")
print(f"Decisão: {decisao_final} | PASS {n_pass} | WARN {n_warn} | FAIL {n_fail}")
if n_fail:
    falhas = checks.loc[checks["status"].eq("FAIL"), "controle"].tolist()
    raise AssertionError(f"Controles reprovados (a evidência não foi gravada): {falhas}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evidência para o notebook 03

# COMMAND ----------

agora = datetime.now(ZoneInfo("America/Sao_Paulo"))
payload = evidencia(checks, results_file=results_path.name,
                    results_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
                    metadata=bundle["metadata"], validation_timestamp=agora.isoformat())
validation_path = results_root / results_path.name.replace("RESULTS_", "VALIDATION_").replace(".csv", ".json")
validation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("Evidência gravada:", validation_path.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Relatório para enviar

# COMMAND ----------

print(f"RELATÓRIO VALIDAÇÃO SPX | {agora:%Y-%m-%d %H:%M} (Brasília)")
print(f"RESULTS: {results_path.name} | mercado {bundle['metadata'].get('market_date')} | decisão {decisao_final} "
      f"(PASS {n_pass}, WARN {n_warn}, FAIL {n_fail})")
with pd.option_context("display.max_rows", 200, "display.max_colwidth", 120, "display.width", 300):
    print(checks[["status", "bloco", "controle", "observado", "critério"]].to_string(index=False))
print("FIM DO RELATÓRIO")
