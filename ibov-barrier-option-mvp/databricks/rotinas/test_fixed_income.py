# Databricks notebook source
# MAGIC %pip install exchange-calendars
# MAGIC %restart_python

# MAGIC %md
# MAGIC # Teste do Módulo de Renda Fixa Nacional

# COMMAND ----------

import sys
from pathlib import Path
from datetime import date

# Adiciona a pasta src ao PATH
project_root = Path("/Workspace/Users/luiz.henrique.felipe.rocha@gmail.com/LuizRocha_Databricks_OptionsPrice/ibov-barrier-option-mvp")
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from ibov_barrier.fixed_income.ntnb import ntnb_price, ntnb_duration

# COMMAND ----------

# Parâmetros de teste
vna = 4000.00          # Valor Nominal Atualizado (exemplo: NTN-B Principal)
taxa_anbima = 6.50     # Taxa IPCA+ (% a.a.)
cupom = 6.00           # Cupom anual (% a.a.)
data_ref = date(2026, 10, 15)
data_venc = date(2035, 5, 15)

print(f"=== Teste de Precificação NTN-B ===")
print(f"VNA: R$ {vna:,.2f}")
print(f"Taxa ANBIMA: {taxa_anbima}% a.a.")
print(f"Cupom: {cupom}% a.a.")
print(f"Data Ref: {data_ref.strftime('%d/%m/%Y')}")
print(f"Vencimento: {data_venc.strftime('%d/%m/%Y')}")

# COMMAND ----------

preco = ntnb_price(vna, taxa_anbima, cupom, data_ref, data_venc)
duration = ntnb_duration(vna, taxa_anbima, cupom, data_ref, data_venc)

print(f"\nPreço Unitário (PU): R$ {preco:,.2f}")
print(f"Duration de Macaulay: {duration:.4f} anos")
