# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Coleta de Dados ANBIMA e Tesouro Direto

# COMMAND ----------

import sys
import requests
import pandas as pd
from datetime import date
from pathlib import Path

# Adiciona a pasta src ao PATH
project_root = Path("/Workspace/Users/luiz.henrique.felipe.rocha@gmail.com/LuizRocha_Databricks_OptionsPrice/ibov-barrier-option-mvp")
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

# COMMAND ----------

# DBTITLE 1,Instala pyettj
# MAGIC %pip install pyettj exchange_calendars lxml

# COMMAND ----------

import pyettj
import pandas as pd
from datetime import date

def baixar_ettj_ipca_anbima(data_ref: date) -> pd.DataFrame:
    """Baixa a ETTJ IPCA real da ANBIMA usando a biblioteca pyettj."""
    try:
        print(f"Buscando dados REAIS da ANBIMA para {data_ref.strftime('%d/%m/%Y')}...")
        # pyettj espera uma string no formato YYYY-MM-DD
        data_str = data_ref.strftime("%Y-%m-%d")
        
        # Chamada real da biblioteca
        df = pyettj.get_ettj(data_str, curva="DIC")
        
        if df is not None and not df.empty:
            # Renomeia as colunas para o padrão do nosso sistema
            df = df.rename(columns={"dias_uteis": "vertice_du", "taxa": "taxa_ipca"})
            df["taxa_ipca"] = df["taxa_ipca"] * 100
            print("✅ ETTJ IPCA REAL (DIC) carregada com sucesso!")
            return df
        else:
            raise ValueError("ANBIMA retornou dados vazios (possível feriado/fim de semana).")
            
    except Exception as e:
        print(f"⚠️ Falha ao buscar dados reais da ANBIMA: {e}")
        print("Usando fallback (mock) para não travar o pipeline...")
        
        dados_mock = {
            "vertice_du": [252, 504, 1260, 2520],
            "taxa_ipca": [6.10, 6.20, 6.35, 6.50]
        }
        return pd.DataFrame(dados_mock)

# COMMAND ----------

# ANBIMA publica ETTJ após o fechamento; se hoje não há dados, usa o último dia útil
for _d in [date(2026, 10, 28), date(2026, 10, 27), date(2026, 10, 26), date(2026, 10, 23)]:
    data_hoje = _d
    df_ettj = baixar_ettj_ipca_anbima(data_hoje)
    if df_ettj is not None and not df_ettj.empty:
        # Verifica se não é mock (mock tem sempre 4 linhas com vértices fixos)
        if len(df_ettj) > 4 or df_ettj["vertice_du"].iloc[0] != 252:
            break

if df_ettj is not None:
    display(df_ettj.head(10))
else:
    print("Falha ao carregar ETTJ.")

# COMMAND ----------

from ibov_barrier.fixed_income.ntnb import ntnb_price, ntnb_duration

vna = 4000.00
cupom = 6.00
data_venc = date(2035, 5, 15)

# Pega a taxa mais próxima de 10 anos (2520 DU)
idx_10_anos = (df_ettj["vertice_du"] - 2520).abs().idxmin()
taxa_10_anos = df_ettj.loc[idx_10_anos, "taxa_ipca"]

preco = ntnb_price(vna, taxa_10_anos, cupom, data_hoje, data_venc)
duration = ntnb_duration(vna, taxa_10_anos, cupom, data_hoje, data_venc)

print(f"\n=== Precificação NTN-B com Dados ANBIMA ===")
print(f"Taxa IPCA (Vértice mais próximo de 10 anos): {taxa_10_anos}% a.a.")
print(f"PU Estimado: R$ {preco:,.2f}")
print(f"Duration: {duration:.4f} anos")

# COMMAND ----------

# DBTITLE 1,Testar LTN com curva PRE da ANBIMA
# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Testar a LTN com dados da ANBIMA (Curva PRE)

from ibov_barrier.fixed_income.ltn import ltn_price, ltn_duration
import pyettj

print("\n=== Buscando Curva PRE para a LTN ===")
try:
    # Pega a curva PRE (Prefixada) real da ANBIMA
    df_pre = pyettj.get_ettj(date(2026, 10, 27).strftime("%Y-%m-%d"), curva="PRE")
    df_pre = df_pre.rename(columns={"dias_uteis": "vertice_du", "taxa": "taxa_pre"})
    df_pre["taxa_pre"] = df_pre["taxa_pre"] * 100
    
    # Pega a taxa de ~1 ano (252 DU) para uma LTN fictícia de 1 ano
    idx_1_ano = (df_pre["vertice_du"] - 252).abs().idxmin()
    taxa_1_ano = df_pre.loc[idx_1_ano, "taxa_pre"]
    
    data_venc_ltn = date(2027, 10, 27)
    preco_ltn = ltn_price(taxa_1_ano, date(2026, 10, 27), data_venc_ltn)
    duration_ltn = ltn_duration(taxa_1_ano, date(2026, 10, 27), data_venc_ltn)
    
    print(f"Taxa PRE (1 ano): {taxa_1_ano:.2f}% a.a.")
    print(f"PU Estimado da LTN: R$ {preco_ltn:.2f}")
    print(f"Duration: {duration_ltn:.4f} anos")
except Exception as e:
    print(f"Erro ao precificar LTN: {e}")