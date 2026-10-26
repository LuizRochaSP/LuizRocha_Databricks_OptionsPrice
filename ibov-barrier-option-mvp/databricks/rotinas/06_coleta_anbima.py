# Databricks notebook source
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
        df = pyettj.get_ettj(data_str, curva="ipca")
        
        if df is not None and not df.empty:
            # Renomeia as colunas para o padrão do nosso sistema
            df = df.rename(columns={"vertice": "vertice_du", "taxa": "taxa_ipca"})
            print("✅ ETTJ IPCA REAL carregada com sucesso!")
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

data_hoje = date.today()
df_ettj = baixar_ettj_ipca_anbima(data_hoje)

if df_ettj is not None:
    display(df_ettj.head(10)) # Mostra os primeiros 10 vértices
else:
    print("Falha ao carregar ETTJ.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Testar a NTN-B com a taxa real

from ibov_barrier.fixed_income.ntnb import ntnb_price, ntnb_duration

vna = 4000.00
cupom = 6.00
data_venc = date(2035, 5, 15)

# Pega a taxa mais próxima de 10 anos (2520 DU) - busca pelo valor mais próximo caso não exista o vértice exato
idx_10_anos = (df_ettj["vertice_du"] - 2520).abs().idxmin()
taxa_10_anos = df_ettj.loc[idx_10_anos, "taxa_ipca"]

preco = ntnb_price(vna, taxa_10_anos, cupom, data_hoje, data_venc)
duration = ntnb_duration(vna, taxa_10_anos, cupom, data_hoje, data_venc)

print(f"\n=== Precificação NTN-B com Dados ANBIMA ===")
print(f"Taxa IPCA (Vértice mais próximo de 10 anos): {taxa_10_anos}% a.a.")
print(f"PU Estimado: R$ {preco:,.2f}")
print(f"Duration: {duration:.4f} anos")
