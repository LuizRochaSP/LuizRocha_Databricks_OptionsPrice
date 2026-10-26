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
# MAGIC %md
# MAGIC ## 1. Coleta da ETTJ IPCA da ANBIMA (via API do pyettj ou CSV)
# MAGIC Como a API do pyettj pode mudar, vamos usar o arquivo CSV oficial da ANBIMA
# MAGIC (disponível publicamente no site deles para download).

def baixar_ettj_ipca_anbima(data_ref: date) -> pd.DataFrame:
    """Baixa a ETTJ IPCA da ANBIMA.
    Nota: Em produção, usar pyettj ou a API oficial. Aqui usamos um mock/CSV.
    """
    # Exemplo de URL de onde a ANBIMA disponibiliza o CSV (pode variar)
    url = "https://www.anbima.com.br/informacoes/est-termo/CZ-down.asp"
    
    try:
        # Tenta fazer o download dos dados
        # Em um cenário real, você faria o parse do arquivo CSV ou chamaria pyettj
        # Aqui simulamos o retorno para não travar o teste por falta de conexão
        print(f"Buscando dados da ANBIMA para {data_ref.strftime('%d/%m/%Y')}...")
        
        # Mock de dados para teste (substitua pela chamada real do pyettj)
        # Exemplo: ettj = pyettj.get_ettj(data_ref, "IPCA")
        dados_mock = {
            "vertice_du": [252, 504, 1260, 2520],
            "taxa_ipca": [6.10, 6.20, 6.35, 6.50]
        }
        df = pd.DataFrame(dados_mock)
        return df
    except Exception as e:
        print(f"Erro ao baixar ETTJ: {e}")
        return None

# COMMAND ----------

data_hoje = date.today()
df_ettj = baixar_ettj_ipca_anbima(data_hoje)

if df_ettj is not None:
    print("ETTJ IPCA carregada com sucesso:")
    display(df_ettj)
else:
    print("Falha ao carregar ETTJ.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Testar a NTN-B com a taxa real

from ibov_barrier.fixed_income.ntnb import ntnb_price, ntnb_duration

# Parâmetros do título
vna = 4000.00
cupom = 6.00
data_venc = date(2035, 5, 15)

# Pega a taxa de 10 anos (2520 DU) do DataFrame
taxa_10_anos = df_ettj[df_ettj["vertice_du"] == 2520]["taxa_ipca"].values[0]

preco = ntnb_price(vna, taxa_10_anos, cupom, data_hoje, data_venc)
duration = ntnb_duration(vna, taxa_10_anos, cupom, data_hoje, data_venc)

print(f"\n=== Precificação NTN-B com Dados ANBIMA ===")
print(f"Taxa IPCA (10 anos): {taxa_10_anos}% a.a.")
print(f"PU Estimado: R$ {preco:,.2f}")
print(f"Duration: {duration:.4f} anos")
