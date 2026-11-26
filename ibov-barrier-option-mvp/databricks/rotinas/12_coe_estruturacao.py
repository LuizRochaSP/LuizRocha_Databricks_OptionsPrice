# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 12 — Estruturação de COE (visão do estruturador)
# MAGIC
# MAGIC Monta um COE com o termsheet dos widgets, sobre o mercado da última rodada aprovada do IBOV, e reporta:
# MAGIC
# MAGIC - a **decomposição do preço de emissão**: título + opções + custos + reservas + margem, em R$ e em % do nominal;
# MAGIC - o **parâmetro resolvido** para a margem alvo (participação, proteção, cap ou strike), quando pedido, ou o
# MAGIC   aviso de margem inviável;
# MAGIC - as **gregas iniciais** do COE (sinal do investidor; o emissor está vendido), por perna, e o DV01 por vértice
# MAGIC   da curva DI.
# MAGIC
# MAGIC **Mercado.** A evidência `VALIDATION_superficieVol_*.json` aprovada mais recente (notebook 02) e o RESULTS dela
# MAGIC (notebook 01), com a conferência do SHA-256, como no notebook 03. A curva DI pré, o q(T) e a superfície de vol
# MAGIC são os do RESULTS. A conferência com o 01 mostra r, q e vol no contrato de referência do pricer.
# MAGIC
# MAGIC **Widgets em %** (100 = 100%). O spread do emissor sobre o DI é digitado (decisão 2 da seção 8 do plano).
# MAGIC
# MAGIC Grava `COE_estruturacao_<carimbo>.json` na pasta de resultados (gravação atômica) e imprime o relatório para
# MAGIC enviar. Primeiro caso: IBOV. O S&P 500 em reais entra no A6.

# COMMAND ----------

# MAGIC %pip install "exchange-calendars==4.13.2" "lxml>=6.0" scipy numpy pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Termsheet, custos e solver (widgets)
dbutils.widgets.text("validacao", "", "Validação VALIDATION_superficieVol_*.json (vazio = a mais recente aprovada)")
dbutils.widgets.text("nominal", "1000000", "Nominal (R$)")
dbutils.widgets.text("nivel_inicial", "", "Nível inicial S0 (vazio = spot da rodada)")
dbutils.widgets.text("prazo_du", "252", "Prazo (DU)")
dbutils.widgets.text("data_vencimento", "", "Vencimento AAAA-MM-DD (vazio = usa o prazo em DU)")
dbutils.widgets.dropdown("direcao", "alta", ["alta", "baixa"], "Direção")
dbutils.widgets.text("protecao", "100", "Proteção (%)")
dbutils.widgets.text("participacao", "100", "Participação (%)")
dbutils.widgets.text("cap", "20", "Cap (%; vazio = sem cap)")
dbutils.widgets.text("strike", "100", "Strike (% de S0)")
dbutils.widgets.text("spread_bp", "100", "Spread do emissor sobre o DI (bp)")
dbutils.widgets.text("preco_emissao", "100", "Preço de emissão (%)")
dbutils.widgets.text("custo_distribuicao", "3", "Custo de distribuição (%)")
dbutils.widgets.text("custo_operacional", "0.5", "Custo operacional (%)")
dbutils.widgets.text("reservas", "0.5", "Reservas (%)")
dbutils.widgets.dropdown("resolver", "nenhum", ["nenhum", "participacao", "protecao", "cap", "strike"],
                         "Parâmetro a resolver")
dbutils.widgets.text("margem_alvo", "2", "Margem alvo (%)")
dbutils.widgets.dropdown("gravar", "sim", ["sim", "não"], "Gravar o JSON da estruturação")

# COMMAND ----------

# DBTITLE 1,Código do projeto e leitura dos widgets
import importlib
import json
import os
import sys
from datetime import date, datetime
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

from ibov_barrier.b3_arquivos import gravar_atomico
from ibov_barrier.coe import Participacao, estruturar
from ibov_barrier.market_date import expected_market_date
from ibov_barrier.produtos import conferir_referencia, estado_ibov, ultima_rodada_aprovada

results_root = Path("/".join(notebook_path.split("/")[:4])) / "ibov-barrier-results"


def numero(nome: str) -> float | None:
    """Valor do widget: aceita "0,5" e "1.000.000"; vazio = None."""
    texto = dbutils.widgets.get(nome).strip().replace(" ", "").replace("_", "")
    if not texto:
        return None
    if "," in texto:
        texto = texto.replace(".", "").replace(",", ".")
    elif texto.count(".") > 1:
        texto = texto.replace(".", "")
    try:
        return float(texto)
    except ValueError:
        raise ValueError(f"widget {nome!r}: {dbutils.widgets.get(nome)!r} não é um número") from None


def obrigatorio(nome: str) -> float:
    valor = numero(nome)
    if valor is None:
        raise ValueError(f"widget {nome!r} vazio")
    return valor


def percentual(nome: str) -> float | None:
    valor = numero(nome)
    return None if valor is None else valor / 100.0


PARAMETRO = dbutils.widgets.get("resolver")
PARAMETRO = None if PARAMETRO == "nenhum" else PARAMETRO
MARGEM_ALVO = percentual("margem_alvo") if PARAMETRO else None
if PARAMETRO and MARGEM_ALVO is None:
    raise ValueError("Informe a margem alvo para resolver o parâmetro.")
CUSTOS = dict(preco_emissao=obrigatorio("preco_emissao") / 100.0,
              custo_distribuicao=obrigatorio("custo_distribuicao") / 100.0,
              custo_operacional=obrigatorio("custo_operacional") / 100.0,
              reservas=obrigatorio("reservas") / 100.0)
SPREAD_BP = obrigatorio("spread_bp")
print("Código de:", SRC, "| resultados em:", results_root)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Rodada aprovada e mercado

# COMMAND ----------

rodada = ultima_rodada_aprovada(results_root, arquivo_validacao=dbutils.widgets.get("validacao").strip() or None)
estado = estado_ibov(rodada, spread_emissor_bp=SPREAD_BP)
superficie = estado.ativo("IBOV").vol

avisos_rodada = []
d1 = expected_market_date(market="B3")
if rodada.data_mercado < d1:
    avisos_rodada.append(f"A rodada aprovada é de {rodada.data_mercado}; o D-1 da B3 é {d1}: o mercado está defasado.")
if rodada.ignoradas:
    avisos_rodada.append("Evidências mais novas puladas (não aprovadas): " + "; ".join(rodada.ignoradas))
if rodada.ressalvas:
    avisos_rodada.append(f"A rodada herda {len(rodada.ressalvas)} ressalva(s) da validação (WARN).")

conferencia = conferir_referencia(estado, rodada.metadata)
limites = {"r(T)": 1e-6, "q(T)": 1e-4, "vol no strike": 1e-3}  # 0,01 bp, 1 bp, 0,1 ponto de vol
for linha in conferencia:
    if linha["diferenca"] is not None and abs(linha["diferenca"]) > limites[linha["grandeza"]]:
        avisos_rodada.append(f"Conferência com o 01: {linha['grandeza']} reconstruído difere do 01 em "
                             f"{1e4 * linha['diferenca']:+.2f} bp no contrato de referência.")

for chave, valor in rodada.proveniencia().items():
    print(f"{chave}: {valor}")
print(f"Spot: {estado.ativo('IBOV').spot:,.2f} | superfície de {superficie.prazo_minimo:.2f} a "
      f"{superficie.prazo_maximo:.2f} ano | curva DI até {int(max(estado.curva('BRL').vertices))} DU")
display(pd.DataFrame(conferencia))
for aviso in avisos_rodada:
    print("AVISO:", aviso)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Termsheet

# COMMAND ----------

nivel_inicial = numero("nivel_inicial") or float(estado.ativo("IBOV").spot)
campos = dict(ativo="IBOV", nivel_inicial=nivel_inicial, nominal=obrigatorio("nominal"),
              protecao=obrigatorio("protecao") / 100.0, participacao=obrigatorio("participacao") / 100.0,
              cap=percentual("cap"), strike=obrigatorio("strike") / 100.0, direcao=dbutils.widgets.get("direcao"),
              nome="notebook 12")
vencimento_txt = dbutils.widgets.get("data_vencimento").strip()
if vencimento_txt:
    # Prazo em DU da B3 entre a data de mercado da rodada e o vencimento.
    termsheet = Participacao.por_datas(data_emissao=rodada.data_mercado,
                                       data_vencimento=date.fromisoformat(vencimento_txt), **campos)
else:
    termsheet = Participacao(vencimento=obrigatorio("prazo_du") / 252.0, **campos)
print(termsheet.descricao())
print(f"Nominal R$ {termsheet.nominal:,.2f} | S0 {termsheet.nivel_inicial:,.2f} | {termsheet.dias_uteis} DU")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Decomposição, solver e gregas iniciais

# COMMAND ----------

est = estruturar(termsheet, estado, parametro=PARAMETRO, margem_alvo=MARGEM_ALVO, avisos=avisos_rodada, **CUSTOS)


def cascata(d):
    return pd.DataFrame([(rotulo, reais, 100.0 * fracao) for rotulo, reais, fracao in d.linhas()],
                        columns=["Rubrica", "R$", "% do nominal"])


display(cascata(est.decomposicao))
if est.solucao is not None:
    print(f"Solver ({est.solucao.situacao}): {est.solucao.mensagem}")
    display(cascata(est.solucao.decomposicao))
display(pd.DataFrame({perna: g.como_dict() for perna, g in est.gregas.items()}))
display(pd.DataFrame({"DU": [int(du) for du in est.dv01_vertices], "DV01 (R$ por +1 bp)": list(est.dv01_vertices.values())}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Gravação

# COMMAND ----------

agora = datetime.now(ZoneInfo("America/Sao_Paulo"))
if dbutils.widgets.get("gravar") == "sim":
    destino = results_root / f"COE_estruturacao_{agora:%Y%m%d_%H%M%S}.json"
    payload = {"schema_version": 1, "timestamp": agora.isoformat(), "notebook": "12_coe_estruturacao",
               **rodada.proveniencia(), "conferencia_01": conferencia, "estruturacao": est.como_dict()}
    gravar_atomico(destino, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
    print("Gravado:", destino.name)
else:
    print("Gravação desligada no widget.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Relatório para enviar

# COMMAND ----------

print(est.texto(proveniencia=rodada.proveniencia(), agora=agora))
