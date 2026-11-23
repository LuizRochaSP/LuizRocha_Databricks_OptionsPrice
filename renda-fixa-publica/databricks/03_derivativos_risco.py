# Databricks notebook source
# MAGIC %md
# MAGIC # Renda Fixa Pública: derivativos de juros e risco (03)
# MAGIC
# MAGIC 1. DI1 e FRA teóricos pela curva pré dos títulos (o projeto não depende de dados da B3).
# MAGIC 2. Swaps DI x Pré e DI x IPCA: taxas par e marcação a mercado de um swap em andamento com o CDI realizado (SGS 12).
# MAGIC 3. Carteira de exemplo: DV01, key-rate DV01, cenários, carry e roll-down, hedge com DI1.
# MAGIC 4. VaR histórico e PCA da curva, com o histórico de taxas do Tesouro Direto.

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

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from renda_fixa import graficos as G
from renda_fixa.calendario import dias_uteis, proximo_dia_util
from renda_fixa.dados import bcb
from renda_fixa.dados.anbima import baixar_mercado_secundario, ler_mercado_secundario, ultimo_disponivel
from renda_fixa.derivativos import DI1, SwapDIIPCA, SwapDIPre, codigo_contrato, taxa_fra
from renda_fixa.mercado import VERTICES_PADRAO, historico_curvas, montar_mercado, variacoes_diarias_bps
from renda_fixa.risco import (Carteira, PosicaoDI1, PosicaoSwap, PosicaoTitulo, cenarios_padrao,
                              contratos_hedge_di1, pca_curva)
from renda_fixa.titulos import criar_titulo

user_home = "/".join(notebook_path.split("/")[:4])
dbutils.widgets.text("data_ref", "", "Data (vazio = mais recente)")
dbutils.widgets.text("pasta_dados", f"{user_home}/rf_dados", "Pasta de dados")
dbutils.widgets.text("swap_inicio", "", "Início do swap em andamento (vazio = 1 ano atrás)")
dbutils.widgets.text("swap_taxa", "14.50", "Taxa pré do swap em andamento (% a.a.)")
dbutils.widgets.text("nivel_var", "0.99", "Nível do VaR")
dbutils.widgets.text("anos_historico", "2", "Histórico para VaR e PCA (anos)")
PASTA = Path(dbutils.widgets.get("pasta_dados").strip())
DATA = dbutils.widgets.get("data_ref").strip()
df = baixar_mercado_secundario(date.fromisoformat(DATA), PASTA / "anbima") if DATA else ultimo_disponivel(date.today(), PASTA / "anbima")
m = montar_mercado(df)
REF, PRE, REAL = m.ref, m.curva_pre, m.curva_real
CURVAS = m.curvas
print("Data de referência:", REF)


def mostrar(fig, nome):
    displayHTML(G.html_figura(fig, f"rf_{nome}_{REF}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. DI1 e FRA teóricos (curva pré dos títulos)

# COMMAND ----------

contratos = []
for ano in range(REF.year + 1, REF.year + 11):
    for mes in (1, 4, 7, 10) if ano <= REF.year + 3 else (1,):
        c = DI1(proximo_dia_util(date(ano, mes, 1)))
        n = c.du(REF)
        if n <= 0:
            continue
        tx = PRE.taxa(n)
        contratos.append({"contrato": str(c), "vencimento": c.vencimento, "du": n, "taxa": tx,
                          "pu": c.pu(tx, REF), "dv01_R$": c.dv01(tx, REF)})
di = pd.DataFrame(contratos)
di["fra_vs_anterior"] = [np.nan] + [taxa_fra(a.taxa, a.du, b.taxa, b.du) for a, b in zip(di.iloc[:-1].itertuples(), di.iloc[1:].itertuples())]
display(di)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Swaps: taxas par e marcação a mercado

# COMMAND ----------

par = []
for anos in (1, 2, 3, 5, 7, 10):
    venc = proximo_dia_util(date(REF.year + anos, REF.month, min(REF.day, 28)))
    linha = {"prazo_anos": anos, "vencimento": venc, "di_x_pre_100cdi": SwapDIPre.taxa_par(PRE, REF, venc),
             "di_x_pre_cdi+1": SwapDIPre.taxa_par(PRE, REF, venc, spread_cdi_aa=1.0)}
    if REAL is not None:
        linha["di_x_ipca_cupom"] = SwapDIIPCA.cupom_par(PRE, REAL, REF, venc)
    par.append(linha)
display(pd.DataFrame(par))

ini_txt = dbutils.widgets.get("swap_inicio").strip()
inicio = date.fromisoformat(ini_txt) if ini_txt else proximo_dia_util(REF - timedelta(days=365))
taxa_swap = float(dbutils.widgets.get("swap_taxa"))
cdi = bcb.serie_sgs(bcb.CDI_DIARIO, inicio - timedelta(days=5), REF, cache_dir=PASTA / "bcb")
fator = bcb.fator_cdi(cdi, inicio, REF)
swap = SwapDIPre(10_000_000, inicio, proximo_dia_util(date(REF.year + 2, 1, 1)), taxa_swap, ativo="pre")
r = swap.avaliar(REF, PRE, fator)
print(f"Swap DI x Pré (recebe {taxa_swap:.2f}% pré, paga 100% CDI), nocional 10 mi, de {inicio} a {swap.vencimento}")
print(f"  CDI acumulado {fator:.8f} | VP pré {r.vp_ativo:,.2f} | VP DI {r.vp_passivo:,.2f} | MtM {r.mtm:,.2f} | DV01 {r.dv01:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Carteira de exemplo

# COMMAND ----------

def preco(titulo, venc):
    s = m.df[(m.df.titulo == titulo) & (m.df.vencimento == venc)]
    return float(s.pu.iloc[0])


ultimo = lambda t: m.df[m.df.titulo == t].vencimento.max()
ltn_v = sorted(m.df[m.df.titulo == "LTN"].vencimento)[len(m.df[m.df.titulo == "LTN"]) // 2]
ntnf_v = ultimo("NTN-F")
ntnb_v = sorted(m.df[m.df.titulo == "NTN-B"].vencimento)[len(m.df[m.df.titulo == "NTN-B"]) // 2]

carteira = Carteira([
    PosicaoTitulo.a_mercado(criar_titulo("LTN", ltn_v), 20_000, preco("LTN", ltn_v), REF, CURVAS),
    PosicaoTitulo.a_mercado(criar_titulo("NTN-F", ntnf_v), 10_000, preco("NTN-F", ntnf_v), REF, CURVAS),
    PosicaoTitulo.a_mercado(criar_titulo("NTN-B", ntnb_v), 2_000, preco("NTN-B", ntnb_v), REF, CURVAS, vna=m.vna("NTN-B")),
    PosicaoSwap(swap, REF, fator),
])
por_posicao = carteira.dv01_por_posicao(REF, CURVAS)
display(por_posicao)
dv01_pre = carteira.dv01(REF, CURVAS, "PRE")
dv01_real = carteira.dv01(REF, CURVAS, "IPCA")
print(f"Valor {carteira.valor(REF, CURVAS):,.2f} | DV01 pré {dv01_pre:,.2f} | DV01 real {dv01_real:,.2f}")

vert = [21, 126, 252, 504, 756, 1260, 1764, 2520, 3780, 5040]
kr = carteira.key_rate_dv01(REF, CURVAS, "PRE", vert)
krr = carteira.key_rate_dv01(REF, CURVAS, "IPCA", vert)
fig = G.figura([G.barras("Pré", [f"{v / 252:.1f}a" for v in vert], kr.values)], "Key-rate DV01 pré (R$ por +1 bp)", "vértice", "R$")
mostrar(fig, "key_rate_pre")
fig = G.figura([G.barras("Real", [f"{v / 252:.1f}a" for v in vert], krr.values)], "Key-rate DV01 real (R$ por +1 bp)", "vértice", "R$")
mostrar(fig, "key_rate_real")

# COMMAND ----------

cen = carteira.cenarios(REF, CURVAS, cenarios_padrao())
fig = G.figura([G.barras("P&L", cen.index, cen.values)], "Cenários de stress (P&L em R$)", "", "R$")
mostrar(fig, "cenarios")
carry21 = carteira.carry_rolldown(REF, CURVAS, 21)
contrato = DI1(proximo_dia_util(date(REF.year + 3, 1, 1)))
q = contratos_hedge_di1(dv01_pre, contrato, PRE.taxa(contrato.du(REF)), REF)
print(f"Carry + roll-down em 21 DU (curvas paradas): {carry21:,.2f}")
print(f"Hedge do DV01 pré com {contrato}: {q:,.0f} contratos em PU ({'vendido' if q < 0 else 'comprado'} em PU = "
      f"{'comprado' if q < 0 else 'vendido'} em taxa)")
hedged = Carteira(carteira.posicoes + [PosicaoDI1(contrato, q, contrato.pu_curva(PRE, REF))])
print(f"DV01 pré depois do hedge: {hedged.dv01(REF, CURVAS, 'PRE'):,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. VaR histórico e PCA (histórico do Tesouro Direto)

# COMMAND ----------

from renda_fixa.dados.tesouro_direto import baixar_tesouro_direto

ANOS_HIST = float(dbutils.widgets.get("anos_historico"))
NIVEL = float(dbutils.widgets.get("nivel_var"))
td = baixar_tesouro_direto(PASTA / "tesouro_direto")
td = td[(td["data_referencia"] <= REF) & (td["data_referencia"] >= REF - timedelta(days=int(365 * ANOS_HIST)))]
hc = historico_curvas(td, VERTICES_PADRAO)
print(f"Histórico Tesouro Direto: {len(hc['PRE'])} dias de curva pré, {len(hc['IPCA'])} de curva real")
var_res = None
if len(hc["PRE"]) >= 30:
    choques_pre = variacoes_diarias_bps(hc["PRE"])  # bps, só dias úteis consecutivos
    maiores = choques_pre.abs().max(axis=1).sort_values(ascending=False).head(5)
    print("Maiores choques diários da curva pré (bps, máximo entre vértices):",
          ", ".join(f"{d}: {v:.0f}" for d, v in maiores.items()))
    var_res = carteira.var_historico(REF, CURVAS, choques_pre, "PRE", NIVEL)
    print(f"VaR {NIVEL:.0%} 1 dia (curva pré, {var_res['n_cenarios']} cenários): {var_res['var']:,.2f} | ES {var_res['es']:,.2f}")
    fig = G.figura([go.Histogram(
        x=var_res["pnls"], nbinsx=60, marker=dict(color=G.QUENTES[2]), name="P&L")],
        f"P&L histórico da carteira (choques da curva pré, {var_res['n_cenarios']} dias)", "R$", "dias")
    mostrar(fig, "var_hist")
    pca = pca_curva(hc["PRE"])
    anos_v = [v / 252 for v in pca.cargas.index]
    fig = G.figura([G.linha(i, f"{c} ({pca.variancia_explicada[i]:.1%})", anos_v, pca.cargas[c]) for i, c in enumerate(pca.cargas.columns)],
                   "PCA da curva pré: nível, inclinação e curvatura", "anos", "carga")
    mostrar(fig, "pca_pre")
    ultimos = hc["PRE"].iloc[[-1, -22, -64]] if len(hc["PRE"]) > 64 else hc["PRE"].iloc[[-1]]
    fig = G.figura([G.linha(i, str(d), [v / 252 for v in ultimos.columns], ultimos.loc[d], "lines+markers") for i, d in enumerate(ultimos.index)],
                   "Curva pré (Tesouro Direto): hoje, 1 mês e 3 meses atrás", "anos", "% a.a.")
    mostrar(fig, "historico_pre")
else:
    print("Histórico curto para VaR e PCA.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Relatório

# COMMAND ----------

print("RELATÓRIO DERIVATIVOS E RISCO", REF)
print(f"Swaps par: " + ", ".join(f"{p['prazo_anos']}a {p['di_x_pre_100cdi']:.3f}%" for p in par))
print(f"Swap em andamento: MtM {r.mtm:,.2f} | DV01 {r.dv01:,.2f} | CDI acumulado {fator:.8f}")
print(f"Carteira: valor {carteira.valor(REF, CURVAS):,.2f} | DV01 pré {dv01_pre:,.2f} | DV01 real {dv01_real:,.2f} | carry 21 DU {carry21:,.2f}")
print("Cenários:", ", ".join(f"{k}: {v:,.0f}" for k, v in cen.items()))
if var_res:
    print(f"VaR {NIVEL:.0%}: {var_res['var']:,.2f} | ES {var_res['es']:,.2f}")
