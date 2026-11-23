"""Renda Fixa Pública Brasil: dashboard do motor (títulos, curvas, derivativos e risco).

Rodar localmente (a partir de renda-fixa-publica/):
    streamlit run dashboard/app.py

Dados: arquivo diário da ANBIMA (títulos), BCB/SGS (CDI) e Tesouro Direto (histórico),
baixados sob demanda com cache em disco (variável RF_CACHE_DIR, padrão: pasta temporária).
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from renda_fixa import graficos as G
from renda_fixa.calendario import dias_uteis, proximo_dia_util
from renda_fixa.dados import bcb
from renda_fixa.dados.anbima import DadoIndisponivel, baixar_mercado_secundario, ultimo_disponivel
from renda_fixa.derivativos import DI1, SwapDIIPCA, SwapDIPre, taxa_fra
from renda_fixa.mercado import (VERTICES_PADRAO, historico_curvas, montar_mercado, tabela_curvas, tabela_titulos,
                                variacoes_diarias_bps)
from renda_fixa.risco import Carteira, PosicaoDI1, PosicaoSwap, PosicaoTitulo, cenarios_padrao, contratos_hedge_di1
from renda_fixa.titulos import criar_titulo

CACHE = Path(os.environ.get("RF_CACHE_DIR", Path(tempfile.gettempdir()) / "rf_dados"))

st.set_page_config(page_title="Renda Fixa Pública", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0B0B0B; color: #E8E9EB; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    h1, h2, h3 { font-weight: 600; letter-spacing: 0.2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def plot(fig, chave: str):
    st.plotly_chart(fig, width="stretch", theme=None, config=G.config_barra(chave), key=chave)


@st.cache_data(show_spinner="Baixando o arquivo da ANBIMA...", ttl=3600)
def carregar_dia(data_txt: str) -> pd.DataFrame:
    if data_txt:
        return baixar_mercado_secundario(date.fromisoformat(data_txt), CACHE / "anbima")
    return ultimo_disponivel(date.today(), CACHE / "anbima")


@st.cache_resource(show_spinner="Construindo curvas e ajustando NSS...")
def mercado(data_txt: str):
    return montar_mercado(carregar_dia(data_txt))


@st.cache_data(show_spinner="Baixando o histórico do Tesouro Direto (~15 MB)...", ttl=6 * 3600)
def historico(ref: date, anos: float) -> dict:
    from renda_fixa.dados.tesouro_direto import baixar_tesouro_direto

    td = baixar_tesouro_direto(CACHE / "tesouro_direto")
    td = td[(td["data_referencia"] <= ref) & (td["data_referencia"] >= ref - timedelta(days=int(365 * anos)))]
    return historico_curvas(td, VERTICES_PADRAO)


@st.cache_data(ttl=3600)
def cdi(inicio: date, fim: date) -> pd.Series:
    return bcb.serie_sgs(bcb.CDI_DIARIO, inicio - timedelta(days=5), fim, cache_dir=CACHE / "bcb")


# ------------------------------------------------------------------ cabeçalho
st.title("Renda Fixa Pública Brasil")
st.caption("Títulos públicos federais, curvas de juros, derivativos de juros e risco. Fonte: ANBIMA, Banco Central e Tesouro Direto.")

with st.sidebar:
    st.header("Data de referência")
    data_txt = st.text_input("AAAA-MM-DD (vazio = mais recente)", "")
    st.caption(f"Cache de dados: {CACHE}")

try:
    m = mercado(data_txt.strip())
except DadoIndisponivel as e:
    st.error(f"Sem dados da ANBIMA: {e}")
    st.stop()

REF, PRE, REAL = m.ref, m.curva_pre, m.curva_real
CURVAS = m.curvas
tt = tabela_titulos(m)

c = st.columns(5)
c[0].metric("Data", REF.strftime("%d/%m/%Y"))
c[1].metric("Títulos", len(tt))
c[2].metric("PU idêntico à ANBIMA", f"{int((tt['erro_pu'] == 0).sum())} de {len(tt)}")
c[3].metric("VNA LFT", f"{m.vna_lft.vna:,.6f}")
c[4].metric("VNA NTN-B", f"{m.vna_ntnb.vna:,.6f}")

aba = st.tabs(["Títulos do dia", "Curvas", "Calculadora", "DI1 e swaps", "Risco de carteira"])

# ------------------------------------------------------------------ títulos
with aba[0]:
    st.subheader("Validação título a título")
    st.caption("PU do motor (convenções de truncamento do Tesouro e da ANBIMA) contra o PU indicativo da ANBIMA. "
               "Resíduo NSS: diferença de preço do título pela curva ajustada, em bps de taxa.")
    filtro = st.multiselect("Títulos", sorted(tt["titulo"].unique()), default=sorted(tt["titulo"].unique()))
    st.dataframe(
        tt[tt["titulo"].isin(filtro)].style.format(
            {"taxa_anbima": "{:.4f}", "taxa_motor": "{:.4f}", "pu_anbima": "{:,.6f}", "pu_motor": "{:,.6f}",
             "erro_pu": "{:.6f}", "duration_mac": "{:.4f}", "duration_mod": "{:.4f}", "dv01": "{:.4f}",
             "convexidade": "{:.2f}", "pu_nss": "{:,.6f}", "residuo_nss_bps": "{:.2f}"}, na_rep="-"),
        width="stretch", hide_index=True, height=560,
    )

# ------------------------------------------------------------------ curvas
with aba[1]:
    tc = tabela_curvas(m, vertices=tuple(range(21, 9000, 21)))
    anos = tc["anos"]
    pts_pre = tt[tt["titulo"].isin(["LTN", "NTN-F"])]
    pts_real = tt[tt["titulo"] == "NTN-B"]
    col = st.columns(2)
    with col[0]:
        plot(G.figura([G.linha(1, "Bootstrap", anos, tc["pre"], "lines"), G.linha(3, "NSS", anos, tc["pre_nss"], "lines"),
                       G.linha(0, "LTN / NTN-F", pts_pre["duration_mac"], pts_pre["taxa_anbima"], "markers")],
                      "Curva prefixada", "anos", "% a.a."), "curva_pre")
    with col[1]:
        plot(G.figura([G.linha(1, "Bootstrap", anos, tc["real"], "lines"), G.linha(3, "NSS", anos, tc["real_nss"], "lines"),
                       G.linha(0, "NTN-B", pts_real["duration_mac"], pts_real["taxa_anbima"], "markers")],
                      "Curva de juro real (IPCA)", "anos", "% a.a."), "curva_real")
    col = st.columns(2)
    with col[0]:
        plot(G.figura([G.linha(2, "Inflação implícita", anos, tc["inflacao_implicita"], "lines")],
                      "Inflação implícita", "anos", "% a.a."), "breakeven")
    with col[1]:
        plot(G.figura([G.linha(4, "Termo pré de 1 ano", anos, tc["termo_pre_1a"], "lines")],
                      "Taxa a termo de 1 ano (pré)", "anos", "% a.a."), "termo")
    st.dataframe(tabela_curvas(m).round(4), width="stretch", hide_index=True)

# ------------------------------------------------------------------ calculadora
with aba[2]:
    col = st.columns([1, 1, 1, 1])
    tipo = col[0].selectbox("Título", ["LTN", "NTN-F", "NTN-B", "NTN-B Principal", "LFT"])
    vencs = sorted(m.df[m.df["titulo"] == tipo]["vencimento"].unique()) if tipo in m.df["titulo"].values else []
    venc = col[1].selectbox("Vencimento", vencs, format_func=lambda d: d.strftime("%d/%m/%Y")) if vencs else \
        col[1].date_input("Vencimento", date(REF.year + 5, 5, 15))
    t = criar_titulo(tipo, venc)
    vna = m.vna(tipo) if t.usa_vna else None
    if t.usa_vna:
        vna = col[2].number_input("VNA", value=float(vna or 4000.0), format="%.6f")
    linha_mkt = m.df[(m.df["titulo"] == tipo) & (m.df["vencimento"] == venc)]
    taxa_padrao = float(linha_mkt["taxa_indicativa"].iloc[0]) if len(linha_mkt) else 10.0
    modo = col[3].radio("Calcular", ["PU a partir da taxa", "Taxa a partir do PU"])
    if modo == "PU a partir da taxa":
        taxa = st.number_input("Taxa (% a.a.)", value=taxa_padrao, format="%.4f", step=0.01)
        pu = t.pu(taxa, REF, vna)
    else:
        pu_in = st.number_input("PU (R$)", value=float(t.pu(taxa_padrao, REF, vna)), format="%.6f")
        taxa = t.taxa(pu_in, REF, vna)
        pu = pu_in
    a = t.analiticos(taxa, REF, vna)
    k = st.columns(6)
    k[0].metric("PU", f"{pu:,.6f}")
    k[1].metric("Taxa", f"{taxa:.4f}%")
    k[2].metric("Prazo (DU)", a.du)
    k[3].metric("Duration mod.", f"{a.duration_modificada:.4f}")
    k[4].metric("DV01 (R$/título)", f"{a.dv01:.4f}")
    k[5].metric("Convexidade", f"{a.convexidade:.2f}")
    fl = pd.DataFrame([{"data": f.data, "du": f.du, "fluxo_pct": f.valor,
                        "vp_pct": f.valor / (1 + taxa / 100) ** (f.du / 252)} for f in t.fluxos(REF)])
    if t.usa_vna:
        fl["fluxo_R$"] = fl["fluxo_pct"] * vna / 100
    else:
        fl["fluxo_R$"] = fl["fluxo_pct"] * 10
    st.dataframe(fl, width="stretch", hide_index=True)

# ------------------------------------------------------------------ DI1 e swaps
with aba[3]:
    st.subheader("DI1 teórico pela curva pré dos títulos")
    linhas = []
    for ano in range(REF.year + 1, REF.year + 11):
        for mes in ((1, 4, 7, 10) if ano <= REF.year + 3 else (1,)):
            ctt = DI1(proximo_dia_util(date(ano, mes, 1)))
            n = ctt.du(REF)
            if n > 0:
                tx = PRE.taxa(n)
                linhas.append({"contrato": str(ctt), "vencimento": ctt.vencimento, "du": n, "taxa": tx,
                               "pu": ctt.pu(tx, REF), "dv01_R$": ctt.dv01(tx, REF)})
    di = pd.DataFrame(linhas)
    di["fra_vs_anterior"] = [np.nan] + [taxa_fra(x.taxa, x.du, y.taxa, y.du) for x, y in
                                        zip(di.iloc[:-1].itertuples(), di.iloc[1:].itertuples())]
    st.dataframe(di.round(4), width="stretch", hide_index=True)

    st.subheader("Taxas par")
    par = []
    for anos_ in (1, 2, 3, 5, 7, 10):
        v = proximo_dia_util(date(REF.year + anos_, REF.month, min(REF.day, 28)))
        par.append({"prazo": f"{anos_}a", "vencimento": v, "DI x Pré (100% CDI)": SwapDIPre.taxa_par(PRE, REF, v),
                    "DI x Pré (CDI + 1%)": SwapDIPre.taxa_par(PRE, REF, v, spread_cdi_aa=1.0),
                    "DI x IPCA (cupom)": SwapDIIPCA.cupom_par(PRE, REAL, REF, v)})
    st.dataframe(pd.DataFrame(par).round(4), width="stretch", hide_index=True)

    st.subheader("Marcação a mercado de swap DI x Pré")
    col = st.columns(5)
    noc = col[0].number_input("Nocional (R$)", value=10_000_000.0, step=1_000_000.0, format="%.2f")
    ini = col[1].date_input("Início", proximo_dia_util(REF - timedelta(days=365)))
    ven = col[2].date_input("Vencimento", proximo_dia_util(date(REF.year + 2, 1, 1)))
    kpre = col[3].number_input("Taxa pré (% a.a.)", value=14.5, format="%.4f")
    lado = col[4].selectbox("Recebe", ["pre", "di"], format_func=lambda x: "Pré" if x == "pre" else "DI")
    col = st.columns(2)
    pct = col[0].number_input("% do CDI", value=100.0)
    spread = col[1].number_input("Spread sobre o CDI (% a.a.)", value=0.0, format="%.4f")
    try:
        fator = bcb.fator_cdi(cdi(ini, REF), ini, REF, pct) if ini < REF else 1.0
        sw = SwapDIPre(noc, ini, ven, kpre, ativo=lado, pct_cdi=pct, spread_cdi_aa=spread)
        r = sw.avaliar(REF, PRE, fator)
        k = st.columns(5)
        k[0].metric("CDI acumulado", f"{fator:.8f}")
        k[1].metric("VP ativo", f"{r.vp_ativo:,.2f}")
        k[2].metric("VP passivo", f"{r.vp_passivo:,.2f}")
        k[3].metric("MtM", f"{r.mtm:,.2f}")
        k[4].metric("DV01", f"{r.dv01:,.2f}")
    except Exception as e:  # SGS fora do ar
        st.error(f"Não foi possível avaliar o swap: {e}")

# ------------------------------------------------------------------ risco
with aba[4]:
    st.subheader("Carteira")
    st.caption("Títulos a mercado (z-spread sobre a curva do dia). Quantidade em títulos; negativa = vendido.")
    base = pd.DataFrame([
        {"titulo": "LTN", "vencimento": sorted(m.df[m.df.titulo == "LTN"].vencimento)[4], "quantidade": 20000},
        {"titulo": "NTN-F", "vencimento": m.df[m.df.titulo == "NTN-F"].vencimento.max(), "quantidade": 10000},
        {"titulo": "NTN-B", "vencimento": sorted(m.df[m.df.titulo == "NTN-B"].vencimento)[7], "quantidade": 2000},
    ])
    opcoes = {t_: sorted(m.df[m.df.titulo == t_].vencimento) for t_ in ("LTN", "NTN-F", "NTN-B", "LFT")}
    ed = st.data_editor(
        base, num_rows="dynamic", width="stretch", hide_index=True,
        column_config={"titulo": st.column_config.SelectboxColumn(options=list(opcoes)),
                       "vencimento": st.column_config.DateColumn(format="DD/MM/YYYY"),
                       "quantidade": st.column_config.NumberColumn(step=100)},
    )
    carteira = Carteira()
    for r_ in ed.dropna().itertuples():
        linha_mkt = m.df[(m.df.titulo == r_.titulo) & (m.df.vencimento == r_.vencimento)]
        if linha_mkt.empty:
            st.warning(f"{r_.titulo} {r_.vencimento} não negociado em {REF}; ignorado.")
            continue
        tit = criar_titulo(r_.titulo, r_.vencimento)
        carteira.adicionar(PosicaoTitulo.a_mercado(tit, float(r_.quantidade), float(linha_mkt.pu.iloc[0]), REF, CURVAS,
                                                   vna=m.vna(r_.titulo)))
    if not carteira.posicoes:
        st.stop()
    dv_pre, dv_real = carteira.dv01(REF, CURVAS, "PRE"), carteira.dv01(REF, CURVAS, "IPCA")
    k = st.columns(4)
    k[0].metric("Valor", f"{carteira.valor(REF, CURVAS):,.2f}")
    k[1].metric("DV01 pré", f"{dv_pre:,.2f}")
    k[2].metric("DV01 real", f"{dv_real:,.2f}")
    k[3].metric("Carry + roll-down 21 DU", f"{carteira.carry_rolldown(REF, CURVAS, 21):,.2f}")
    st.dataframe(carteira.dv01_por_posicao(REF, CURVAS).round(2), width="stretch", hide_index=True)

    vert = [21, 126, 252, 504, 756, 1260, 1764, 2520, 3780, 5040]
    rot = [f"{v / 252:.1f}a" for v in vert]
    col = st.columns(2)
    with col[0]:
        plot(G.figura([G.barras("Pré", rot, carteira.key_rate_dv01(REF, CURVAS, "PRE", vert).values)],
                      "Key-rate DV01 pré (R$ por +1 bp)", "vértice", "R$"), "kr_pre")
    with col[1]:
        plot(G.figura([G.barras("Real", rot, carteira.key_rate_dv01(REF, CURVAS, "IPCA", vert).values)],
                      "Key-rate DV01 real (R$ por +1 bp)", "vértice", "R$"), "kr_real")
    cen = carteira.cenarios(REF, CURVAS, cenarios_padrao())
    plot(G.figura([G.barras("P&L", cen.index, cen.values)], "Cenários de stress (P&L em R$)", "", "R$"), "cenarios")

    contrato = DI1(proximo_dia_util(date(REF.year + 3, 1, 1)))
    q = contratos_hedge_di1(dv_pre, contrato, PRE.taxa(contrato.du(REF)), REF)
    st.info(f"Hedge do DV01 pré com {contrato}: {abs(q):,.0f} contratos "
            f"{'comprados em taxa (vendidos em PU)' if q < 0 else 'vendidos em taxa (comprados em PU)'}.")

    st.subheader("VaR histórico (curva pré, Tesouro Direto)")
    col = st.columns(3)
    anos_h = col[0].number_input("Anos de histórico", value=2.0, min_value=0.5, max_value=20.0, step=0.5)
    nivel = col[1].selectbox("Nível", [0.95, 0.975, 0.99], index=2)
    if col[2].button("Calcular VaR"):
        hc = historico(REF, anos_h)
        choques = variacoes_diarias_bps(hc["PRE"])
        res = carteira.var_historico(REF, CURVAS, choques, "PRE", nivel)
        k = st.columns(3)
        k[0].metric(f"VaR {nivel:.1%} 1 dia", f"{res['var']:,.2f}")
        k[1].metric("Expected Shortfall", f"{res['es']:,.2f}")
        k[2].metric("Cenários", res["n_cenarios"])
        fig = G.estilo(go.Figure(go.Histogram(x=res["pnls"], nbinsx=60, marker=dict(color=G.QUENTES[2]))),
                       "P&L histórico da carteira", 400, "R$", "dias")
        plot(fig, "var_hist")
