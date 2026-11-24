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
from renda_fixa.exibicao import br, csv_br, tabela, tem_vazios
from renda_fixa.derivativos import DI1, SwapDIIPCA, SwapDIPre, taxa_fra
from renda_fixa.mercado import (VERTICES_PADRAO, comparar_di1, curvas_para_grafico, historico_curvas, montar_mercado, tabela_curvas,
                                tabela_titulos, variacoes_diarias_bps)
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


def mostrar(df: pd.DataFrame, chave: str, formatos: dict | None = None, casas: int = 4, **kw):
    """Tabela no padrão brasileiro. Com vazios ("—") as colunas viram texto na tela, então vai junto um
    CSV com os números puros, que o Excel em português abre direto."""
    st.dataframe(tabela(df, formatos, casas), width="stretch", hide_index=True, **kw)
    if tem_vazios(df):
        st.download_button("Baixar CSV (números para Excel)", csv_br(df), f"{chave}_{REF:%Y%m%d}.csv",
                           "text/csv", key=f"csv_{chave}")


def eixo_x(fig, x_max: float):
    """Eixo de prazo a partir de zero (o automático chegava a -20 anos)."""
    fig.update_xaxes(range=[0, float(x_max) * 1.03], autorange=False)
    return fig


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


@st.cache_data(show_spinner="Baixando os ajustes da B3...", ttl=3600)
def ajustes_b3(ref: date) -> pd.DataFrame | None:
    from renda_fixa.dados.b3 import baixar_ajustes_b3

    try:
        return baixar_ajustes_b3(ref, CACHE / "b3")
    except DadoIndisponivel:
        return None


@st.cache_data(ttl=3600)
def cdi(inicio: date, fim: date) -> pd.Series:
    return bcb.serie_sgs(bcb.CDI_DIARIO, inicio - timedelta(days=5), fim, cache_dir=CACHE / "bcb")


# ------------------------------------------------------------------ cabeçalho
st.title("Renda Fixa Pública Brasil")
st.caption("Títulos públicos federais, curvas de juros, derivativos de juros e risco. Fonte: ANBIMA, B3, Banco Central e Tesouro Direto.")

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

# duas linhas de 3: com 6 lado a lado os valores cortavam em janelas estreitas
c = st.columns(3)
c[0].metric("Data", REF.strftime("%d/%m/%Y"))
c[1].metric("Títulos", len(tt))
c[2].metric("PU idêntico à ANBIMA", f"{int((tt['erro_pu'] == 0).sum())} de {len(tt)}")
c = st.columns(3)
c[0].metric("VNA LFT", br(m.vna_lft.vna, 6))
c[1].metric("VNA NTN-B", br(m.vna_ntnb.vna, 6))
c[2].metric("VNA NTN-C", br(m.vna_ntnc.vna, 6) if m.vna_ntnc else "—")

aba = st.tabs(["Títulos do dia", "Curvas", "Calculadora", "DI1 e swaps", "Risco de carteira"])

# ------------------------------------------------------------------ títulos
with aba[0]:
    st.subheader("Validação título a título")
    st.caption("PU do motor (convenções de truncamento do Tesouro e da ANBIMA) contra o PU indicativo da ANBIMA. "
               "Resíduo NSS: diferença de preço do título pela curva ajustada, em bps de taxa.")
    filtro = st.multiselect("Títulos", sorted(tt["titulo"].unique()), default=sorted(tt["titulo"].unique()))
    st.caption("LFT (pós-fixada) e NTN-C (um único título, IGP-M) ficam fora do ajuste NSS: "
               "pu_nss e residuo_nss_bps aparecem como \"—\".")
    mostrar(tt[tt["titulo"].isin(filtro)], "titulos",
            {"taxa_anbima": "{:.4f}", "taxa_motor": "{:.4f}", "pu_anbima": "{:,.6f}", "pu_motor": "{:,.6f}",
             "erro_pu": "{:.6f}", "duration_mac": "{:.4f}", "duration_mod": "{:.4f}", "dv01": "{:.4f}",
             "convexidade": "{:.2f}", "pu_nss": "{:,.6f}", "residuo_nss_bps": "{:.2f}"}, height=560)

# ------------------------------------------------------------------ curvas
with aba[1]:
    tc = curvas_para_grafico(m)
    anos = tc["anos"]
    pts_pre = tt[tt["titulo"].isin(["LTN", "NTN-F"])]
    pts_real = tt[tt["titulo"] == "NTN-B"]
    ate = lambda c_: anos[tc[c_].notna()].max()  # último prazo com valor na coluna
    st.caption("Curvas desenhadas só na faixa coberta pelos títulos (sem extrapolação). "
               "Pontos: taxa ANBIMA de cada título na sua duration.")
    col = st.columns(2)
    with col[0]:
        fig = G.figura([G.linha(1, "Bootstrap", anos, tc["pre"], "lines"), G.linha(3, "NSS", anos, tc["pre_nss"], "lines"),
                        G.linha(0, "LTN / NTN-F", pts_pre["duration_mac"], pts_pre["taxa_anbima"], "markers")],
                       "Curva prefixada", "anos", "% a.a.")
        plot(eixo_x(fig, ate("pre")), "curva_pre")
    with col[1]:
        fig = G.figura([G.linha(1, "Bootstrap", anos, tc["real"], "lines"), G.linha(3, "NSS", anos, tc["real_nss"], "lines"),
                        G.linha(0, "NTN-B", pts_real["duration_mac"], pts_real["taxa_anbima"], "markers")],
                       "Curva de juro real (IPCA)", "anos", "% a.a.")
        plot(eixo_x(fig, ate("real")), "curva_real")
    col = st.columns(2)
    with col[0]:
        fig = G.figura([G.linha(2, "Inflação implícita", anos, tc["inflacao_implicita"], "lines")],
                       "Inflação implícita", "anos", "% a.a.")
        plot(eixo_x(fig, ate("inflacao_implicita")), "breakeven")
    with col[1]:
        # a NSS dá a termo suave; a do bootstrap é serrilhada por construção (taxa constante entre títulos)
        fig = G.figura([G.linha(4, "NSS", anos, tc["termo_pre_1a_nss"], "lines"),
                        G.linha(1, "Bootstrap", anos, tc["termo_pre_1a"], "lines")],
                       "Taxa a termo de 1 ano (pré)", "início do período (anos)", "% a.a.")
        fig.data[1].update(opacity=0.45)
        plot(eixo_x(fig, ate("termo_pre_1a")), "termo")
    mostrar(tabela_curvas(m), "curvas")

    st.subheader("Curva de ajustes de DI1 (B3) x curva pré dos títulos")
    aj = ajustes_b3(REF)
    if aj is None:
        st.info(f"A B3 ainda não publicou o arquivo do pregão de {REF:%d/%m/%Y}.")
    else:
        cmp_di1 = comparar_di1(m, aj)
        dentro = cmp_di1[~cmp_di1["extrapolado"]]
        k = st.columns(4)
        k[0].metric("Contratos DI1", len(cmp_di1))
        k[1].metric("PU de ajuste reproduzido", f"{int((cmp_di1['erro_pu'] == 0).sum())} de {len(cmp_di1)}")
        k[2].metric("Spread médio", f"{br(dentro['spread_bps'].mean(), 1, True)} bps")
        k[3].metric("Faixa do spread",
                    f"{br(dentro['spread_bps'].min(), 1, True)} a {br(dentro['spread_bps'].max(), 1, True)} bps")
        col = st.columns(2)
        with col[0]:
            fig = G.figura([G.linha(1, "Títulos (bootstrap)", anos, tc["pre"], "lines"),
                            G.linha(0, "DI1 (ajuste B3)", cmp_di1["du"] / 252.0, cmp_di1["taxa_di1"], "lines+markers"),
                            G.linha(3, "Títulos (NSS)", anos, tc["pre_nss"], "lines")],
                           "DI1 x títulos prefixados", "anos", "% a.a.")
            plot(eixo_x(fig, max(ate("pre"), cmp_di1["du"].max() / 252.0)), "di1_vs_titulos")
        with col[1]:
            plot(G.figura([G.barras("DI1 - títulos (bps)", dentro["contrato"], dentro["spread_bps"])],
                          "Spread DI1 - títulos (+ = DI1 acima)", "", "bps"), "spread_di1")
        st.caption("Depois do último prefixado a curva dos títulos é extrapolada com taxa constante "
                   "(coluna extrapolado); esses contratos ficam fora do gráfico de spread.")
        mostrar(cmp_di1, "di1_b3")

# ------------------------------------------------------------------ calculadora
with aba[2]:
    col = st.columns([1, 1, 1, 1])
    tipo = col[0].selectbox("Título", ["LTN", "NTN-F", "NTN-B", "NTN-B Principal", "NTN-C", "LFT"])
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
    k[0].metric("PU", br(pu, 6))
    k[1].metric("Taxa", f"{br(taxa, 4)}%")
    k[2].metric("Prazo (DU)", a.du)
    k[3].metric("Duration mod.", br(a.duration_modificada, 4))
    k[4].metric("DV01 (R$/título)", br(a.dv01, 4))
    k[5].metric("Convexidade", br(a.convexidade, 2))
    fl = pd.DataFrame([{"data": f.data, "du": f.du, "fluxo_pct": f.valor,
                        "vp_pct": f.valor / (1 + taxa / 100) ** (f.du / 252)} for f in t.fluxos(REF)])
    if t.usa_vna:
        fl["fluxo_R$"] = fl["fluxo_pct"] * vna / 100
    else:
        fl["fluxo_R$"] = fl["fluxo_pct"] * 10
    mostrar(fl, "fluxos", {"fluxo_pct": "{:,.4f}", "vp_pct": "{:,.4f}", "fluxo_R$": "{:,.2f}"})

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
    mostrar(di, "di1_teorico", {"pu": "{:,.2f}", "dv01_R$": "{:,.4f}"})

    st.subheader("Taxas par")
    par = []
    for anos_ in (1, 2, 3, 5, 7, 10):
        v = proximo_dia_util(date(REF.year + anos_, REF.month, min(REF.day, 28)))
        par.append({"prazo": f"{anos_}a", "vencimento": v, "DI x Pré (100% CDI)": SwapDIPre.taxa_par(PRE, REF, v),
                    "DI x Pré (CDI + 1%)": SwapDIPre.taxa_par(PRE, REF, v, spread_cdi_aa=1.0),
                    "DI x IPCA (cupom)": SwapDIIPCA.cupom_par(PRE, REAL, REF, v)})
    mostrar(pd.DataFrame(par), "taxas_par")

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
    sw = None
    try:
        fator = bcb.fator_cdi(cdi(ini, REF), ini, REF, pct) if ini < REF else 1.0
        sw = SwapDIPre(noc, ini, ven, kpre, ativo=lado, pct_cdi=pct, spread_cdi_aa=spread)
        r = sw.avaliar(REF, PRE, fator)
        k = st.columns(5)
        k[0].metric("CDI acumulado", br(fator, 8))
        k[1].metric("VP ativo", br(r.vp_ativo))
        k[2].metric("VP passivo", br(r.vp_passivo))
        k[3].metric("MtM", br(r.mtm))
        k[4].metric("DV01", br(r.dv01))
    except Exception as e:  # SGS fora do ar
        st.error(f"Não foi possível avaliar o swap: {e}")

# ------------------------------------------------------------------ risco
with aba[4]:
    st.subheader("Carteira")
    st.caption("Títulos a mercado (z-spread sobre a curva do dia). Quantidade em títulos; negativa = vendido.")
    base = pd.DataFrame([
        # mesma carteira do notebook 03: LTN e NTN-B do meio da lista, NTN-F mais longa
        {"titulo": "LTN", "vencimento": sorted(m.df[m.df.titulo == "LTN"].vencimento)[(m.df.titulo == "LTN").sum() // 2],
         "quantidade": 20000},
        {"titulo": "NTN-F", "vencimento": m.df[m.df.titulo == "NTN-F"].vencimento.max(), "quantidade": 10000},
        {"titulo": "NTN-B", "vencimento": sorted(m.df[m.df.titulo == "NTN-B"].vencimento)[(m.df.titulo == "NTN-B").sum() // 2],
         "quantidade": 2000},
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
    if sw is not None and st.checkbox("Incluir o swap DI x Pré da aba \"DI1 e swaps\" (como no notebook 03)",
                                      value=True):
        carteira.adicionar(PosicaoSwap(sw, REF, fator))
    if not carteira.posicoes:
        st.stop()
    dv_pre, dv_real = carteira.dv01(REF, CURVAS, "PRE"), carteira.dv01(REF, CURVAS, "IPCA")
    k = st.columns(4)
    k[0].metric("Valor", br(carteira.valor(REF, CURVAS)))
    k[1].metric("DV01 pré", br(dv_pre))
    k[2].metric("DV01 real", br(dv_real))
    k[3].metric("Carry + roll-down 21 DU", br(carteira.carry_rolldown(REF, CURVAS, 21)))
    mostrar(carteira.dv01_por_posicao(REF, CURVAS), "carteira", casas=2)

    vert = [21, 126, 252, 504, 756, 1260, 1764, 2520, 3780, 5040]
    rot = [f"{br(v / 252, 1)}a" for v in vert]
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
    st.info(f"Hedge do DV01 pré com {contrato}: {br(abs(q), 0)} contratos "
            f"{'comprados em taxa (vendidos em PU)' if q < 0 else 'vendidos em taxa (comprados em PU)'}.")

    st.subheader("VaR histórico (curva pré, Tesouro Direto)")
    col = st.columns(3)
    anos_h = col[0].number_input("Anos de histórico", value=2.0, min_value=0.5, max_value=20.0, step=0.5)
    nivel = col[1].selectbox("Nível", [0.95, 0.975, 0.99], index=2, format_func=lambda x: f"{br(100 * x, 1)}%")
    if col[2].button("Calcular VaR"):
        hc = historico(REF, anos_h)
        choques = variacoes_diarias_bps(hc["PRE"])
        res = carteira.var_historico(REF, CURVAS, choques, "PRE", nivel)
        k = st.columns(3)
        k[0].metric(f"VaR {br(100 * nivel, 1)}% 1 dia", br(res["var"]))
        k[1].metric("Expected Shortfall", br(res["es"]))
        k[2].metric("Cenários", res["n_cenarios"])
        fig = G.estilo(go.Figure(go.Histogram(x=res["pnls"], nbinsx=60, marker=dict(color=G.QUENTES[2]))),
                       "P&L histórico da carteira", 400, "R$", "dias")
        plot(fig, "var_hist")
