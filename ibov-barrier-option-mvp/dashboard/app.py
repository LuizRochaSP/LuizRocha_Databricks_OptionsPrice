"""Dashboard Streamlit — Databricks Apps.

Roda em container dedicado e acessa arquivos do Workspace via SDK.
"""
import json
import os
from io import BytesIO, StringIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from databricks.sdk import WorkspaceClient

import desk
import formato_br
import mercado as mk
import multiativos
import pipeline

st.set_page_config(
    page_title="Dashboard Renda Fixa",
    page_icon="",
    layout="wide",
)
formato_br.instalar(st)  # números no padrão brasileiro em textos, tabelas e gráficos

# ---- CSS institucional (dark theme bank-grade) ----
st.markdown(
    """
    <style>
    /* Links e textos: sem o azul padrão do Streamlit */
    a, .stApp a, [data-testid="stMarkdownContainer"] a {
        color: #E8E9EB !important;
        text-decoration: none !important;
    }
    [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF;
    }

    /* Cartões de métricas: bordas finas, fundo escuro, números monoespaçados */
    div[data-testid="stMetric"],
    [data-testid="metric-container"] {
        background-color: #1E2128 !important;
        border: 1px solid #2C3038 !important;
        border-radius: 8px;
        padding: 12px 16px !important;
    }
    div[data-testid="stMetricValue"],
    [data-testid="stMetricValue"] {
        font-family: "Courier New", Courier, monospace !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] {
        color: #9CA3AF !important;
        font-size: 0.85em !important;
    }

    /* Botão Calcular Preço: azul institucional B3 */
    .stButton > button,
    div[data-testid="stButton"] > button {
        background-color: #00A3E0 !important;
        border: 1px solid #00A3E0 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover,
    div[data-testid="stButton"] > button:hover {
        background-color: #0091C7 !important;
        border-color: #0091C7 !important;
        color: #FFFFFF !important;
    }

    /* Remove o azul dos títulos das seções */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    /* Remove o azul das tags do multiselect */
    span[data-baseweb="tag"] {
        background-color: #2C3038 !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RESULTS_BASE_PATH = os.environ.get(
    "RESULTS_BASE_PATH",
    "/Workspace/Users/luiz.henrique.felipe.rocha@gmail.com/ibov-barrier-results",
)

SUPPORTED_INSTRUMENTS = {
    "option_barrier": {
        "label": "Opção com Barreira (IBOV)",
        "status": "active",
        "slug": "ibov",  # ?ativo=ibov na URL
        "description": "Call down-and-out sobre Ibovespa, precificada por Monte Carlo.",
    },
    "option_barrier_spx": {
        "label": "Opção com Barreira (S&P 500)",
        "status": "active",
        "slug": "spx",
        "description": "Superfície do SPX (notebook 07), Quanto/Compo em BRL e vol local (notebook 09).",
    },
    "option_vanilla": {
        "label": "Opção Europeia Simples",
        "status": "planned",
        "description": "Call/Put europeia sobre IBOV — em desenvolvimento.",
    },
    "swap_di_pre": {
        "label": "Swap DI x Pré",
        "status": "planned",
        "description": "Precificação de swap via curva DI — em desenvolvimento.",
    },
    "future_ind": {
        "label": "Futuro de Ibovespa",
        "status": "planned",
        "description": "Precificação e análise de futuros — em desenvolvimento.",
    },
}


@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()


@st.cache_data(ttl=300)
def list_paths(base_path: str):
    """Todos os arquivos da pasta de resultados, mais recente primeiro: uma chamada à API a cada 5 min,
    compartilhada por todas as listas abaixo (antes eram cinco)."""
    try:
        return sorted((e.path for e in get_workspace_client().workspace.list(base_path)), reverse=True)
    except Exception as e:
        st.error(f"Erro ao listar {base_path}: {e}")
        return []


def _com_prefixo(base_path: str, prefix: str, ext: str):
    return [p for p in list_paths(base_path) if p.split("/")[-1].startswith(prefix) and p.endswith(ext)]


def list_available_results(base_path: str):
    return _com_prefixo(base_path, "RESULTS_superficieVol_", ".csv")


def list_all_names(base_path: str):
    """Nomes de todos os arquivos da pasta de resultados (para checar RESULTS -> VALIDATION -> RISK)."""
    return [p.split("/")[-1] for p in list_paths(base_path)]


def list_results_by_prefix(base_path: str, prefix: str):
    """Arquivos RESULTS de outros mercados (07: SPX, 09: Quanto), mais recente primeiro."""
    return _com_prefixo(base_path, prefix, ".csv")


def list_available_risk(base_path: str):
    return _com_prefixo(base_path, "RISK_superficieVol_", ".json")


def limpar_caches():
    list_paths.clear()


def painel_pipeline(rodada_inicial):
    """Botão para rodar a cadeia toda e o andamento da rodada (atualiza sozinho enquanto roda)."""
    pl = pipeline.Pipeline(get_workspace_client(), os.environ.get("PIPELINE_JOB_NAME", pipeline.JOB_PADRAO),
                           os.environ.get("PIPELINE_JOB_ID"))
    try:
        if pl.job_id() is None:
            st.warning(f"O job `{pl.nome}` não existe ou o app não tem permissão para vê-lo. "
                       "Rode os notebooks 00 a 03 no folder de produção, nessa ordem.")
            return
        prog = rodada_inicial if rodada_inicial is not None else pl.ultima()
    except Exception as e:
        st.warning(f"Não foi possível consultar o job `{pl.nome}`: {e}")
        return

    if prog is not None and prog.ativo:
        @st.fragment(run_every=15)
        def andamento():
            atual = pl.acompanhar(prog.run_id)
            if not atual.ativo:
                limpar_caches()
                st.rerun(scope="app")
            inicio = (datetime.fromtimestamp(atual.inicio_ms / 1000).strftime("%H:%M")
                      if atual.inicio_ms else "—")
            etapas = pipeline.andamento_texto(atual.tarefas)
            st.info(f"Pipeline rodando desde {inicio} (rodada {atual.run_id}). {etapas}. "
                    "Esta área atualiza a cada 15 s e a página recarrega sozinha ao terminar.")
        andamento()
        return

    if prog is not None and prog.resultado and prog.resultado != "SUCCESS":
        resumo = pipeline.resumo_falha(prog.tarefas)
        st.error(f"A última rodada do pipeline ({prog.run_id}) não terminou: "
                 f"{resumo or pipeline.estado_pt(None, prog.resultado)}. "
                 f"Veja o log da etapa em Jobs > {pl.nome} no Databricks.")
    if st.button("Rodar pipeline completo (00 → 01 → 02 → 03)", key="rodar_pipeline"):
        try:
            run_id = pl.disparar()
            st.session_state["rodada_pipeline"] = run_id
            st.rerun()
        except Exception as e:
            st.error(f"Não foi possível iniciar o job `{pl.nome}`: {e}")


@st.cache_data(ttl=300)
def load_multiativos(file_path: str):
    frame = load_result(file_path)
    if frame is None:
        return None
    try:
        return multiativos.parse_bundle(frame)
    except Exception as e:
        st.error(f"Erro ao ler {file_path.split('/')[-1]}: {e}")
        return None


@st.cache_data(ttl=300)
def load_result(file_path: str):
    try:
        w = get_workspace_client()
        with w.workspace.download(file_path) as f:
            # BytesIO explícito: o handle do SDK (_StreamingResponse) tem .mode=None,
            # e o pandas tenta "b"in handle.mode ao detectar modo binário TypeError.
            return pd.read_csv(BytesIO(f.read()))
    except Exception as e:
        st.error(f"Erro ao carregar {file_path}: {e}")
        return None


def extract_json_object(bundle, name):
    rows = bundle.loc[bundle["object_name"] == name, "payload_json"]
    if len(rows) == 0:
        return None
    try:
        return json.loads(rows.iloc[0])
    except (json.JSONDecodeError, TypeError):
        return None


def extract_dataframe(bundle, name):
    rows = bundle.loc[bundle["object_name"] == name, "payload_json"]
    if len(rows) == 0:
        return None
    payload = rows.iloc[0]
    try:
        return pd.read_json(StringIO(payload), orient="table")
    except Exception:
        try:
            return pd.read_json(StringIO(payload))
        except Exception:
            return None


def load_risk_json(file_path: str):
    try:
        w = get_workspace_client()
        response = w.workspace.download(file_path)
        return json.loads(response.read().decode())
    except Exception as e:
        st.error(f"Erro ao carregar {file_path}: {e}")
        return None


@st.cache_resource(max_entries=4)
def mercado_ibov(file_path: str):
    """Spot, curvas e superfície do arquivo do 01 para o Desk (o arquivo não muda: o nome tem data e hora)."""
    b = load_result(file_path)
    if b is None:
        return None
    try:
        return mk.mercado_ibov(extract_json_object(b, "metadata"), extract_dataframe(b, "curve"),
                               extract_dataframe(b, "surface_data"))
    except Exception:
        return None


@st.cache_resource(max_entries=4)
def mercado_spx(file_path: str):
    """Spot, r(T), q(T) e superfície do arquivo do 07 para o Desk."""
    try:
        return mk.mercado_spx(load_multiativos(file_path))
    except Exception:
        return None


@st.cache_data(max_entries=4)
def grade_spx(file_path: str):
    """Grade da superfície 3D do SPX (interpolação feita uma vez por arquivo)."""
    b = load_multiativos(file_path)
    return multiativos.surface_grid(b["options"]) if b and "options" in b else None


def cabecalho():
    st.title("Quantitative Pricing & Risk Dashboard")
    st.markdown("**Motor de precificação, análise de risco e simulação de estruturas**")
    st.markdown("---")


def rodape(nome_arquivo: str):
    st.markdown("---")
    st.caption(f"{nome_arquivo} | Cache: 5 min | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Databricks Apps")


def pagina_spx():
    """S&P 500: só as abas do SPX rodam (o IBOV não é carregado nem desenhado)."""
    spx_files = list_results_by_prefix(RESULTS_BASE_PATH, multiativos.SPX_PREFIX)
    quanto_files = list_results_by_prefix(RESULTS_BASE_PATH, multiativos.QUANTO_PREFIX)
    spx_path = None
    if spx_files:
        spx_path = st.sidebar.selectbox(
            "Arquivo RESULTS", options=spx_files, key="spx_file",
            format_func=lambda p: multiativos.file_label(p, multiativos.SPX_PREFIX),
            help="Execuções do notebook 07 (superfície do SPX), mais recente primeiro.",
        )
    else:
        st.sidebar.warning("Nenhum resultado do notebook 07")
    b = load_multiativos(spx_path) if spx_path else None

    cabecalho()
    if b is not None:
        st.success(f"Resultado carregado: **{spx_path.split('/')[-1]}**")
    else:
        st.info(multiativos.SEM_SPX + " O Structuring Desk funciona com valores de reserva.")

    tabs = st.tabs(["Resumo do Gestor", "Superfície e Smiles", "Taxa e Dividendos",
                    "Quanto / Compo (BRL)", "Vol Local", "Structuring Desk"])
    with tabs[0]:
        multiativos.render_spx_resumo(st, b, spx_files, load_multiativos)
    with tabs[1]:
        multiativos.render_spx_superficie(st, b, grade_spx(spx_path) if b is not None else None)
    with tabs[2]:
        multiativos.render_spx_taxas(st, b)
    with tabs[3]:
        multiativos.render_quanto(st, quanto_files, load_multiativos)
    with tabs[4]:
        multiativos.render_local_vol(st, quanto_files, load_multiativos)
    with tabs[5]:
        ref = mk.referencia_quanto(load_multiativos(quanto_files[0])) if quanto_files else None
        desk.render(desk.SPX, mercado_spx(spx_path) if b is not None else None, ref)
    rodape(spx_path.split("/")[-1] if spx_path else "sem arquivo do SPX")


st.sidebar.header("Configurações")

# O instrumento vem primeiro: ele decide quais arquivos e abas aparecem. ?ativo=spx abre direto no S&P 500.
st.sidebar.markdown("**Tipo de Instrumento**")
ativos = [k for k, v in SUPPORTED_INSTRUMENTS.items() if v["status"] == "active"]
instrument_choice = st.sidebar.selectbox(
    "Selecione o instrumento",
    options=ativos,
    index=next((i for i, k in enumerate(ativos)
                if SUPPORTED_INSTRUMENTS[k]["slug"] == st.query_params.get("ativo")), 0),
    format_func=lambda x: SUPPORTED_INSTRUMENTS[x]["label"],
    key="instrumento",
)
st.query_params["ativo"] = SUPPORTED_INSTRUMENTS[instrument_choice]["slug"]

if instrument_choice == "option_barrier_spx":
    pagina_spx()
    st.stop()

# ---- IBOV ----
available_files = list_available_results(RESULTS_BASE_PATH)

if not available_files:
    st.sidebar.error("Nenhum RESULTS encontrado")
    st.error(
        f"**Nenhum arquivo RESULTS em** `{RESULTS_BASE_PATH}`\n\n"
        "Execute `01_main_option_pricer.py` primeiro."
    )
    st.stop()

def ts_of(path: str) -> str:
    return path.split("/")[-1].replace("RESULTS_superficieVol_", "").replace(".csv", "")


# completude de cada execução: RESULTS (01) -> VALIDATION (02) -> RISK (03)
exec_status = pipeline.status_execucoes(list_all_names(RESULTS_BASE_PATH))
available_files = sorted(available_files, key=ts_of, reverse=True)
default_ts = exec_status["ts"].iloc[pipeline.indice_padrao(exec_status)] if len(exec_status) else None
selected_file_path = st.sidebar.selectbox(
    "Arquivo RESULTS",
    options=available_files,
    index=next((i for i, f in enumerate(available_files) if ts_of(f) == default_ts), 0),
    format_func=lambda x: pipeline.rotulo(ts_of(x), exec_status),
    help="Abre na execução completa mais recente (com validação do 02 e risco do 03).",
)

bundle = load_result(selected_file_path)

if bundle is None:
    st.stop()

selected_filename = selected_file_path.split("/")[-1]

cabecalho()

st.success(f"Resultado carregado: **{selected_filename}**")

# informação incompleta: diz o que falta e oferece rodar a cadeia toda
latest_ts = ts_of(available_files[0])
falta_sel = pipeline.motivo(ts_of(selected_file_path), exec_status)
falta_ult = pipeline.motivo(latest_ts, exec_status) if latest_ts != ts_of(selected_file_path) else None
rodada = st.session_state.get("rodada_pipeline")
if falta_sel or falta_ult or rodada:
    if falta_sel:
        st.error(f"Informação incompleta. {falta_sel}")
    elif falta_ult:
        st.warning(f"A execução mais recente está incompleta e não foi aberta. {falta_ult}")
    try:
        prog_inicial = (pipeline.Pipeline(get_workspace_client(),
                                          os.environ.get("PIPELINE_JOB_NAME", pipeline.JOB_PADRAO),
                                          os.environ.get("PIPELINE_JOB_ID")).acompanhar(rodada)
                        if rodada else None)
    except Exception:
        prog_inicial = None
    if prog_inicial is not None and not prog_inicial.ativo:
        st.session_state.pop("rodada_pipeline", None)
        if prog_inicial.resultado == "SUCCESS":
            st.success(f"Pipeline concluído (rodada {prog_inicial.run_id}): a execução nova já está na lista.")
            prog_inicial = None
    if falta_sel or falta_ult or (prog_inicial is not None and prog_inicial.ativo):
        painel_pipeline(prog_inicial)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Resumo do Gestor", "Superfície e Volatilidade", "Curvas de Mercado",
     "Motor de Precificação", "Controle e Ingestão", "Structuring Desk"])

with tab1:
    st.header("Resumo do Pricer")

    metadata = extract_json_object(bundle, "metadata")
    pricing = extract_json_object(bundle, "pricing_result")

    if metadata and pricing:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Data de Mercado", metadata.get("market_date", "—"))
        col2.metric("Spot IBOV", f"{metadata.get('spot', 0):,.2f}")
        col3.metric("Strike Alvo", f"{metadata.get('target_strike', 0):,.2f}")
        col4.metric("Vol. Implícita", f"{metadata.get('target_iv', 0):.2%}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Preço Vanilla (BS)", f"{pricing.get('vanilla_price', 0):,.2f}")
        col6.metric("Preço Down-and-Out", f"{pricing.get('barrier_price', 0):,.2f}")
        col7.metric("Desconto Barreira", f"{pricing.get('barrier_discount_pct', 0):.2%}")
        col8.metric("P(knock-out)", f"{pricing.get('knock_out_probability', 0):.2%}")

        if "analytical_price"in pricing:
            col9, col10 = st.columns(2)
            col9.metric("Preço Analítico (Reiner-Rubinstein)", f"{pricing.get('analytical_price', 0):,.2f}")
            col10.metric("Erro MC vs Analítico", f"{pricing.get('analytical_error_pct', 0):.4%}")

        with st.expander("Ver parâmetros do contrato e mercado"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Contrato**")
                st.json(pricing.get("contract", {}))
            with col_b:
                st.markdown("**Mercado**")
                st.json(pricing.get("market", {}))

    st.markdown("---")
    st.header("Risco & Gregas")

    selected_ts = (
        selected_file_path.split("/")[-1]
        .replace("RESULTS_superficieVol_", "")
        .replace(".csv", "")
    )
    risk_files = list_available_risk(RESULTS_BASE_PATH)
    risk_match = next(
        (p for p in risk_files if selected_ts in p.split("/")[-1]), None
    )

    if risk_match is None:
        st.error(
            (pipeline.motivo(selected_ts, exec_status) or "Nenhum arquivo RISK correspondente a este RESULTS.")
            + "\n\nUse o botão \"Rodar pipeline completo\" no topo da página, ou escolha na barra lateral "
            "uma execução marcada como completa."
        )
    else:
        risk = load_risk_json(risk_match)
        if risk:
            m = risk.get("metrics", {})

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Preço (MC)", f"{m.get('price', 0):,.2f}")
            col_r2.metric("Delta", f"{m.get('delta', 0):.4f}")
            col_r3.metric("Gamma", f"{m.get('gamma', 0):.6f}")
            col_r4.metric("Vega (1%)", f"{m.get('vega_1pct', 0):,.2f}")

            col_r5, col_r6, col_r7, col_r8 = st.columns(4)
            col_r5.metric("Theta (1d)", f"{m.get('theta_1day', 0):,.2f}")
            col_r6.metric("Rho (1%)", f"{m.get('rho_1pct', 0):,.2f}")
            col_r7.metric("DV01", f"{m.get('dv01', 0):.4f}")
            col_r8.metric("Convexidade (1bp)", f"{m.get('rate_convexity_1bp', 0):.6f}")

            st.markdown("---")
            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            col_v1.metric("VaR 99% (1d)", f"{m.get('var', 0):,.2f}")
            col_v2.metric("Expected Shortfall", f"{m.get('expected_shortfall', 0):,.2f}")
            col_v3.metric("PnL médio", f"{m.get('pnl_mean', 0):,.2f}")
            col_v4.metric("PnL std", f"{m.get('pnl_std', 0):,.2f}")

            st.caption(
                f"VaR/ES: {m.get('confidence_level', 0.99):.0%} de confiança, horizonte de "
                f"{m.get('horizon_days', 1)} dia(s), {m.get('scenarios', 0):,} cenários — "
                f"{m.get('methodology', '—')}"
            )

            limitations = risk.get("limitations", [])
            if limitations:
                with st.expander("Limitações do cálculo de risco"):
                    for lim in limitations:
                        st.markdown(f"- {lim}")

            st.caption(
                f"Validação: {risk.get('validation_decision', '—')} "
                f"({risk.get('validation_warn_count', 0)} ressalvas) | "
                f"Fonte: `{risk_match.split('/')[-1]}`"
            )

    st.markdown("---")
    st.header("Histórico de Execuções")

    if len(available_files) > 1:
        selected_hist = st.multiselect(
            "Comparar múltiplas execuções",
            options=available_files,
            default=available_files[:2],
            format_func=lambda x: x.split("/")[-1].replace("RESULTS_superficieVol_", "").replace(".csv", ""),
        )

        if selected_hist:
            comparison_data = []
            for fpath in selected_hist:
                bundle_hist = load_result(fpath)
                if bundle_hist is None:
                    continue
                meta_hist = extract_json_object(bundle_hist, "metadata")
                price_hist = extract_json_object(bundle_hist, "pricing_result")
                if meta_hist and price_hist:
                    comparison_data.append({
                        "Arquivo": fpath.split("/")[-1],
                        "Data": meta_hist.get("market_date", "—"),
                        "Spot": meta_hist.get("spot", 0),
                        "Vanilla": price_hist.get("vanilla_price", 0),
                        "DAO": price_hist.get("barrier_price", 0),
                        "Desconto %": price_hist.get("barrier_discount_pct", 0),
                        "P(KO) %": price_hist.get("knock_out_probability", 0),
                    })

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)

                if len(comparison_df) > 1:
                    # eixo por execução (data e hora do arquivo): várias execuções no mesmo dia de
                    # mercado ficavam empilhadas num eixo de datas com marcações em milissegundos
                    stamps = comparison_df["Arquivo"].str.extract(r"(\d{8}_\d{6})")[0]
                    comparison_df["Execução"] = pd.to_datetime(stamps, format="%Y%m%d_%H%M%S",
                                                               errors="coerce").dt.strftime("%d/%m %H:%M")
                    comparison_df = comparison_df.sort_values("Arquivo")
                    fig_comp = px.line(
                        comparison_df, x="Execução", y=["Vanilla", "DAO"],
                        markers=True, title="Evolução de Preços",
                        labels={"value": "Preço (pontos)", "variable": "Tipo"},
                        hover_data=["Data"],
                    )
                    fig_comp.update_xaxes(type="category")
                    st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Apenas 1 arquivo disponível.")

with tab2:
    st.header("Superfície de Volatilidade Implícita")

    iv_data = extract_dataframe(bundle, "iv_data")

    if iv_data is not None and len(iv_data) > 0:
        iv_data = iv_data.copy()
        iv_data["T_anos"] = iv_data.get("T", 0.5)

        x_col = "log_moneyness"if "log_moneyness"in iv_data.columns else "strike"
        z_col = "implied_vol"if "implied_vol"in iv_data.columns else "volatility"

        if x_col in iv_data.columns and z_col in iv_data.columns:
            # Superfície 3D real (MATLAB style): interpolação cúbica sobre grade regular
            from scipy.interpolate import griddata

            x_vals = iv_data[x_col].to_numpy(dtype=float)
            y_vals = iv_data["T_anos"].to_numpy(dtype=float)
            z_vals = iv_data[z_col].to_numpy(dtype=float)

            xi = np.linspace(x_vals.min(), x_vals.max(), 80)
            yi = np.linspace(y_vals.min(), y_vals.max(), 80)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x_vals, y_vals), z_vals, (X, Y), method="cubic")

            fig_3d = go.Figure(
                data=[
                    go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale="Viridis",
                        colorbar=dict(title="IV", tickformat=".1%"),
                        contours=dict(
                            z=dict(show=True, usecolormap=True, project_z=True)
                        ),
                    )
                ]
            )
            fig_3d.update_layout(
                title="Superfície de Volatilidade Implícita",
                scene=dict(
                    xaxis_title="Log-moneyness ln(K/F)",
                    yaxis_title="Prazo (anos)",
                    zaxis_title="IV",
                    zaxis=dict(tickformat=".1%"),
                ),
                height=600,
            )
            st.plotly_chart(
                fig_3d,
                use_container_width=True,
                config={
                    'displayModeBar': True, # Mostra a barra de ferramentas (com botão de expandir)
                    'scrollZoom': True, # Permite dar zoom usando o scroll do mouse
                    'displaylogo': False, # Remove o logo do Plotly da barra
                    'modeBarButtonsToRemove': ['sendDataToCloud'] # Limpa botões inúteis
                }
            )

            col_x, col_y = st.columns(2)
            col_x.metric("Opções com IV válido", len(iv_data))
            col_y.metric("Vencimentos distintos", iv_data["T_anos"].nunique())

    st.markdown("---")
    st.header("Smile de Volatilidade por Vencimento")

    if iv_data is not None and len(iv_data) > 0:
        smile_df = iv_data.copy()
        group_col = None
        if "maturity_date"in smile_df.columns:
            group_col = "maturity_date"
        elif "T"in smile_df.columns:
            smile_df["T_group"] = (smile_df["T"] * 10).round() / 10
            group_col = "T_group"

        k_col = "log_moneyness"if "log_moneyness"in smile_df.columns else "strike"
        v_col = "implied_vol"if "implied_vol"in smile_df.columns else "volatility"
        type_col = "option_type"if "option_type"in smile_df.columns else None

        if group_col and k_col in smile_df.columns and v_col in smile_df.columns:
            if type_col:
                types_available = sorted(smile_df[type_col].dropna().unique().tolist())
                selected_types = st.multiselect(
                    "Filtrar por tipo de opção",
                    options=types_available,
                    default=types_available,
                )
                if selected_types:
                    smile_df = smile_df[smile_df[type_col].isin(selected_types)]

            counts = smile_df.groupby(group_col)[k_col].count()
            valid_groups = counts[counts >= 3].index.tolist()

            if valid_groups:
                smile_filtered = smile_df[smile_df[group_col].isin(valid_groups)].copy()
                if group_col == "maturity_date":  # legenda só com a data (sem 00:00:00)
                    smile_filtered[group_col] = smile_filtered[group_col].astype(str).str[:10]
                fig_smile = px.line(
                    smile_filtered.sort_values(k_col),
                    x=k_col, y=v_col, color=group_col, markers=True,
                    title="Smile de Volatilidade Implícita por Vencimento",
                    labels={k_col: "Log-moneyness ln(K/F)", v_col: "IV", group_col: "Vencimento"},
                )
                fig_smile.update_traces(line_width=2, marker_size=7)
                fig_smile.update_layout(
                    yaxis_tickformat=".1%", hovermode="x unified", height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                )
                fig_smile.add_vline(
                    x=0.0, line_dash="dash", line_color="gray", opacity=0.5,
                    annotation_text="ATM", annotation_position="top",
                )
                st.plotly_chart(fig_smile, use_container_width=True)

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Vencimentos no smile", len(valid_groups))
                col_b.metric("Strikes totais", len(smile_filtered))
                col_c.metric(
                    "IV Mín / Máx",
                    f"{smile_filtered[v_col].min():.2%} / {smile_filtered[v_col].max():.2%}",
                )

                with st.expander("Ver dados do smile"):
                    st.dataframe(smile_filtered, use_container_width=True)
            else:
                st.info("Nenhum vencimento com 3 strikes.")

with tab3:
    st.header("Curva de Juros (DI)")

    curve_df = extract_dataframe(bundle, "curve")
    y_col = None

    if curve_df is not None and len(curve_df) > 0:
        curve_df = curve_df.copy()
        if "T"in curve_df.columns:
            curve_df["T_plot"] = curve_df["T"]
        elif "tenor_bd"in curve_df.columns:
            curve_df["T_plot"] = curve_df["tenor_bd"] / 252.0
        else:
            curve_df["T_plot"] = range(len(curve_df))

        for candidate in ["r_cont", "r", "rate"]:
            if candidate in curve_df.columns:
                y_col = candidate
                break

        if y_col:
            fig = px.line(
                curve_df, x="T_plot", y=y_col, markers=True,
                title="Curva Prefixada (r contínua)",
                labels={"T_plot": "Prazo (anos)", y_col: "Taxa contínua (% a.a.)"},
            )
            fig.update_traces(line_color="#1f77b4", marker_size=8)
            fig.update_layout(yaxis_tickformat=".2%", hovermode="x unified", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Coluna de taxa não encontrada.")

    st.markdown("---")
    st.header("Dividend Yield Implícito q(T)")

    if curve_df is not None and len(curve_df) > 0:
        q_col = None
        for candidate in ["q_cont", "q", "dividend_yield"]:
            if candidate in curve_df.columns:
                q_col = candidate
                break

        if q_col:
            fig_q = px.line(
                curve_df, x="T_plot", y=q_col, markers=True,
                title="Dividend Yield Implícito (q contínuo)",
                labels={"T_plot": "Prazo (anos)", q_col: "q (% a.a.)"},
            )
            fig_q.update_traces(line_color="#ff7f0e", marker_size=8)
            fig_q.update_layout(yaxis_tickformat=".2%", hovermode="x unified", height=400)
            st.plotly_chart(fig_q, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("q Médio", f"{curve_df[q_col].mean():.4%}")
            col2.metric("q Mínimo", f"{curve_df[q_col].min():.4%}")
            col3.metric("q Máximo", f"{curve_df[q_col].max():.4%}")

            st.info(
                f"**Limitação conhecida:** q(T) médio de {curve_df[q_col].mean():.2%} a.a. está abaixo do "
                "histórico NEFIN/USP (5-7%). Ver README."
            )

with tab4:
    st.header("Diagnóstico da Simulação Monte Carlo")

    sim_diag = extract_json_object(bundle, "simulation_diagnostics")
    if sim_diag:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Caminhos", f"{sim_diag.get('paths', 0):,}")
        col2.metric("Passos", f"{sim_diag.get('steps', 0):,}")
        col3.metric("Seed", sim_diag.get("seed", "—"))
        col4.metric("Monitoramento", sim_diag.get("monitoring", "—"))

        if pricing:
            col5, col6, col7 = st.columns(3)
            col5.metric("Erro-padrão", f"{pricing.get('standard_error', 0):,.4f}")
            col6.metric("IC 95% (inf)", f"{pricing.get('ci_low_95', 0):,.2f}")
            col7.metric("IC 95% (sup)", f"{pricing.get('ci_high_95', 0):,.2f}")

    st.markdown("---")
    st.subheader("Visualização da Simulação Monte Carlo")

    if pricing and sim_diag:
        try:
            market_block = pricing.get("market", {})
            contract_block = pricing.get("contract", {})

            spot = float(market_block.get("spot", 0))
            r = float(market_block.get("rate", 0))
            q_val = float(market_block.get("dividend_yield", 0))
            sigma = float(market_block.get("volatility", 0))
            strike = float(contract_block.get("strike", 0))
            barrier = float(contract_block.get("barrier", 0))
            maturity = float(contract_block.get("maturity", 0.5))
            seed = int(sim_diag.get("seed", 42))
            monitoring = str(sim_diag.get("monitoring", "brownian_bridge"))

            if spot > 0 and sigma > 0 and maturity > 0:
                st.markdown("**Caminhos simulados (amostra de 50)**")

                n_paths_plot = 50
                n_steps_plot = int(sim_diag.get("steps", 126))
                dt_plot = maturity / n_steps_plot

                rng = np.random.default_rng(seed + 1)
                z_plot = rng.standard_normal((n_paths_plot, n_steps_plot))
                drift = (r - q_val - 0.5 * sigma ** 2) * dt_plot
                diffusion = sigma * np.sqrt(dt_plot)
                log_paths_plot = np.log(spot) + np.cumsum(drift + diffusion * z_plot, axis=1)
                paths_plot = np.exp(log_paths_plot)
                paths_with_t0 = np.concatenate(
                    [np.full((n_paths_plot, 1), spot), paths_plot], axis=1
                )
                time_grid = np.linspace(0, maturity, n_steps_plot + 1)

                fig_paths = go.Figure()
                for i in range(n_paths_plot):
                    fig_paths.add_trace(go.Scatter(
                        x=time_grid, y=paths_with_t0[i], mode="lines",
                        line=dict(color="rgba(31,119,180,0.3)", width=1),
                        showlegend=False, hoverinfo="skip",
                    ))
                # rótulos dentro da área do gráfico (à direita eles saíam cortados); spot em branco,
                # já que a linha preta sumia no fundo escuro
                fig_paths.add_hline(y=spot, line_dash="solid", line_color="white",
                                    annotation_text=f"Spot {spot:,.0f}", annotation_position="top left")
                fig_paths.add_hline(y=strike, line_dash="dash", line_color="orange",
                                    annotation_text=f"Strike {strike:,.0f}", annotation_position="top left")
                fig_paths.add_hline(y=barrier, line_dash="dot", line_color="red",
                                    annotation_text=f"Barreira {barrier:,.0f}", annotation_position="bottom left")
                fig_paths.update_layout(
                    title="Trajetórias do IBOV sob medida neutra ao risco",
                    xaxis_title="Tempo (anos)", yaxis_title="Spot IBOV (pontos)",
                    height=450, showlegend=False,
                )
                st.plotly_chart(fig_paths, use_container_width=True)

                st.caption(
                    f"Monitoramento: **{monitoring}** | "
                    f"GBM com r={r:.2%}, q={q_val:.2%}, σ={sigma:.2%}"
                )

                col_conv, col_hist = st.columns(2)

                with col_conv:
                    st.markdown("**Convergência do preço MC**")
                    n_paths_conv = 5000
                    rng_conv = np.random.default_rng(seed)
                    z_conv = rng_conv.standard_normal((n_paths_conv, n_steps_plot))
                    uniforms_conv = rng_conv.random((n_paths_conv, n_steps_plot))

                    drift_c = (r - q_val - 0.5 * sigma ** 2) * dt_plot
                    diff_c = sigma * np.sqrt(dt_plot)
                    log_paths_c = np.log(spot) + np.cumsum(drift_c + diff_c * z_conv, axis=1)

                    prev = np.concatenate(
                        (np.full((n_paths_conv, 1), np.log(spot)), log_paths_c[:, :-1]), axis=1
                    )
                    log_barrier = np.log(barrier)
                    if monitoring == "discrete":
                        alive_c = np.all(log_paths_c > log_barrier, axis=1)
                    else:
                        above = (prev > log_barrier) & (log_paths_c > log_barrier)
                        cross_prob = np.ones_like(log_paths_c)
                        cross_prob[above] = np.exp(
                            -2.0 * (prev[above] - log_barrier) * (log_paths_c[above] - log_barrier)
                            / (sigma ** 2 * dt_plot)
                        )
                        alive_c = np.all(above & (uniforms_conv > cross_prob), axis=1)

                    terminal = np.exp(log_paths_c[:, -1])
                    disc = np.exp(-r * maturity)
                    payoffs = disc * np.maximum(terminal - strike, 0.0) * alive_c

                    running_mean = np.cumsum(payoffs) / np.arange(1, n_paths_conv + 1)
                    x_paths = np.arange(1, n_paths_conv + 1)

                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=x_paths, y=running_mean, mode="lines",
                        line=dict(color="#1f77b4", width=2), name="Preço MC",
                    ))
                    fig_conv.add_hline(
                        y=pricing.get("barrier_price", 0),
                        line_dash="dash", line_color="green",
                        annotation_text=f"Final {pricing.get('barrier_price', 0):,.2f}",
                        annotation_position="right",
                    )
                    fig_conv.update_layout(
                        xaxis_title="Nº de caminhos", yaxis_title="Preço estimado",
                        height=380, showlegend=False,
                        margin=dict(l=40, r=40, t=20, b=40),
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)

                with col_hist:
                    st.markdown("**Distribuição de payoffs**")
                    payoffs_ativos = payoffs[payoffs > 0]
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=payoffs_ativos, nbinsx=50,
                        marker_color="rgba(31,119,180,0.7)",
                    ))
                    fig_hist.add_vline(
                        x=float(payoffs.mean()), line_dash="dash", line_color="red",
                        annotation_text=f"Média {payoffs.mean():,.2f}",
                        annotation_position="top",
                    )
                    fig_hist.update_layout(
                        xaxis_title="Payoff descontado", yaxis_title="Frequência",
                        height=380, showlegend=False,
                        margin=dict(l=40, r=40, t=20, b=40),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                col_mc1, col_mc2, col_mc3, col_mc4 = st.columns(4)
                col_mc1.metric("Payoffs > 0", f"{(payoffs > 0).mean():.2%}")
                col_mc2.metric("Payoff médio", f"{payoffs.mean():,.2f}")
                col_mc3.metric("Payoff máx", f"{payoffs.max():,.2f}")
                col_mc4.metric("Payoff std", f"{payoffs.std():,.2f}")
        except Exception as e:
            st.warning(f"Erro ao gerar visualização MC: {e}")

with tab5:
    st.header("Validação e Qualidade")

    if pricing:
        market_block = pricing.get("market", {})
        contract_block = pricing.get("contract", {})

        forward_stored = market_block.get("forward", 0)
        spot = market_block.get("spot", 0)
        rate = market_block.get("rate", 0)
        q_val = market_block.get("dividend_yield", 0)
        maturity = contract_block.get("maturity", 1)

        forward_recalc = spot * np.exp((rate - q_val) * maturity) if spot > 0 else 0
        forward_ok = abs(forward_stored - forward_recalc) < 0.01

        barrier_price = pricing.get("barrier_price", 0)
        vanilla_price = pricing.get("vanilla_price", 0)
        se = pricing.get("standard_error", 0)
        ko = pricing.get("knock_out_probability", 0)
        ci_low = pricing.get("ci_low_95", 0)
        ci_high = pricing.get("ci_high_95", 0)

        checks = {
            "Preço não-negativo": barrier_price >= 0,
            "Barreira Vanilla + 5×SE": barrier_price <= vanilla_price + (5.0 * se),
            "P(knock-out) em [0; 1]": 0 <= ko <= 1,
            "Erro-padrão não-negativo": se >= 0,
            "IC contém preço": ci_low <= barrier_price <= ci_high,
            "Forward reconstruído": forward_ok,
            "Erro MC vs Analítico < 1%": pricing.get("analytical_error_pct", 0) < 0.01,
        }

        pass_count = fail_count = 0
        for label, result in checks.items():
            if result:
                st.success(f"{label}")
                pass_count += 1
            else:
                st.error(f"{label}")
                fail_count += 1

        col1, col2, col3 = st.columns(3)
        col1.metric("PASS", pass_count)
        col2.metric("FAIL", fail_count)
        total = pass_count + fail_count
        col3.metric("Taxa", f"{pass_count / total * 100:.1f}%"if total > 0 else "—")

    st.markdown("---")
    st.header("Exportar Dados")
    market_date = metadata.get("market_date", "unknown") if metadata else "unknown"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="Download RESULTS (CSV)",
            data=bundle.to_csv(index=False),
            file_name=f"results_{market_date}.csv",
            mime="text/csv",
        )
    with col2:
        if curve_df is not None and len(curve_df) > 0:
            st.download_button(
                label="Download Curva DI (CSV)",
                data=curve_df.to_csv(index=False),
                file_name=f"curve_{market_date}.csv",
                mime="text/csv",
            )
    with col3:
        if iv_data is not None and len(iv_data) > 0:
            st.download_button(
                label="Download IV Data (CSV)",
                data=iv_data.to_csv(index=False),
                file_name=f"iv_{market_date}.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.header("Dados Brutos")
    with st.expander("Ver bundle RESULTS completo"):
        st.dataframe(bundle, use_container_width=True)
    with st.expander("Ver lista de objetos no bundle"):
        st.write(sorted(bundle["object_name"].unique()))

with tab6:
    # Structuring Desk (desk.py): o mesmo simulador do S&P 500, com valores iniciais e o modo
    # "dados de mercado do dia" vindos deste arquivo do 01
    desk.render(desk.IBOV, mercado_ibov(selected_file_path))

rodape(selected_filename)
