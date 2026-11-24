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
import paineis
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


# Pipeline de cada ativo: job (nome por variável de ambiente), botão e chave da rodada na sessão.
FLUXOS = {
    "ibov": {"job_env": "PIPELINE_JOB_NAME", "job_id_env": "PIPELINE_JOB_ID", "job_padrao": pipeline.JOB_PADRAO,
             "botao": "Rodar pipeline completo (00 → 01 → 02 → 03)", "botao_key": "rodar_pipeline",
             "chave": "rodada_pipeline", "manual": "Rode os notebooks 00 a 03 no folder de produção, nessa ordem."},
    "spx": {"job_env": "PIPELINE_JOB_NAME_SPX", "job_id_env": "PIPELINE_JOB_ID_SPX", "job_padrao": pipeline.JOB_SPX,
            "botao": "Rodar pipeline do S&P 500 (07 → 11 → 03, e o 09)", "botao_key": "rodar_pipeline_spx",
            "chave": "rodada_pipeline_spx",
            "manual": "Rode os notebooks 07, 11, 03 (ativo=spx) e 09 no folder de produção, nessa ordem."},
}


def _pipeline(fluxo: str) -> pipeline.Pipeline:
    f = FLUXOS[fluxo]
    return pipeline.Pipeline(get_workspace_client(), os.environ.get(f["job_env"], f["job_padrao"]),
                             os.environ.get(f["job_id_env"]))


def aviso_execucao(fluxo: str, falta_sel: str | None, falta_ult: str | None):
    """Informação incompleta: diz o que falta e oferece rodar a cadeia toda do ativo."""
    chave = FLUXOS[fluxo]["chave"]
    rodada = st.session_state.get(chave)
    if not (falta_sel or falta_ult or rodada):
        return
    if falta_sel:
        st.error(f"Informação incompleta. {falta_sel}")
    elif falta_ult:
        st.warning(f"A execução mais recente está incompleta e não foi aberta. {falta_ult}")
    try:
        prog_inicial = _pipeline(fluxo).acompanhar(rodada) if rodada else None
    except Exception:
        prog_inicial = None
    if prog_inicial is not None and not prog_inicial.ativo:
        st.session_state.pop(chave, None)
        if prog_inicial.resultado == "SUCCESS":
            st.success(f"Pipeline concluído (rodada {prog_inicial.run_id}): a execução nova já está na lista.")
            prog_inicial = None
    if falta_sel or falta_ult or (prog_inicial is not None and prog_inicial.ativo):
        painel_pipeline(prog_inicial, fluxo)


def painel_pipeline(rodada_inicial, fluxo: str = "ibov"):
    """Botão para rodar a cadeia toda e o andamento da rodada (atualiza sozinho enquanto roda)."""
    f = FLUXOS[fluxo]
    pl = _pipeline(fluxo)
    try:
        if pl.job_id() is None:
            st.warning(f"O job `{pl.nome}` não existe ou o app não tem permissão para vê-lo. {f['manual']}")
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
    if st.button(f["botao"], key=f["botao_key"]):
        try:
            run_id = pl.disparar()
            st.session_state[f["chave"]] = run_id
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


def _ts_spx(path: str) -> str:
    return path.split("/")[-1].replace(multiativos.SPX_PREFIX, "").replace(".csv", "")


def _irmao(path: str, tipo: str) -> str | None:
    """VALIDATION_/RISK_ do mesmo carimbo de um RESULTS (None se o arquivo não existe na pasta)."""
    nome = path.split("/")[-1].replace("RESULTS_", f"{tipo}_").replace(".csv", ".json")
    return next((p for p in list_paths(RESULTS_BASE_PATH) if p.split("/")[-1] == nome), None)


def pagina_spx():
    """S&P 500: só as abas do SPX rodam (o IBOV não é carregado nem desenhado)."""
    spx_files = sorted(list_results_by_prefix(RESULTS_BASE_PATH, multiativos.SPX_PREFIX), key=_ts_spx, reverse=True)
    quanto_files = list_results_by_prefix(RESULTS_BASE_PATH, multiativos.QUANTO_PREFIX)
    # completude: RESULTS (07) -> VALIDATION (11) -> RISK (03 com ativo=spx)
    status = pipeline.status_execucoes(list_all_names(RESULTS_BASE_PATH), "spx_superficie")
    spx_path = None
    if spx_files:
        padrao = status["ts"].iloc[pipeline.indice_padrao(status)] if len(status) else None
        spx_path = st.sidebar.selectbox(
            "Arquivo RESULTS", options=spx_files, key="spx_file",
            index=next((i for i, f in enumerate(spx_files) if _ts_spx(f) == padrao), 0),
            format_func=lambda p: pipeline.rotulo(_ts_spx(p), status),
            help="Abre na execução completa mais recente (com validação do 11 e risco do 03).",
        )
    else:
        st.sidebar.warning("Nenhum resultado do notebook 07")
    b = load_multiativos(spx_path) if spx_path else None
    meta = (b or {}).get("metadata", {})
    pricing = (b or {}).get("pricing_result")

    cabecalho()
    if b is not None:
        st.success(f"Resultado carregado: **{spx_path.split('/')[-1]}**")
        ts_sel, ts_ult = _ts_spx(spx_path), _ts_spx(spx_files[0])
        falta_sel = pipeline.motivo(ts_sel, status)
        falta_ult = pipeline.motivo(ts_ult, status) if ts_ult != ts_sel else None
    else:
        st.info(multiativos.SEM_SPX + " O Structuring Desk funciona com valores de reserva.")
        falta_sel = falta_ult = None
    aviso_execucao("spx", falta_sel, falta_ult)

    tabs = st.tabs(["Resumo do Gestor", "Superfície e Smiles", "Taxa e Dividendos", "Motor de Precificação",
                    "Controle e Ingestão", "Quanto / Compo (BRL)", "Vol Local", "Structuring Desk"])
    with tabs[0]:
        if b is not None:
            st.header("Resumo do Pricer")
            if pricing:
                paineis.resumo_pricer(meta, pricing, "Spot SPX")
                st.caption(f"Call down-and-out de referência: strike {meta.get('strike_pct', 0):.0%} e barreira "
                           f"{meta.get('barrier_pct', 0):.0%} do spot, {meta.get('target_maturity', 0):.2f} ano "
                           "(as regras do notebook 01 aplicadas à superfície do S&P 500).")
            else:
                st.info("Este arquivo do 07 é anterior à precificação de referência: rode o pipeline do S&P 500 "
                        "para ter preço, gregas e VaR.")
            st.markdown("---")
        multiativos.render_spx_mercado(st, b)
        if b is not None:
            st.markdown("---")
            risk_path = _irmao(spx_path, "RISK")
            paineis.risco(load_risk_json(risk_path) if risk_path else None,
                          risk_path.split("/")[-1] if risk_path else None,
                          (pipeline.motivo(_ts_spx(spx_path), status) or "Sem arquivo de risco para esta execução.")
                          + "\n\nUse o botão do pipeline no topo da página, ou escolha na barra lateral uma "
                            "execução marcada como completa.")
            st.markdown("---")
            multiativos.render_spx_historico(st, spx_files, load_multiativos)
    with tabs[1]:
        multiativos.render_spx_superficie(st, b, grade_spx(spx_path) if b is not None else None)
    with tabs[2]:
        multiativos.render_spx_taxas(st, b)
    with tabs[3]:
        if pricing:
            paineis.motor(pricing, b.get("simulation_diagnostics"), "S&P 500")
        else:
            st.info("Sem precificação de referência neste arquivo do 07.")
    with tabs[4]:
        controle_spx(b, spx_path, pricing)
    with tabs[5]:
        multiativos.render_quanto(st, quanto_files, load_multiativos)
    with tabs[6]:
        multiativos.render_local_vol(st, quanto_files, load_multiativos)
    with tabs[7]:
        ref = mk.referencia_quanto(load_multiativos(quanto_files[0])) if quanto_files else None
        desk.render(desk.SPX, mercado_spx(spx_path) if b is not None else None, ref)
    rodape(spx_path.split("/")[-1] if spx_path else "sem arquivo do SPX")


def controle_spx(b, spx_path, pricing):
    """Controle e Ingestão do S&P 500: checagens do preço, controles do notebook 11 e exportação."""
    st.header("Validação e Qualidade")
    if b is None:
        st.info(multiativos.SEM_SPX)
        return
    if pricing:
        paineis.controles_precificacao(pricing)
    st.markdown("---")
    st.header("Validação do Notebook 11")
    val_path = _irmao(spx_path, "VALIDATION")
    val = load_risk_json(val_path) if val_path else None
    if val is None:
        st.info("Esta execução ainda não foi validada pelo notebook 11 (ou reprovou: com FAIL ele não grava a "
                "evidência).")
    else:
        c = st.columns(4)
        c[0].metric("Decisão", val.get("decision", "—"))
        c[1].metric("PASS", val.get("pass_count", 0))
        c[2].metric("WARN", val.get("warn_count", 0))
        c[3].metric("FAIL", val.get("fail_count", 0))
        checks = pd.DataFrame(val.get("checks") or val.get("warnings") or [])
        if not checks.empty:
            st.dataframe(checks, use_container_width=True, hide_index=True)
        st.caption(f"Validado em {multiativos.quando(val.get('validation_timestamp'))} | "
                   f"Fonte: `{val_path.split('/')[-1]}`")
    st.markdown("---")
    st.header("Exportar Dados")
    md = b["metadata"].get("market_date", "unknown")
    col1, col2, col3 = st.columns(3)
    with col1:
        raw = load_result(spx_path)
        if raw is not None:
            st.download_button("Download RESULTS (CSV)", raw.to_csv(index=False), f"results_spx_{md}.csv", "text/csv")
    with col2:
        st.download_button("Download Forwards, r e q (CSV)", b["forwards"].to_csv(index=False),
                           f"forwards_spx_{md}.csv", "text/csv")
    with col3:
        st.download_button("Download Superfície (CSV)", b["options"].to_csv(index=False),
                           f"superficie_spx_{md}.csv", "text/csv")


st.sidebar.header("Configurações")

# O instrumento vem primeiro: ele decide quais arquivos e abas aparecem. ?ativo=spx abre direto no S&P 500.
st.sidebar.markdown("**Tipo de Instrumento**")
ativos = [k for k, v in SUPPORTED_INSTRUMENTS.items() if v["status"] == "active"]
# A URL só define o valor inicial (primeira carga da sessão). Não passar index= calculado a cada execução:
# no Streamlit 1.38 (o do Databricks Apps) o index entra na identidade do widget, e um index tirado da URL
# (que guarda a escolha anterior) recriava o combo a cada troca e o clique se perdia.
if "instrumento" not in st.session_state:
    st.session_state["instrumento"] = next(
        (k for k in ativos if SUPPORTED_INSTRUMENTS[k]["slug"] == st.query_params.get("ativo")), ativos[0])
instrument_choice = st.sidebar.selectbox(
    "Selecione o instrumento",
    options=ativos,
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
aviso_execucao("ibov", falta_sel, falta_ult)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Resumo do Gestor", "Superfície e Volatilidade", "Curvas de Mercado",
     "Motor de Precificação", "Controle e Ingestão", "Structuring Desk"])

with tab1:
    st.header("Resumo do Pricer")

    metadata = extract_json_object(bundle, "metadata")
    pricing = extract_json_object(bundle, "pricing_result")

    if metadata and pricing:
        paineis.resumo_pricer(metadata, pricing, "Spot IBOV")

    st.markdown("---")
    selected_ts = ts_of(selected_file_path)
    risk_files = list_available_risk(RESULTS_BASE_PATH)
    risk_match = next((p for p in risk_files if selected_ts in p.split("/")[-1]), None)
    paineis.risco(
        load_risk_json(risk_match) if risk_match else None,
        risk_match.split("/")[-1] if risk_match else None,
        (pipeline.motivo(selected_ts, exec_status) or "Nenhum arquivo RISK correspondente a este RESULTS.")
        + "\n\nUse o botão \"Rodar pipeline completo\" no topo da página, ou escolha na barra lateral "
        "uma execução marcada como completa.",
    )

    st.markdown("---")
    st.header("Histórico de Execuções")

    if len(available_files) > 1:
        selected_hist = st.multiselect(
            "Comparar múltiplas execuções",
            options=available_files,
            default=available_files[:2],
            format_func=lambda x: pipeline.data_hora(ts_of(x)),
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
                        "_ts": ts_of(fpath),
                        "Execução": pipeline.data_hora(ts_of(fpath)),
                        "Data": meta_hist.get("market_date", "—"),
                        "Spot": meta_hist.get("spot", 0),
                        "Vanilla": price_hist.get("vanilla_price", 0),
                        "Down-and-Out": price_hist.get("barrier_price", 0),
                        "Desconto": price_hist.get("barrier_discount_pct", 0),
                        "P(knock-out)": price_hist.get("knock_out_probability", 0),
                    })

            if comparison_data:
                # a mais antiga primeiro: o gráfico lê da esquerda para a direita
                comparison_df = (pd.DataFrame(comparison_data).sort_values("_ts").drop(columns="_ts")
                                 .reset_index(drop=True))
                paineis.historico_precos(comparison_df, "pontos")
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
    paineis.motor(pricing, extract_json_object(bundle, "simulation_diagnostics"), "IBOV")

with tab5:
    st.header("Validação e Qualidade")

    if pricing:
        paineis.controles_precificacao(pricing)

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
