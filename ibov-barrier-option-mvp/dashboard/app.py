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

from types import SimpleNamespace

from ibov_barrier.pricing import price_down_and_out_call, MarketData, BarrierContract
from ibov_barrier.pricing_barriers import (
    price_down_and_out_put,
    price_barrier_strangle,
    price_barrier_risk_reversal,
    price_one_touch_up,
    price_no_touch_up,
    price_barrier_collar,
    price_barrier_call_spread,
    price_barrier_put_spread,
    price_knock_out_forward,
)
from ibov_barrier.pricing_extensions import (
    barrier_hit_probability_analytical,
    double_no_touch_probability_analytical,
    price_double_no_touch,
    price_double_one_touch,
    price_knock_in_forward,
    price_knock_in_straddle,
    price_no_touch_down,
    price_one_touch_down,
    price_range_accrual,
    price_single_barrier,
)
from ibov_barrier.pricing_double import (
    double_knock_out_analytical,
    double_barrier_option_analytical,
    price_double_barrier,
    price_double_barrier_option,
)
from ibov_barrier.pricing_path_barriers import (
    parisian_barrier_quadrature,
    price_parisian_barrier,
    price_soft_barrier,
    price_step_barrier,
    soft_barrier_quadrature,
    step_barrier_quadrature,
)

# Família B completa: dupla barreira (teto e chão monitorados simultaneamente).
DOUBLE_BARRIER_STRATEGIES = ("Double Knock-Out", "Double Knock-In", "Double Barrier KI/KO")

# Barreiras dependentes da trajetória (roadmap 8, 9, 10): Parisian, Step e Soft (janela).
PATH_BARRIER_STRATEGIES = ("Parisian Barrier", "Step Barrier", "Soft Barrier (Janela)")

# Convenções de ordem KI/KO: rótulo no Desk -> (parâmetro do motor, benchmark exibido)
KIKO_CONVENTIONS = {
    "KI_first (KIKO)": ("KI_first", "KIKO = KO simples − DKO"),
    "KO_after_KI (KOKI)": ("KO_after_KI", "integral do primeiro toque × Reiner-Rubinstein"),
    "KO_first": ("KO_first", "densidade de saída pelo KO × vanilla"),
    "Sequential": ("sequential", "Knock-In simples − KOKI"),
}

# Família A: nome no Desk -> (tipo, direção da barreira, knock) do precificador genérico com rebate.
SINGLE_BARRIER_SPECS = {
    "Down-and-Out Call": ("call", "down", "out"),
    "Up-and-Out Call": ("call", "up", "out"),
    "Down-and-In Call": ("call", "down", "in"),
    "Up-and-In Call": ("call", "up", "in"),
    "Down-and-Out Put": ("put", "down", "out"),
    "Up-and-Out Put": ("put", "up", "out"),
    "Down-and-In Put": ("put", "down", "in"),
    "Up-and-In Put": ("put", "up", "in"),
}

st.set_page_config(
    page_title="Dashboard Renda Fixa",
    page_icon="",
    layout="wide",
)

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
        "description": "Call down-and-out sobre Ibovespa, precificada por Monte Carlo.",
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
def list_available_results(base_path: str):
    try:
        w = get_workspace_client()
        files = []
        for entry in w.workspace.list(base_path):
            name = entry.path.split("/")[-1]
            if name.startswith("RESULTS_superficieVol_") and name.endswith(".csv"):
                files.append(entry.path)
        return sorted(files, reverse=True)
    except Exception as e:
        st.error(f"Erro ao listar arquivos: {e}")
        return []


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


@st.cache_data(ttl=300)
def list_available_risk(base_path: str):
    try:
        w = get_workspace_client()
        files = []
        for entry in w.workspace.list(base_path):
            name = entry.path.split("/")[-1]
            if name.startswith("RISK_superficieVol_") and name.endswith(".json"):
                files.append(entry.path)
        return sorted(files, reverse=True)
    except Exception as e:
        st.error(f"Erro ao listar arquivos de risco: {e}")
        return []


def load_risk_json(file_path: str):
    try:
        w = get_workspace_client()
        response = w.workspace.download(file_path)
        return json.loads(response.read().decode())
    except Exception as e:
        st.error(f"Erro ao carregar {file_path}: {e}")
        return None


st.sidebar.header("Configurações")

available_files = list_available_results(RESULTS_BASE_PATH)

if not available_files:
    st.sidebar.error("Nenhum RESULTS encontrado")
    st.error(
        f"**Nenhum arquivo RESULTS em** `{RESULTS_BASE_PATH}`\n\n"
        "Execute `01_main_option_pricer.py` primeiro."
    )
    st.stop()

selected_file_path = st.sidebar.selectbox(
    "Arquivo RESULTS",
    options=available_files,
    format_func=lambda x: x.split("/")[-1].replace("RESULTS_superficieVol_", "").replace(".csv", ""),
)

st.sidebar.markdown("**Tipo de Instrumento**")
instrument_choice = st.sidebar.selectbox(
    "Selecione o instrumento",
    options=["option_barrier"],
    format_func=lambda x: SUPPORTED_INSTRUMENTS[x]["label"],
)

bundle = load_result(selected_file_path)

if bundle is None:
    st.stop()

selected_filename = selected_file_path.split("/")[-1]

st.title("Quantitative Pricing & Risk Dashboard")
st.markdown("**Motor de precificação, análise de risco e simulação de estruturas**")
st.markdown("---")

st.success(f"Resultado carregado: **{selected_filename}**")

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
        st.info(
            "Nenhum arquivo RISK correspondente a este RESULTS.\n\n"
            "Execute o notebook `03_analise_risco` após a validação."
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
                    fig_comp = px.line(
                        comparison_df, x="Data", y=["Vanilla", "DAO"],
                        markers=True, title="Evolução de Preços",
                        labels={"value": "Preço (pontos)", "variable": "Tipo"},
                    )
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
                "**Limitação conhecida:** q(T) ~0.8% a.a. está abaixo do "
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
    st.subheader("b Visualização da Simulação Monte Carlo")

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
                fig_paths.add_hline(y=spot, line_dash="solid", line_color="black",
                                    annotation_text=f"Spot {spot:,.0f}", annotation_position="right")
                fig_paths.add_hline(y=strike, line_dash="dash", line_color="orange",
                                    annotation_text=f"Strike {strike:,.0f}", annotation_position="right")
                fig_paths.add_hline(y=barrier, line_dash="dot", line_color="red",
                                    annotation_text=f"Barreira {barrier:,.0f}", annotation_position="right")
                fig_paths.update_layout(
                    title="Trajetórias do IBOV sob medida neutra ao risco",
                    xaxis_title="Tempo (anos)", yaxis_title="Spot IBOV (pontos)",
                    height=450, showlegend=False,
                )
                st.plotly_chart(fig_paths, use_container_width=True)

                st.caption(
                    f"Monitoramento: **{monitoring}** | "
                    f"GBM com r={r:.2%}, q={q_val:.2%}, ={sigma:.2%}"
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
            "P(knock-out) [0,1]": 0 <= ko <= 1,
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
    st.header("Structuring Desk - Simulador de Opções com Barreira")
    st.markdown("Monte a estrutura e visualize o payoff e o preço em tempo real.")

    col_esq, col_dir = st.columns([1, 2])

    with col_esq:
        st.subheader("Parâmetros da Operação")
        sim_strategy = st.selectbox(
            "Estratégia",
            [
                "Down-and-Out Call", "Up-and-Out Call",
                "Down-and-In Call", "Up-and-In Call",
                "Down-and-Out Put", "Up-and-Out Put",
                "Down-and-In Put", "Up-and-In Put",
                "Parisian Barrier", "Step Barrier", "Soft Barrier (Janela)",
                "Double Knock-Out", "Double Knock-In", "Double Barrier KI/KO",
                "Knock-Out Straddle", "Knock-In Straddle",
                "Barrier Strangle", "Barrier Risk Reversal",
                "Barrier Collar", "Barrier Call Spread", "Barrier Put Spread",
                "Knock-Out Forward", "Knock-In Forward",
                "One-Touch Up", "No-Touch Up", "One-Touch Down", "No-Touch Down",
                "Double One-Touch", "Double No-Touch", "Range Accrual",
            ],
        )
        # Down = barreira inferior; Up = barreira superior.
        # Knock-In: a opção nasce morta e só "acorda" se tocar a barreira.
        # Knock-Out: a opção nasce viva e morre se tocar a barreira.
        # Knock-Out/Knock-In Straddle e Forward usam barreira para baixo (down).
        # Estruturas combinadas (Família C): duas pernas Knock-Out com a mesma barreira.
        is_combined = sim_strategy in (
            "Barrier Strangle", "Barrier Risk Reversal",
            "Barrier Collar", "Barrier Call Spread", "Barrier Put Spread",
        )
        # Binários de toque (Família D): payoff fixo condicionado ao toque da barreira
        is_binary = sim_strategy in ("One-Touch Up", "No-Touch Up", "One-Touch Down", "No-Touch Down")
        is_one_touch = sim_strategy.startswith("One-Touch") or sim_strategy == "Double One-Touch"
        # Produtos de duas barreiras (túnel L < S0 < U): Double Touch e Range Accrual
        is_double = sim_strategy in ("Double One-Touch", "Double No-Touch")
        is_range = sim_strategy == "Range Accrual"
        # Família B: DKO/DKI (túnel) e KI/KO (KI de um lado, KO do outro, ordem parametrizável)
        is_family_b = sim_strategy in DOUBLE_BARRIER_STRATEGIES
        is_sequential = sim_strategy == "Double Barrier KI/KO"
        is_two_barrier = is_double or is_range or is_family_b
        is_single = sim_strategy in SINGLE_BARRIER_SPECS
        # Barreiras dependentes da trajetória: call/put, Down/Up, Out/In escolhidos no Desk
        is_path_barrier = sim_strategy in PATH_BARRIER_STRATEGIES
        is_step = sim_strategy == "Step Barrier"
        path_option_type, path_direction, path_knock = "call", "down", "out"
        if is_path_barrier:
            path_option_type = st.radio("Tipo da Opção", ["Call", "Put"], horizontal=True,
                                        key="path_type").lower()
            path_direction = "up" if st.radio("Direção da Barreira", ["Down (inferior)", "Up (superior)"],
                                              horizontal=True, key="path_dir").startswith("Up") else "down"
            path_knock = "in" if st.radio("Knock", ["Knock-Out", "Knock-In"], horizontal=True,
                                          key="path_knock") == "Knock-In" else "out"

        dbl_option_type = "call"
        ki_is_lower = True
        if is_family_b:
            dbl_label = st.radio("Tipo da Opção", ["Call", "Put"], horizontal=True)
            dbl_option_type = dbl_label.lower()
            if is_sequential:
                ki_label = st.radio("Barreira Knock-In", ["Inferior (L)", "Superior (U)"], horizontal=True,
                                    help="A barreira oposta é a Knock-Out.")
                ki_is_lower = ki_label.startswith("Inferior")
                sim_convention_label = st.selectbox(
                    "Convenção de Ordem KI/KO", list(KIKO_CONVENTIONS),
                    help="KI_first: nasce no KI; o KO mata a qualquer momento. "
                         "KO_after_KI: nasce no KI; o KO só vale após a ativação. "
                         "KO_first: nasce viva; o KO só mata se vier antes do KI (KI primeiro = vanilla). "
                         "Sequential: só paga se tocar o KI e depois o KO.",
                )
                sim_convention, sim_convention_bench = KIKO_CONVENTIONS[sim_convention_label]

        # Variante Up (barreira superior) para Collar, Spreads e Strangle
        combined_direction = "down"
        if sim_strategy in ("Barrier Strangle", "Barrier Collar", "Barrier Call Spread", "Barrier Put Spread"):
            combined_label = st.radio("Direção da Barreira", ["Down (inferior)", "Up (superior)"], horizontal=True)
            combined_direction = "up" if combined_label.startswith("Up") else "down"

        barrier_is_down = (
            sim_strategy.startswith("Down")
            or sim_strategy in ("Knock-Out Straddle", "Knock-In Straddle", "Knock-Out Forward", "Knock-In Forward")
            or sim_strategy.endswith("Touch Down")
            or (is_combined and combined_direction == "down")
            or (is_path_barrier and path_direction == "down")
        )
        is_knock_in = ("and-In" in sim_strategy or sim_strategy.startswith("Knock-In")
                       or (is_path_barrier and path_knock == "in"))
        has_strike = not (is_binary or is_double or is_range)

        sim_spot = st.number_input("Spot Inicial (R$)", min_value=10000.0, max_value=500000.0, value=172000.0, step=1000.0)
        sim_vol = st.slider("Volatilidade (%)", min_value=5.0, max_value=100.0, value=20.0, step=1.0) / 100.0
        sim_rate = st.slider("Taxa de Juros (%)", min_value=0.0, max_value=30.0, value=12.0, step=0.5) / 100.0

        sim_strike_pct = st.slider("Strike (% do Spot)", min_value=50, max_value=150, value=90, step=1) if has_strike else 100
        sim_strike = sim_spot * (sim_strike_pct / 100.0)
        # 2ª perna (put) das estruturas combinadas da Família C
        if is_combined:
            sim_strike_2_pct = st.slider("Strike da 2ª Perna (% do Spot)", min_value=50, max_value=150, value=110, step=1)
            sim_strike_2 = sim_spot * (sim_strike_2_pct / 100.0)
        # Payout dos binários de toque (Família D) / Notional do Range Accrual
        if is_binary or is_double:
            sim_payout = st.number_input("Payout (R$)", min_value=1.0, max_value=1_000_000.0, value=100.0, step=10.0)
        sim_accrual_rate = None
        if is_range:
            sim_payout = st.number_input("Notional (R$)", min_value=1.0, max_value=100_000_000.0, value=1_000_000.0, step=10_000.0)
            sim_accrual_rate = st.number_input(
                "Taxa de Accrual (% a.a.)", min_value=0.1, max_value=100.0, value=10.0, step=0.5,
                help="Cupom = Notional × taxa × prazo × fração dos fixings dentro da faixa.",
            ) / 100.0
        # Rebate (Família A): pago no vencimento se a opção morrer (KO) ou nunca ativar (KI)
        sim_rebate = 0.0
        if is_single or is_path_barrier or sim_strategy in ("Double Knock-Out", "Double Knock-In"):
            sim_rebate = st.number_input(
                "Rebate (R$)", min_value=0.0, max_value=1_000_000.0, value=0.0, step=10.0,
                help="Pago no vencimento: Knock-Out se a barreira for tocada; Knock-In se nunca for tocada.",
            )

        # TRAVA DE UX: O range do slider muda conforme a barreira (Down/Up)
        sim_barrier = None
        sim_lower = sim_upper = None
        if is_range:
            range_variant = st.radio("Tipo de Faixa", ["Túnel (L e U)", "Down (acima de L)", "Up (abaixo de U)"], horizontal=True)
            if range_variant != "Up (abaixo de U)":
                sim_lower = sim_spot * st.slider("Limite Inferior L (% do Spot)", min_value=50, max_value=99, value=80, step=1) / 100.0
            if range_variant != "Down (acima de L)":
                sim_upper = sim_spot * st.slider("Limite Superior U (% do Spot)", min_value=101, max_value=150, value=120, step=1) / 100.0
        elif is_double or is_family_b:
            sim_lower = sim_spot * st.slider("Barreira Inferior L (% do Spot)", min_value=50, max_value=99, value=80, step=1) / 100.0
            sim_upper = sim_spot * st.slider("Barreira Superior U (% do Spot)", min_value=101, max_value=150, value=120, step=1) / 100.0
        elif is_step:
            lo_pct, hi_pct = (50, 99) if barrier_is_down else (101, 150)
            step_1 = st.slider("Barreira 1 (% do Spot)", min_value=lo_pct, max_value=hi_pct,
                               value=80 if barrier_is_down else 120, step=1)
            step_2 = st.slider("Barreira 2 (% do Spot)", min_value=lo_pct, max_value=hi_pct,
                               value=90 if barrier_is_down else 110, step=1,
                               help="Nível que passa a valer a partir da data de mudança.")
            step_change_pct = st.slider("Data da Mudança (% do prazo)", min_value=5, max_value=95, value=50, step=5)
            sim_barrier = sim_spot * step_1 / 100.0
            sim_barrier_2 = sim_spot * step_2 / 100.0
        elif barrier_is_down:
            sim_barrier_pct = st.slider("Barreira (% do Spot)", min_value=50, max_value=99, value=80, step=1)
            sim_barrier = sim_spot * (sim_barrier_pct / 100.0)
        else:
            sim_barrier_pct = st.slider("Barreira (% do Spot)", min_value=101, max_value=150, value=120, step=1)
            sim_barrier = sim_spot * (sim_barrier_pct / 100.0)

        sim_tenor = st.selectbox("Vencimento", ["1 Mês (21 DU)", "3 Meses (63 DU)", "6 Meses (126 DU)", "1 Ano (252 DU)"], index=2)
        tenor_map = {"1 Mês (21 DU)": 21/252, "3 Meses (63 DU)": 63/252, "6 Meses (126 DU)": 126/252, "1 Ano (252 DU)": 1.0}
        sim_maturity = tenor_map[sim_tenor]
        # Monitoramento em dias úteis: 1 fechamento por DU; datas arredondadas para o DU mais próximo.
        n_days = int(round(sim_maturity * 252))
        dt_day = sim_maturity / n_days
        if sim_strategy == "Parisian Barrier":
            sim_window = st.slider("Fechamentos Além da Barreira (DU)", min_value=1, max_value=min(20, n_days),
                                   value=min(5, n_days), step=1,
                                   help="O evento de barreira só ocorre após esse número de fechamentos além do nível.")
            sim_cumulative = st.checkbox("Contagem cumulativa (ParAsian)",
                                         help="Desmarcado: dias CONSECUTIVOS (zera ao voltar). Marcado: total de dias no prazo.")
        if is_step:
            step_change_time = max(1, min(n_days - 1, round(step_change_pct / 100 * n_days))) * dt_day
            sim_schedule = [(0.0, sim_barrier), (step_change_time, sim_barrier_2)]
        if sim_strategy == "Soft Barrier (Janela)":
            win_lo, win_hi = st.slider("Janela de Monitoramento (% do prazo)", min_value=0, max_value=100,
                                       value=(50, 100), step=5,
                                       help="A barreira só vale dentro da janela; fora dela o ativo pode cruzar o nível.")
            win_start = min(round(win_lo / 100 * n_days), n_days - 1)
            win_end = max(round(win_hi / 100 * n_days), win_start + 1)
            sim_windows = [(win_start * dt_day, win_end * dt_day)]

    with col_dir:
        st.subheader("Diagrama de Payoff e Precificação")

        # GRÁFICO DE PAYOFF
        s_t = np.linspace(sim_spot * 0.5, sim_spot * 1.5, 200)
        if is_range:
            # Range Accrual: cada fixing dentro da faixa acumula Notional/N.
            # Perfil = Notional se o ativo permanecesse no nível S em todos os fixings.
            lo_level = sim_lower if sim_lower is not None else 0.0
            hi_level = sim_upper if sim_upper is not None else np.inf
            sim_coupon = sim_payout * sim_accrual_rate * sim_maturity
            payoff = np.where((s_t >= lo_level) & (s_t <= hi_level), sim_coupon, 0.0)
        elif is_family_b:
            # Família B: vanilla condicionada às duas barreiras (perfil terminal indicativo).
            vanilla_payoff = np.maximum(s_t - sim_strike, 0) if dbl_option_type == "call" else np.maximum(sim_strike - s_t, 0)
            inside = (s_t > sim_lower) & (s_t < sim_upper)
            if sim_strategy == "Double Knock-Out":
                payoff = np.where(inside, vanilla_payoff, sim_rebate)
            elif sim_strategy == "Double Knock-In":
                payoff = np.where(inside, sim_rebate, vanilla_payoff)
            elif sim_convention in ("KI_first", "KO_after_KI"):
                # Terminar além da barreira KO implica estrutura morta nessas convenções.
                beyond_ko = s_t >= sim_upper if ki_is_lower else s_t <= sim_lower
                payoff = np.where(beyond_ko, 0.0, vanilla_payoff)
            else:
                # KO_first / Sequential: a sobrevivência depende só da ordem dos toques,
                # não do nível terminal; o perfil exibido é a vanilla condicional.
                payoff = vanilla_payoff
        elif is_double:
            # Double One-Touch paga fora do túnel (toque implícito); Double No-Touch paga dentro.
            inside = (s_t > sim_lower) & (s_t < sim_upper)
            payoff = np.where(inside, 0.0, sim_payout) if is_one_touch else np.where(inside, sim_payout, 0.0)
        elif is_binary:
            # Família D: payoff binário — payout fixo do cliente.
            # One-Touch paga além da barreira (toque implícito no terminal);
            # No-Touch paga aquém dela (barreira jamais tocada nesse cenário).
            touched_region = s_t <= sim_barrier if barrier_is_down else s_t >= sim_barrier
            payoff = np.where(touched_region if is_one_touch else ~touched_region, sim_payout, 0.0)
        elif is_combined:
            # Família C: estruturas de duas pernas Knock-Out (Strike 1 + Strike 2)
            if sim_strategy == "Barrier Strangle":
                payoff = np.maximum(s_t - sim_strike, 0) + np.maximum(sim_strike_2 - s_t, 0)
            elif sim_strategy == "Barrier Risk Reversal":
                payoff = np.maximum(s_t - sim_strike, 0) - np.maximum(sim_strike_2 - s_t, 0)
            elif sim_strategy == "Barrier Collar":
                payoff = np.maximum(sim_strike - s_t, 0) - np.maximum(s_t - sim_strike_2, 0)
            elif sim_strategy == "Barrier Call Spread":
                payoff = np.maximum(s_t - sim_strike, 0) - np.maximum(s_t - sim_strike_2, 0)
            else:  # Barrier Put Spread
                payoff = np.maximum(sim_strike - s_t, 0) - np.maximum(sim_strike_2 - s_t, 0)
        elif sim_strategy in ("Knock-Out Forward", "Knock-In Forward"):
            # Forward sintético: payoff linear; KO morre no crash, KI só nasce no crash
            payoff = s_t - sim_strike
        elif is_path_barrier:
            # Path-dependent: viver/morrer depende da trajetória, não do nível terminal.
            # O perfil exibido é a vanilla condicional ao estado (viva/ativada) no vencimento.
            if path_option_type == "call":
                payoff = np.maximum(s_t - sim_strike, 0)
            else:
                payoff = np.maximum(sim_strike - s_t, 0)
        elif "Put" in sim_strategy:
            payoff = np.maximum(sim_strike - s_t, 0)
        elif sim_strategy in ("Knock-Out Straddle", "Knock-In Straddle"):
            payoff = np.maximum(s_t - sim_strike, 0) + np.maximum(sim_strike - s_t, 0)
        else:  # Calls
            payoff = np.maximum(s_t - sim_strike, 0)

        # Região morta do payoff:
        # Knock-Out zero ONDE a barreira FOI tocada; Knock-In zero onde NÃO foi tocada.
        # Com rebate, a região morta paga o rebate em vez de zero.
        # Binários e produtos de túnel (Família D) já carregam a própria região no payoff.
        if is_binary or is_two_barrier or is_path_barrier:
            pass
        elif is_knock_in:
            if barrier_is_down:
                payoff[s_t > sim_barrier] = sim_rebate
            else:
                payoff[s_t < sim_barrier] = sim_rebate
        else:
            if barrier_is_down:
                payoff[s_t <= sim_barrier] = sim_rebate
            else:
                payoff[s_t >= sim_barrier] = sim_rebate

        fig_payoff = go.Figure()
        if is_range:
            # Range Accrual não tem payoff linear: barra com o cupom acumulado se S ficar na faixa.
            fig_payoff.add_trace(go.Scatter(x=s_t, y=payoff, mode='lines', name='Cupom', fill='tozeroy',
                                            line=dict(color='#2ca02c', width=2, shape='hv'),
                                            fillcolor='rgba(44, 160, 44, 0.35)'))
        else:
            fig_payoff.add_trace(go.Scatter(x=s_t, y=payoff, mode='lines', name='Payoff', line=dict(color='#2ca02c', width=3)))
        if is_two_barrier:
            for level, label, pos in ((sim_lower, "L", "top left"), (sim_upper, "U", "top right")):
                if level is not None:
                    if is_sequential:
                        label += " (KI)" if (label == "L") == ki_is_lower else " (KO)"
                    fig_payoff.add_vline(x=level, line_dash="dash", line_color="red", annotation_text=f"Barreira {label}", annotation_position=pos)
        elif is_step:
            fig_payoff.add_vline(x=sim_barrier, line_dash="dash", line_color="red",
                                 annotation_text="Barreira 1", annotation_position="top left")
            fig_payoff.add_vline(x=sim_barrier_2, line_dash="dash", line_color="#d62728",
                                 annotation_text=f"Barreira 2 (a partir de {step_change_time:.2f} ano)",
                                 annotation_position="bottom left")
        else:
            barrier_label = "Barreira (Knock-In)" if is_knock_in else ("Barreira" if is_binary else "Barreira (Knock-Out)")
            fig_payoff.add_vline(x=sim_barrier, line_dash="dash", line_color="red", annotation_text=barrier_label, annotation_position="top left")
        if has_strike:
            fig_payoff.add_vline(x=sim_strike, line_dash="dot", line_color="orange", annotation_text="Strike", annotation_position="top right")
        if is_range:
            fig_payoff.add_hline(y=sim_coupon, line_dash="dash", line_color="#00A3E0",
                                 annotation_text=f"Cupom máximo {sim_accrual_rate:.2%} a.a. × {sim_maturity:.2f} ano (R$ {sim_coupon:,.2f})")
        elif is_binary or is_double:
            # Linha horizontal tracejada no nível do Payout/Notional escolhido
            level_label = "Notional" if is_range else "Payout"
            fig_payoff.add_hline(y=sim_payout, line_dash="dash", line_color="#00A3E0", annotation_text=f"{level_label} (R$ {sim_payout:,.0f})")
        payoff_title = "Diagrama de Accrual (cupom se S ficar na faixa em todos os fixings)" if is_range else "Payoff no Vencimento"
        if is_path_barrier:
            payoff_title = ("Payoff no Vencimento (se " + ("viva" if path_knock == "out" else "ativada")
                            + "; o evento de barreira depende da trajetória)")
        fig_payoff.update_layout(title=payoff_title, xaxis_title="Preço do Ativo no Vencimento (R$)", yaxis_title="Lucro/Prejuízo (R$)", height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_payoff, use_container_width=True)

        # TRAVAS DE UX COM MENSAGENS AMIGÁVEIS
        is_invalid = False

        # Travas de UX: Knock-Out nasce além da barreira = morta; Knock-In nasce além = JÁ ATIVA (vanilla).
        # Binários: barreira superior já tocada = One-Touch GANHO integral (não é erro).
        if is_two_barrier:
            pass  # sliders garantem L < Spot < U
        elif is_binary:
            if (not barrier_is_down) and sim_spot >= sim_barrier:
                st.info("Barreira superior já tocada: o One-Touch paga o Payout integral (o No-Touch vale zero).")
            elif barrier_is_down and sim_spot <= sim_barrier:
                st.info("Barreira inferior já tocada: o One-Touch paga o Payout integral (o No-Touch vale zero).")
        elif not is_knock_in:
            if barrier_is_down and sim_spot <= sim_barrier:
                st.error("**Opa, chefe!** O Spot já está abaixo ou na barreira. A opção morreu antes de começar! Diminua a barreira ou aumente o Spot.")
                is_invalid = True
            elif (not barrier_is_down) and sim_spot >= sim_barrier:
                st.error("**Opa, chefe!** O Spot já está acima da barreira. A opção morreu antes de começar! Aumente a barreira ou diminua o Spot.")
                is_invalid = True
        else:
            if barrier_is_down and sim_spot <= sim_barrier:
                st.info("A opção JÁ está ativa (spot tocou a barreira) — equivale à vanilla.")
            elif (not barrier_is_down) and sim_spot >= sim_barrier:
                st.info("A opção JÁ está ativa (spot tocou a barreira) — equivale à vanilla.")

        if has_strike and barrier_is_down and (not is_knock_in) and sim_strike < sim_barrier:
            st.warning("**Atenção:** O Strike está abaixo da Barreira. Essa estrutura é atípica.")

        if not is_invalid:
            if st.button("Calcular Preço", use_container_width=True, type="primary"):
                with st.spinner("Rodando Monte Carlo com 50.000 caminhos..."):
                    market_sim = MarketData(spot=sim_spot, rate=sim_rate, dividend_yield=0.02, volatility=sim_vol)
                    contract_sim = BarrierContract(strike=sim_strike, barrier=sim_barrier, maturity=sim_maturity)

                    try:
                        mc = dict(paths=50_000, steps=126, seed=42)
                        benchmark = None  # benchmark analítico (Famílias B e D), quando disponível
                        if is_family_b:
                            # Família B: motor com instante de toque por passo (L e U simultâneas)
                            bench_args = (sim_spot, sim_strike)
                            mkt_args = (sim_maturity, sim_rate, 0.02, sim_vol)
                            if is_sequential:
                                ki_level, ko_level = (sim_lower, sim_upper) if ki_is_lower else (sim_upper, sim_lower)
                                result = price_double_barrier_option(
                                    market_sim, strike=sim_strike, knock_in=ki_level, knock_out=ko_level,
                                    maturity=sim_maturity, option_type=dbl_option_type,
                                    convention=sim_convention, **mc,
                                )
                                benchmark = double_barrier_option_analytical(
                                    *bench_args, ki_level, ko_level, *mkt_args,
                                    option_type=dbl_option_type, convention=sim_convention,
                                )
                            else:
                                knock = "out" if sim_strategy == "Double Knock-Out" else "in"
                                result = price_double_barrier(
                                    market_sim, strike=sim_strike, lower=sim_lower, upper=sim_upper,
                                    maturity=sim_maturity, option_type=dbl_option_type, knock=knock,
                                    rebate=sim_rebate, **mc,
                                )
                                # Ikeda-Kunitomo (DKO) + rebate via P(Double No-Touch) analítica
                                dko = double_knock_out_analytical(*bench_args, sim_lower, sim_upper, *mkt_args,
                                                                  option_type=dbl_option_type)
                                p_dnt = double_no_touch_probability_analytical(sim_spot, sim_lower, sim_upper, *mkt_args)
                                rebate_pv = sim_rebate * np.exp(-sim_rate * sim_maturity)
                                if knock == "out":
                                    benchmark = dko + rebate_pv * (1.0 - p_dnt)
                                else:
                                    benchmark = result.vanilla_price - dko + rebate_pv * p_dnt
                        elif is_path_barrier:
                            # Roadmap 8-10: 1 passo por dia útil (fechamentos diários)
                            path_kw = dict(option_type=path_option_type, direction=path_direction,
                                           knock=path_knock, rebate=sim_rebate)
                            path_mc = dict(paths=50_000, steps=n_days, seed=42)
                            mkt_args = (sim_maturity, sim_rate, 0.02, sim_vol)
                            if sim_strategy == "Parisian Barrier":
                                result = price_parisian_barrier(
                                    market_sim, strike=sim_strike, barrier=sim_barrier, maturity=sim_maturity,
                                    window=sim_window, cumulative=sim_cumulative, **path_kw, **path_mc,
                                )
                                benchmark = parisian_barrier_quadrature(
                                    sim_spot, sim_strike, sim_barrier, *mkt_args, steps=n_days,
                                    window=sim_window, cumulative=sim_cumulative, **path_kw,
                                )
                                path_bench_text = ("cadeia de Markov exata do monitoramento diário "
                                                   "(log-preço × contador de dias além da barreira)")
                            elif is_step:
                                result = price_step_barrier(
                                    market_sim, strike=sim_strike, schedule=sim_schedule, maturity=sim_maturity,
                                    **path_kw, **path_mc,
                                )
                                benchmark = step_barrier_quadrature(sim_spot, sim_strike, *mkt_args,
                                                                    schedule=sim_schedule, **path_kw)
                                path_bench_text = ("monitoramento contínuo, densidade de transição morta "
                                                   "na barreira encadeada por degrau")
                            else:
                                result = price_soft_barrier(
                                    market_sim, strike=sim_strike, barrier=sim_barrier, maturity=sim_maturity,
                                    windows=sim_windows, **path_kw, **path_mc,
                                )
                                benchmark = soft_barrier_quadrature(sim_spot, sim_strike, sim_barrier, *mkt_args,
                                                                    windows=sim_windows, **path_kw)
                                path_bench_text = ("monitoramento contínuo na janela, densidade de transição "
                                                   "morta na barreira encadeada por segmento")
                        elif is_single:
                            # Família A (8 tipos) com rebate opcional; rebate=0 reproduz os
                            # precificadores dedicados exatamente (mesmos caminhos e CV).
                            opt_type, direction, knock = SINGLE_BARRIER_SPECS[sim_strategy]
                            result = price_single_barrier(
                                market_sim, contract_sim, option_type=opt_type, direction=direction,
                                knock=knock, rebate=sim_rebate, **mc,
                            )
                        elif is_range:
                            # Range Accrual: 1 fixing por dia útil do prazo
                            n_fixings = int(round(sim_maturity * 252))
                            result = price_range_accrual(
                                market_sim, sim_maturity, lower=sim_lower, upper=sim_upper,
                                notional=sim_payout, paths=50_000, steps=n_fixings, seed=42,
                                accrual_rate=sim_accrual_rate,
                            )
                            benchmark = result.analytical_price
                        elif is_double:
                            double_fn = price_double_one_touch if is_one_touch else price_double_no_touch
                            result = double_fn(
                                market_sim, lower=sim_lower, upper=sim_upper, maturity=sim_maturity,
                                payout=sim_payout, **mc,
                            )
                            p_dnt = double_no_touch_probability_analytical(
                                sim_spot, sim_lower, sim_upper, sim_maturity, sim_rate, 0.02, sim_vol
                            )
                            p_pay = (1.0 - p_dnt) if is_one_touch else p_dnt
                            benchmark = sim_payout * np.exp(-sim_rate * sim_maturity) * p_pay
                        elif sim_strategy in ("One-Touch Down", "No-Touch Down"):
                            down_fn = price_one_touch_down if is_one_touch else price_no_touch_down
                            result = down_fn(market_sim, contract_sim, payout=sim_payout, **mc)
                            p_hit = barrier_hit_probability_analytical(
                                sim_spot, sim_barrier, sim_maturity, sim_rate, 0.02, sim_vol
                            )
                            p_pay = p_hit if is_one_touch else 1.0 - p_hit
                            benchmark = sim_payout * np.exp(-sim_rate * sim_maturity) * p_pay
                        elif sim_strategy == "Knock-In Straddle":
                            # Call DIC + Put DIP, mesmo Strike: só nasce se tocar a barreira
                            result = price_knock_in_straddle(market_sim, contract_sim, **mc)
                        elif sim_strategy == "Knock-In Forward":
                            # Call DIC - Put DIP, mesmo Strike: forward que só nasce no crash
                            result = price_knock_in_forward(market_sim, contract_sim, **mc)
                        elif sim_strategy == "Barrier Strangle":
                            # Família C: Call KO (Strike 1) + Put KO (Strike 2), mesma barreira
                            result = price_barrier_strangle(market_sim, contract_sim, sim_strike_2, direction=combined_direction, paths=50_000, steps=126, seed=42)
                        elif sim_strategy == "Barrier Risk Reversal":
                            # Família C: Call DOC (Strike 1) - Put DOC (Strike 2); negativo = crédito líquido
                            result = price_barrier_risk_reversal(market_sim, contract_sim, sim_strike_2, paths=50_000, steps=126, seed=42)
                        elif sim_strategy == "One-Touch Up":
                            # Família D: binário — payout fixo se tocar a barreira superior
                            result = price_one_touch_up(market_sim, contract_sim, payout=sim_payout, paths=50_000, steps=126, seed=42)
                        elif sim_strategy == "No-Touch Up":
                            # Família D: binário — payout fixo se NUNCA tocar a barreira superior
                            result = price_no_touch_up(market_sim, contract_sim, payout=sim_payout, paths=50_000, steps=126, seed=42)
                        elif sim_strategy == "Barrier Collar":
                            # Família C+: Put DOC (Strike 1) - Call DOC (Strike 2); negativo = crédito
                            result = price_barrier_collar(market_sim, contract_sim, sim_strike_2, direction=combined_direction, paths=50_000, steps=126, seed=42)
                        elif sim_strategy == "Barrier Call Spread":
                            # Família C+: Call KO (Strike 1) - Call KO (Strike 2)
                            result = price_barrier_call_spread(market_sim, contract_sim, sim_strike_2, direction=combined_direction, paths=50_000, steps=126, seed=42)
                        elif sim_strategy == "Barrier Put Spread":
                            # Família C+: Put KO (Strike 1) - Put KO (Strike 2)
                            result = price_barrier_put_spread(market_sim, contract_sim, sim_strike_2, direction=combined_direction, paths=50_000, steps=126, seed=42)
                        elif sim_strategy == "Knock-Out Forward":
                            # Call DOC - Put DOC, mesmo Strike: forward que morre no crash
                            result = price_knock_out_forward(market_sim, contract_sim, paths=50_000, steps=126, seed=42)
                        else:
                            # Knock-Out Straddle: perna call + perna put, mesmo Strike e
                            # Barreira down. Mesma seed mesmos caminhos e mesmos eventos
                            # de knock-out nas duas pernas; P(KO) é idêntica entre pernas.
                            result_call = price_down_and_out_call(market_sim, contract_sim, paths=50_000, steps=126, seed=42)
                            result_put = price_down_and_out_put(market_sim, contract_sim, paths=50_000, steps=126, seed=42)
                            result = SimpleNamespace(
                                price=result_call.price + result_put.price,
                                vanilla_price=result_call.vanilla_price + result_put.vanilla_price,
                                knock_out_probability=result_call.knock_out_probability,
                            )

                        if sim_strategy == "Barrier Risk Reversal" and result.price < 0:
                            st.success("Cálculo concluído! Preço negativo = crédito líquido RECEBIDO (a venda da put custeia a call).")
                        elif sim_strategy != "Barrier Risk Reversal" and abs(result.price) <= 0.01:
                            st.warning("**Calma lá!** Com esses parâmetros a opção está valendo praticamente zero. Tente ajustar o Strike ou a Barreira.")
                        else:
                            st.success("Cálculo concluído com sucesso! Estrutura pronta para o cliente.")

                        col_r1, col_r2, col_r3 = st.columns(3)
                        col_r1.metric("Preço da Estrutura", f"R$ {result.price:,.2f}")
                        if benchmark is not None:
                            # Família D: benchmark analítico (monitoramento contínuo) ao lado do MC
                            col_r2.metric("Benchmark Analítico", f"R$ {benchmark:,.2f}")
                        else:
                            col_r2.metric("Preço Vanilla", f"R$ {result.vanilla_price:,.2f}")
                        if is_range:
                            col_r3.metric("Fração Esperada na Faixa", f"{result.expected_fraction:.2%}")
                            st.caption(f"Cupom máximo R$ {sim_payout * sim_accrual_rate * sim_maturity:,.2f} "
                                       f"({sim_accrual_rate:.2%} a.a. × {sim_maturity:.2f} ano) | "
                                       f"{result.observations} fixings (1 por dia útil) | IC 95%: "
                                       f"[R$ {result.ci_low:,.2f}; R$ {result.ci_high:,.2f}]")
                        else:
                            # Knock-Ins: o campo reporta P(morrer sem ativação)
                            if is_binary or is_double:
                                p_label = "P(Tocado)" if is_one_touch else "P(Não Tocado)"
                            elif is_family_b:
                                p_label = {"Double Knock-Out": "P(Knock-out)",
                                           "Double Knock-In": "P(Sem Ativação)"}.get(sim_strategy, "P(Não Viva em T)")
                            else:
                                p_label = "P(Sem Ativação)" if is_knock_in else "P(Knock-out)"
                            col_r3.metric(p_label, f"{result.knock_out_probability:.2%}")
                        if sim_rebate > 0 and (is_single or is_family_b or is_path_barrier):
                            never_activates = is_knock_in or sim_strategy == "Double Knock-In"
                            st.caption(f"Inclui rebate de R$ {sim_rebate:,.2f} pago no vencimento "
                                       + ("se a opção nunca ativar." if never_activates else "se a barreira for tocada."))
                        if is_family_b:
                            st.caption("Benchmark analítico: monitoramento contínuo "
                                       + ("(Ikeda-Kunitomo)." if not is_sequential else
                                          f"({sim_convention_bench}).")
                                       + " O MC usa Brownian Bridge, então a diferença é só ruído estatístico.")
                        if is_path_barrier:
                            if sim_strategy == "Parisian Barrier":
                                detail = (f"Evento após {sim_window} fechamento(s) "
                                          + ("no total" if sim_cumulative else "consecutivos")
                                          + " além da barreira. ")
                            elif is_step:
                                detail = (f"Barreira R$ {sim_barrier:,.0f} até {step_change_time:.3f} ano, "
                                          f"depois R$ {sim_barrier_2:,.0f}. ")
                            else:
                                detail = (f"Barreira monitorada de {sim_windows[0][0]:.3f} a "
                                          f"{sim_windows[0][1]:.3f} ano. ")
                            st.caption(detail + f"{n_days} passos (1 por dia útil). Benchmark: {path_bench_text}. "
                                       f"IC 95% do MC: [R$ {result.ci_low:,.2f}; R$ {result.ci_high:,.2f}].")
                    except Exception:
                        st.error("**Ops!** Não conseguimos calcular essa estrutura. Verifique os parâmetros.")

st.markdown("---")
st.caption(
    f"{selected_filename} | "
    f"Cache: 5 min | "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Databricks Apps"
)
