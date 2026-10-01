"""Dashboard Streamlit — Databricks Apps.

Roda em container dedicado e acessa arquivos do Workspace via SDK.
"""
import json
import os
from io import StringIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from databricks.sdk import WorkspaceClient

st.set_page_config(
    page_title="Dashboard Renda Fixa",
    page_icon="📊",
    layout="wide",
)

RESULTS_BASE_PATH = os.environ.get(
    "RESULTS_BASE_PATH",
    "/Workspace/Users/luiz.henrique.felipe.rocha@gmail.com/ibov-barrier-results",
)

SUPPORTED_INSTRUMENTS = {
    "option_barrier": {
        "label": "📊 Opção com Barreira (IBOV)",
        "status": "active",
        "description": "Call down-and-out sobre Ibovespa, precificada por Monte Carlo.",
    },
    "option_vanilla": {
        "label": "📈 Opção Europeia Simples",
        "status": "planned",
        "description": "Call/Put europeia sobre IBOV — em desenvolvimento.",
    },
    "swap_di_pre": {
        "label": "💱 Swap DI x Pré",
        "status": "planned",
        "description": "Precificação de swap via curva DI — em desenvolvimento.",
    },
    "future_ind": {
        "label": "📅 Futuro de Ibovespa",
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
            return pd.read_csv(f)
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


st.sidebar.header("⚙️ Configurações")

available_files = list_available_results(RESULTS_BASE_PATH)

if not available_files:
    st.sidebar.error("Nenhum RESULTS encontrado")
    st.error(
        f"⚠️ **Nenhum arquivo RESULTS em** `{RESULTS_BASE_PATH}`\n\n"
        "Execute `01_main_option_pricer.py` primeiro."
    )
    st.stop()

selected_file_path = st.sidebar.selectbox(
    "📁 Arquivo RESULTS",
    options=available_files,
    format_func=lambda x: x.split("/")[-1].replace("RESULTS_superficieVol_", "").replace(".csv", ""),
)

st.sidebar.markdown("**Tipo de Instrumento**")
instrument_choice = st.sidebar.selectbox(
    "Selecione o instrumento",
    options=["option_barrier"],
    format_func=lambda x: SUPPORTED_INSTRUMENTS[x]["label"],
)

with st.sidebar.expander("🔮 Próximos instrumentos"):
    for key, info in SUPPORTED_INSTRUMENTS.items():
        if info["status"] == "planned":
            st.markdown(f"**{info['label']}**")
            st.caption(info["description"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Seções do Dashboard**")

show_metadata = st.sidebar.checkbox("📋 Metadata", value=True)
show_validation = st.sidebar.checkbox("✅ Validação", value=True)
show_export = st.sidebar.checkbox("📥 Exportação", value=True)
show_history = st.sidebar.checkbox("📈 Histórico", value=False)
show_raw_data = st.sidebar.checkbox("📊 Dados Brutos", value=False)
show_mc_viz = st.sidebar.checkbox("🎲 Visualização Monte Carlo", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Databricks Apps · Renda Fixa")

bundle = load_result(selected_file_path)

if bundle is None:
    st.stop()

selected_filename = selected_file_path.split("/")[-1]

st.title("📊 Dashboard do Projeto de Renda Fixa")
st.markdown("**Opção com barreira sobre Ibovespa — pipeline end-to-end**")
st.markdown("---")

st.success(f"✓ Resultado carregado: **{selected_filename}**")

st.header("1️⃣ Resumo do Pricer")

metadata = extract_json_object(bundle, "metadata")
pricing = extract_json_object(bundle, "pricing_result")

if metadata and pricing:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Data de Mercado", metadata.get("market_date", "—"))
    col2.metric("📈 Spot IBOV", f"{metadata.get('spot', 0):,.2f}")
    col3.metric("🎯 Strike Alvo", f"{metadata.get('target_strike', 0):,.2f}")
    col4.metric("📊 Vol. Implícita", f"{metadata.get('target_iv', 0):.2%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("💰 Preço Vanilla (BS)", f"{pricing.get('vanilla_price', 0):,.2f}")
    col6.metric("💎 Preço Down-and-Out", f"{pricing.get('barrier_price', 0):,.2f}")
    col7.metric("📉 Desconto Barreira", f"{pricing.get('barrier_discount_pct', 0):.2%}")
    col8.metric("🎲 P(knock-out)", f"{pricing.get('knock_out_probability', 0):.2%}")

    if show_metadata:
        with st.expander("📋 Ver parâmetros do contrato e mercado"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Contrato**")
                st.json(pricing.get("contract", {}))
            with col_b:
                st.markdown("**Mercado**")
                st.json(pricing.get("market", {}))

st.markdown("---")

st.header("2️⃣ Curva de Juros (DI)")

curve_df = extract_dataframe(bundle, "curve")
y_col = None

if curve_df is not None and len(curve_df) > 0:
    curve_df = curve_df.copy()
    if "T" in curve_df.columns:
        curve_df["T_plot"] = curve_df["T"]
    elif "tenor_bd" in curve_df.columns:
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

st.header("3️⃣ Dividend Yield Implícito q(T)")

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
            "⚠️ **Limitação conhecida:** q(T) ~0.8% a.a. está abaixo do "
            "histórico NEFIN/USP (5-7%). Ver README."
        )

st.markdown("---")

st.header("4️⃣ Superfície de Volatilidade Implícita")

iv_data = extract_dataframe(bundle, "iv_data")

if iv_data is not None and len(iv_data) > 0:
    iv_data = iv_data.copy()
    iv_data["T_anos"] = iv_data.get("T", 0.5)

    x_col = "log_moneyness" if "log_moneyness" in iv_data.columns else "strike"
    z_col = "implied_vol" if "implied_vol" in iv_data.columns else "volatility"

    if x_col in iv_data.columns and z_col in iv_data.columns:
        fig_3d = px.scatter_3d(
            iv_data, x=x_col, y="T_anos", z=z_col, color=z_col,
            color_continuous_scale="Viridis",
            title="Superfície de Volatilidade Implícita",
            labels={x_col: "Log-moneyness ln(K/F)", "T_anos": "Prazo (anos)", z_col: "IV"},
        )
        fig_3d.update_layout(height=600, scene=dict(zaxis=dict(tickformat=".1%")))
        st.plotly_chart(fig_3d, use_container_width=True)

        col_x, col_y = st.columns(2)
        col_x.metric("🎯 Opções com IV válido", len(iv_data))
        col_y.metric("📅 Vencimentos distintos", iv_data["T_anos"].nunique())

st.markdown("---")

st.header("4️⃣b Smile de Volatilidade por Vencimento")

if iv_data is not None and len(iv_data) > 0:
    smile_df = iv_data.copy()
    group_col = None
    if "maturity_date" in smile_df.columns:
        group_col = "maturity_date"
    elif "T" in smile_df.columns:
        smile_df["T_group"] = (smile_df["T"] * 10).round() / 10
        group_col = "T_group"

    k_col = "log_moneyness" if "log_moneyness" in smile_df.columns else "strike"
    v_col = "implied_vol" if "implied_vol" in smile_df.columns else "volatility"
    type_col = "option_type" if "option_type" in smile_df.columns else None

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

            with st.expander("📋 Ver dados do smile"):
                st.dataframe(smile_filtered, use_container_width=True)
        else:
            st.info("Nenhum vencimento com ≥3 strikes.")

st.markdown("---")

st.header("5️⃣ Diagnóstico da Simulação Monte Carlo")

sim_diag = extract_json_object(bundle, "simulation_diagnostics")
if sim_diag:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Caminhos", f"{sim_diag.get('paths', 0):,}")
    col2.metric("📊 Passos", f"{sim_diag.get('steps', 0):,}")
    col3.metric("🌱 Seed", sim_diag.get("seed", "—"))
    col4.metric("📈 Monitoramento", sim_diag.get("monitoring", "—"))

    if pricing:
        col5, col6, col7 = st.columns(3)
        col5.metric("📉 Erro-padrão", f"{pricing.get('standard_error', 0):,.4f}")
        col6.metric("📊 IC 95% (inf)", f"{pricing.get('ci_low_95', 0):,.2f}")
        col7.metric("📊 IC 95% (sup)", f"{pricing.get('ci_high_95', 0):,.2f}")

if show_mc_viz:
    st.subheader("5️⃣b Visualização da Simulação Monte Carlo")

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

st.markdown("---")

if show_validation:
    st.header("6️⃣ Validação e Qualidade")

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
            "Barreira ≤ Vanilla + 5×SE": barrier_price <= vanilla_price + (5.0 * se),
            "P(knock-out) ∈ [0,1]": 0 <= ko <= 1,
            "Erro-padrão não-negativo": se >= 0,
            "IC contém preço": ci_low <= barrier_price <= ci_high,
            "Forward reconstruído": forward_ok,
        }

        pass_count = fail_count = 0
        for label, result in checks.items():
            if result:
                st.success(f"✅ {label}")
                pass_count += 1
            else:
                st.error(f"❌ {label}")
                fail_count += 1

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ PASS", pass_count)
        col2.metric("❌ FAIL", fail_count)
        total = pass_count + fail_count
        col3.metric("📊 Taxa", f"{pass_count / total * 100:.1f}%" if total > 0 else "—")

    st.markdown("---")

if show_export:
    st.header("7️⃣ Exportar Dados")
    market_date = metadata.get("market_date", "unknown") if metadata else "unknown"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📥 Download RESULTS (CSV)",
            data=bundle.to_csv(index=False),
            file_name=f"results_{market_date}.csv",
            mime="text/csv",
        )
    with col2:
        if curve_df is not None and len(curve_df) > 0:
            st.download_button(
                label="📥 Download Curva DI (CSV)",
                data=curve_df.to_csv(index=False),
                file_name=f"curve_{market_date}.csv",
                mime="text/csv",
            )
    with col3:
        if iv_data is not None and len(iv_data) > 0:
            st.download_button(
                label="📥 Download IV Data (CSV)",
                data=iv_data.to_csv(index=False),
                file_name=f"iv_{market_date}.csv",
                mime="text/csv",
            )
    st.markdown("---")

if show_history:
    st.header("8️⃣ Histórico de Execuções")

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
    st.markdown("---")

if show_raw_data:
    st.header("9️⃣ Dados Brutos")
    with st.expander("📊 Ver bundle RESULTS completo"):
        st.dataframe(bundle, use_container_width=True)
    with st.expander("📋 Ver lista de objetos no bundle"):
        st.write(sorted(bundle["object_name"].unique()))

st.markdown("---")
st.caption(
    f"📁 {selected_filename} | "
    f"🔄 Cache: 5 min | "
    f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"☁️ Databricks Apps"
)
