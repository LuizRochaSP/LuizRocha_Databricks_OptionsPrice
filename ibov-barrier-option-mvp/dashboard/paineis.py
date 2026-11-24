"""Painéis comuns ao IBOV e ao S&P 500: resumo do pricer, risco e gregas, histórico de preços, motor de Monte
Carlo e controles da precificação.

Recebem os objetos já lidos (dicionários do RESULTS e do RISK) e o nome do ativo; o IBOV (notebook 01) e o
S&P 500 (notebook 07) gravam a precificação com as mesmas chaves (``pricing_result`` e ``simulation_diagnostics``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import multiativos as ma

METODOLOGIA_PT = {
    "delta-gamma Monte Carlo with lognormal spot shocks": "Monte Carlo delta-gamma com choques lognormais no spot",
}


def resumo_pricer(metadata: dict, pricing: dict, nome_spot: str):
    """Data, spot, strike, vol, preços vanilla / down-and-out / analítico e o contrato."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data de Mercado", str(metadata.get("market_date", "—")))
    col2.metric(nome_spot, f"{metadata.get('spot', 0):,.2f}")
    col3.metric("Strike Alvo", f"{metadata.get('target_strike', 0):,.2f}")
    col4.metric("Vol. Implícita", f"{metadata.get('target_iv', 0):.2%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Preço Vanilla (BS)", f"{pricing.get('vanilla_price', 0):,.2f}")
    col6.metric("Preço Down-and-Out", f"{pricing.get('barrier_price', 0):,.2f}")
    col7.metric("Desconto Barreira", f"{pricing.get('barrier_discount_pct', 0):.2%}")
    col8.metric("P(knock-out)", f"{pricing.get('knock_out_probability', 0):.2%}")

    if "analytical_price" in pricing:
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


def risco(risk: dict | None, fonte: str | None, falta: str | None):
    """Gregas, DV01, VaR e ES do notebook 03. Sem arquivo de risco, mostra ``falta`` (o que não rodou)."""
    st.header("Risco & Gregas")
    if risk is None:
        if falta:
            st.error(falta)
        return
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

    metodologia = m.get("methodology", "—")
    st.caption(
        f"VaR/ES: {m.get('confidence_level', 0.99):.0%} de confiança, horizonte de "
        f"{m.get('horizon_days', 1)} dia(s), {m.get('scenarios', 0):,} cenários — "
        f"{METODOLOGIA_PT.get(metodologia, metodologia)}"
    )

    limitations = risk.get("limitations", [])
    if limitations:
        with st.expander("Limitações do cálculo de risco"):
            for lim in limitations:
                st.markdown(f"- {lim}")

    st.caption(
        f"Validação: {risk.get('validation_decision', '—')} "
        f"({risk.get('validation_warn_count', 0)} ressalvas) | "
        f"Fonte: `{fonte or '—'}`"
    )


HISTORICO_FMT = {"Spot": "{:,.2f}", "Vanilla": "{:,.2f}", "Down-and-Out": "{:,.2f}", "Desconto": "{:.2%}",
                 "P(knock-out)": "{:.2%}", "Vol ATM 1m": "{:.2%}", "Vol ATM 12m": "{:.2%}", "r 1a": "{:.3%}",
                 "q 1a": "{:.3%}"}


def historico_precos(hist: pd.DataFrame, unidade: str):
    """Tabela (percentuais em %, preços com 2 casas, sem índice) e o gráfico Vanilla x Down-and-Out."""
    fmt = {k: v for k, v in HISTORICO_FMT.items() if k in hist.columns}
    st.dataframe(ma.sem_vazios(hist, fmt), use_container_width=True, hide_index=True)
    precos = [c for c in ("Vanilla", "Down-and-Out") if c in hist.columns]
    if len(hist) > 1 and precos and hist[precos].notna().any().any():
        ma._chart(st, ma.fig_historico_precos(hist, unidade))


# ---------------------------------------------------------------- motor de Monte Carlo

@st.cache_data(max_entries=8)
def _simulacao(spot, r, q_val, sigma, strike, barrier, maturity, seed, monitoring, n_steps):
    """Amostra de 50 trajetórias e 5.000 caminhos para convergência e payoffs (refeita só quando o contrato muda)."""
    n_paths_plot = 50
    dt_plot = maturity / n_steps
    rng = np.random.default_rng(seed + 1)
    z_plot = rng.standard_normal((n_paths_plot, n_steps))
    drift = (r - q_val - 0.5 * sigma ** 2) * dt_plot
    diffusion = sigma * np.sqrt(dt_plot)
    log_paths_plot = np.log(spot) + np.cumsum(drift + diffusion * z_plot, axis=1)
    paths_with_t0 = np.concatenate([np.full((n_paths_plot, 1), spot), np.exp(log_paths_plot)], axis=1)
    time_grid = np.linspace(0, maturity, n_steps + 1)

    n_paths_conv = 5000
    rng_conv = np.random.default_rng(seed)
    z_conv = rng_conv.standard_normal((n_paths_conv, n_steps))
    uniforms_conv = rng_conv.random((n_paths_conv, n_steps))
    log_paths_c = np.log(spot) + np.cumsum(drift + diffusion * z_conv, axis=1)
    prev = np.concatenate((np.full((n_paths_conv, 1), np.log(spot)), log_paths_c[:, :-1]), axis=1)
    log_barrier = np.log(barrier)
    if monitoring == "discrete":
        alive_c = np.all(log_paths_c > log_barrier, axis=1)
    else:
        above = (prev > log_barrier) & (log_paths_c > log_barrier)
        cross_prob = np.ones_like(log_paths_c)
        cross_prob[above] = np.exp(
            -2.0 * (prev[above] - log_barrier) * (log_paths_c[above] - log_barrier) / (sigma ** 2 * dt_plot)
        )
        alive_c = np.all(above & (uniforms_conv > cross_prob), axis=1)
    terminal = np.exp(log_paths_c[:, -1])
    payoffs = np.exp(-r * maturity) * np.maximum(terminal - strike, 0.0) * alive_c
    running_mean = np.cumsum(payoffs) / np.arange(1, n_paths_conv + 1)
    return time_grid, paths_with_t0, running_mean, payoffs


def motor(pricing: dict | None, sim_diag: dict | None, ativo: str):
    """Diagnóstico da simulação (caminhos, passos, semente, monitoramento) e a visualização do Monte Carlo."""
    st.header("Diagnóstico da Simulação Monte Carlo")
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
    if not (pricing and sim_diag):
        return
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
        if not (spot > 0 and sigma > 0 and maturity > 0):
            return
        n_steps_plot = int(sim_diag.get("steps", 126))
        time_grid, paths_with_t0, running_mean, payoffs = _simulacao(
            spot, r, q_val, sigma, strike, barrier, maturity, seed, monitoring, n_steps_plot)

        st.markdown("**Caminhos simulados (amostra de 50)**")
        fig_paths = go.Figure()
        for i in range(len(paths_with_t0)):
            fig_paths.add_trace(go.Scatter(
                x=time_grid, y=paths_with_t0[i], mode="lines",
                line=dict(color="rgba(255,138,31,0.30)", width=1),  # laranja da rampa quente, translúcido
                showlegend=False, hoverinfo="skip",
            ))
        # rótulos dentro da área do gráfico; spot em branco sobre o fundo preto
        fig_paths.add_hline(y=spot, line_dash="solid", line_color="white",
                            annotation_text=f"Spot {spot:,.0f}", annotation_position="top left")
        fig_paths.add_hline(y=strike, line_dash="dash", line_color=ma.WARM[1],
                            annotation_text=f"Strike {strike:,.0f}", annotation_position="top left")
        fig_paths.add_hline(y=barrier, line_dash="dot", line_color=ma.WARM[3],
                            annotation_text=f"Barreira {barrier:,.0f}", annotation_position="bottom left")
        fig_paths.update_layout(xaxis_title="Tempo (anos)", yaxis_title=f"Spot {ativo} (pontos)", showlegend=False)
        ma._chart(st, ma.style(fig_paths, f"Trajetórias do {ativo} sob medida neutra ao risco", height=450))

        st.caption(f"Monitoramento: **{monitoring}** | GBM com r={r:.2%}, q={q_val:.2%}, σ={sigma:.2%}")

        col_conv, col_hist = st.columns(2)
        with col_conv:
            st.markdown("**Convergência do preço MC**")
            fig_conv = go.Figure(go.Scatter(x=np.arange(1, len(running_mean) + 1), y=running_mean, mode="lines",
                                            line=ma._line(2), name="Preço MC"))
            fig_conv.add_hline(y=pricing.get("barrier_price", 0), line_dash="dash", line_color=ma.WARM[0],
                               annotation_text=f"Final {pricing.get('barrier_price', 0):,.2f}",
                               annotation_position="right")
            fig_conv.update_layout(xaxis_title="Nº de caminhos", yaxis_title="Preço estimado", showlegend=False)
            ma._chart(st, ma.style(fig_conv, "Convergência", height=380))

        with col_hist:
            st.markdown("**Distribuição de payoffs**")
            fig_hist = go.Figure(go.Histogram(x=payoffs[payoffs > 0], nbinsx=50, marker_color=ma.WARM[2]))
            fig_hist.add_vline(x=float(payoffs.mean()), line_dash="dash", line_color=ma.WARM[3],
                               annotation_text=f"Média {payoffs.mean():,.2f}", annotation_position="top")
            fig_hist.update_layout(xaxis_title="Payoff descontado", yaxis_title="Frequência", showlegend=False)
            ma._chart(st, ma.style(fig_hist, "Payoffs", height=380))

        col_mc1, col_mc2, col_mc3, col_mc4 = st.columns(4)
        col_mc1.metric("Payoffs > 0", f"{(payoffs > 0).mean():.2%}")
        col_mc2.metric("Payoff médio", f"{payoffs.mean():,.2f}")
        col_mc3.metric("Payoff máx", f"{payoffs.max():,.2f}")
        col_mc4.metric("Payoff std", f"{payoffs.std():,.2f}")
    except Exception as e:
        st.warning(f"Erro ao gerar visualização MC: {e}")


def controles_precificacao(pricing: dict):
    """Checagens da precificação gravada (as mesmas para os dois ativos)."""
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
    col3.metric("Taxa", f"{pass_count / total * 100:.1f}%" if total > 0 else "—")
