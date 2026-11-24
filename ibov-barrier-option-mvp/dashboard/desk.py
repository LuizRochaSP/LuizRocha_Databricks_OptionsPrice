"""Structuring Desk: simulador de opções com barreira para o IBOV e o S&P 500.

O ativo vem do combo "Tipo de Instrumento" da barra lateral. O que muda entre ativos fica em ``AtivoDesk``
(moeda, unidade do nível, faixa do spot, valores de reserva); os 30 precificadores são os mesmos.

- Valores iniciais de spot, r, q e vol: do último arquivo do ativo (01 para o IBOV, 07 para o SPX), no prazo
  padrão de 6 meses (vol ATM). Sem arquivo, valores de reserva fixos.
- "Usar dados de mercado do dia": spot do arquivo, r(T) e q(T) no prazo escolhido e vol da superfície no strike
  (vol ATM nas estruturas sem strike), com as regras de ``mercado.py``.
- S&P 500 liquidado em R$ (Quanto): drift do SPX ajustado por -rho sigma_S sigma_X e desconto em r BRL
  (``quanto_market_data``); payouts e rebates em R$ entram em pontos (divididos pelo câmbio fixado) e o preço
  volta para R$ multiplicado pelo câmbio. Câmbio, r BRL, vol do câmbio e correlação vêm do notebook 09.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import mercado as mk
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
from ibov_barrier.pricing_quanto import QuantoMarket, quanto_market_data

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

STRATEGIES = [
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
]

TENORS = [("1 Mês", 21), ("3 Meses", 63), ("6 Meses", 126), ("1 Ano", 252)]


@dataclass(frozen=True)
class AtivoDesk:
    chave: str
    nome: str
    moeda: str          # moeda do preço (sem Quanto)
    nivel: str          # unidade do spot, strike e barreira nos rótulos
    dias: str           # rótulo dos dias do prazo
    eixo_x: str         # eixo do diagrama de payoff
    rotulo_taxa: str
    spot_min: float
    spot_max: float
    spot_step: float
    taxa_max: float     # % a.a.
    reserva: dict       # spot, rate, q e vol sem arquivo de mercado
    quanto: bool        # aceita liquidação em R$ (Quanto)


IBOV = AtivoDesk(
    chave="ibov", nome="IBOV", moeda="R$", nivel="R$", dias="DU", eixo_x="Preço do Ativo no Vencimento (R$)",
    rotulo_taxa="Taxa de Juros (%)", spot_min=10_000.0, spot_max=500_000.0, spot_step=1_000.0, taxa_max=30.0,
    reserva={"spot": 172_000.0, "rate": 0.12, "q": 0.02, "vol": 0.20}, quanto=False,
)
SPX = AtivoDesk(
    chave="spx", nome="S&P 500", moeda="US$", nivel="pontos", dias="pregões",
    eixo_x="Nível do S&P 500 no Vencimento (pontos)", rotulo_taxa="Taxa de Juros em US$ (%)",
    spot_min=500.0, spot_max=20_000.0, spot_step=10.0, taxa_max=15.0,
    reserva={"spot": 7_000.0, "rate": 0.045, "q": 0.012, "vol": 0.18}, quanto=True,
)
QUANTO_RESERVA = {"X0": 5.40, "r_brl": 0.14, "vol_fx": 0.15, "rho": -0.25, "data": None, "janela_rho": None}
LIQ_USD, LIQ_QUANTO = "US$ (opção em dólar)", "R$ (Quanto, câmbio fixo)"


def md(texto: str) -> str:
    """"$" escapado para st.caption/markdown: dois "$" na mesma linha viram fórmula LaTeX no Streamlit."""
    return texto.replace("$", "\\$")


def _faixa(valor: float, lo: float, hi: float, casas: int) -> float:
    return float(min(max(round(valor, casas), lo), hi))


def render(ativo: AtivoDesk, mercado: mk.MercadoDia | None = None, quanto_ref: dict | None = None):
    """Desenha a aba. ``mercado``: arquivo do dia do ativo; ``quanto_ref``: última execução do notebook 09."""
    st.header(f"Structuring Desk - Simulador de Opções com Barreira ({ativo.nome})")
    st.markdown("Monte a estrutura e visualize o payoff e o preço em tempo real.")

    def nivel(valor: float, casas: int = 0) -> str:
        """Nível do ativo para legendas (com "$" escapado)."""
        return f"R\\$ {valor:,.{casas}f}" if ativo.nivel == "R$" else f"{valor:,.{casas}f} pontos"

    col_esq, col_dir = st.columns([1, 2])

    with col_esq:
        st.subheader("Parâmetros da Operação")
        # a estratégia fica escolhida quando o ativo troca na barra lateral
        sim_strategy = st.selectbox("Estratégia", STRATEGIES, key="desk_estrategia")
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

        # Valores iniciais: arquivo do dia do ativo (prazo de 6 meses, vol ATM); sem arquivo, os de reserva.
        # A chave leva a data do arquivo: escolher outro arquivo devolve os campos ao novo padrão.
        pad = mk.padroes(mercado, ativo.reserva)
        tag = f"desk_{ativo.chave}_{pad['origem'] or 'reserva'}"
        usar_mercado = st.toggle(
            "Usar dados de mercado do dia", key=f"desk_{ativo.chave}_mercado", disabled=mercado is None,
            help=("Spot do arquivo, r(T) e q(T) no prazo escolhido e vol da superfície no strike "
                  "(vol ATM nas estruturas sem strike)." if mercado is not None else
                  "Sem arquivo de mercado deste ativo na pasta de resultados."),
        ) and mercado is not None
        quanto = ativo.quanto and st.radio(
            "Liquidação", [LIQ_USD, LIQ_QUANTO], horizontal=True, key="desk_spx_liquidacao",
            help="Quanto: paga em R$ o payoff em pontos vezes o câmbio fixado no contrato.",
        ) == LIQ_QUANTO
        moeda = "R$" if quanto else ativo.moeda
        moeda_md = md(moeda)

        if usar_mercado:
            sim_spot = mercado.spot
            st.markdown(f"Spot do {ativo.nome} em {mercado.data}: **{sim_spot:,.2f}**")
        else:
            sim_spot = st.number_input(
                f"Spot Inicial ({ativo.nivel})", min_value=ativo.spot_min, max_value=ativo.spot_max,
                value=_faixa(pad["spot"], ativo.spot_min, ativo.spot_max, 2), step=ativo.spot_step,
                key=f"{tag}_spot")
            sim_vol = st.slider("Volatilidade (%)", min_value=5.0, max_value=100.0,
                                value=_faixa(pad["vol"] * 100, 5.0, 100.0, 1), step=0.1, key=f"{tag}_vol") / 100.0
            sim_rate = st.slider(ativo.rotulo_taxa, min_value=0.0, max_value=ativo.taxa_max,
                                 value=_faixa(pad["rate"] * 100, 0.0, ativo.taxa_max, 2), step=0.01,
                                 key=f"{tag}_rate") / 100.0
            sim_q = st.slider("Dividend Yield q (%)", min_value=-10.0, max_value=10.0,
                              value=_faixa(pad["q"] * 100, -10.0, 10.0, 2), step=0.01, key=f"{tag}_q") / 100.0
            if pad["origem"]:
                st.caption(f"Valores iniciais do mercado de {pad['origem']} (prazo de 6 meses, vol ATM).")

        sim_strike_pct = st.slider("Strike (% do Spot)", min_value=50, max_value=150, value=90, step=1) if has_strike else 100
        sim_strike = sim_spot * (sim_strike_pct / 100.0)
        # 2ª perna (put) das estruturas combinadas da Família C
        if is_combined:
            sim_strike_2_pct = st.slider("Strike da 2ª Perna (% do Spot)", min_value=50, max_value=150, value=110, step=1)
            sim_strike_2 = sim_spot * (sim_strike_2_pct / 100.0)
        # Payout dos binários de toque (Família D) / Notional do Range Accrual
        if is_binary or is_double:
            sim_payout = st.number_input(f"Payout ({moeda})", min_value=1.0, max_value=1_000_000.0, value=100.0, step=10.0)
        sim_accrual_rate = None
        if is_range:
            sim_payout = st.number_input(f"Notional ({moeda})", min_value=1.0, max_value=100_000_000.0, value=1_000_000.0, step=10_000.0)
            sim_accrual_rate = st.number_input(
                "Taxa de Accrual (% a.a.)", min_value=0.1, max_value=100.0, value=10.0, step=0.5,
                help="Cupom = Notional × taxa × prazo × fração dos fixings dentro da faixa.",
            ) / 100.0
        # Rebate (Família A): pago no vencimento se a opção morrer (KO) ou nunca ativar (KI)
        sim_rebate = 0.0
        if is_single or is_path_barrier or sim_strategy in ("Double Knock-Out", "Double Knock-In"):
            sim_rebate = st.number_input(
                f"Rebate ({moeda})", min_value=0.0, max_value=1_000_000.0, value=0.0, step=10.0,
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

        tenor_map = {f"{nome} ({du} {ativo.dias})": du / 252 for nome, du in TENORS}
        sim_tenor = st.selectbox("Vencimento", list(tenor_map), index=2, key=f"desk_{ativo.chave}_prazo")
        sim_maturity = tenor_map[sim_tenor]
        # Monitoramento diário: 1 fechamento por pregão (252 por ano nas duas bolsas); datas arredondadas
        # para o pregão mais próximo.
        n_days = int(round(sim_maturity * 252))
        dt_day = sim_maturity / n_days
        if sim_strategy == "Parisian Barrier":
            sim_window = st.slider(f"Fechamentos Além da Barreira ({ativo.dias})", min_value=1, max_value=min(20, n_days),
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

        if usar_mercado:
            # Vol no strike (1ª perna nas estruturas de duas pernas); sem strike (binários, túnel, Range
            # Accrual), no forward. Mesma regra do notebook 01: linear em ln(K/F) x prazo.
            k_ref = sim_strike if has_strike else mercado.forward(sim_maturity)
            sim_vol, regra_vol = mercado.vol(k_ref, sim_maturity)
            sim_rate, sim_q = mercado.r(sim_maturity), mercado.q(sim_maturity)
            st.markdown(f"**Mercado de {mercado.data}, prazo de {sim_maturity:.2f} ano**")
            st.markdown(f"- Vol {'no strike' if has_strike else 'ATM'}: **{sim_vol:.2%}**\n"
                        f"- r(T): **{sim_rate:.3%}**\n"
                        f"- q(T): **{sim_q:.3%}**")
            nota = f"Fonte: {mercado.fonte}. Vol por interpolação {regra_vol} em ln(K/F) × prazo"
            if regra_vol != "linear":
                nota += f" (ponto fora dos dados; a superfície cobre prazos de {mercado.cobertura()})"
            st.caption(nota + ".")

        if quanto:
            ref = quanto_ref or QUANTO_RESERVA
            qtag = f"desk_quanto_{ref['data'] or 'reserva'}"
            st.markdown("**Câmbio e correlação (Quanto)**")
            # campos numéricos (não sliders): o valor inicial é o do 09 sem arredondar para o passo do controle
            fx_fixed = st.number_input("Câmbio fixado X0 (R\\$ por US\\$)", min_value=0.5, max_value=50.0,
                                       value=float(ref["X0"]), step=0.01, format="%.4f", key=f"{qtag}_x0")
            r_brl = st.number_input("Taxa de Juros em R$ (% a.a.)", min_value=0.0, max_value=50.0,
                                    value=float(ref["r_brl"] * 100), step=0.05, format="%.3f",
                                    key=f"{qtag}_rbrl") / 100.0
            vol_fx = st.number_input("Vol do USDBRL (% a.a.)", min_value=0.0, max_value=100.0,
                                     value=float(ref["vol_fx"] * 100), step=0.5, format="%.2f",
                                     key=f"{qtag}_volfx") / 100.0
            rho = st.number_input("Correlação S&P 500 x USDBRL", min_value=-1.0, max_value=1.0,
                                  value=float(ref["rho"]), step=0.05, format="%.3f", key=f"{qtag}_rho")
            q_mkt = QuantoMarket(spot=sim_spot, rate_usd=sim_rate, dividend_yield=sim_q, vol_asset=sim_vol,
                                 fx_spot=fx_fixed, rate_brl=r_brl, vol_fx=vol_fx, correlation=rho)
            # drift do S&P 500 = r US$ - q - rho sigma_S sigma_X; desconto em r BRL
            m_quanto = quanto_market_data(q_mkt)
            r_calc, q_calc, escala = m_quanto.rate, m_quanto.dividend_yield, fx_fixed
            st.caption(f"Ajuste Quanto no drift do S&P 500: {q_mkt.quanto_adjustment * 1e4:+.0f} bps. "
                       + (f"Valores iniciais do notebook 09 (mercado de {ref['data']})." if quanto_ref else
                          "Sem execução do notebook 09: valores ilustrativos."))
        else:
            r_calc, q_calc, escala = sim_rate, sim_q, 1.0

    # Payout, rebate e notional na unidade do motor (pontos do índice). No Quanto eles são digitados em R$:
    # entram divididos pelo câmbio fixado e o resultado volta para R$ multiplicado por ele (o preço é
    # linear nesses valores). Sem Quanto, escala = 1 e nada muda.
    pay_c = sim_payout / escala if (is_binary or is_double or is_range) else None
    reb_c = sim_rebate / escala

    with col_dir:
        st.subheader("Diagrama de Payoff e Precificação")

        # GRÁFICO DE PAYOFF
        s_t = np.linspace(sim_spot * 0.5, sim_spot * 1.5, 200)
        if is_range:
            # Range Accrual: cada fixing dentro da faixa acumula Notional/N.
            # Perfil = Notional se o ativo permanecesse no nível S em todos os fixings.
            lo_level = sim_lower if sim_lower is not None else 0.0
            hi_level = sim_upper if sim_upper is not None else np.inf
            sim_coupon = pay_c * sim_accrual_rate * sim_maturity
            payoff = np.where((s_t >= lo_level) & (s_t <= hi_level), sim_coupon, 0.0)
        elif is_family_b:
            # Família B: vanilla condicionada às duas barreiras (perfil terminal indicativo).
            vanilla_payoff = np.maximum(s_t - sim_strike, 0) if dbl_option_type == "call" else np.maximum(sim_strike - s_t, 0)
            inside = (s_t > sim_lower) & (s_t < sim_upper)
            if sim_strategy == "Double Knock-Out":
                payoff = np.where(inside, vanilla_payoff, reb_c)
            elif sim_strategy == "Double Knock-In":
                payoff = np.where(inside, reb_c, vanilla_payoff)
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
            payoff = np.where(inside, 0.0, pay_c) if is_one_touch else np.where(inside, pay_c, 0.0)
        elif is_binary:
            # Família D: payoff binário — payout fixo do cliente.
            # One-Touch paga além da barreira (toque implícito no terminal);
            # No-Touch paga aquém dela (barreira jamais tocada nesse cenário).
            touched_region = s_t <= sim_barrier if barrier_is_down else s_t >= sim_barrier
            payoff = np.where(touched_region if is_one_touch else ~touched_region, pay_c, 0.0)
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
                payoff[s_t > sim_barrier] = reb_c
            else:
                payoff[s_t < sim_barrier] = reb_c
        else:
            if barrier_is_down:
                payoff[s_t <= sim_barrier] = reb_c
            else:
                payoff[s_t >= sim_barrier] = reb_c
        # pontos -> moeda do preço (no Quanto, x câmbio fixado)
        payoff = payoff * escala

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
            fig_payoff.add_hline(y=sim_coupon * escala, line_dash="dash", line_color="#00A3E0",
                                 annotation_text=f"Cupom máximo {sim_accrual_rate:.2%} a.a. × {sim_maturity:.2f} ano ({moeda} {sim_coupon * escala:,.2f})")
        elif is_binary or is_double:
            # Linha horizontal tracejada no nível do Payout/Notional escolhido
            level_label = "Notional" if is_range else "Payout"
            fig_payoff.add_hline(y=sim_payout, line_dash="dash", line_color="#00A3E0", annotation_text=f"{level_label} ({moeda} {sim_payout:,.0f})")
        payoff_title = "Diagrama de Accrual (cupom se S ficar na faixa em todos os fixings)" if is_range else "Payoff no Vencimento"
        if is_path_barrier:
            payoff_title = ("Payoff no Vencimento (se " + ("viva" if path_knock == "out" else "ativada")
                            + "; o evento de barreira depende da trajetória)")
        if quanto:
            payoff_title += f" em R$ (pontos × {fx_fixed:,.4f})"
        fig_payoff.update_layout(title=payoff_title, xaxis_title=ativo.eixo_x, yaxis_title=f"Lucro/Prejuízo ({moeda})", height=350, margin=dict(l=20, r=20, t=40, b=20))
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
                    # r_calc/q_calc: os do ativo; no Quanto, desconto em r BRL e drift ajustado (quanto_market_data)
                    market_sim = MarketData(spot=sim_spot, rate=r_calc, dividend_yield=q_calc, volatility=sim_vol)
                    contract_sim = BarrierContract(strike=sim_strike, barrier=sim_barrier, maturity=sim_maturity)

                    try:
                        mc = dict(paths=50_000, steps=126, seed=42)
                        benchmark = None  # benchmark analítico (Famílias B e D), quando disponível
                        if is_family_b:
                            # Família B: motor com instante de toque por passo (L e U simultâneas)
                            bench_args = (sim_spot, sim_strike)
                            mkt_args = (sim_maturity, r_calc, q_calc, sim_vol)
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
                                    rebate=reb_c, **mc,
                                )
                                # Ikeda-Kunitomo (DKO) + rebate via P(Double No-Touch) analítica
                                dko = double_knock_out_analytical(*bench_args, sim_lower, sim_upper, *mkt_args,
                                                                  option_type=dbl_option_type)
                                p_dnt = double_no_touch_probability_analytical(sim_spot, sim_lower, sim_upper, *mkt_args)
                                rebate_pv = reb_c * np.exp(-r_calc * sim_maturity)
                                if knock == "out":
                                    benchmark = dko + rebate_pv * (1.0 - p_dnt)
                                else:
                                    benchmark = result.vanilla_price - dko + rebate_pv * p_dnt
                        elif is_path_barrier:
                            # Roadmap 8-10: 1 passo por pregão (fechamentos diários)
                            path_kw = dict(option_type=path_option_type, direction=path_direction,
                                           knock=path_knock, rebate=reb_c)
                            path_mc = dict(paths=50_000, steps=n_days, seed=42)
                            mkt_args = (sim_maturity, r_calc, q_calc, sim_vol)
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
                                knock=knock, rebate=reb_c, **mc,
                            )
                        elif is_range:
                            # Range Accrual: 1 fixing por pregão do prazo
                            n_fixings = int(round(sim_maturity * 252))
                            result = price_range_accrual(
                                market_sim, sim_maturity, lower=sim_lower, upper=sim_upper,
                                notional=pay_c, paths=50_000, steps=n_fixings, seed=42,
                                accrual_rate=sim_accrual_rate,
                            )
                            benchmark = result.analytical_price
                        elif is_double:
                            double_fn = price_double_one_touch if is_one_touch else price_double_no_touch
                            result = double_fn(
                                market_sim, lower=sim_lower, upper=sim_upper, maturity=sim_maturity,
                                payout=pay_c, **mc,
                            )
                            p_dnt = double_no_touch_probability_analytical(
                                sim_spot, sim_lower, sim_upper, sim_maturity, r_calc, q_calc, sim_vol
                            )
                            p_pay = (1.0 - p_dnt) if is_one_touch else p_dnt
                            benchmark = pay_c * np.exp(-r_calc * sim_maturity) * p_pay
                        elif sim_strategy in ("One-Touch Down", "No-Touch Down"):
                            down_fn = price_one_touch_down if is_one_touch else price_no_touch_down
                            result = down_fn(market_sim, contract_sim, payout=pay_c, **mc)
                            p_hit = barrier_hit_probability_analytical(
                                sim_spot, sim_barrier, sim_maturity, r_calc, q_calc, sim_vol
                            )
                            p_pay = p_hit if is_one_touch else 1.0 - p_hit
                            benchmark = pay_c * np.exp(-r_calc * sim_maturity) * p_pay
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
                            result = price_one_touch_up(market_sim, contract_sim, payout=pay_c, paths=50_000, steps=126, seed=42)
                        elif sim_strategy == "No-Touch Up":
                            # Família D: binário — payout fixo se NUNCA tocar a barreira superior
                            result = price_no_touch_up(market_sim, contract_sim, payout=pay_c, paths=50_000, steps=126, seed=42)
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

                        # pontos do índice -> moeda do preço (no Quanto, x câmbio fixado)
                        price_out = result.price * escala
                        if sim_strategy == "Barrier Risk Reversal" and price_out < 0:
                            st.success("Cálculo concluído! Preço negativo = crédito líquido RECEBIDO (a venda da put custeia a call).")
                        elif sim_strategy != "Barrier Risk Reversal" and abs(price_out) <= 0.01:
                            st.warning("**Calma lá!** Com esses parâmetros a opção está valendo praticamente zero. Tente ajustar o Strike ou a Barreira.")
                        else:
                            st.success("Cálculo concluído com sucesso! Estrutura pronta para o cliente.")

                        col_r1, col_r2, col_r3 = st.columns(3)
                        col_r1.metric("Preço da Estrutura", f"{moeda} {price_out:,.2f}")
                        if benchmark is not None:
                            # Família D: benchmark analítico (monitoramento contínuo) ao lado do MC
                            col_r2.metric("Benchmark Analítico", f"{moeda} {benchmark * escala:,.2f}")
                        else:
                            col_r2.metric("Preço Vanilla", f"{moeda} {result.vanilla_price * escala:,.2f}")
                        if is_range:
                            col_r3.metric("Fração Esperada na Faixa", f"{result.expected_fraction:.2%}")
                            # "R\$": dois "$" na mesma legenda viram fórmula LaTeX no Streamlit
                            st.caption(f"Cupom máximo {moeda_md} {sim_payout * sim_accrual_rate * sim_maturity:,.2f} "
                                       f"({sim_accrual_rate:.2%} a.a. × {sim_maturity:.2f} ano) | "
                                       f"{result.observations} fixings (1 por pregão) | IC 95%: "
                                       f"[{moeda_md} {result.ci_low * escala:,.2f}; {moeda_md} {result.ci_high * escala:,.2f}]")
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
                            st.caption(f"Inclui rebate de {moeda_md} {sim_rebate:,.2f} pago no vencimento "
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
                                detail = (f"Barreira {nivel(sim_barrier)} até {step_change_time:.3f} ano, "
                                          f"depois {nivel(sim_barrier_2)}. ")
                            else:
                                detail = (f"Barreira monitorada de {sim_windows[0][0]:.3f} a "
                                          f"{sim_windows[0][1]:.3f} ano. ")
                            st.caption(detail + f"{n_days} passos (1 por pregão). Benchmark: {path_bench_text}. "
                                       f"IC 95% do MC: [{moeda_md} {result.ci_low * escala:,.2f}; "
                                       f"{moeda_md} {result.ci_high * escala:,.2f}].")
                        usados = (f"Parâmetros usados: spot {sim_spot:,.2f} | vol {sim_vol:.2%} | "
                                  f"r {sim_rate:.3%} | q {sim_q:.3%}")
                        if quanto:
                            usados += (f" | Quanto: câmbio {fx_fixed:,.4f}, r BRL {r_brl:.3%}, vol USDBRL "
                                       f"{vol_fx:.2%}, correlação {rho:+.2f} (drift {r_calc - q_calc:.3%}, "
                                       f"preço em pontos × câmbio)")
                        st.caption(md(usados) + ("" if not usar_mercado else f" | mercado de {mercado.data}"))
                    except Exception:
                        st.error("**Ops!** Não conseguimos calcular essa estrutura. Verifique os parâmetros.")
