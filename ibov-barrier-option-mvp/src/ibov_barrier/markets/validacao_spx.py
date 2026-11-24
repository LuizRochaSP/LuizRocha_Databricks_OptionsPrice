"""Validação do RESULTS do S&P 500 (notebook 07) antes do risco (notebook 03).

Mesmas regras e mesma evidência do notebook 02 do IBOV: cada controle sai PASS, WARN ou FAIL; a decisão é
APROVADO (nenhum WARN), APROVADO COM RESSALVAS (só WARN) ou REPROVADO (algum FAIL), e o notebook 11 só grava
``VALIDATION_spx_superficie_<ts>.json`` sem FAIL. O 03 recusa qualquer RESULTS alterado depois (SHA-256).

Blocos: integridade, superfície, forwards e carry, precificação (replay determinístico do Monte Carlo).
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from ..analytical import price_down_and_out_call_analytical
from ..pricing import BarrierContract, MarketData, price_down_and_out_call
from .referencia import curvas_no_prazo, vol_no_ponto

OBJETOS = ("metadata", "forwards", "options", "pricing_result", "simulation_diagnostics")
COLUNAS = ["bloco", "controle", "status", "observado", "critério", "interpretação"]
ORDEM = pd.CategoricalDtype(["FAIL", "WARN", "PASS"], ordered=True)


def _faixa(valor, pass_ate, warn_ate):
    """PASS até ``pass_ate``, WARN até ``warn_ate``, FAIL acima (comparação em módulo)."""
    v = abs(valor)
    return "PASS" if v <= pass_ate else "WARN" if v <= warn_ate else "FAIL"


def validar_spx(b: dict, *, d1_esperado: date | None = None, replay: bool = True) -> pd.DataFrame:
    """Controles do RESULTS do SPX já lido (``results_bundle.read_bundle``)."""
    linhas = []

    def add(bloco, controle, status, observado, criterio, interpretacao):
        linhas.append(dict(zip(COLUNAS, (bloco, controle, status, str(observado), criterio, interpretacao))))

    # ---- 1. integridade
    faltam = [o for o in OBJETOS if o not in b]
    add("Integridade", "Objetos do RESULTS", "FAIL" if faltam else "PASS",
        ", ".join(faltam) or "todos presentes", "metadata, forwards, options, pricing_result, simulation_diagnostics",
        "Sem eles não há como reconstruir o preço nem calcular o risco." if faltam else "RESULTS completo.")
    if faltam:
        return pd.DataFrame(linhas, columns=COLUNAS)
    meta, opts, fwd = b["metadata"], b["options"], b["forwards"]
    pr, sim = b["pricing_result"], b["simulation_diagnostics"]
    md = str(meta.get("market_date"))
    if d1_esperado is not None:
        ok = md == d1_esperado.isoformat()
        add("Integridade", "Data de mercado", "PASS" if ok else "WARN", md, f"D-1 NYSE = {d1_esperado}",
            "Fotografia do último pregão." if ok else "Não é o último pregão: use só para análise histórica.")

    # ---- 2. superfície
    n = len(opts)
    add("Superfície", "Pontos na superfície", "PASS" if n >= 5000 else "WARN" if n >= 3000 else "FAIL", n,
        "≥ 5.000 (WARN de 3.000 a 5.000)", "Quantidade de cotações limpas com vol implícita.")
    nv = int(opts["expiration"].nunique())
    add("Superfície", "Vencimentos com vol", "PASS" if nv >= 20 else "WARN" if nv >= 10 else "FAIL", nv,
        "≥ 20 (WARN de 10 a 20)", "Cobertura da estrutura a termo.")
    vmin, vmax = float(opts["implied_vol"].min()), float(opts["implied_vol"].max())
    ok = 0.03 <= vmin and vmax <= 3.0
    add("Superfície", "Faixa da vol implícita", "PASS" if ok else "FAIL", f"{vmin:.2%} a {vmax:.2%}",
        "entre 3% e 300%", "Vol fora da faixa indica cotação ou inversão de Black-Scholes com problema.")
    cob = meta.get("cobertura")
    add("Superfície", "Cobertura frente ao histórico", "WARN" if cob else "PASS", cob or "normal",
        "sem alerta do notebook 07", "Dia com menos pontos que o habitual." if cob else "Cobertura normal.")
    if "arb" in b and len(b["arb"]):
        antes, rem = float(b["arb"]["antes"].sum()), float(b["arb"]["removidos"].sum())
        share = rem / antes if antes else 0.0
        add("Superfície", "Cortes por arbitragem vertical/butterfly",
            "PASS" if share <= 0.15 else "WARN" if share <= 0.30 else "FAIL", f"{share:.1%}",
            "≤ 15% (WARN até 30%)", "Parcela das cotações removidas por violar não arbitragem.")

    # ---- 3. forwards e carry
    avisos = meta.get("avisos") or []
    add("Forwards e carry", "Avisos da paridade", "WARN" if avisos else "PASS", len(avisos), "nenhum",
        "; ".join(map(str, avisos))[:300] or "Sem vencimento fora da curva r(T).")
    faixa = tuple(meta.get("faixa q") or (0.0, 0.03))
    longos = fwd.loc[fwd["T"] >= float(meta.get("T minimo taxa", 0.25))]
    fora = int((~longos["q"].between(*faixa)).sum())
    add("Forwards e carry", "q(T) nos prazos a partir de 3 meses", "PASS" if fora == 0 else "WARN",
        f"{fora} de {len(longos)} fora", f"dentro de [{faixa[0]:.0%}, {faixa[1]:.0%}]",
        "Dividendo implícito fora do razoável sugere forward ou spot inconsistentes.")
    r1, q1 = curvas_no_prazo(fwd, 1.0)
    sofr = meta.get("sofr")
    if sofr is None:
        add("Forwards e carry", "r 1 ano - SOFR", "WARN", "sem SOFR", "SOFR do dia disponível",
            "Sem referência externa para a taxa da paridade.")
    else:
        spread = (r1 - float(sofr)) * 1e4
        add("Forwards e carry", "r 1 ano - SOFR", "PASS" if -25 <= spread <= 200 else "WARN", f"{spread:+.0f} bps",
            "entre -25 e +200 bps", "A taxa da paridade costuma ficar 75 a 115 bps acima da SOFR (notebook 08).")
    gap = meta.get("spot_gap_bps")
    if gap is None:
        add("Forwards e carry", "Spot: fechamento x paridade", "WARN", "sem diagnóstico",
            "RESULTS gravado com o diagnóstico de spot", "Arquivo anterior ao diagnóstico de spot do notebook 07.")
    else:
        add("Forwards e carry", "Spot: fechamento x paridade", _faixa(gap, 25, 100), f"{float(gap):+.1f} bps",
            "|diferença| ≤ 25 bps (WARN até 100)",
            f"Spot usado: {meta.get('spot_fonte', '—')}. Diferença grande indica fechamento do índice e cotações "
            "das opções em horários diferentes.")

    # ---- 4. precificação
    c, m = pr["contract"], pr["market"]
    S, K, H, T = float(m["spot"]), float(c["strike"]), float(c["barrier"]), float(c["maturity"])
    r, q, vol = float(m["rate"]), float(m["dividend_yield"]), float(m["volatility"])
    preco, vanilla, se = float(pr["barrier_price"]), float(pr["vanilla_price"]), float(pr["standard_error"])
    add("Precificação", "Contrato coerente", "PASS" if 0 < H < S and K > 0 and T > 0 else "FAIL",
        f"K {K:,.2f} | H {H:,.2f} | T {T:.4f}", "0 < barreira < spot, strike e prazo positivos",
        "Down-and-out com a barreira abaixo do spot.")
    r_ref, q_ref = curvas_no_prazo(fwd, T)
    # tolerâncias na precisão do arquivo: o to_json do pandas grava as tabelas com 10 dígitos
    ok = abs(r - r_ref) < 1e-9 and abs(q - q_ref) < 1e-9
    add("Precificação", "r(T) e q(T) reproduzidos", "PASS" if ok else "FAIL", f"r {r:.4%} | q {q:.4%}",
        "iguais aos forwards do arquivo", "A taxa e o dividendo usados saem da paridade do próprio dia.")
    f_ref = S * math.exp((r - q) * T)
    ok = abs(float(m.get("forward", f_ref)) / f_ref - 1) < 1e-9
    add("Precificação", "Forward reconstruído", "PASS" if ok else "FAIL", f"{float(m.get('forward', f_ref)):,.2f}",
        "S·exp((r - q)·T)", "Forward coerente com spot, taxa e dividendo.")
    v_ref, metodo = vol_no_ponto(opts, K, T, f_ref)
    add("Precificação", "Vol no strike reproduzida", "PASS" if abs(vol - v_ref) < 1e-8 else "FAIL",
        f"{vol:.4%} ({metodo})", "igual à superfície do arquivo",
        "Mesma regra do notebook 01: linear em ln(K/F) x prazo; fora dos dados, o ponto mais próximo.")
    add("Precificação", "Preço não negativo", "PASS" if preco >= 0 else "FAIL", f"{preco:,.4f}", "≥ 0", "")
    add("Precificação", "Barreira ≤ vanilla + 5 erros-padrão", "PASS" if preco <= vanilla + 5 * se else "FAIL",
        f"{preco:,.2f} ≤ {vanilla + 5 * se:,.2f}", "a barreira só retira valor", "")
    pko = float(pr["knock_out_probability"])
    add("Precificação", "P(knock-out) em [0; 1]", "PASS" if 0 <= pko <= 1 else "FAIL", f"{pko:.4%}", "[0; 1]", "")
    ok = float(pr["ci_low_95"]) <= preco <= float(pr["ci_high_95"])
    add("Precificação", "IC 95% contém o preço", "PASS" if ok else "FAIL",
        f"[{float(pr['ci_low_95']):,.2f}; {float(pr['ci_high_95']):,.2f}]", "inferior ≤ preço ≤ superior", "")
    analitico = price_down_and_out_call_analytical(spot=S, strike=K, barrier=H, maturity=T, rate=r,
                                                   dividend_yield=q, volatility=vol)
    erro = abs(preco - analitico) / analitico if analitico > 0 else 0.0
    add("Precificação", "MC x Reiner-Rubinstein", "PASS" if erro < 0.01 else "WARN" if erro < 0.02 else "FAIL",
        f"{erro:.4%}", "< 1% (WARN até 2%)", "Monitoramento contínuo (ponte browniana) contra a fórmula fechada.")
    if replay:
        res = price_down_and_out_call(MarketData(S, r, q, vol), BarrierContract(K, H, T), paths=int(sim["paths"]),
                                      steps=int(sim["steps"]), seed=int(sim["seed"]),
                                      monitoring=str(sim["monitoring"]))
        ok = abs(res.price - preco) <= 1e-9 * max(1.0, abs(preco))
        add("Precificação", "Replay determinístico do Monte Carlo", "PASS" if ok else "FAIL",
            f"{res.price:,.6f}", f"= {preco:,.6f} (mesma semente)",
            "Mesmos parâmetros e semente reproduzem o preço gravado.")
    return pd.DataFrame(linhas, columns=COLUNAS)


def decisao(checks: pd.DataFrame) -> tuple[str, int, int, int]:
    """(decisão, PASS, WARN, FAIL), com a regra do notebook 02."""
    n = checks["status"].value_counts()
    n_pass, n_warn, n_fail = (int(n.get(s, 0)) for s in ("PASS", "WARN", "FAIL"))
    if n_fail:
        return "REPROVADO", n_pass, n_warn, n_fail
    return ("APROVADO COM RESSALVAS" if n_warn else "APROVADO"), n_pass, n_warn, n_fail


def ordenar(checks: pd.DataFrame) -> pd.DataFrame:
    out = checks.assign(_o=checks["status"].astype(ORDEM))
    return out.sort_values(["_o", "bloco"], kind="stable").drop(columns="_o").reset_index(drop=True)


def evidencia(checks: pd.DataFrame, *, results_file: str, results_sha256: str, metadata: dict,
              validation_timestamp: str) -> dict:
    """JSON de aprovação no formato do notebook 02 (o 03 lê decision, fail_count e results_sha256)."""
    dec, n_pass, n_warn, n_fail = decisao(checks)
    warns = checks.loc[checks["status"].eq("WARN"), ["bloco", "controle", "observado", "critério", "interpretação"]]
    return {
        "schema_version": 1,
        "validation_timestamp": validation_timestamp,
        "results_file": results_file,
        "results_sha256": results_sha256,
        "results_schema_version": metadata.get("schema_version"),
        "execution_timestamp": metadata.get("execution_timestamp"),
        "market_date": str(metadata.get("market_date")),
        "decision": dec,
        "approved_for_risk": n_fail == 0,
        "pass_count": n_pass,
        "warn_count": n_warn,
        "fail_count": n_fail,
        "warnings": warns.to_dict(orient="records"),
        "checks": checks.to_dict(orient="records"),
    }
