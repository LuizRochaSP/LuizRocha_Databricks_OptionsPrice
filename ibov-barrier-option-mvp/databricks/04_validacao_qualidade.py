# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "lxml",
# ]
# ///
# MAGIC %md
# MAGIC # Validação e qualidade dos dados e da superfície de volatilidade
# MAGIC
# MAGIC Este notebook responde a uma pergunta diferente de “o código executou?”:
# MAGIC
# MAGIC > **Os dados, as curvas e as volatilidades são suficientemente consistentes para serem usados na precificação e no risco?**
# MAGIC
# MAGIC Os controles são classificados em:
# MAGIC
# MAGIC | Status | Significado |
# MAGIC |---|---|
# MAGIC | `PASS` | controle aprovado |
# MAGIC | `WARN` | resultado utilizável com ressalva e investigação |
# MAGIC | `FAIL` | falha material; não se deve promover o resultado sem correção |
# MAGIC
# MAGIC O notebook `03_superficie_volatilidade` continua sendo a fonte única da leitura e
# MAGIC transformação dos dados. Este notebook apenas valida seus objetos e resultados.

# COMMAND ----------

# MAGIC %run ./03_superficie_volatilidade

# COMMAND ----------

# Comentário: configura tolerâncias transparentes dos controles de qualidade.
STRICT_MODE = False
MAX_RELATIVE_SPREAD = 0.30
MAX_REPRICING_ERROR = 0.01
MAX_PARITY_ERROR_PCT_SPOT = 0.005
MIN_POINTS_PER_SMILE = 3
MIN_SURFACE_MATURITIES = 2
MIN_SURFACE_POINTS = 6
CURVE_ABS_RATE_LIMIT = 1.00
MONOTONICITY_TOLERANCE = 0.01
CONVEXITY_SLOPE_TOLERANCE = 0.01

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Estrutura do relatório
# MAGIC
# MAGIC Cada teste registra o bloco avaliado, o valor observado, o critério usado e uma
# MAGIC interpretação. Os testes não escondem linhas rejeitadas: as exceções ficam em tabelas
# MAGIC próprias para investigação.

# COMMAND ----------

# Comentário: cria o coletor padronizado de resultados PASS, WARN e FAIL.
quality_checks = []


def add_check(block, control, status, observed, criterion, detail):
    if status not in {"PASS", "WARN", "FAIL"}:
        raise ValueError(f"Status inválido: {status}")
    quality_checks.append({
        "bloco": str(block),
        "controle": str(control),
        "status": status,
        "observado": str(observed),
        "critério": str(criterion),
        "interpretação": str(detail),
    })


def status_from_count(count, warning_only=False):
    if count == 0:
        return "PASS"
    return "WARN" if warning_only else "FAIL"


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Integridade dos arquivos e da fotografia de mercado
# MAGIC
# MAGIC Uma fotografia válida precisa ter os quatro arquivos, ZIPs íntegros, XMLs legíveis e
# MAGIC uma única data de avaliação. A seleção automática da última pasta completa evita usar
# MAGIC acidentalmente uma data parcial.

# COMMAND ----------

# Comentário: valida presença, integridade ZIP, conteúdo XML e coerência da pasta selecionada.
required_archives = {"IN": in_zip, "IR": ir_zip, "SPRD": sprd_zip, "PR": pr_zip}
missing_archives = [name for name, path in required_archives.items() if not path.exists()]
add_check(
    "Arquivos", "Quatro fontes obrigatórias", status_from_count(len(missing_archives)),
    f"{4 - len(missing_archives)}/4", "IN, IR, SPRD e PR presentes",
    "Todos os insumos foram localizados." if not missing_archives else f"Ausentes: {missing_archives}",
)

archive_diagnostics = []
for source, zip_path in required_archives.items():
    diagnostic = {"fonte": source, "arquivo": zip_path.name, "zip_ok": False, "xmls": 0, "erro": ""}
    try:
        with ZipFile(zip_path) as archive:
            bad_member = archive.testzip()
            xml_members = [x for x in archive.namelist() if x.lower().endswith(".xml")]
            diagnostic["zip_ok"] = bad_member is None
            diagnostic["xmls"] = len(xml_members)
            if bad_member:
                diagnostic["erro"] = f"Membro corrompido: {bad_member}"
            elif not xml_members:
                diagnostic["erro"] = "ZIP sem XML"
    except Exception as exc:
        diagnostic["erro"] = f"{type(exc).__name__}: {exc}"
    archive_diagnostics.append(diagnostic)

archive_diagnostics = pd.DataFrame(archive_diagnostics)
bad_archives = archive_diagnostics.loc[
    (~archive_diagnostics["zip_ok"]) | archive_diagnostics["xmls"].eq(0)
]
add_check(
    "Arquivos", "Integridade dos ZIPs e XMLs", status_from_count(len(bad_archives)),
    f"{len(archive_diagnostics) - len(bad_archives)}/{len(archive_diagnostics)} íntegros",
    "ZIP íntegro e ao menos um XML por fonte",
    "Arquivos podem ser processados." if bad_archives.empty else "Há arquivo corrompido ou sem XML.",
)

folder_date_ok = market_folder.name == valuation_date.isoformat()
add_check(
    "Arquivos", "Data da pasta e data de avaliação", "PASS" if folder_date_ok else "FAIL",
    f"pasta={market_folder.name}; avaliação={valuation_date.isoformat()}", "datas idênticas",
    "A fotografia está coerente." if folder_date_ok else "A data da pasta diverge da avaliação.",
)
display(archive_diagnostics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Cadastro, preços e liquidez
# MAGIC
# MAGIC Aqui separamos “há um preço” de “há um preço representativo”. Um último negócio pode
# MAGIC existir sem bid/ask executável; por isso ele é aceito na construção inicial, mas recebe
# MAGIC uma ressalva de qualidade.

# COMMAND ----------

# Comentário: mede duplicidades, campos essenciais, origem do preço e largura do spread.
catalog_duplicate_count = int(catalog["ticker"].duplicated().sum())
price_duplicate_count = int(prices["ticker"].duplicated().sum())
add_check(
    "Cadastro", "Tickers únicos após tratamento",
    status_from_count(catalog_duplicate_count + price_duplicate_count),
    f"cadastro={catalog_duplicate_count}; preços={price_duplicate_count}", "zero duplicidades",
    "Chaves estão únicas." if catalog_duplicate_count + price_duplicate_count == 0 else "Rever regra de deduplicação.",
)

essential_columns = ["ticker", "strike", "maturity_date", "option_type", "option_style"]
missing_essential = int(catalog[essential_columns].isna().any(axis=1).sum())
add_check(
    "Cadastro", "Campos essenciais preenchidos", status_from_count(missing_essential),
    f"{missing_essential} linha(s) incompleta(s)", "zero linhas incompletas",
    "Cadastro utilizável." if missing_essential == 0 else "Há instrumentos sem atributo essencial.",
)

positive_prices = options["market_price"].gt(0)
add_check(
    "Preços", "Preço de mercado positivo",
    "PASS" if positive_prices.all() else "WARN",
    f"{int(positive_prices.sum())}/{len(options)}", "100% positivos",
    "Todas as linhas têm preço válido." if positive_prices.all() else "Linhas sem preço positivo serão filtradas.",
)

options_quality = options.copy()
options_quality["relative_spread"] = np.where(
    valid_spread,
    (options_quality["best_ask"] - options_quality["best_bid"])
    / ((options_quality["best_ask"] + options_quality["best_bid"]) / 2.0),
    np.nan,
)
mid_share = safe_ratio(options_quality["price_source"].eq("mid").sum(), len(options_quality))
add_check(
    "Liquidez", "Preço baseado em bid/ask",
    "PASS" if mid_share >= 0.50 else "WARN",
    f"{mid_share:.1%}" if np.isfinite(mid_share) else "sem observações", "ao menos 50% por mid",
    "Maioria baseada em cotação bilateral." if mid_share >= 0.50 else "Predominam últimos negócios; possível defasagem.",
)

wide_spreads = options_quality.loc[options_quality["relative_spread"].gt(MAX_RELATIVE_SPREAD)].copy()
quoted_count = int(options_quality["relative_spread"].notna().sum())
wide_share = safe_ratio(len(wide_spreads), quoted_count)
add_check(
    "Liquidez", "Spread relativo das cotações",
    "PASS" if len(wide_spreads) == 0 else "WARN",
    f"{len(wide_spreads)}/{quoted_count} acima de {MAX_RELATIVE_SPREAD:.0%}",
    f"spread relativo <= {MAX_RELATIVE_SPREAD:.0%}",
    "Cotações dentro do limite." if len(wide_spreads) == 0 else "Spreads largos reduzem a precisão da volatilidade.",
)

liquid_mask = (
    options_quality["trades"].fillna(0).ge(MIN_TRADES)
    & options_quality["open_interest"].fillna(0).ge(MIN_OPEN_INTEREST)
)
liquid_share = safe_ratio(liquid_mask.sum(), len(options_quality))
add_check(
    "Liquidez", "Negócios e posição em aberto",
    "PASS" if liquid_share >= 0.50 else "WARN",
    f"{int(liquid_mask.sum())}/{len(options_quality)} ({liquid_share:.1%})",
    f"negócios >= {MIN_TRADES} e posição >= {MIN_OPEN_INTEREST}",
    "A maioria atende ao filtro." if liquid_share >= 0.50 else "A amostra tem baixa liquidez.",
)

if not wide_spreads.empty:
    display(
        wide_spreads[["ticker", "best_bid", "best_ask", "relative_spread", "trades", "open_interest"]]
        .sort_values("relative_spread", ascending=False)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Curvas de juros e carry implícito
# MAGIC
# MAGIC Para cada vencimento de futuro do índice:
# MAGIC
# MAGIC $$F_{0,T}=S_0e^{[r(T)-q(T)]T}$$
# MAGIC
# MAGIC O controle reconstrói o futuro a partir de S₀, r(T) e q(T). Como q(T) foi
# MAGIC inferido dessa identidade, o erro deve ser apenas numérico.

# COMMAND ----------

# Comentário: valida domínio, finitude, ordenação e reconstrução dos futuros da curva.
curve_numeric = curve[["T", "r", "q", "future"]].replace([np.inf, -np.inf], np.nan)
curve_missing = int(curve_numeric.isna().any(axis=1).sum())
curve_domain_failures = int(((curve["T"] <= 0) | (curve["future"] <= 0)).sum())
curve_ordered = bool(curve["T"].is_monotonic_increasing and curve["T"].is_unique)
extreme_rates = int((curve[["r", "q"]].abs() > CURVE_ABS_RATE_LIMIT).any(axis=1).sum())

add_check(
    "Curvas", "Valores finitos e domínio positivo",
    status_from_count(curve_missing + curve_domain_failures),
    f"ausentes={curve_missing}; domínio inválido={curve_domain_failures}",
    "T e futuro positivos; r e q finitos", "Domínio adequado." if not curve_missing + curve_domain_failures else "Curva contém valores inválidos.",
)
add_check(
    "Curvas", "Prazos ordenados e únicos", "PASS" if curve_ordered else "FAIL",
    str(curve_ordered), "T crescente e sem duplicidade",
    "Interpolação unidimensional é bem definida." if curve_ordered else "A interpolação pode ser ambígua.",
)
add_check(
    "Curvas", "Magnitude de r(T) e q(T)", "PASS" if extreme_rates == 0 else "WARN",
    f"{extreme_rates} ponto(s) com |taxa| > {CURVE_ABS_RATE_LIMIT:.0%}",
    f"|r| e |q| <= {CURVE_ABS_RATE_LIMIT:.0%}",
    "Sem magnitude extrema." if extreme_rates == 0 else "Investigar unidade, contrato ou outlier.",
)

curve_validation = curve.copy()
curve_validation["future_rebuilt"] = spot * np.exp(
    (curve_validation["r"] - curve_validation["q"]) * curve_validation["T"]
)
curve_validation["future_error"] = curve_validation["future_rebuilt"] - curve_validation["future"]
max_future_error = float(curve_validation["future_error"].abs().max())
add_check(
    "Curvas", "Reconstrução do futuro", "PASS" if max_future_error <= 0.01 else "FAIL",
    f"erro máximo={max_future_error:.6f}", "erro absoluto máximo <= 0,01 ponto",
    "Identidade spot-forward preservada." if max_future_error <= 0.01 else "Curvas não reproduzem o futuro observado.",
)
display(curve_validation.style.format({
    "T": "{:.4f}", "r": "{:.4%}", "q": "{:.4%}", "future": "{:,.2f}",
    "future_rebuilt": "{:,.2f}", "future_error": "{:.6f}",
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Não arbitragem e inversão de Black–Scholes
# MAGIC
# MAGIC Antes de calcular volatilidade, o preço deve respeitar os limites europeus:
# MAGIC
# MAGIC $$\max(0,S_0e^{-qT}-Ke^{-rT})\leq C\leq S_0e^{-qT}$$
# MAGIC
# MAGIC $$\max(0,Ke^{-rT}-S_0e^{-qT})\leq P\leq Ke^{-rT}$$
# MAGIC
# MAGIC Depois da inversão, o Black–Scholes com σ̂ precisa reconstruir o preço de
# MAGIC mercado dentro da tolerância numérica.

# COMMAND ----------

# Comentário: identifica violações dos limites e mede o erro de reprecificação da volatilidade.
arbitrage_test = filtered.copy()
bounds = arbitrage_test.apply(arbitrage_bounds, axis=1, result_type="expand")
bounds.columns = ["lower_bound", "upper_bound"]
arbitrage_test = pd.concat([arbitrage_test, bounds], axis=1)
arbitrage_test["inside_bounds"] = (
    arbitrage_test["market_price"].gt(arbitrage_test["lower_bound"])
    & arbitrage_test["market_price"].lt(arbitrage_test["upper_bound"])
)
arbitrage_violations = arbitrage_test.loc[~arbitrage_test["inside_bounds"]].copy()
bound_pass_share = safe_ratio(arbitrage_test["inside_bounds"].sum(), len(arbitrage_test))
add_check(
    "Não arbitragem", "Limites europeus de preço",
    "PASS" if arbitrage_violations.empty else "WARN",
    f"{int(arbitrage_test['inside_bounds'].sum())}/{len(arbitrage_test)} ({bound_pass_share:.1%})",
    "preço estritamente dentro dos limites",
    "Todos os preços são invertíveis." if arbitrage_violations.empty else "Violações foram excluídas da superfície.",
)

repricing = iv_data.copy()
repricing["model_price"] = repricing.apply(
    lambda row: black_scholes_price(
        row["option_type"], spot, row["strike"], row["T"], row["r"], row["q"], row["implied_vol"]
    ),
    axis=1,
)
repricing["repricing_error"] = repricing["model_price"] - repricing["market_price"]
repricing_failures = repricing.loc[repricing["repricing_error"].abs().gt(MAX_REPRICING_ERROR)].copy()
max_repricing_error = float(repricing["repricing_error"].abs().max()) if len(repricing) else np.nan
add_check(
    "Volatilidade", "Reconstrução do preço observado",
    "PASS" if len(repricing) and repricing_failures.empty else "FAIL",
    f"erro máximo={max_repricing_error:.8f}" if np.isfinite(max_repricing_error) else "sem volatilidades",
    f"erro absoluto <= {MAX_REPRICING_ERROR:.2f}",
    "Inversão numérica convergiu." if len(repricing) and repricing_failures.empty else "Há falha de convergência ou amostra vazia.",
)

iv_finite = np.isfinite(iv_data["implied_vol"]).all() if len(iv_data) else False
iv_in_range = iv_data["implied_vol"].between(MIN_IV, MAX_IV).all() if len(iv_data) else False
add_check(
    "Volatilidade", "Volatilidades finitas e no intervalo", "PASS" if iv_finite and iv_in_range else "FAIL",
    f"n={len(iv_data)}; min={iv_data['implied_vol'].min():.2%}; max={iv_data['implied_vol'].max():.2%}" if len(iv_data) else "n=0",
    f"{MIN_IV:.0%} <= IV <= {MAX_IV:.0%}",
    "Amostra numérica válida." if iv_finite and iv_in_range else "A superfície não possui amostra válida.",
)

if not arbitrage_violations.empty:
    display(arbitrage_violations[[
        "ticker", "option_type", "strike", "maturity_date", "market_price", "lower_bound", "upper_bound"
    ]])
if not repricing_failures.empty:
    display(repricing_failures[["ticker", "market_price", "model_price", "repricing_error", "implied_vol"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Paridade put-call e forma dos preços por strike
# MAGIC
# MAGIC Para mesmo strike e vencimento, a paridade europeia é:
# MAGIC
# MAGIC $$C-P=S_0e^{-qT}-Ke^{-rT}$$
# MAGIC
# MAGIC Para calls de mesmo vencimento, o preço deve ser decrescente e convexo em K. Esses
# MAGIC testes são diagnósticos: cotações assíncronas ou último negócio podem produzir alertas
# MAGIC sem que isso represente uma arbitragem realmente executável.

# COMMAND ----------

# Comentário: cruza calls e puts comparáveis e avalia paridade, monotonicidade e convexidade.
parity_source = filtered[[
    "option_type", "maturity_date", "strike", "market_price", "T", "r", "q"
]].drop_duplicates(["option_type", "maturity_date", "strike"])
calls = parity_source.loc[parity_source["option_type"].eq("CALL")].copy()
puts = parity_source.loc[parity_source["option_type"].eq("PUT")].copy()
parity = calls.merge(
    puts,
    on=["maturity_date", "strike"],
    suffixes=("_call", "_put"),
    how="inner",
)
if len(parity):
    parity["parity_rhs"] = (
        spot * np.exp(-parity["q_call"] * parity["T_call"])
        - parity["strike"] * np.exp(-parity["r_call"] * parity["T_call"])
    )
    parity["parity_error"] = (
        parity["market_price_call"] - parity["market_price_put"] - parity["parity_rhs"]
    )
    parity["parity_error_pct_spot"] = parity["parity_error"].abs() / spot
    parity_failures = parity.loc[
        parity["parity_error_pct_spot"].gt(MAX_PARITY_ERROR_PCT_SPOT)
    ].copy()
    parity_status = "PASS" if parity_failures.empty else "WARN"
    parity_observed = f"{len(parity_failures)}/{len(parity)} acima da tolerância"
    parity_detail = "Pares consistentes." if parity_failures.empty else "Investigar sincronismo e liquidez de calls/puts."
else:
    parity_failures = parity.copy()
    parity_status = "WARN"
    parity_observed = "nenhum par call-put comparável"
    parity_detail = "Não foi possível testar a paridade com esta amostra."

add_check(
    "Não arbitragem", "Paridade put-call", parity_status, parity_observed,
    f"erro absoluto/spot <= {MAX_PARITY_ERROR_PCT_SPOT:.2%}", parity_detail,
)

shape_rows = []
for maturity, group in calls.groupby("maturity_date"):
    group = group.sort_values("strike")
    if len(group) < 3:
        continue
    strikes = group["strike"].to_numpy(dtype=float)
    call_prices = group["market_price"].to_numpy(dtype=float)
    price_differences = np.diff(call_prices)
    slopes = np.diff(call_prices) / np.diff(strikes)
    monotonic_violations = int((price_differences > MONOTONICITY_TOLERANCE).sum())
    convexity_violations = int((np.diff(slopes) < -CONVEXITY_SLOPE_TOLERANCE).sum())
    shape_rows.append({
        "vencimento": maturity,
        "pontos": len(group),
        "violações_monotonicidade": monotonic_violations,
        "violações_convexidade": convexity_violations,
    })

shape_diagnostics = pd.DataFrame(shape_rows)
if len(shape_diagnostics):
    total_monotonic = int(shape_diagnostics["violações_monotonicidade"].sum())
    total_convexity = int(shape_diagnostics["violações_convexidade"].sum())
    shape_status = "PASS" if total_monotonic + total_convexity == 0 else "WARN"
    shape_observed = f"monotonicidade={total_monotonic}; convexidade={total_convexity}"
    shape_detail = "Forma consistente." if shape_status == "PASS" else "Possível efeito de liquidez, assincronia ou arbitragem estática."
else:
    shape_status, shape_observed = "WARN", "sem vencimento com 3 calls"
    shape_detail = "Amostra insuficiente para testar a forma por strike."
add_check(
    "Não arbitragem", "Calls decrescentes e convexas no strike", shape_status,
    shape_observed, "zero violações", shape_detail,
)
if len(shape_diagnostics):
    display(shape_diagnostics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cobertura dos smiles e da superfície
# MAGIC
# MAGIC A interpolação linear só é válida dentro do casco convexo formado pelas observações.
# MAGIC Quando o alvo fica fora dele, o vizinho mais próximo é uma extrapolação operacional,
# MAGIC não uma interpolação confiável. Por isso o uso de `nearest` gera `WARN`.

# COMMAND ----------

# Comentário: mede densidade por vencimento e verifica a cobertura do ponto-alvo.
smile_coverage = (
    iv_data.groupby("maturity_date")
    .agg(
        pontos=("ticker", "size"),
        strikes=("strike", "nunique"),
        k_min=("log_moneyness", "min"),
        k_max=("log_moneyness", "max"),
        iv_min=("implied_vol", "min"),
        iv_max=("implied_vol", "max"),
    )
    .reset_index()
)
thin_smiles = smile_coverage.loc[smile_coverage["strikes"].lt(MIN_POINTS_PER_SMILE)]
dense_smiles = int(smile_coverage["strikes"].ge(MIN_POINTS_PER_SMILE).sum())
add_check(
    "Superfície", "Pontos mínimos por smile",
    "PASS" if dense_smiles >= MIN_SURFACE_MATURITIES else "WARN",
    f"{dense_smiles} vencimento(s) com >= {MIN_POINTS_PER_SMILE} strikes",
    f">= {MIN_SURFACE_MATURITIES} vencimentos densos",
    "Há base para uma superfície inicial." if dense_smiles >= MIN_SURFACE_MATURITIES else "Cobertura temporal ou por strike insuficiente.",
)

surface_ready = len(surface_data) >= MIN_SURFACE_POINTS and surface_data["T"].nunique() >= MIN_SURFACE_MATURITIES
add_check(
    "Superfície", "Quantidade mínima bidimensional", "PASS" if surface_ready else "FAIL",
    f"pontos={len(surface_data)}; vencimentos={surface_data['T'].nunique()}",
    f">= {MIN_SURFACE_POINTS} pontos e >= {MIN_SURFACE_MATURITIES} vencimentos",
    "Grade bidimensional pode ser construída." if surface_ready else "Amostra insuficiente para a superfície.",
)

target_inside_linear_surface = interpolation_method == "linear" and np.isfinite(target_iv)
add_check(
    "Superfície", "Cobertura do ponto-alvo",
    "PASS" if target_inside_linear_surface else "WARN",
    interpolation_method, "interpolação linear dentro do casco convexo",
    "Alvo coberto pelas observações." if target_inside_linear_surface else "Volatilidade-alvo é aproximação pelo vizinho mais próximo.",
)

target_parameters_finite = bool(np.isfinite([
    spot, TARGET_STRIKE, TARGET_MATURITY, target_r[0], target_q[0], target_iv
]).all())
target_domain_valid = bool(
    spot > 0 and TARGET_STRIKE > 0 and TARGET_MATURITY > 0 and target_iv > 0
)
add_check(
    "Prontidão", "Parâmetros-alvo finitos e positivos",
    "PASS" if target_parameters_finite and target_domain_valid else "FAIL",
    f"S={spot:.2f}; K={TARGET_STRIKE:.2f}; T={TARGET_MATURITY:.4f}; IV={target_iv:.4%}",
    "S, K, T e IV positivos; todos os parâmetros finitos",
    "Entrada numérica disponível para a barreira." if target_parameters_finite and target_domain_valid else "Não executar a precificação com esses parâmetros.",
)
display(smile_coverage.style.format({
    "k_min": "{:.4f}", "k_max": "{:.4f}", "iv_min": "{:.2%}", "iv_max": "{:.2%}",
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Painel consolidado e decisão de uso
# MAGIC
# MAGIC A decisão segue uma regra simples:
# MAGIC
# MAGIC - **APROVADO:** nenhum `FAIL` e nenhum `WARN`;
# MAGIC - **APROVADO COM RESSALVAS:** nenhum `FAIL`, mas há `WARN`;
# MAGIC - **REPROVADO:** existe ao menos um `FAIL`.
# MAGIC
# MAGIC `STRICT_MODE=True` interrompe a execução quando houver falha material. No modo padrão,
# MAGIC o notebook mostra todo o diagnóstico sem interromper a análise.

# COMMAND ----------

# Comentário: consolida os testes, resume os status e determina a decisão final da fotografia.
quality_report = pd.DataFrame(quality_checks)
status_order = pd.CategoricalDtype(["FAIL", "WARN", "PASS"], ordered=True)
quality_report["status_sort"] = quality_report["status"].astype(status_order)
quality_report = quality_report.sort_values(["status_sort", "bloco", "controle"]).drop(columns="status_sort")

status_summary = (
    quality_report["status"].value_counts()
    .reindex(["PASS", "WARN", "FAIL"], fill_value=0)
    .rename_axis("status")
    .reset_index(name="quantidade")
)
n_fail = int((quality_report["status"] == "FAIL").sum())
n_warn = int((quality_report["status"] == "WARN").sum())

if n_fail:
    final_decision = "REPROVADO"
    final_message = "Há falha material. Corrija os itens FAIL antes de usar o resultado em produção."
elif n_warn:
    final_decision = "APROVADO COM RESSALVAS"
    final_message = "O resultado pode apoiar análise exploratória, com as ressalvas explicitadas."
else:
    final_decision = "APROVADO"
    final_message = "Todos os controles definidos foram aprovados."

decision_table = pd.DataFrame({
    "data_mercado": [valuation_date.isoformat()],
    "decisão": [final_decision],
    "PASS": [int((quality_report["status"] == "PASS").sum())],
    "WARN": [n_warn],
    "FAIL": [n_fail],
    "mensagem": [final_message],
})

display(decision_table)
display(status_summary)
display(quality_report)

print(f"Data de mercado: {valuation_date:%d/%m/%Y}")
print(f"Decisão final: {final_decision}")
print(final_message)

if STRICT_MODE and n_fail:
    failed_names = quality_report.loc[quality_report["status"].eq("FAIL"), "controle"].tolist()
    raise AssertionError(f"Controles materiais reprovados: {failed_names}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próxima evolução
# MAGIC
# MAGIC Este primeiro notebook de qualidade cobre a fotografia de mercado, as curvas e a
# MAGIC superfície. Depois de integrar esses parâmetros ao preço da opção com barreira,
# MAGIC acrescentaremos controles específicos do modelo:
# MAGIC
# MAGIC 1. 0 ≤ V_barreira ≤ V_vanilla;
# MAGIC 2. intervalo de confiança e erro-padrão de Monte Carlo;
# MAGIC 3. convergência por número de caminhos e passos;
# MAGIC 4. comparação entre monitoramento discreto e Brownian Bridge;
# MAGIC 5. reprodutibilidade com semente fixa;
# MAGIC 6. estabilidade das gregas e dos cenários de risco.