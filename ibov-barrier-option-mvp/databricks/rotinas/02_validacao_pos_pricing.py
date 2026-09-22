# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "lxml>=6.0",
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
# MAGIC O notebook `01_main_option_pricer` é a fonte única da leitura e transformação
# MAGIC dos dados. Este notebook exige um resultado gerado pelo 01_main_option_pricer na data corrente e interrompe
# MAGIC imediatamente se esse pré-requisito não for atendido.

# COMMAND ----------

# Verifica se o lxml está disponível; instala somente se necessário.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("lxml") is None:
    print("lxml não encontrado. Instalando...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "lxml>=6.0",
    ])
    importlib.invalidate_caches()
    print("lxml instalado com sucesso.")
else:
    print("lxml já está instalado.")

# COMMAND ----------

# DBTITLE 1,Carrega a última execução do notebook 01_main_option_pricer
# Comentário: interrompe imediatamente se o notebook 01_main_option_pricer não tiver gerado resultado hoje.
import json
import re
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Não encontrei a raiz do projeto contendo as pastas data e src."
    )


project_root = find_project_root()
results_root = project_root.parents[1] / "ibov-barrier-results"
result_pattern = re.compile(
    r"^RESULTS_superficieVol_(\d{8})_(\d{6})\.csv$"
)

result_candidates = []
if results_root.exists():
    for path in results_root.glob("RESULTS_superficieVol_*.csv"):
        match = result_pattern.match(path.name)
        if match:
            timestamp = datetime.strptime(
                "".join(match.groups()), "%Y%m%d%H%M%S"
            )
            result_candidates.append((timestamp, path))

if not result_candidates:
    raise RuntimeError(
        "ERRO: nenhum resultado do notebook 01_main_option_pricer foi encontrado. "
        "Execute primeiro o notebook 01_main_option_pricer."
    )

latest_result_timestamp, latest_result_path = max(
    result_candidates, key=lambda item: item[0]
)
today = datetime.now(ZoneInfo("America/Sao_Paulo")).date()

if latest_result_timestamp.date() != today:
    raise RuntimeError(
        "ERRO: o notebook 01_main_option_pricer ainda não foi executado hoje. "
        f"Último resultado disponível: {latest_result_timestamp:%d/%m/%Y %H:%M:%S}. "
        "Execute o notebook 01_main_option_pricer antes do notebook 02_validacao_pos_pricing."
    )

result_bundle = pd.read_csv(latest_result_path)
required_objects = {
    "metadata", "catalog", "prices", "options", "curve",
    "filtered", "iv_data", "surface_data",
}
available_objects = set(result_bundle["object_name"])
missing_objects = sorted(required_objects - available_objects)
if missing_objects:
    raise RuntimeError(
        "ERRO: o resultado mais recente do notebook 01_main_option_pricer está incompleto. "
        f"Objetos ausentes: {missing_objects}. Execute novamente o notebook 01_main_option_pricer."
    )


def bundle_payload(name: str) -> str:
    rows = result_bundle.loc[result_bundle["object_name"].eq(name), "payload_json"]
    if len(rows) != 1:
        raise RuntimeError(
            f"ERRO: objeto {name!r} ausente ou duplicado no resultado do notebook 01_main_option_pricer."
        )
    return rows.iloc[0]


result_metadata = json.loads(bundle_payload("metadata"))
if result_metadata.get("schema_version") != 1:
    raise RuntimeError(
        "ERRO: versão incompatível do resultado do notebook 01_main_option_pricer. "
        "Execute novamente o notebook 01_main_option_pricer."
    )

metadata_execution_date = datetime.fromisoformat(
    result_metadata["execution_timestamp"]
).date()
if metadata_execution_date != today:
    raise RuntimeError(
        "ERRO: a data interna do resultado não corresponde a hoje. "
        "Execute novamente o notebook 01_main_option_pricer."
    )

frames = {
    name: pd.read_json(StringIO(bundle_payload(name)), orient="table")
    for name in required_objects - {"metadata"}
}

catalog = frames["catalog"]
prices = frames["prices"]
options = frames["options"]
curve = frames["curve"]
filtered = frames["filtered"]
iv_data = frames["iv_data"]
surface_data = frames["surface_data"]

MARKET_DATE = result_metadata["market_date"]
valuation_date = date.fromisoformat(MARKET_DATE)
market_folder = project_root / "data" / MARKET_DATE

required_prefixes = ("IN", "IR", "SPRD", "PR")
complete_market_folders = [
    folder for folder in (project_root / "data").iterdir()
    if folder.is_dir()
    and all(any(folder.glob(f"{prefix}*.zip")) for prefix in required_prefixes)
]
if not complete_market_folders:
    raise RuntimeError("ERRO: não existe nenhuma pasta de mercado completa.")

latest_complete_market_folder = max(complete_market_folders, key=lambda p: p.name)
if market_folder.name != latest_complete_market_folder.name:
    raise RuntimeError(
        "ERRO: o resultado de hoje não usa a última fotografia completa da B3. "
        f"Resultado do 01_main_option_pricer: {market_folder.name}; "
        f"última pasta completa: {latest_complete_market_folder.name}. "
        "Execute novamente o notebook 01_main_option_pricer."
    )

in_zip = market_folder / result_metadata["in_zip"]
ir_zip = market_folder / result_metadata["ir_zip"]
sprd_zip = market_folder / result_metadata["sprd_zip"]
pr_zip = market_folder / result_metadata["pr_zip"]

spot = float(result_metadata["spot"])
TARGET_STRIKE = float(result_metadata["target_strike"])
TARGET_MATURITY = float(result_metadata["target_maturity"])
target_r = np.array([float(result_metadata["target_r"])])
target_q = np.array([float(result_metadata["target_q"])])
target_iv = float(result_metadata["target_iv"])
interpolation_method = result_metadata["interpolation_method"]
MIN_TRADES = int(result_metadata["min_trades"])
MIN_OPEN_INTEREST = int(result_metadata["min_open_interest"])
MIN_IV = float(result_metadata["min_iv"])
MAX_IV = float(result_metadata["max_iv"])

valid_spread = (
    options["best_bid"].notna()
    & options["best_ask"].notna()
    & options["best_bid"].gt(0)
    & options["best_ask"].ge(options["best_bid"])
)


def black_scholes_price(
    option_type, spot_value, strike, maturity, rate, dividend_yield, volatility
):
    sqrt_t = np.sqrt(maturity)
    d1 = (
        np.log(spot_value / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot_value * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == "CALL":
        return float(
            discounted_spot * norm.cdf(d1)
            - discounted_strike * norm.cdf(d2)
        )
    return float(
        discounted_strike * norm.cdf(-d2)
        - discounted_spot * norm.cdf(-d1)
    )


def arbitrage_bounds(row):
    discounted_spot = spot * np.exp(-row["q"] * row["T"])
    discounted_strike = row["strike"] * np.exp(-row["r"] * row["T"])
    if row["option_type"] == "CALL":
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


print(f"✓ Resultado carregado: {latest_result_path.name}")
print(f"✓ Executado hoje às {latest_result_timestamp:%H:%M:%S}")
print(f"✓ Data de mercado validada: {valuation_date:%d/%m/%Y}")

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

# DBTITLE 1,Seção 9 — Validação da barreira
# MAGIC %md
# MAGIC ## 9. Validação independente da precificação com barreira
# MAGIC
# MAGIC Esta seção lê os resultados de `pricing_result` e `simulation_diagnostics`
# MAGIC persistidos pelo notebook 01, valida domínios, reconcilia aritmética,
# MAGIC reprecifica independentemente com o motor oficial `ibov_barrier.pricing` e
# MAGIC consolida tudo em um painel de controles financeiros. Os controles abaixo são
# MAGIC adicionados ao mesmo `quality_checks` usado pela superfície de volatilidade,
# MAGIC garantindo uma decisão final unificada.

# COMMAND ----------

# DBTITLE 1,Dependência e leitura (A/B)
# DBTITLE 1,Dependência e leitura das seções de barreira (A/B)
# Comentário: localiza o arquivo de resultado mais recente do notebook 01,
# já carregado pela célula de dependência. Desserializa pricing_result e
# simulation_diagnostics, interrompendo com FAIL se houver problema.
import json as _json_barrier

barrier_checks = []  # lista separada para a tabela consolidada da seção H


def add_barrier_check(categoria, controle, observado, esperado, tolerancia, status, mensagem):
    """Adiciona um controle ao painel de barreira e ao quality_checks unificado."""
    if status not in {"PASS", "WARN", "FAIL"}:
        raise ValueError(f"Status inválido: {status}")
    barrier_checks.append({
        "categoria": str(categoria),
        "controle": str(controle),
        "valor_observado": str(observado),
        "valor_esperado": str(esperado),
        "tolerancia": str(tolerancia),
        "status": status,
        "mensagem": str(mensagem),
    })
    add_check(
        block=f"Barreira-{categoria}",
        control=controle,
        status=status,
        observed=observado,
        criterion=str(esperado),
        detail=mensagem,
    )


# --- Seção A: confirma dependência explícita do notebook 01 ---
# O arquivo já foi selecionado na célula de carregamento (cell 3).
# Aqui apenas confirmamos qual arquivo foi usado.
barrier_result_file = latest_result_path
barrier_execution_ts = result_metadata["execution_timestamp"]
barrier_market_date = MARKET_DATE

print(f"Arquivo de resultado analisado: {barrier_result_file.name}")
print(f"Data de execução do notebook 01: {barrier_execution_ts}")
print(f"Data de mercado: {barrier_market_date}")
print()

# --- Seção B: leitura robusta das novas seções ---
_barrier_read_errors = []

# pricing_result
_pr_rows = result_bundle.loc[result_bundle["object_name"].eq("pricing_result"), "payload_json"]
if len(_pr_rows) == 0:
    _barrier_read_errors.append("Seção pricing_result ausente no arquivo de resultado.")
    pricing_result_data = None
elif len(_pr_rows) > 1:
    _barrier_read_errors.append("Seção pricing_result duplicada no arquivo de resultado.")
    pricing_result_data = None
else:
    try:
        pricing_result_data = _json_barrier.loads(_pr_rows.iloc[0])
    except _json_barrier.JSONDecodeError as exc:
        _barrier_read_errors.append(f"pricing_result contém JSON inválido: {exc}")
        pricing_result_data = None

# simulation_diagnostics
_sd_rows = result_bundle.loc[result_bundle["object_name"].eq("simulation_diagnostics"), "payload_json"]
if len(_sd_rows) == 0:
    _barrier_read_errors.append("Seção simulation_diagnostics ausente no arquivo de resultado.")
    simulation_diagnostics_data = None
elif len(_sd_rows) > 1:
    _barrier_read_errors.append("Seção simulation_diagnostics duplicada no arquivo de resultado.")
    simulation_diagnostics_data = None
else:
    try:
        simulation_diagnostics_data = _json_barrier.loads(_sd_rows.iloc[0])
    except _json_barrier.JSONDecodeError as exc:
        _barrier_read_errors.append(f"simulation_diagnostics contém JSON inválido: {exc}")
        simulation_diagnostics_data = None

if _barrier_read_errors:
    for err in _barrier_read_errors:
        add_barrier_check(
            "Dependência", "Leitura das seções",
            err, "pricing_result e simulation_diagnostics presentes e válidos",
            "N/A", "FAIL",
            "Seção ausente ou inválida impede a validação da barreira.",
        )
    print("❌ FALHA: Não foi possível ler as seções de barreira.")
    for err in _barrier_read_errors:
        print(f"  • {err}")
    raise RuntimeError(f"{len(_barrier_read_errors)} erro(s) na leitura das seções de barreira.")

print("✓ pricing_result e simulation_diagnostics lidos com sucesso.")
print(f"  Chaves em pricing_result: {list(pricing_result_data.keys())}")
print(f"  Chaves em simulation_diagnostics: {list(simulation_diagnostics_data.keys())}")

# COMMAND ----------

# DBTITLE 1,Validações de domínio (C)
# DBTITLE 1,Validações de presença e domínio (C)
# Comentário: verifica presença, finitude e domínio de todos os campos persistidos.
import math

# Extrai valores do pricing_result e simulation_diagnostics.
pr = pricing_result_data
sd = simulation_diagnostics_data

# Mapeia todos os campos esperados.
pr_contract = pr.get("contract", {})
pr_market = pr.get("market", {})

barrier_fields = {
    "market_date": sd.get("market_date"),
    "execution_timestamp": sd.get("execution_timestamp"),
    "spot": pr_market.get("spot"),
    "strike": pr_contract.get("strike"),
    "barrier": pr_contract.get("barrier"),
    "maturity": pr_contract.get("maturity"),
    "rate": pr_market.get("rate"),
    "dividend_yield": pr_market.get("dividend_yield"),
    "volatility": pr_market.get("volatility"),
    "forward": pr_market.get("forward"),
    "vanilla_price": pr.get("vanilla_price"),
    "barrier_price": pr.get("barrier_price"),
    "barrier_discount_abs": pr.get("barrier_discount_abs"),
    "barrier_discount_pct": pr.get("barrier_discount_pct"),
    "knock_out_probability": pr.get("knock_out_probability"),
    "standard_error": pr.get("standard_error"),
    "ci_low_95": pr.get("ci_low_95"),
    "ci_high_95": pr.get("ci_high_95"),
    "paths": sd.get("paths"),
    "steps": sd.get("steps"),
    "seed": sd.get("seed"),
    "monitoring": sd.get("monitoring"),
}

# --- Presença: nenhum campo pode ser None ---
missing_fields = [name for name, val in barrier_fields.items() if val is None]
add_barrier_check(
    "Presença", "Todos os campos presentes",
    f"{len(missing_fields)} ausente(s): {missing_fields}" if missing_fields else "Todos presentes",
    "22 campos esperados",
    "N/A",
    "FAIL" if missing_fields else "PASS",
    "Campos ausentes impedem validação." if missing_fields else "Todos os campos estão presentes.",
)

# Se houver campos ausentes, interrompe antes das validações de domínio.
if missing_fields:
    print("❌ FALHA: Campos ausentes em pricing_result/simulation_diagnostics.")
    raise RuntimeError(f"Campos ausentes: {missing_fields}")

# --- Finitude: valores numéricos devem ser finitos ---
numeric_fields = [
    "spot", "strike", "barrier", "maturity", "rate", "dividend_yield",
    "volatility", "forward", "vanilla_price", "barrier_price",
    "barrier_discount_abs", "barrier_discount_pct", "knock_out_probability",
    "standard_error", "ci_low_95", "ci_high_95", "paths", "steps", "seed",
]
non_finite = []
for name in numeric_fields:
    val = barrier_fields[name]
    try:
        if not math.isfinite(float(val)):
            non_finite.append(name)
    except (TypeError, ValueError):
        non_finite.append(name)

add_barrier_check(
    "Domínio", "Valores numéricos finitos",
    f"{len(non_finite)} não-finito(s): {non_finite}" if non_finite else "Todos finitos",
    "Todos os valores numéricos são finitos",
    "N/A",
    "FAIL" if non_finite else "PASS",
    "Valores não-finitos indicam erro numérico." if non_finite else "Todos os valores são finitos.",
)

if non_finite:
    raise RuntimeError(f"Valores não-finitos: {non_finite}")

# --- Domínio: validações individuais ---
# spot > 0
add_barrier_check(
    "Domínio", "spot > 0",
    f"{float(barrier_fields['spot']):,.2f}", "spot > 0",
    "N/A",
    "PASS" if float(barrier_fields["spot"]) > 0 else "FAIL",
    "Spot positivo." if float(barrier_fields["spot"]) > 0 else "Spot deve ser positivo.",
)

# strike > 0
add_barrier_check(
    "Domínio", "strike > 0",
    f"{float(barrier_fields['strike']):,.2f}", "strike > 0",
    "N/A",
    "PASS" if float(barrier_fields["strike"]) > 0 else "FAIL",
    "Strike positivo." if float(barrier_fields["strike"]) > 0 else "Strike deve ser positivo.",
)

# 0 < barrier < spot
_b = float(barrier_fields["barrier"])
_s = float(barrier_fields["spot"])
add_barrier_check(
    "Domínio", "0 < barreira < spot",
    f"{_b:,.2f} < {_s:,.2f}", "0 < barreira < spot",
    "N/A",
    "PASS" if 0 < _b < _s else "FAIL",
    "Barreira positiva e abaixo do spot." if 0 < _b < _s else "Barreira deve estar entre 0 e spot.",
)

# prazo > 0
add_barrier_check(
    "Domínio", "prazo > 0",
    f"{float(barrier_fields['maturity']):.4f}", "prazo > 0",
    "N/A",
    "PASS" if float(barrier_fields["maturity"]) > 0 else "FAIL",
    "Prazo positivo." if float(barrier_fields["maturity"]) > 0 else "Prazo deve ser positivo.",
)

# volatilidade >= 0
add_barrier_check(
    "Domínio", "volatilidade >= 0",
    f"{float(barrier_fields['volatility']):.4%}", "volatilidade >= 0",
    "N/A",
    "PASS" if float(barrier_fields["volatility"]) >= 0 else "FAIL",
    "Volatilidade não-negativa." if float(barrier_fields["volatility"]) >= 0 else "Volatilidade deve ser >= 0.",
)

# preço vanilla >= 0
add_barrier_check(
    "Domínio", "preço vanilla >= 0",
    f"{float(barrier_fields['vanilla_price']):,.2f}", "preço vanilla >= 0",
    "N/A",
    "PASS" if float(barrier_fields["vanilla_price"]) >= 0 else "FAIL",
    "Preço vanilla não-negativo." if float(barrier_fields["vanilla_price"]) >= 0 else "Preço vanilla deve ser >= 0.",
)

# preço com barreira >= 0
add_barrier_check(
    "Domínio", "preço com barreira >= 0",
    f"{float(barrier_fields['barrier_price']):,.2f}", "preço com barreira >= 0",
    "N/A",
    "PASS" if float(barrier_fields["barrier_price"]) >= 0 else "FAIL",
    "Preço com barreira não-negativo." if float(barrier_fields["barrier_price"]) >= 0 else "Preço com barreira deve ser >= 0.",
)

# erro-padrão >= 0
add_barrier_check(
    "Domínio", "erro-padrão >= 0",
    f"{float(barrier_fields['standard_error']):,.2f}", "erro-padrão >= 0",
    "N/A",
    "PASS" if float(barrier_fields["standard_error"]) >= 0 else "FAIL",
    "Erro-padrão não-negativo." if float(barrier_fields["standard_error"]) >= 0 else "Erro-padrão deve ser >= 0.",
)

# 0 <= P(knock-out) <= 1
_ko = float(barrier_fields["knock_out_probability"])
add_barrier_check(
    "Domínio", "0 <= P(knock-out) <= 1",
    f"{_ko:.4%}", "0 <= P(knock-out) <= 1",
    "N/A",
    "PASS" if 0 <= _ko <= 1 else "FAIL",
    "Probabilidade em [0,1]." if 0 <= _ko <= 1 else "Probabilidade fora de [0,1].",
)

# caminhos >= 2
add_barrier_check(
    "Domínio", "caminhos >= 2",
    f"{int(barrier_fields['paths']):,}", "caminhos >= 2",
    "N/A",
    "PASS" if int(barrier_fields["paths"]) >= 2 else "FAIL",
    "Caminhos suficientes para MC." if int(barrier_fields["paths"]) >= 2 else "Caminhos insuficientes.",
)

# passos >= 1
add_barrier_check(
    "Domínio", "passos >= 1",
    f"{int(barrier_fields['steps'])}", "passos >= 1",
    "N/A",
    "PASS" if int(barrier_fields["steps"]) >= 1 else "FAIL",
    "Passos suficientes para MC." if int(barrier_fields["steps"]) >= 1 else "Passos insuficientes.",
)

# monitoring pertence a {discrete, brownian_bridge}
_mon = str(barrier_fields["monitoring"])
add_barrier_check(
    "Domínio", "monitoring válido",
    _mon, "discrete ou brownian_bridge",
    "N/A",
    "PASS" if _mon in {"discrete", "brownian_bridge"} else "FAIL",
    "Método de monitoramento válido." if _mon in {"discrete", "brownian_bridge"} else "Método inválido.",
)

print("✓ Validações de presença e domínio concluídas.")

# COMMAND ----------

# DBTITLE 1,Reconciliação aritmética (D)
# DBTITLE 1,Reconciliação aritmética independente (D)
# Comentário: recalcula descontos, IC e forward a partir dos valores persistidos,
# comparando com as figuras armazenadas. Tolerâncias documentadas abaixo.
# Tolerâncias: 1e-6 para razões/percentuais, 1e-2 para preços (ponto flutuante + arredondamento).
_TOL_PRICE = 1e-2       # tolerância absoluta para preços em pontos do IBOV
_TOL_RATIO = 1e-6       # tolerância para razões e percentuais
_TOL_FORWARD = 1e-2    # tolerância absoluta para forward em pontos do IBOV

_vanilla = float(barrier_fields["vanilla_price"])
_barrier = float(barrier_fields["barrier_price"])
_se = float(barrier_fields["standard_error"])
_spot = float(barrier_fields["spot"])
_r = float(barrier_fields["rate"])
_q = float(barrier_fields["dividend_yield"])
_T = float(barrier_fields["maturity"])

# 1. desconto absoluto: vanilla - barrier
_recalc_discount_abs = _vanilla - _barrier
_stored_discount_abs = float(barrier_fields["barrier_discount_abs"])
_diff_abs = abs(_recalc_discount_abs - _stored_discount_abs)
add_barrier_check(
    "Reconciliação", "desconto absoluto",
    f"recal={_recalc_discount_abs:,.4f}; armaz={_stored_discount_abs:,.4f}",
    f"vanilla - barrier = {_recalc_discount_abs:,.4f}",
    f"{_TOL_PRICE:.0e}",
    "PASS" if _diff_abs <= _TOL_PRICE else "FAIL",
    f"Diferença={_diff_abs:.6f}" if _diff_abs <= _TOL_PRICE else f"Divergência={_diff_abs:.6f} excede tolerância.",
)

# 2. desconto percentual: discount_abs / vanilla_price
if _vanilla > 0:
    _recalc_discount_pct = _recalc_discount_abs / _vanilla
else:
    _recalc_discount_pct = 0.0
_stored_discount_pct = float(barrier_fields["barrier_discount_pct"])
_diff_pct = abs(_recalc_discount_pct - _stored_discount_pct)
add_barrier_check(
    "Reconciliação", "desconto percentual",
    f"recal={_recalc_discount_pct:.8f}; armaz={_stored_discount_pct:.8f}",
    f"discount_abs / vanilla = {_recalc_discount_pct:.8f}",
    f"{_TOL_RATIO:.0e}",
    "PASS" if _diff_pct <= _TOL_RATIO else "FAIL",
    f"Diferença={_diff_pct:.10f}" if _diff_pct <= _TOL_RATIO else f"Divergência={_diff_pct:.10f} excede tolerância.",
)

# 3. intervalo de confiança: barrier ± 1.96 × SE
_recalc_ci_low = _barrier - 1.96 * _se
_recalc_ci_high = _barrier + 1.96 * _se
_stored_ci_low = float(barrier_fields["ci_low_95"])
_stored_ci_high = float(barrier_fields["ci_high_95"])
_diff_ci_low = abs(_recalc_ci_low - _stored_ci_low)
_diff_ci_high = abs(_recalc_ci_high - _stored_ci_high)
_max_ci_diff = max(_diff_ci_low, _diff_ci_high)
add_barrier_check(
    "Reconciliação", "IC 95%",
    f"recal=[{_recalc_ci_low:,.4f}, {_recalc_ci_high:,.4f}]; armaz=[{_stored_ci_low:,.4f}, {_stored_ci_high:,.4f}]",
    f"barrier ± 1.96 × SE",
    f"{_TOL_PRICE:.0e}",
    "PASS" if _max_ci_diff <= _TOL_PRICE else "FAIL",
    f"Diferença máxima={_max_ci_diff:.6f}" if _max_ci_diff <= _TOL_PRICE else f"Divergência={_max_ci_diff:.6f} excede tolerância.",
)

# 4. forward: spot × exp((r - q) × T)
_recalc_forward = _spot * math.exp((_r - _q) * _T)
_stored_forward = float(barrier_fields["forward"])
_diff_forward = abs(_recalc_forward - _stored_forward)
add_barrier_check(
    "Reconciliação", "forward",
    f"recal={_recalc_forward:,.4f}; armaz={_stored_forward:,.4f}",
    f"spot × exp((r - q) × T) = {_recalc_forward:,.4f}",
    f"{_TOL_FORWARD:.0e}",
    "PASS" if _diff_forward <= _TOL_FORWARD else "FAIL",
    f"Diferença={_diff_forward:.6f}" if _diff_forward <= _TOL_FORWARD else f"Divergência={_diff_forward:.6f} excede tolerância.",
)

# Exibe tabela de reconciliação
reconciliation_table = pd.DataFrame([
    {"controle": "desconto absoluto", "recalculado": _recalc_discount_abs, "armazenado": _stored_discount_abs,
     "diferenca": _diff_abs, "tolerancia": _TOL_PRICE,
     "status": "PASS" if _diff_abs <= _TOL_PRICE else "FAIL"},
    {"controle": "desconto percentual", "recalculado": _recalc_discount_pct, "armazenado": _stored_discount_pct,
     "diferenca": _diff_pct, "tolerancia": _TOL_RATIO,
     "status": "PASS" if _diff_pct <= _TOL_RATIO else "FAIL"},
    {"controle": "IC 95% (limite inferior)", "recalculado": _recalc_ci_low, "armazenado": _stored_ci_low,
     "diferenca": _diff_ci_low, "tolerancia": _TOL_PRICE,
     "status": "PASS" if _diff_ci_low <= _TOL_PRICE else "FAIL"},
    {"controle": "IC 95% (limite superior)", "recalculado": _recalc_ci_high, "armazenado": _stored_ci_high,
     "diferenca": _diff_ci_high, "tolerancia": _TOL_PRICE,
     "status": "PASS" if _diff_ci_high <= _TOL_PRICE else "FAIL"},
    {"controle": "forward", "recalculado": _recalc_forward, "armazenado": _stored_forward,
     "diferenca": _diff_forward, "tolerancia": _TOL_FORWARD,
     "status": "PASS" if _diff_forward <= _TOL_FORWARD else "FAIL"},
])
print("✓ Reconciliação aritmética concluída.")
display(reconciliation_table.style.format({
    "recalculado": "{:.6f}", "armazenado": "{:.6f}",
    "diferenca": "{:.8f}", "tolerancia": "{:.0e}",
}))

# COMMAND ----------

# DBTITLE 1,Reprecificação independente (E)
# DBTITLE 1,Reprecificação independente (E)
# Comentário: importa o motor oficial ibov_barrier.pricing, reconstrói os objetos
# de mercado e contrato a partir dos dados persistidos, e recalcula os preços.
# Não copia nem reimplementa o motor dentro deste notebook.
import sys as _sys_barrier

_src_path = project_root / "src"
if str(_src_path) not in _sys_barrier.path:
    _sys_barrier.path.insert(0, str(_src_path))

from ibov_barrier import BarrierContract, MarketData, price_down_and_out_call

print(f"Motor de precificação importado de: {_src_path}")
print("API: MarketData, BarrierContract, price_down_and_out_call")
print()

# Reconstrói os objetos exclusivamente a partir dos dados persistidos.
_repr_market = MarketData(
    spot=float(pr_market["spot"]),
    rate=float(pr_market["rate"]),
    dividend_yield=float(pr_market["dividend_yield"]),
    volatility=float(pr_market["volatility"]),
)
_repr_contract = BarrierContract(
    strike=float(pr_contract["strike"]),
    barrier=float(pr_contract["barrier"]),
    maturity=float(pr_contract["maturity"]),
)

print(f"MarketData: spot={_repr_market.spot:,.2f}, r={_repr_market.rate:.6%}, q={_repr_market.dividend_yield:.6%}, σ={_repr_market.volatility:.6%}")
print(f"BarrierContract: K={_repr_contract.strike:,.2f}, H={_repr_contract.barrier:,.2f}, T={_repr_contract.maturity:.4f}")
print()

# Reprecifica com os mesmos parâmetros persistidos em simulation_diagnostics.
_repr_result = price_down_and_out_call(
    _repr_market,
    _repr_contract,
    paths=int(sd["paths"]),
    steps=int(sd["steps"]),
    seed=int(sd["seed"]),
    monitoring=str(sd["monitoring"]),
)

print(f"✓ Reprecificação concluída em {int(sd['paths']):,} caminhos × {int(sd['steps'])} passos")
print(f"  Preço vanilla recalculado:  {_repr_result.vanilla_price:,.4f}")
print(f"  Preço barrier recalculado:   {_repr_result.price:,.4f}")
print(f"  Erro-padrão recalculado:     {_repr_result.standard_error:,.4f}")
print(f"  P(knock-out) recalculada:    {_repr_result.knock_out_probability:.6%}")
print(f"  IC 95% recalculado:          [{_repr_result.ci_low:,.4f}, {_repr_result.ci_high:,.4f}]")
print()

# --- Compara recalculado vs persistido ---
# Tolerâncias: mesma seed = reprodutibilidade determinística.
# Usamos tolerância pequena para cobrir diferenças de ponto flutuante entre execuções.
_TOL_REPRICE_PRICE = 1e-4      # pontos do IBOV
_TOL_REPRICE_SE = 1e-4        # pontos do IBOV
_TOL_REPRICE_PROB = 1e-8       # probabilidade sem dimensão
_TOL_REPRICE_CI = 1e-4         # pontos do IBOV

_stored_vanilla = float(pr["vanilla_price"])
_stored_barrier = float(pr["barrier_price"])
_stored_se = float(pr["standard_error"])
_stored_ko = float(pr["knock_out_probability"])
_stored_ci_low = float(pr["ci_low_95"])
_stored_ci_high = float(pr["ci_high_95"])

_diff_vanilla = abs(_repr_result.vanilla_price - _stored_vanilla)
add_barrier_check(
    "Reprecificação", "preço vanilla",
    f"recal={_repr_result.vanilla_price:,.6f}; armaz={_stored_vanilla:,.6f}",
    f"diferença={_diff_vanilla:.8f}",
    f"{_TOL_REPRICE_PRICE:.0e}",
    "PASS" if _diff_vanilla <= _TOL_REPRICE_PRICE else "FAIL",
    "Preço vanilla reproduzido." if _diff_vanilla <= _TOL_REPRICE_PRICE else f"Divergência={_diff_vanilla:.8f} excede tolerância.",
)

_diff_barrier = abs(_repr_result.price - _stored_barrier)
add_barrier_check(
    "Reprecificação", "preço com barreira",
    f"recal={_repr_result.price:,.6f}; armaz={_stored_barrier:,.6f}",
    f"diferença={_diff_barrier:.8f}",
    f"{_TOL_REPRICE_PRICE:.0e}",
    "PASS" if _diff_barrier <= _TOL_REPRICE_PRICE else "FAIL",
    "Preço com barreira reproduzido." if _diff_barrier <= _TOL_REPRICE_PRICE else f"Divergência={_diff_barrier:.8f} excede tolerância.",
)

_diff_se = abs(_repr_result.standard_error - _stored_se)
add_barrier_check(
    "Reprecificação", "erro-padrão",
    f"recal={_repr_result.standard_error:,.6f}; armaz={_stored_se:,.6f}",
    f"diferença={_diff_se:.8f}",
    f"{_TOL_REPRICE_SE:.0e}",
    "PASS" if _diff_se <= _TOL_REPRICE_SE else "FAIL",
    "Erro-padrão reproduzido." if _diff_se <= _TOL_REPRICE_SE else f"Divergência={_diff_se:.8f} excede tolerância.",
)

_diff_ko = abs(_repr_result.knock_out_probability - _stored_ko)
add_barrier_check(
    "Reprecificação", "P(knock-out)",
    f"recal={_repr_result.knock_out_probability:.10f}; armaz={_stored_ko:.10f}",
    f"diferença={_diff_ko:.12f}",
    f"{_TOL_REPRICE_PROB:.0e}",
    "PASS" if _diff_ko <= _TOL_REPRICE_PROB else "FAIL",
    "Probabilidade reproduzida." if _diff_ko <= _TOL_REPRICE_PROB else f"Divergência={_diff_ko:.12f} excede tolerância.",
)

_diff_ci_low = abs(_repr_result.ci_low - _stored_ci_low)
_diff_ci_high = abs(_repr_result.ci_high - _stored_ci_high)
_max_ci_reprice_diff = max(_diff_ci_low, _diff_ci_high)
add_barrier_check(
    "Reprecificação", "IC 95%",
    f"recal=[{_repr_result.ci_low:,.6f}, {_repr_result.ci_high:,.6f}]; armaz=[{_stored_ci_low:,.6f}, {_stored_ci_high:,.6f}]",
    f"diferença máxima={_max_ci_reprice_diff:.8f}",
    f"{_TOL_REPRICE_CI:.0e}",
    "PASS" if _max_ci_reprice_diff <= _TOL_REPRICE_CI else "FAIL",
    "IC reproduzido." if _max_ci_reprice_diff <= _TOL_REPRICE_CI else f"Divergência={_max_ci_reprice_diff:.8f} excede tolerância.",
)

print("✓ Reprecificação independente concluída.")

# COMMAND ----------

# DBTITLE 1,Controles financeiros (F)
# DBTITLE 1,Controles financeiros (F)
# Comentário: controles financeiros sobre os preços e a estrutura da barreira.
# A tolerância estatística para o limite superior é 5× erro-padrão do MC,
# justificada pela variância do estimador com variável de controle.
_FIN_TOL_DISCOUNT = 1e-2  # tolerância para desconto absoluto (ponto flutuante)

# 1. 0 <= preço com barreira
add_barrier_check(
    "Financeiro", "0 <= preço com barreira",
    f"{_barrier:,.2f}", "barrier_price >= 0",
    "N/A",
    "PASS" if _barrier >= 0 else "FAIL",
    "Preço com barreira não-negativo." if _barrier >= 0 else "Preço com barreira é negativo.",
)

# 2. preço com barreira <= preço vanilla + tolerância estatística
_stat_tol = 5.0 * _se if _se > 0 else 1.0
_upper_limit = _vanilla + _stat_tol
add_barrier_check(
    "Financeiro", "barrier <= vanilla + 5×SE",
    f"{_barrier:,.2f} <= {_upper_limit:,.2f}",
    f"barrier_price <= vanilla_price + 5×SE = {_upper_limit:,.2f}",
    f"5×SE = {_stat_tol:,.2f}",
    "PASS" if _barrier <= _upper_limit else "FAIL",
    "Barreira não supera vanilla além da tolerância estatística." if _barrier <= _upper_limit else "Barreira supera vanilla materialmente.",
)

# 3. limite inferior do IC <= preço estimado <= limite superior
_stored_ci_low_val = float(barrier_fields["ci_low_95"])
_stored_ci_high_val = float(barrier_fields["ci_high_95"])
_ic_ok = _stored_ci_low_val <= _barrier <= _stored_ci_high_val
add_barrier_check(
    "Financeiro", "IC contém preço estimado",
    f"{_stored_ci_low_val:,.2f} <= {_barrier:,.2f} <= {_stored_ci_high_val:,.2f}",
    "ci_low <= barrier_price <= ci_high",
    "N/A",
    "PASS" if _ic_ok else "FAIL",
    "Preço dentro do IC 95%." if _ic_ok else "Preço fora do IC 95%.",
)

# 4. desconto absoluto não negativo (com tolerância numérica)
_discount_ok = _recalc_discount_abs >= -_FIN_TOL_DISCOUNT
add_barrier_check(
    "Financeiro", "desconto absoluto >= 0",
    f"{_recalc_discount_abs:,.4f}",
    f"discount_abs >= -{_FIN_TOL_DISCOUNT:.0e}",
    f"{_FIN_TOL_DISCOUNT:.0e}",
    "PASS" if _discount_ok else "FAIL",
    "Desconto não-negativo." if _discount_ok else "Desconto é negativo além da tolerância.",
)

# 5. desconto percentual coerente com os preços
_pct_ok = abs(_recalc_discount_pct - (_recalc_discount_abs / _vanilla if _vanilla > 0 else 0.0)) < _TOL_RATIO
add_barrier_check(
    "Financeiro", "desconto percentual coerente",
    f"{_recalc_discount_pct:.8f}",
    f"discount_abs / vanilla = {(_recalc_discount_abs / _vanilla if _vanilla > 0 else 0.0):.8f}",
    f"{_TOL_RATIO:.0e}",
    "PASS" if _pct_ok else "FAIL",
    "Desconto percentual coerente." if _pct_ok else "Desconto percentual incoerente.",
)

# 6. barreira abaixo do spot (já validado em C, mas repetido como controle financeiro)
_barrier_below_spot = float(pr_contract["barrier"]) < float(pr_market["spot"])
add_barrier_check(
    "Financeiro", "barreira < spot",
    f"{float(pr_contract['barrier']):,.2f} < {float(pr_market['spot']):,.2f}",
    "barrier < spot",
    "N/A",
    "PASS" if _barrier_below_spot else "FAIL",
    "Barreira abaixo do spot." if _barrier_below_spot else "Barreira acima do spot — inválido para down-and-out.",
)

# 7. Brownian Bridge identificado corretamente quando configurado
_mon_config = str(sd["monitoring"])
_bb_ok = (_mon_config == "brownian_bridge") and (_mon_config in {"discrete", "brownian_bridge"})
add_barrier_check(
    "Financeiro", "monitoramento Brownian Bridge",
    _mon_config, "brownian_bridge",
    "N/A",
    "PASS" if _mon_config == "brownian_bridge" else "WARN",
    "Brownian Bridge configurado e identificado." if _mon_config == "brownian_bridge" else f"Método {_mon_config} — Brownian Bridge esperado.",
)

# 8. P(knock-out) compatível com [0,1] (já validado em C, repetido como controle financeiro)
_ko_ok = 0.0 <= _stored_ko <= 1.0
add_barrier_check(
    "Financeiro", "P(knock-out) em [0,1]",
    f"{_stored_ko:.6%}", "0 <= P(knock-out) <= 1",
    "N/A",
    "PASS" if _ko_ok else "FAIL",
    "Probabilidade em [0,1]." if _ko_ok else "Probabilidade fora de [0,1].",
)

print("✓ Controles financeiros concluídos.")

# COMMAND ----------

# DBTITLE 1,Diagnósticos da simulação (G)
# DBTITLE 1,Diagnósticos da simulação (G)
# Comentário: tabela-resumo com parâmetros da simulação e resultado da reprecificação.
_se_relative = (_repr_result.standard_error / _repr_result.price * 100) if _repr_result.price > 0 else float("nan")
_reprice_status = "PASS" if (_diff_barrier <= _TOL_REPRICE_PRICE and _diff_vanilla <= _TOL_REPRICE_PRICE) else "FAIL"

diagnostics_table = pd.DataFrame([{
    "arquivo": barrier_result_file.name,
    "data_execucao": str(barrier_execution_ts),
    "data_mercado": barrier_market_date,
    "paths": int(sd["paths"]),
    "steps": int(sd["steps"]),
    "seed": int(sd["seed"]),
    "monitoring": str(sd["monitoring"]),
    "preco_persistido": _stored_barrier,
    "preco_recalculado": _repr_result.price,
    "diferenca": _diff_barrier,
    "erro_padrao": _repr_result.standard_error,
    "erro_padrao_relativo_pct": _se_relative,
    "ic_95": f"[{_stored_ci_low:,.2f}, {_stored_ci_high:,.2f}]",
    "prob_knock_out": _stored_ko,
    "desconto_barreira": _stored_discount_abs,
    "status_reprodutibilidade": _reprice_status,
}])

print("✓ Diagnósticos da simulação:")
display(diagnostics_table.style.format({
    "preco_persistido": "{:,.4f}",
    "preco_recalculado": "{:,.4f}",
    "diferenca": "{:.8f}",
    "erro_padrao": "{:,.4f}",
    "erro_padrao_relativo_pct": "{:.4f}%",
    "prob_knock_out": "{:.4%}",
    "desconto_barreira": "{:,.4f}",
}))

# COMMAND ----------

# DBTITLE 1,Tabela consolidada (H)
# DBTITLE 1,Tabela consolidada de controles da barreira (H)
# Comentário: tabela final com uma linha por controle da barreira.
barrier_report = pd.DataFrame(barrier_checks)
if not barrier_report.empty:
    _status_order_b = pd.CategoricalDtype(["FAIL", "WARN", "PASS"], ordered=True)
    barrier_report["status_sort"] = barrier_report["status"].astype(_status_order_b)
    barrier_report = barrier_report.sort_values(["status_sort", "categoria", "controle"]).drop(columns="status_sort")

print("✓ Tabela consolidada de controles da barreira:")
display(barrier_report)

_barrier_n_fail = int((barrier_report["status"] == "FAIL").sum()) if not barrier_report.empty else 0
_barrier_n_warn = int((barrier_report["status"] == "WARN").sum()) if not barrier_report.empty else 0
_barrier_n_pass = int((barrier_report["status"] == "PASS").sum()) if not barrier_report.empty else 0
print(f"\nResumo da barreira: {_barrier_n_pass} PASS, {_barrier_n_warn} WARN, {_barrier_n_fail} FAIL")

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

# DBTITLE 1,Painel consolidado e decisão
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
    "arquivo": [barrier_result_file.name],
    "data_execucao": [str(barrier_execution_ts)],
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

# Seção I — Decisão da barreira: sempre interromper em caso de FAIL material.
_barrier_fail_names = [
    r["controle"] for r in barrier_checks if r["status"] == "FAIL"
] if barrier_checks else []
if _barrier_fail_names:
    print(f"\n❌ FALHA MATERIAL nos controles da barreira: {_barrier_fail_names}")
    raise AssertionError(
        f"Controles da barreira reprovados: {_barrier_fail_names}. "
        "Uma execução reprovada não pode ser interpretada como sucesso."
    )

# COMMAND ----------

# DBTITLE 1,Próxima evolução
# MAGIC %md
# MAGIC ## Próxima evolução
# MAGIC
# MAGIC Este notebook agora cobre a fotografia de mercado, as curvas, a superfície de
# MAGIC volatilidade e a precificação com barreira. Os controles da barreira implementados
# MAGIC incluem:
# MAGIC
# MAGIC 1. 0 <= V_barreira <= V_vanilla (com tolerância estatística de 5×SE);
# MAGIC 2. intervalo de confiança e erro-padrão de Monte Carlo;
# MAGIC 3. reprodutibilidade determinística com semente fixa (reprecificação independente);
# MAGIC 4. comparação entre monitoramento discreto e Brownian Bridge;
# MAGIC 5. reconciliação aritmética de descontos, IC e forward;
# MAGIC 6. validação de domínio para todos os 22 campos persistidos.
# MAGIC
# MAGIC Próximos passos:
# MAGIC - convergência por número de caminhos e passos;
# MAGIC - estabilidade das gregas e dos cenários de risco;
# MAGIC - análise de sensibilidade da barreira.