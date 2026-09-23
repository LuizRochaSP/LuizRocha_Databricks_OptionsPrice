# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = ["exchange-calendars==4.13.2", "lxml>=6.0"]
# ///
# MAGIC %md
# MAGIC # 03 — Análise de risco da opção com barreira
# MAGIC
# MAGIC Consome exclusivamente um resultado aprovado pelo notebook 02 e calcula gregas,
# MAGIC DV01, convexidade de taxa, VaR e Expected Shortfall.

# COMMAND ----------

# DBTITLE 1,Parâmetros de risco
RISK_HORIZON_DAYS = 1
RISK_CONFIDENCE_LEVEL = 0.99
RISK_SCENARIOS = 100_000
RISK_SEED = 42
RISK_ANNUAL_DRIFT = 0.0

# COMMAND ----------

# DBTITLE 1,Dependência explícita da validação
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Não encontrei a raiz do projeto contendo data e src.")


project_root = find_project_root()
import sys
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
from ibov_barrier.market_date import expected_market_date, require_market_date, validate_snapshot

results_root = project_root.parents[1] / "ibov-barrier-results"
validation_pattern = re.compile(r"^VALIDATION_superficieVol_(\d{8})_(\d{6})\.json$")
validation_candidates = []
for path in results_root.glob("VALIDATION_superficieVol_*.json"):
    match = validation_pattern.match(path.name)
    if match:
        stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        validation_candidates.append((stamp, path))

if not validation_candidates:
    raise RuntimeError("Nenhuma evidência de validação encontrada. Execute primeiro o notebook 02.")

validation_stamp, validation_path = max(validation_candidates, key=lambda item: item[0])
today = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
if validation_stamp.date() != today:
    raise RuntimeError("A validação mais recente não foi gerada hoje. Execute os notebooks 01 e 02.")

validation = json.loads(validation_path.read_text(encoding="utf-8"))
if validation.get("schema_version") != 1:
    raise RuntimeError("Versão incompatível da evidência de validação.")
if validation.get("fail_count") != 0 or not validation.get("approved_for_risk"):
    raise RuntimeError("O resultado não foi aprovado para análise de risco.")
if validation.get("decision") not in {"APROVADO", "APROVADO COM RESSALVAS"}:
    raise RuntimeError("Decisão de validação incompatível com a análise de risco.")

results_path = results_root / validation["results_file"]
if not results_path.is_file():
    raise RuntimeError(f"RESULTS aprovado não encontrado: {results_path}")
actual_sha256 = hashlib.sha256(results_path.read_bytes()).hexdigest()
if actual_sha256 != validation.get("results_sha256"):
    raise RuntimeError("O RESULTS foi alterado depois da validação (SHA-256 divergente).")

print(f"✓ Validação: {validation_path.name}")
print(f"✓ RESULTS aprovado: {results_path.name}")
print(f"✓ Decisão: {validation['decision']} | WARN={validation['warn_count']} | FAIL=0")

# COMMAND ----------

# DBTITLE 1,Reconstrói mercado, contrato e simulação
bundle = pd.read_csv(results_path)


def json_object(name: str) -> dict:
    rows = bundle.loc[bundle["object_name"].eq(name), "payload_json"]
    if len(rows) != 1:
        raise RuntimeError(f"Objeto {name!r} ausente ou duplicado no RESULTS.")
    return json.loads(rows.iloc[0])


metadata = json_object("metadata")
pricing_result = json_object("pricing_result")
simulation = json_object("simulation_diagnostics")
if metadata.get("schema_version") != 2:
    raise RuntimeError("O notebook 03 exige RESULTS schema_version=2.")

src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ibov_barrier import BarrierContract, MarketData, RiskConfig, calculate_risk_metrics

require_market_date(metadata["market_date"], today)
if validation.get("market_date") != metadata["market_date"]:
    raise RuntimeError("Data de mercado divergente entre RESULTS e VALIDATION.")

market_values = pricing_result["market"]
contract_values = pricing_result["contract"]
market = MarketData(
    spot=float(market_values["spot"]),
    rate=float(market_values["rate"]),
    dividend_yield=float(market_values["dividend_yield"]),
    volatility=float(market_values["volatility"]),
)
contract = BarrierContract(
    strike=float(contract_values["strike"]),
    barrier=float(contract_values["barrier"]),
    maturity=float(contract_values["maturity"]),
)

# COMMAND ----------

# DBTITLE 1,Calcula gregas, DV01, convexidade, VaR e ES
risk_config = RiskConfig(
    horizon_days=RISK_HORIZON_DAYS,
    confidence_level=RISK_CONFIDENCE_LEVEL,
    scenarios=RISK_SCENARIOS,
    seed=RISK_SEED,
    annual_drift=RISK_ANNUAL_DRIFT,
)
risk_result = calculate_risk_metrics(
    market,
    contract,
    paths=int(simulation["paths"]),
    steps=int(simulation["steps"]),
    seed=int(simulation["seed"]),
    monitoring=str(simulation["monitoring"]),
    config=risk_config,
)

risk_table = pd.DataFrame([{
    "data_mercado": metadata["market_date"],
    "arquivo_results": results_path.name,
    "decisao_validacao": validation["decision"],
    "warn_validacao": validation["warn_count"],
    **asdict(risk_result),
}])
display(risk_table)

if risk_result.expected_shortfall + 1e-12 < risk_result.var:
    raise AssertionError("Expected Shortfall não pode ser inferior ao VaR.")

print("✓ Métricas de risco calculadas.")
print(f"✓ VaR {100 * risk_result.confidence_level:.2f}%: {risk_result.var:,.2f}")
print(f"✓ Expected Shortfall: {risk_result.expected_shortfall:,.2f}")

# COMMAND ----------

# DBTITLE 1,Ressalvas herdadas do notebook 02
warnings_table = pd.DataFrame(validation.get("warnings", []))
if warnings_table.empty:
    print("✓ A validação não registrou ressalvas.")
else:
    print(f"⚠ A análise herda {len(warnings_table)} ressalva(s) de qualidade de mercado.")
    display(warnings_table)

# COMMAND ----------

# DBTITLE 1,Persiste resultado de risco
risk_filename = validation_path.name.replace("VALIDATION_", "RISK_")
risk_path = results_root / risk_filename
risk_payload = {
    "schema_version": 1,
    "risk_timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
    "validation_file": validation_path.name,
    "results_file": results_path.name,
    "results_sha256": actual_sha256,
    "market_date": metadata["market_date"],
    "validation_decision": validation["decision"],
    "validation_warn_count": int(validation["warn_count"]),
    "metrics": asdict(risk_result),
    "limitations": [
        "VaR e ES model-based por aproximação delta-gamma do risco de spot.",
        "Não inclui choques conjuntos de volatilidade e curva de juros.",
        "A aproximação pode perder precisão perto da barreira.",
    ],
}
risk_path.write_text(json.dumps(risk_payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✓ Resultado de risco salvo em: {risk_path}")
