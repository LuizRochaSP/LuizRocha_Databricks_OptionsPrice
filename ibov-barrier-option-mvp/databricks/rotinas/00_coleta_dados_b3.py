# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "exchange-calendars==4.13.2",
#   "lxml>=6.0",
# ]
# ///
# DBTITLE 1,Título
# MAGIC %md
# MAGIC # Download de boletins diários da B3
# MAGIC
# MAGIC Este notebook baixa automaticamente os arquivos públicos da B3 (boletins diários) via **pesquisa por pregão** e os salva na pasta `data/YYYY-MM-DD/` do projeto.
# MAGIC
# MAGIC ## Arquivos baixados
# MAGIC
# MAGIC | Arquivo | Conteúdo | Especificação B3 |
# MAGIC |---|---|---|
# MAGIC | `IR` | Arquivo de Índices (Ibovespa à vista) | BVBG.087.01 IndexReport |
# MAGIC | `PR` | Boletim de Negociação (preços por instrumento) | BVBG.086.01 PriceReport |
# MAGIC | `SPRD` | Boletim Simplificado de Derivativos | BVBG.187.01 Simplified Price Report |
# MAGIC | `IN` | Cadastro de Instrumentos | BVBG.028.02 Instruments File |
# MAGIC
# MAGIC ## Como funciona
# MAGIC
# MAGIC 1. O usuário informa uma data ou intervalo de datas.
# MAGIC 2. O notebook monta a URL de download da B3 no formato `pesquisapregao/download?filelist=IR{YYMMDD}.zip,...`.
# MAGIC 3. A B3 retorna um ZIP externo contendo os 4 ZIPs individuais.
# MAGIC 4. O notebook extrai cada ZIP individual para `data/YYYY-MM-DD/`.
# MAGIC 5. Datas sem pregão (finais de semana, feriados) retornam um ZIP vazio e são puladas automaticamente.
# MAGIC 6. Datas já baixadas são puladas por padrão (a menos que `OVERWRITE = True`).

# COMMAND ----------

# DBTITLE 1,Configuração
# Comentário: configura as datas e os parâmetros de download.
from datetime import date, timedelta

# Data única ou intervalo. Para uma única data, use START_DATE = END_DATE.
START_DATE = "2026-09-24"
END_DATE = "2026-09-24"
# Ambos None: rotina D-1. Intervalo explícito: coleta histórica, limitada a D-1.

# Sobrescrever datas já baixadas.
OVERWRITE = False

# Arquivos a baixar (prefixos da B3).
FILE_PREFIXES = ["IR", "PR", "SPRD", "IN"]

# URL base do serviço de download da B3.
B3_DOWNLOAD_URL = "https://www.b3.com.br/pesquisapregao/download"
B3_REFERER = (
    "https://www.b3.com.br/pt_br/market-data-e-indices/"
    "servicos-de-dados/market-data/historico/boletins-diarios/"
    "pesquisa-por-pregao/pesquisa-por-pregao/"
)

# Tamanho mínimo aceitável para o ZIP externo (em bytes).
# Respostas menores indicam que não houve pregão na data.
MIN_DOWNLOAD_SIZE = 1000

# COMMAND ----------

# DBTITLE 1,Funções auxiliares
# Comentário: funções auxiliares para localizar a raiz do projeto, formatar datas e baixar arquivos.
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZipFile
import io
import requests


def find_project_root() -> Path:
    """Localiza a raiz do projeto procurando pelas pastas data e src."""
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Não encontrei a raiz do projeto. Confirme que existem as pastas data e src."
    )


def format_b3_date(d: date) -> str:
    """Formata a data no padrão YYMMDD usado pela B3 nos nomes de arquivo."""
    return d.strftime("%y%m%d")


def build_filelist(d: date, prefixes: list[str]) -> str:
    """Monta a lista de arquivos para o parâmetro filelist da URL."""
    yy_mm_dd = format_b3_date(d)
    return ",".join(f"{prefix}{yy_mm_dd}.zip" for prefix in prefixes)


def is_folder_complete(folder: Path, prefixes: list[str]) -> bool:
    """Verifica se a pasta já contém todos os arquivos esperados."""
    for prefix in prefixes:
        if not list(folder.glob(f"{prefix}*.zip")):
            return False
    return True


def download_and_extract(d: date, prefixes: list[str], data_root: Path, overwrite: bool) -> dict:
    """Baixa e extrai os arquivos da B3 para uma data específica."""
    folder = data_root / d.isoformat()

    if folder.exists() and is_folder_complete(folder, prefixes) and not overwrite:
        return {"data": d.isoformat(), "status": "skip", "motivo": "já baixado"}

    folder.mkdir(parents=True, exist_ok=True)

    filelist = build_filelist(d, prefixes)
    params = {"filelist": filelist}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": B3_REFERER,
    }

    try:
        resp = requests.get(B3_DOWNLOAD_URL, params=params, headers=headers, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"data": d.isoformat(), "status": "erro", "motivo": f"HTTP: {exc}"}

    if len(resp.content) < MIN_DOWNLOAD_SIZE:
        return {"data": d.isoformat(), "status": "vazio", "motivo": "sem pregão"}

    try:
        outer_zip = ZipFile(io.BytesIO(resp.content))
        extracted = []
        for member_name in outer_zip.namelist():
            if not member_name.lower().endswith(".zip"):
                continue
            member_data = outer_zip.read(member_name)
            dest_path = folder / member_name
            dest_path.write_bytes(member_data)
            extracted.append(member_name)

        if not extracted:
            return {"data": d.isoformat(), "status": "vazio", "motivo": "ZIP sem arquivos esperados"}

        return {
            "data": d.isoformat(),
            "status": "ok",
            "arquivos": extracted,
            "tamanho_total": len(resp.content),
        }
    except Exception as exc:
        return {"data": d.isoformat(), "status": "erro", "motivo": f"ZIP: {exc}"}

# COMMAND ----------

# DBTITLE 1,Execução do download
# Comentário: executa o download para cada data no intervalo, pulando finais de semana.
project_root = find_project_root()
data_root = project_root / "data"
import sys
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
from ibov_barrier.market_date import expected_market_date, require_market_date, validate_snapshot

from ibov_barrier.market_date import b3_calendar
expected_date = expected_market_date()
calendar = b3_calendar()

if (START_DATE is None) != (END_DATE is None):
    raise ValueError("Informe ambas as datas ou deixe ambas como None.")
start = expected_date if START_DATE is None else date.fromisoformat(START_DATE)
end = expected_date if END_DATE is None else date.fromisoformat(END_DATE)
if end > expected_date:
    raise ValueError(f"A coleta deve terminar até D-1 B3: {expected_date}.")

if end < start:
    raise ValueError("END_DATE deve ser maior ou igual a START_DATE.")

results = []
current = start
while current <= end:
    if not calendar.is_session(current.isoformat()):
        results.append({"data": current.isoformat(), "status": "skip", "motivo": "sem sessão no calendário B3"})
        current += timedelta(days=1)
        continue

    print(f"Baixando {current.isoformat()}...", end=" ")
    result = download_and_extract(current, FILE_PREFIXES, data_root, OVERWRITE)
    results.append(result)
    if result["status"] not in {"ok", "skip"}:
        raise RuntimeError(f"Coleta incompleta: {result}")
    validate_snapshot(data_root, current)

    if result["status"] == "ok":
        size_mb = result.get("tamanho_total", 0) / 1e6
        n_files = len(result.get("arquivos", []))
        print(f"OK ({n_files} arquivos, {size_mb:.1f} MB)")
    elif result["status"] == "skip":
        print(f"{result['motivo']}")
    elif result["status"] == "vazio":
        print(f"{result['motivo']}")
    else:
        print(f"ERRO: {result['motivo']}")

    current += timedelta(days=1)

# COMMAND ----------

# DBTITLE 1,Resumo
# Comentário: resume os resultados e lista o conteúdo das pastas baixadas.
import pandas as pd

summary = pd.DataFrame(results)
display(summary)

ok_count = (summary["status"] == "ok").sum()
skip_count = (summary["status"] == "skip").sum()
vazio_count = (summary["status"] == "vazio").sum()
erro_count = (summary["status"] == "erro").sum()

print(f"\nResumo: {ok_count} baixados, {skip_count} pulados, {vazio_count} sem pregão, {erro_count} erros")

# Lista as pastas recém-baixadas.
for result in results:
    if result["status"] != "ok":
        continue
    folder = data_root / result["data"]
    files = sorted(folder.glob("*.zip"))
    print(f"\n{result['data']} ({len(files)} arquivos):")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")