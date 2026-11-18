"""Resolução da chave da ThetaData: um único lugar, ordem fixa, nunca no código nem no repositório.

Ordem de busca (a primeira que existir vence):
  1. argumento explícito `api_key=` (testes, chamadas pontuais);
  2. variável de ambiente THETADATA_API_KEY;
  3. Databricks Secret `quantita/thetadata_api_key` (só quando roda dentro do Databricks);
  4. arquivo local: THETADATA_KEY_FILE, ou ~/.thetadata/tdk.txt (fora do OneDrive e do repositório).

Para trocar a chave:
  - local: `python tools/set_thetadata_key.py` (pede a chave sem mostrar na tela e grava o arquivo);
  - Databricks: `databricks secrets put-secret quantita thetadata_api_key`.
Nenhum código muda na troca.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "THETADATA_API_KEY"
FILE_ENV_VAR = "THETADATA_KEY_FILE"
SECRET_SCOPE, SECRET_KEY = "quantita", "thetadata_api_key"
DEFAULT_KEY_FILE = Path.home() / ".thetadata" / "tdk.txt"


@dataclass(frozen=True)
class ResolvedKey:
    value: str
    source: str  # de onde veio, para log; nunca o valor

    def __repr__(self) -> str:  # evita vazar a chave em prints e tracebacks
        return f"ResolvedKey(source={self.source!r}, value={mask(self.value)!r})"


def mask(key: str) -> str:
    """Mostra só o começo e o fim: td1_prod_abcd…wxyz."""
    return f"{key[:12]}…{key[-4:]}" if len(key) > 20 else "***"


def _clean(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = raw.strip().strip('"').strip("'")
    if not key:
        return None
    if any(ch.isspace() for ch in key):
        raise ValueError("Chave ThetaData com espaços ou quebras de linha internas; regrave o arquivo/secret.")
    return key


def key_file_path() -> Path:
    return Path(os.environ.get(FILE_ENV_VAR) or DEFAULT_KEY_FILE).expanduser()


def _databricks_secret() -> str | None:
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        return None
    try:
        from databricks.sdk.runtime import dbutils  # disponível em notebooks, jobs e apps do Databricks
        return dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY)
    except Exception:  # noqa: BLE001 - secret ausente ou sem permissão: tenta a próxima fonte
        return None


def resolve_thetadata_key(api_key: str | None = None) -> ResolvedKey:
    candidates = (
        ("argumento", lambda: api_key),
        (f"variável de ambiente {ENV_VAR}", lambda: os.environ.get(ENV_VAR)),
        (f"Databricks Secret {SECRET_SCOPE}/{SECRET_KEY}", _databricks_secret),
        (f"arquivo {key_file_path()}",
         lambda: key_file_path().read_text(encoding="utf-8") if key_file_path().is_file() else None),
    )
    for source, getter in candidates:
        key = _clean(getter())
        if key:
            return ResolvedKey(key, source)
    raise RuntimeError(
        "Chave ThetaData não encontrada. Configure uma das opções: "
        f"(local) python tools/set_thetadata_key.py -> {key_file_path()}; "
        f"(sessão) variável {ENV_VAR}; "
        f"(Databricks) databricks secrets put-secret {SECRET_SCOPE} {SECRET_KEY}.")
