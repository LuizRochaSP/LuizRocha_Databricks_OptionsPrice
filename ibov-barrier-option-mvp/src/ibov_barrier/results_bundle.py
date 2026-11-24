"""Arquivo de resultados no formato do notebook 01 (lido pelo dashboard).

Um CSV com uma linha por objeto: `object_name`, `object_type` ("json" ou "dataframe") e `payload_json`.
DataFrames vão com `to_json(orient="table")`; dicionários, listas e Series vão como JSON.
O dashboard tem um leitor próprio (dashboard/multiativos.py) porque o app só publica a pasta dashboard/.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


def _default(value):
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return {str(k): v for k, v in value.items()}
    raise TypeError(f"tipo não serializável: {type(value).__name__}")


def _normalize(value):
    """Converte para tipos JSON puros; NaN/inf viram null (JSON padrão não tem NaN)."""
    if isinstance(value, pd.Series):
        value = {str(k): v for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_normalize(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return _default(value)


def _frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.index.name is None and out.index.nlevels == 1 and pd.api.types.is_integer_dtype(out.index):
        # índice posicional (inclusive fora de ordem, depois de sort_values): não vira coluna
        out = out.reset_index(drop=True)
    else:
        # índice com significado (nomeado, datas, MultiIndex) vira coluna; o payload é gravado com
        # index=False, então uma coluna "index" não colide mais com o índice do orient="table"
        out = out.reset_index()
    out.columns = [str(c) for c in out.columns]
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: v.isoformat() if isinstance(v, (date, datetime)) else v)
    return out.reset_index(drop=True)


def bundle_rows(objects: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    for name, obj in objects.items():
        if isinstance(obj, pd.DataFrame):
            rows.append({"object_name": name, "object_type": "dataframe",
                         "payload_json": _frame(obj).to_json(orient="table", date_format="iso", index=False)})
        else:
            text = json.dumps(_normalize(obj), ensure_ascii=False, allow_nan=False)
            rows.append({"object_name": name, "object_type": "json", "payload_json": text})
    return pd.DataFrame(rows, columns=["object_name", "object_type", "payload_json"])


def write_bundle(path: Path | str, objects: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_rows(objects).to_csv(path, index=False)
    return path


def read_bundle(source) -> dict:
    """Lê o CSV (caminho, buffer ou DataFrame) e devolve {nome: dict | DataFrame}."""
    frame = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    out = {}
    for row in frame.itertuples(index=False):
        if row.object_type == "dataframe":
            out[row.object_name] = read_frame(row.payload_json)
        else:
            out[row.object_name] = json.loads(row.payload_json)
    return out


def read_frame(payload: str) -> pd.DataFrame:
    """DataFrame de um payload orient="table".

    Arquivos gravados antes da correção do _frame podem ter dois campos "index" no schema, que o
    pandas recusa ("Index data must be 1-dimensional"). Nesse caso lê os registros direto e descarta
    as colunas de índice posicional.
    """
    try:
        return pd.read_json(StringIO(payload), orient="table")
    except ValueError:
        data = json.loads(payload)
        cols = [f["name"] for f in data["schema"]["fields"] if f["name"] not in ("index", "level_0")]
        return pd.DataFrame(data["data"]).reindex(columns=cols)


def results_filename(kind: str, when: datetime) -> str:
    """RESULTS_<kind>_<AAAAMMDD_HHMMSS>.csv; o dashboard separa os tipos pelo prefixo."""
    return f"RESULTS_{kind}_{when:%Y%m%d_%H%M%S}.csv"
