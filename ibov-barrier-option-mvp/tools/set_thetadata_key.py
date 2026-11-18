"""Grava ou troca a chave da ThetaData nesta máquina, sem mostrar a chave na tela.

    python tools/set_thetadata_key.py            # pede a chave (entrada oculta) e grava
    python tools/set_thetadata_key.py --status   # mostra de onde a chave está sendo lida (mascarada)
    python tools/set_thetadata_key.py --test     # faz uma chamada leve à API para validar (Python 3.12+)

Destino: THETADATA_KEY_FILE, ou ~/.thetadata/tdk.txt, fora do OneDrive e do repositório.
O arquivo fica legível só pelo seu usuário (icacls no Windows, chmod 600 nos demais).
No Databricks não use este script: `databricks secrets put-secret quantita thetadata_api_key`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ibov_barrier.markets.credentials import key_file_path, mask, resolve_thetadata_key  # noqa: E402


def _inside_sync_or_repo(path: Path) -> str | None:
    text = str(path.resolve()).lower()
    if "onedrive" in text or "dropbox" in text or "google drive" in text:
        return "pasta sincronizada com a nuvem"
    for parent in path.resolve().parents:
        if (parent / ".git").exists():
            return f"repositório git ({parent})"
    return None


def _restrict(path: Path) -> None:
    if os.name == "nt":
        user = os.environ.get("USERNAME", "")
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                       check=True, capture_output=True)
    else:
        path.chmod(0o600)


def save() -> None:
    path = key_file_path()
    problem = _inside_sync_or_repo(path)
    if problem:
        sys.exit(f"Recusado: {path} está dentro de {problem}. Use outro THETADATA_KEY_FILE.")
    key = getpass.getpass("Cole a chave da ThetaData (não aparece na tela): ").strip()
    if not key or any(ch.isspace() for ch in key):
        sys.exit("Chave vazia ou com espaços; nada foi gravado.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(key, encoding="utf-8")
    _restrict(tmp)
    tmp.replace(path)  # troca atômica: nunca fica um arquivo pela metade
    print(f"Chave {mask(key)} gravada em {path} (acesso só do seu usuário).")
    if os.environ.get("THETADATA_API_KEY"):
        print("Atenção: a variável THETADATA_API_KEY está definida e tem prioridade sobre o arquivo.")


def status() -> None:
    resolved = resolve_thetadata_key()
    print(f"Chave {mask(resolved.value)} lida de: {resolved.source}")


def test() -> None:
    resolved = resolve_thetadata_key()
    from thetadata import ThetaClient
    client = ThetaClient(api_key=resolved.value, dataframe_type="pandas")
    day = dt.date.today() - dt.timedelta(days=7)
    frame = client.interest_rate_history_eod("SOFR", day, dt.date.today())
    print(f"OK: chave {mask(resolved.value)} ({resolved.source}) autenticou; {len(frame)} dias de SOFR.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true")
    group.add_argument("--test", action="store_true")
    args = parser.parse_args()
    status() if args.status else test() if args.test else save()
