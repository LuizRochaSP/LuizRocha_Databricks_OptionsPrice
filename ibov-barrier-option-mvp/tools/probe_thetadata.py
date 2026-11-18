"""Sonda da biblioteca `thetadata` (Python 3.12+): descobre a API real antes de fixar o adaptador SPX.

Não grava nada e não imprime a chave. Uso (local ou célula de notebook Databricks):

    pip install thetadata
    THETADATA_API_KEY=... python tools/probe_thetadata.py            # só introspecção
    THETADATA_API_KEY=... python tools/probe_thetadata.py --call NOME --kwargs '{"symbol": "SPXW", ...}'

No Databricks, a chave vem de `dbutils.secrets.get("quantita", "thetadata_api_key")` e é passada
por variável de ambiente; nunca escrever a chave no código ou no notebook.

O que anotar no ARQUITETURA_MULTIATIVOS.md depois de rodar:
  1. nome do cliente e da chamada EOD de opções (todas as opções de uma série numa data);
  2. colunas retornadas e unidades (strike em dólares ou milésimos; right C/P ou CALL/PUT; expiration);
  3. tipo de retorno (pandas ou outro);
  4. se há EOD do índice SPX e taxas (SOFR) no plano contratado;
  5. horário em que o EOD de D-1 fica disponível.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys


def describe(obj, prefix=""):
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        attr = getattr(obj, name, None)
        if callable(attr):
            try:
                sig = str(inspect.signature(attr))
            except (TypeError, ValueError):
                sig = "(...)"
            doc = (inspect.getdoc(attr) or "").splitlines()
            print(f"  {prefix}{name}{sig}  # {doc[0] if doc else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--call", help="método do cliente a chamar")
    parser.add_argument("--kwargs", default="{}", help="argumentos JSON da chamada")
    args = parser.parse_args()

    if sys.version_info < (3, 12):
        sys.exit(f"thetadata exige Python 3.12+; este é {sys.version.split()[0]}")
    try:
        import thetadata
    except ImportError:
        sys.exit("pip install thetadata")
    from importlib.metadata import version
    print(f"thetadata {version('thetadata')}")

    print("\nMódulo thetadata:")
    describe(thetadata)
    classes = [(n, c) for n, c in inspect.getmembers(thetadata, inspect.isclass) if not n.startswith("_")]
    for name, cls in classes:
        print(f"\nClasse {name}{inspect.signature(cls) if callable(cls) else ''}:")
        describe(cls, prefix=f"{name}.")

    if not args.call:
        return
    key = os.environ.get("THETADATA_API_KEY")
    if not key:
        sys.exit("Defina THETADATA_API_KEY (via Databricks Secret).")
    client_cls = next((c for n, c in classes if "client" in n.lower()), None)
    if client_cls is None:
        sys.exit("Nenhuma classe *Client encontrada; ajuste a sonda com os nomes listados acima.")
    params = inspect.signature(client_cls).parameters
    kw = {p: key for p in params if "key" in p.lower()} or {}
    client = client_cls(**kw) if kw else client_cls(key)
    result = getattr(client, args.call)(**json.loads(args.kwargs))
    print(f"\nRetorno de {args.call}: {type(result)}")
    try:
        import pandas as pd
        frame = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        print(frame.dtypes)
        print(frame.head(10).to_string())
        print(f"linhas: {len(frame)}")
    except Exception as exc:  # noqa: BLE001 - sonda: mostrar o que vier
        print(f"Não converteu para DataFrame ({exc}); repr: {result!r:.2000}")


if __name__ == "__main__":
    main()
