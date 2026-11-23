"""Histórico de taxas e preços do Tesouro Direto (Tesouro Transparente), de 2004 em diante.

Para que serve: o arquivo diário da ANBIMA só fica público por poucos dias, e o histórico
do Tesouro Direto é público e longo. Ele serve para curvas históricas (VaR, PCA), não para
validar PU: as taxas têm 2 casas, são da manhã e trazem o spread do varejo. Por isso a
"taxa_indicativa" aqui é a média entre a taxa de compra e a de venda.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd

URL_TD = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/"
    "resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
)

NOMES = {
    "Tesouro Prefixado": "LTN",
    "Tesouro Prefixado com Juros Semestrais": "NTN-F",
    "Tesouro IPCA+ com Juros Semestrais": "NTN-B",
    "Tesouro IPCA+": "NTN-B Principal",
    "Tesouro Selic": "LFT",
}


def ler_tesouro_direto(texto_ou_bytes) -> pd.DataFrame:
    """Converte o CSV do Tesouro Direto no formato de colunas do arquivo ANBIMA.

    Títulos fora do escopo (IGP-M, Renda+, Educa+) são descartados.
    """
    if isinstance(texto_ou_bytes, bytes):
        texto_ou_bytes = texto_ou_bytes.decode("latin-1")
    df = pd.read_csv(io.StringIO(texto_ou_bytes), sep=";", decimal=",", dtype={"Tipo Titulo": str})
    df = df[df["Tipo Titulo"].isin(NOMES)].copy()
    out = pd.DataFrame(
        {
            "titulo": df["Tipo Titulo"].map(NOMES),
            "data_referencia": pd.to_datetime(df["Data Base"], format="%d/%m/%Y").dt.date,
            "vencimento": pd.to_datetime(df["Data Vencimento"], format="%d/%m/%Y").dt.date,
            "taxa_compra": df["Taxa Compra Manha"].astype(float),
            "taxa_venda": df["Taxa Venda Manha"].astype(float),
            "pu_compra": df["PU Compra Manha"].astype(float),
            "pu_venda": df["PU Venda Manha"].astype(float),
        }
    )
    out["taxa_indicativa"] = (out["taxa_compra"] + out["taxa_venda"]) / 2.0
    # Fora do horário de compra o Tesouro publica taxa de compra zerada: usa só a de venda.
    so_venda = (out["taxa_compra"] == 0) & (out["titulo"] != "LFT")
    out.loc[so_venda, "taxa_indicativa"] = out.loc[so_venda, "taxa_venda"]
    return out.sort_values(["data_referencia", "titulo", "vencimento"]).reset_index(drop=True)


def baixar_tesouro_direto(cache_dir: str | Path | None = None, max_idade_horas: float = 12.0, tentativas: int = 3) -> pd.DataFrame:
    """Baixa o histórico completo (~15 MB), com cache em ``cache_dir/tesouro_direto.csv``."""
    cache = Path(cache_dir) / "tesouro_direto.csv" if cache_dir else None
    if cache and cache.exists() and (time.time() - cache.stat().st_mtime) < max_idade_horas * 3600:
        return ler_tesouro_direto(cache.read_bytes())
    import requests

    erro = None
    for i in range(tentativas):
        try:
            r = requests.get(URL_TD, timeout=120)
            r.raise_for_status()
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(r.content)
            return ler_tesouro_direto(r.content)
        except Exception as exc:
            erro = exc
            time.sleep(3.0 * (i + 1))
    if cache and cache.exists():  # cache vencido ainda é dado real
        return ler_tesouro_direto(cache.read_bytes())
    raise RuntimeError(f"Falha ao baixar o histórico do Tesouro Direto: {erro}")
