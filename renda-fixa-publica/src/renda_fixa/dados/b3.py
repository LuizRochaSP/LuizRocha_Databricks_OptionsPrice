"""Ajustes diários dos futuros de juros da B3 (DI1 e DAP).

Fonte: arquivo público "TradeInformationConsolidatedFile" do pregão (arquivos.b3.com.br), um
por dia, com histórico de vários anos. Para cada instrumento negociado traz o PU de ajuste
(``AdjstdQt``) e a taxa de ajuste (``AdjstdQtTax``, % a.a.). O download é em dois passos: a API
devolve um token e o token baixa o CSV (~4 MB). O cache guarda só as linhas de DI1 e DAP.

Contratos sem negócio no dia não aparecem no arquivo. Dia sem pregão (feriado) ou arquivo ainda
não publicado levanta ``DadoIndisponivel``; nunca há dado fictício.
"""
from __future__ import annotations

import io
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

from ..derivativos.futuros import vencimento_dap, vencimento_di1
from .anbima import DadoIndisponivel

URL_TOKEN = "https://arquivos.b3.com.br/api/download/requestname?fileName=TradeInformationConsolidatedFile&date={data}"
URL_DOWNLOAD = "https://arquivos.b3.com.br/api/download?token={token}"
_TICKER = re.compile(r"^(DI1|DAP)([FGHJKMNQUVXZ])(\d{2})$")
COLUNAS = ["data_referencia", "contrato", "ativo", "vencimento", "pu_ajuste", "taxa_ajuste",
           "taxa_minima", "taxa_maxima", "taxa_ultima", "negocios", "contratos", "volume"]


def ler_ajustes_b3(texto: str) -> pd.DataFrame:
    """Linhas de DI1 e DAP do arquivo consolidado do pregão, já tipadas."""
    linhas = texto.splitlines()
    try:
        ini = next(i for i, l in enumerate(linhas) if l.startswith("RptDt;"))
    except StopIteration as exc:
        raise ValueError("Cabeçalho 'RptDt;' não encontrado no arquivo da B3.") from exc
    bruto = pd.read_csv(io.StringIO("\n".join(linhas[ini:])), sep=";", decimal=",", dtype=str)
    bruto = bruto[bruto["TckrSymb"].str.match(_TICKER, na=False)].copy()
    if bruto.empty:
        return pd.DataFrame(columns=COLUNAS)

    def num(c):
        return pd.to_numeric(bruto[c].str.replace(",", ".", regex=False), errors="coerce")

    ativo = bruto["TckrSymb"].str[:3]
    venc = [vencimento_di1(t) if a == "DI1" else vencimento_dap(t) for t, a in zip(bruto["TckrSymb"], ativo)]
    out = pd.DataFrame({
        "data_referencia": pd.to_datetime(bruto["RptDt"]).dt.date,
        "contrato": bruto["TckrSymb"],
        "ativo": ativo,
        "vencimento": venc,
        "pu_ajuste": num("AdjstdQt"),
        "taxa_ajuste": num("AdjstdQtTax"),
        "taxa_minima": num("MinPric"),
        "taxa_maxima": num("MaxPric"),
        "taxa_ultima": num("LastPric"),
        "negocios": num("TradQty"),
        "contratos": num("FinInstrmQty"),
        "volume": num("NtlFinVol"),
    })
    out = out[out["pu_ajuste"].notna() & out["taxa_ajuste"].notna()]
    return out.sort_values(["ativo", "vencimento"]).reset_index(drop=True)


def _cache_texto(df: pd.DataFrame) -> str:
    return df.to_csv(sep=";", index=False)


def baixar_ajustes_b3(
    data: date,
    cache_dir: str | Path | None = None,
    tentativas: int = 3,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Ajustes de DI1 e DAP do pregão de ``data`` (com cache ``b3_AAAAMMDD.csv``)."""
    cache = Path(cache_dir) / f"b3_{data:%Y%m%d}.csv" if cache_dir else None
    if cache and cache.exists():
        df = pd.read_csv(cache, sep=";")
        for c in ("data_referencia", "vencimento"):
            df[c] = pd.to_datetime(df[c]).dt.date
        return df

    import requests

    cab = {"User-Agent": "Mozilla/5.0"}
    erro: Exception | None = None
    for i in range(tentativas):
        try:
            r = requests.get(URL_TOKEN.format(data=data.isoformat()), headers=cab, timeout=timeout)
            if r.status_code in (400, 404):
                raise DadoIndisponivel(f"B3 sem arquivo de pregão para {data:%d/%m/%Y}.")
            r.raise_for_status()
            token = r.json().get("token") or r.json()["redirectUrl"].split("token=")[1]
            arq = requests.get(URL_DOWNLOAD.format(token=token), headers=cab, timeout=timeout)
            if arq.status_code in (400, 404) or not arq.content:
                raise DadoIndisponivel(f"B3 sem arquivo de pregão para {data:%d/%m/%Y}.")
            arq.raise_for_status()
            df = ler_ajustes_b3(arq.content.decode("utf-8-sig", errors="replace"))
            if df.empty or (df["data_referencia"] != data).any():
                raise DadoIndisponivel(f"Arquivo da B3 de {data:%d/%m/%Y} sem ajustes de DI1/DAP.")
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(_cache_texto(df), encoding="utf-8")
            return df
        except DadoIndisponivel:
            raise
        except Exception as exc:  # rede, 5xx, JSON inesperado
            erro = exc
            time.sleep(2.0 * (i + 1))
    raise DadoIndisponivel(f"Falha ao baixar o arquivo da B3 de {data:%d/%m/%Y}: {erro}")


def ajustes_di1(df: pd.DataFrame) -> list[tuple[date, float]]:
    """(vencimento, taxa de ajuste) dos DI1, no formato de ``derivativos.curva_di1``."""
    sub = df[df["ativo"] == "DI1"]
    return list(zip(sub["vencimento"], sub["taxa_ajuste"].astype(float)))
