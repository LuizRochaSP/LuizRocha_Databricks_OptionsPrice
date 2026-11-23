"""Taxas e PUs indicativos dos títulos públicos (mercado secundário ANBIMA).

Fonte: arquivo diário público ``https://www.anbima.com.br/informacoes/merc-sec/arqs/msAAMMDD.txt``,
separado por ``@`` e com vírgula decimal. Um arquivo inexistente (404) significa dia sem
divulgação (feriado ou ainda não publicado): a função levanta ``DadoIndisponivel``, nunca
devolve dado fictício.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..arredondamento import truncar
from ..titulos import criar_titulo

URL_MS = "https://www.anbima.com.br/informacoes/merc-sec/arqs/ms{aammdd}.txt"

COLUNAS = [
    "titulo",
    "data_referencia",
    "codigo_selic",
    "data_base",
    "vencimento",
    "taxa_compra",
    "taxa_venda",
    "taxa_indicativa",
    "pu",
    "desvio_padrao",
    "intervalo_inf_d0",
    "intervalo_sup_d0",
    "intervalo_inf_d1",
    "intervalo_sup_d1",
    "criterio",
]
_NUMERICAS = COLUNAS[5:14]
_DATAS = ["data_referencia", "data_base", "vencimento"]


class DadoIndisponivel(RuntimeError):
    """O arquivo do dia não existe ou não pôde ser baixado."""


def ler_mercado_secundario(texto: str) -> pd.DataFrame:
    """Converte o texto do arquivo msAAMMDD.txt num DataFrame tipado."""
    linhas = texto.splitlines()
    try:
        ini = next(i for i, l in enumerate(linhas) if l.startswith("Titulo@"))
    except StopIteration as exc:
        raise ValueError("Cabeçalho 'Titulo@' não encontrado no arquivo da ANBIMA.") from exc
    registros = []
    for l in linhas[ini + 1 :]:
        partes = l.strip().split("@")
        if len(partes) < len(COLUNAS):
            continue
        registros.append(partes[: len(COLUNAS)])
    df = pd.DataFrame(registros, columns=COLUNAS)
    for c in _NUMERICAS:
        df[c] = pd.to_numeric(df[c].str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    for c in _DATAS:
        df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce").dt.date
    df["titulo"] = df["titulo"].str.strip()
    return df.sort_values(["titulo", "vencimento"]).reset_index(drop=True)


def baixar_mercado_secundario(
    data: date,
    cache_dir: str | Path | None = None,
    tentativas: int = 3,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Baixa (ou lê do cache) o arquivo ANBIMA do dia ``data``."""
    nome = f"ms{data:%y%m%d}.txt"
    cache = Path(cache_dir) / nome if cache_dir else None
    if cache and cache.exists():
        return ler_mercado_secundario(cache.read_text(encoding="latin-1"))

    import requests

    url = URL_MS.format(aammdd=f"{data:%y%m%d}")
    ultimo_erro: Exception | None = None
    for i in range(tentativas):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                raise DadoIndisponivel(f"ANBIMA sem arquivo para {data:%d/%m/%Y} ({url}).")
            r.raise_for_status()
            texto = r.content.decode("latin-1")
            df = ler_mercado_secundario(texto)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(texto, encoding="latin-1")
            return df
        except DadoIndisponivel:
            raise
        except Exception as exc:  # rede, 5xx
            ultimo_erro = exc
            time.sleep(2.0 * (i + 1))
    raise DadoIndisponivel(f"Falha ao baixar {url}: {ultimo_erro}")


def ultimo_disponivel(ate: date, cache_dir: str | Path | None = None, max_dias: int = 10) -> pd.DataFrame:
    """Arquivo mais recente em ou antes de ``ate`` (volta até ``max_dias`` dias corridos)."""
    for k in range(max_dias + 1):
        d = ate - timedelta(days=k)
        if d.weekday() >= 5:
            continue
        try:
            return baixar_mercado_secundario(d, cache_dir)
        except DadoIndisponivel:
            continue
    raise DadoIndisponivel(f"Nenhum arquivo ANBIMA entre {ate - timedelta(days=max_dias)} e {ate}.")


@dataclass(frozen=True)
class VNAImplicito:
    vna: float
    consistente: bool  # True se um único VNA (6 casas) reproduz o PU de todos os vencimentos
    n_titulos: int


def vna_implicito(df: pd.DataFrame, titulo: str) -> VNAImplicito:
    """VNA do dia implícito nos PUs da ANBIMA para ``titulo`` (NTN-B ou LFT).

    Cada vencimento restringe o VNA a um intervalo [PU / c, (PU + 1e-6) / c), com c a cotação
    truncada. A interseção dos intervalos devolve o VNA com 6 casas que reproduz todos os PUs.
    """
    sub = df[(df["titulo"] == titulo) & df["taxa_indicativa"].notna() & df["pu"].notna()]
    if sub.empty:
        raise ValueError(f"Sem {titulo} no arquivo.")
    lo, hi, pontos = -math.inf, math.inf, []
    for _, r in sub.iterrows():
        t = criar_titulo(titulo, r["vencimento"])
        cot = truncar(t.cotacao_bruta(float(r["taxa_indicativa"]), r["data_referencia"]), 4) / 100.0
        a, b = r["pu"] / cot, (r["pu"] + 1e-6) / cot
        lo, hi = max(lo, a), min(hi, b)
        pontos.append(a)
    candidato = math.ceil(round(lo * 1e6, 3)) / 1e6
    if candidato < hi:
        return VNAImplicito(candidato, True, len(sub))
    mediana = float(pd.Series(pontos).median())
    return VNAImplicito(truncar(mediana, 6), False, len(sub))
