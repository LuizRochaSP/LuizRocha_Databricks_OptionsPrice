"""Séries do Banco Central (SGS) com retries, janelas de até 10 anos e cache em parquet.

Códigos úteis:
- 433: IPCA, variação mensal (%)
- 11: Selic diária (% a.d.), usada no VNA da LFT
- 12: CDI diário (% a.d.), usado na ponta DI dos swaps e no ajuste do DI1
- 1178: Selic anualizada base 252 (% a.a.)
- 4389: CDI anualizado base 252 (% a.a.)
- 432: meta Selic (% a.a.)
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd

URL_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json&dataInicial={ini}&dataFinal={fim}"

IPCA_MENSAL = 433
SELIC_DIARIA = 11
CDI_DIARIO = 12
SELIC_ANUAL = 1178
CDI_ANUAL = 4389
META_SELIC = 432


def _baixar_janela(codigo: int, ini: date, fim: date, tentativas: int, timeout: float) -> pd.Series:
    import requests

    url = URL_SGS.format(codigo=codigo, ini=f"{ini:%d/%m/%Y}", fim=f"{fim:%d/%m/%Y}")
    erro: Exception | None = None
    for i in range(tentativas):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return pd.Series(dtype=float)
            r.raise_for_status()
            dados = r.json()
            if not dados:
                return pd.Series(dtype=float)
            df = pd.DataFrame(dados)
            idx = pd.to_datetime(df["data"], format="%d/%m/%Y").dt.date
            return pd.Series(pd.to_numeric(df["valor"], errors="coerce").values, index=idx, name=codigo)
        except Exception as exc:
            erro = exc
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"SGS {codigo} falhou ({ini} a {fim}): {erro}")


def serie_sgs(
    codigo: int,
    inicio: date,
    fim: date,
    cache_dir: str | Path | None = None,
    tentativas: int = 4,
    timeout: float = 30.0,
) -> pd.Series:
    """Série SGS entre ``inicio`` e ``fim`` (inclusive), indexada por ``date``.

    Com ``cache_dir``, guarda ``sgs_<codigo>.parquet`` e só baixa o que falta.
    """
    cache = Path(cache_dir) / f"sgs_{codigo}.parquet" if cache_dir else None
    atual = pd.Series(dtype=float)
    if cache and cache.exists():
        df = pd.read_parquet(cache)
        atual = pd.Series(df["valor"].values, index=pd.to_datetime(df["data"]).dt.date, name=codigo)

    faltando: list[tuple[date, date]] = []
    if atual.empty:
        faltando.append((inicio, fim))
    else:
        if inicio < min(atual.index):
            faltando.append((inicio, min(atual.index)))
        if fim > max(atual.index):
            faltando.append((max(atual.index), fim))

    partes = [atual]
    for a, b in faltando:
        ano = a.year
        while date(ano, 1, 1) <= b:
            ja = max(a, date(ano, 1, 1))
            jb = min(b, date(ano + 9, 12, 31))
            partes.append(_baixar_janela(codigo, ja, jb, tentativas, timeout))
            ano += 10

    s = pd.concat([p for p in partes if not p.empty]) if any(not p.empty for p in partes) else pd.Series(dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if cache and not s.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"data": pd.to_datetime(list(s.index)), "valor": s.values}).to_parquet(cache, index=False)
    return s[(s.index >= inicio) & (s.index <= fim)]


def fator_cdi(cdi_diario_pct: pd.Series, inicio: date, fim: date, pct_cdi: float = 100.0) -> float:
    """Fator acumulado do CDI de ``inicio`` (inclusive) a ``fim`` (exclusive), a ``pct_cdi``% do CDI.

    Convenção B3/CETIP: produto de (1 + TDI_k x p), com TDI_k a taxa diária (fração) arredondada
    na 8ª casa e o produto arredondado na 16ª. A série SGS 12 já é a taxa diária em % a.d.
    """
    from ..arredondamento import arredondar

    s = cdi_diario_pct[(cdi_diario_pct.index >= inicio) & (cdi_diario_pct.index < fim)]
    fator = 1.0
    for taxa in s.values:
        fator = arredondar(fator * (1.0 + arredondar(float(taxa) / 100.0, 8) * pct_cdi / 100.0), 16)
    return fator
