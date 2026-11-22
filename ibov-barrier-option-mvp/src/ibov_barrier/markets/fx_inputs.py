"""Entradas em BRL para Quanto/Compo: PTAX (BCB), taxa DI no prazo e fechamentos do SPX.

- **PTAX:** série SGS 1 do Banco Central (dólar venda, BRL por USD), pública e sem chave.
- **Taxa BRL:** curva DI1 do arquivo SPRD da B3 (a mesma do pricer do IBOV, notebook 00/01); como
  alternativa, a curva PRE da ANBIMA via `pyettj` (notebook 06). As duas vêm em taxa efetiva 252 DU e
  são convertidas para contínua ACT/365 no prazo do contrato, a mesma base de tempo do SPX.
- **Fechamentos do SPX:** `index_history_eod` da ThetaData, para vol e correlação históricas.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
PTAX_SELL = 1           # SGS 1: dólar americano (venda), PTAX
SGS_MAX_YEARS = 10      # séries diárias do SGS aceitam no máximo 10 anos por consulta


def parse_sgs(payload) -> pd.Series:
    """JSON do SGS ([{"data": "dd/mm/aaaa", "valor": "5,4321"}, ...]) -> Series indexada por data."""
    frame = pd.DataFrame(payload)
    if frame.empty or not {"data", "valor"}.issubset(frame.columns):
        raise ValueError("Resposta do SGS vazia ou sem as colunas data/valor.")
    values = pd.to_numeric(frame["valor"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    index = pd.to_datetime(frame["data"], format="%d/%m/%Y").dt.date
    return pd.Series(values.to_numpy(), index=index, name="ptax").dropna().sort_index()


def fetch_sgs(code: int, start: date, end: date, *, session=None, timeout: float = 30.0) -> pd.Series:
    """Baixa uma série diária do SGS em blocos de até 10 anos."""
    if session is None:
        import requests
        session = requests
    parts, cursor = [], start
    while cursor <= end:
        stop = min(end, date(cursor.year + SGS_MAX_YEARS, cursor.month, 1) - timedelta(days=1))
        resp = session.get(SGS_URL.format(code=code), timeout=timeout, params={
            "formato": "json", "dataInicial": cursor.strftime("%d/%m/%Y"), "dataFinal": stop.strftime("%d/%m/%Y")})
        resp.raise_for_status()
        parts.append(parse_sgs(resp.json()))
        cursor = stop + timedelta(days=1)
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_ptax(start: date, end: date, **kw) -> pd.Series:
    return fetch_sgs(PTAX_SELL, start, end, **kw)


def value_on_or_before(series: pd.Series, when: date) -> tuple[date, float]:
    """Último valor com data <= `when` (a PTAX não sai em feriados brasileiros)."""
    sub = series.loc[[d for d in series.index if d <= when]]
    if sub.empty:
        raise ValueError(f"Sem valor até {when}.")
    return sub.index[-1], float(sub.iloc[-1])


def continuous_from_effective(rate_252: float, business_days: float, maturity_365: float) -> float:
    """Taxa efetiva 252 DU -> contínua ACT/365 no mesmo prazo: DF = (1 + r)^(-DU/252), r_c = -ln(DF)/T."""
    if business_days <= 0 or maturity_365 <= 0:
        raise ValueError("business_days and maturity must be positive")
    log_df = -business_days / 252.0 * np.log1p(rate_252)
    return float(-log_df / maturity_365)


def latest_sprd(data_root: Path, on_or_before: date, *, lookback_days: int = 10) -> tuple[date, Path]:
    """SPRD mais recente em `data_root/<AAAA-MM-DD>/SPRD<aammdd>.zip` com data <= `on_or_before`."""
    root = Path(data_root)
    for k in range(lookback_days + 1):
        d = on_or_before - timedelta(days=k)
        path = root / d.isoformat() / f"SPRD{d:%y%m%d}.zip"
        if path.is_file():
            return d, path
    raise FileNotFoundError(f"Nenhum SPRD em {root} entre {on_or_before - timedelta(days=lookback_days)} e "
                            f"{on_or_before} (rode o notebook 00 no Git folder que tem a pasta data).")


def brl_rate_from_sprd(data_root: Path, ref_date: date, expiry: date, maturity_365: float) -> dict:
    """Taxa BRL contínua ACT/365 até `expiry`, pela curva DI1 (flat forward) do SPRD mais recente."""
    from ibov_barrier.di_curve import DiscountCurve, business_days, load_di1_ind

    b3_date, path = latest_sprd(data_root, ref_date)
    di1, _ = load_di1_ind(path, b3_date)
    if di1.empty:
        raise ValueError(f"{path.name} sem contratos DI1.")
    curve = DiscountCurve.from_di1(di1["tenor_bd"].to_numpy(), di1["rate"].to_numpy())
    du = business_days(b3_date, expiry)
    effective = curve.rate(du)
    return {"fonte": f"DI1 B3 ({path.name})", "data curva": b3_date, "DU": du, "taxa efetiva 252": effective,
            "r_brl": continuous_from_effective(effective, du, maturity_365), "vértices": len(di1)}


def ettj_rate(frame: pd.DataFrame, business_days: float) -> float:
    """Taxa efetiva 252 (decimal) no prazo, interpolada em DU numa tabela de ETTJ (ANBIMA/pyettj).

    Aceita os formatos usuais: colunas de prazo em DU ('vertice_du', 'dias_uteis', 'Dias Úteis', ...) e de
    taxa ('taxa', 'taxa_pre', 'DI x pré 252', ...), em % ou decimal. Prazo só em dias corridos é rejeitado.
    """
    cols = {c: str(c).strip().lower() for c in frame.columns}
    du_col = next((c for c, n in cols.items() if "util" in n or "útei" in n or n in {"du", "vertice_du"}), None)
    rate_col = next((c for c, n in cols.items() if "252" in n), None) or \
        next((c for c, n in cols.items() if c != du_col and ("taxa" in n or "pre" in n or "pré" in n)), None)
    if du_col is None or rate_col is None:
        raise ValueError(f"ETTJ sem colunas de DU e taxa reconhecíveis: {list(frame.columns)}")
    df = pd.DataFrame({"du": pd.to_numeric(frame[du_col], errors="coerce"),
                       "taxa": pd.to_numeric(frame[rate_col].astype(str).str.replace(",", ".", regex=False),
                                             errors="coerce")}).dropna().sort_values("du")
    if df.empty:
        raise ValueError("ETTJ sem linhas numéricas.")
    if df["taxa"].abs().median() > 1.0:
        df["taxa"] /= 100.0
    # Interpolação linear no log do fator de desconto (equivale a flat forward entre vértices).
    log_df = -df["du"] / 252.0 * np.log1p(df["taxa"])
    target = float(np.interp(business_days, df["du"], log_df))
    if business_days > df["du"].iloc[-1] or business_days < df["du"].iloc[0]:
        # Fora da tabela: taxa constante do vértice mais próximo.
        edge = df.iloc[-1] if business_days > df["du"].iloc[-1] else df.iloc[0]
        return float(edge["taxa"])
    return float(np.expm1(-target * 252.0 / business_days))


def index_closes(frame: pd.DataFrame) -> pd.Series:
    """Fechamentos da ThetaData (`index_history_eod`) -> Series por data, com a coluna `close`."""
    if "close" not in frame.columns:
        raise ValueError(f"Sem coluna close: {list(frame.columns)}")
    candidates = [c for c in frame.columns if any(k in str(c).lower() for k in ("date", "time", "created"))]
    for col in candidates:
        try:
            when = pd.to_datetime(frame[col])
        except (ValueError, TypeError):
            continue
        out = pd.Series(pd.to_numeric(frame["close"], errors="coerce").to_numpy(), index=when.dt.date,
                        name="spx").dropna()
        out = out[out > 0]
        return out[~out.index.duplicated(keep="last")].sort_index()
    raise ValueError(f"Sem coluna de data reconhecível: {list(frame.columns)}")
