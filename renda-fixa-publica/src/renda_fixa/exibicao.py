"""Números no padrão brasileiro para o dashboard: ponto no milhar e vírgula no decimal.

- ``br``: um número formatado (métricas, textos);
- ``tabela``: Styler para ``st.dataframe`` com vazios como "—" (o st.dataframe ignora o ``na_rep``
  do Styler e mostraria "None");
- ``csv_br``: CSV para o Excel em português (separador ";", decimal ","), com os números puros.
"""
from __future__ import annotations

import pandas as pd

SEM_MILHAR = ("du",)  # contagens de dias úteis ficam sem separador


def _troca(texto: str) -> str:
    return texto.replace(",", "\0").replace(".", ",").replace("\0", ".")


def br(valor: float, casas: int = 2, sinal: bool = False) -> str:
    """1234567.891 -> "1.234.567,89"; com ``sinal`` -> "+1.234.567,89"."""
    if valor is None or pd.isna(valor):
        return "—"
    return _troca(f"{valor:{'+' if sinal else ''},.{casas}f}")


def _formato_padrao(col: pd.Series, nome: str, casas: int) -> str:
    validos = col.dropna()
    inteiro = len(validos) > 0 and bool((validos % 1 == 0).all())
    milhar = "" if nome in SEM_MILHAR else ","
    return f"{{:{milhar}.0f}}" if inteiro else f"{{:{milhar}.{casas}f}}"


def _numericas(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas de texto que só têm números e vazios (null do JSON, None) viram numéricas."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            num = pd.to_numeric(out[c], errors="coerce")
            if num.notna().sum() == out[c].notna().sum():
                out[c] = num
    return out


def _e_numero(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)


def tabela(df: pd.DataFrame, formatos: dict | None = None, casas: int = 4):
    """Styler com números no padrão brasileiro e vazios como "—".

    Colunas numéricas sem vazios seguem numéricas (ordenáveis e com o valor completo no download
    da própria tabela); as que têm vazios viram texto já formatado. ``formatos`` usa a sintaxe do
    Python ("{:,.2f}"): a vírgula do milhar e o ponto do decimal são trocados na exibição.
    """
    formatos = formatos or {}
    out, fmt = _numericas(df), {}
    for c in out.columns:
        col = out[c]
        if not _e_numero(col):
            continue
        f = formatos.get(c) or _formato_padrao(col, c, casas)
        if col.isna().any():
            out[c] = col.map(lambda v, f=f: "—" if pd.isna(v) else _troca(f.format(v)))
        else:
            fmt[c] = f
    return out.style.format(fmt, decimal=",", thousands=".")


def tem_vazios(df: pd.DataFrame) -> bool:
    """Se alguma coluna numérica tem vazios (aí a tabela mostra texto e vale oferecer o CSV)."""
    out = _numericas(df)
    return any(_e_numero(out[c]) and out[c].isna().any() for c in out.columns)


def csv_br(df: pd.DataFrame) -> bytes:
    """CSV que o Excel em português abre direto: ";" entre colunas, vírgula decimal, vazio = célula vazia."""
    return _numericas(df).to_csv(sep=";", decimal=",", index=False).encode("utf-8-sig")
