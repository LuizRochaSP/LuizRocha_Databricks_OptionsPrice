"""Leitura de DI1/IND do SPRD da B3 e da ETTJ da ANBIMA."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from lxml import etree

from .conventions import (
    business_days,
    first_business_day,
    nearest_wednesday,
)

MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


def _latest_xml(archive: ZipFile):
    members = [m for m in archive.infolist() if m.filename.lower().endswith(".xml")]
    if not members:
        raise ValueError("ZIP sem XML")
    return max(members, key=lambda m: m.date_time)


def _contract_maturity(ticker: str) -> date:
    month = MONTH_CODES[ticker[-3]]
    year = 2000 + int(ticker[-2:])
    return first_business_day(year, month) if ticker.startswith("DI1") \
        else nearest_wednesday(year, month)


def load_di1_ind(sprd_zip: Path, valuation_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (di1, ind) com colunas [ticker, maturity, tenor_bd, ...]."""
    with ZipFile(sprd_zip) as archive:
        member = _latest_xml(archive)
        with archive.open(member) as f:
            rows = []
            for _, el in etree.iterparse(
                f, events=("end",), tag="{*}PricRpt", huge_tree=True
            ):
                ticker = el.findtext("{*}SctyId/{*}TckrSymb", default="")
                if ticker.startswith(("DI1", "IND")):
                    row = {"ticker": ticker}
                    for child in el.iter():
                        tag = etree.QName(child).localname
                        if tag in {"AdjstdQt", "AdjstdQtTax", "TradDt"}:
                            row[tag] = (child.text or "").strip()
                    rows.append(row)
                el.clear()
                while el.getprevious() is not None:
                    del el.getparent()[0]

    df = pd.DataFrame(rows)
    df["maturity"] = df["ticker"].map(_contract_maturity)
    df["tenor_bd"] = df["maturity"].apply(lambda m: business_days(valuation_date, m))

    di = df[df["ticker"].str.startswith("DI1")].copy()
    di["rate"] = pd.to_numeric(di["AdjstdQtTax"], errors="coerce") / 100.0
    di = di.dropna(subset=["rate", "tenor_bd"]).drop_duplicates("ticker")
    di = di[di["tenor_bd"] > 0].sort_values("tenor_bd")

    ind = df[df["ticker"].str.startswith("IND")].copy()
    ind["future"] = pd.to_numeric(ind["AdjstdQt"], errors="coerce")
    ind = ind.dropna(subset=["future", "tenor_bd"]).drop_duplicates("ticker")
    ind = ind[ind["tenor_bd"] > 0].sort_values("tenor_bd")

    return (
        di[["ticker", "maturity", "tenor_bd", "rate"]].reset_index(drop=True),
        ind[["ticker", "maturity", "tenor_bd", "future"]].reset_index(drop=True),
    )


def load_anbima_ettj(csv_path: Path) -> pd.DataFrame:
    """Carrega ETTJ da ANBIMA. Esperado CSV com colunas [vertice_du, taxa]."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"vertice_du", "taxa"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV deve ter colunas {required}; encontrado {df.columns.tolist()}")
    df["taxa"] = pd.to_numeric(df["taxa"], errors="coerce") / 100.0
    return df.dropna().sort_values("vertice_du").reset_index(drop=True)