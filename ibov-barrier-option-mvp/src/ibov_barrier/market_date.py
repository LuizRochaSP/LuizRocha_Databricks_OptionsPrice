"""D-1 de negociação B3, independente das pastas disponíveis em disco."""
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from zipfile import ZipFile

from lxml import etree

CALENDAR_NAME = "BVMF"
CALENDAR_VERSION = "4.13.2"
SAO_PAULO = ZoneInfo("America/Sao_Paulo")
PREFIXES = ("IN", "IR", "PR", "SPRD")


def execution_date(now: datetime | date | None = None) -> date:
    if now is None:
        return datetime.now(SAO_PAULO).date()
    if isinstance(now, datetime):
        if now.tzinfo is None:
            raise ValueError("Informe datetime com fuso horário explícito.")
        return now.astimezone(SAO_PAULO).date()
    if isinstance(now, date):
        return now
    raise TypeError("Data de execução inválida.")


def b3_calendar():
    from importlib.metadata import version
    import exchange_calendars

    if version("exchange-calendars") != CALENDAR_VERSION:
        raise RuntimeError(f"Configure exchange-calendars=={CALENDAR_VERSION} no ambiente.")
    return exchange_calendars.get_calendar(CALENDAR_NAME)


def expected_market_date(now: datetime | date | None = None) -> date:
    """Última sessão estritamente anterior a D (inclusive se D não for sessão)."""
    yesterday = execution_date(now) - timedelta(days=1)
    # Fora da cobertura do calendário, o provedor levanta erro; não há fallback.
    return b3_calendar().date_to_session(yesterday.isoformat(), direction="previous").date()


def require_market_date(value: str, now: datetime | date | None = None) -> date:
    expected = expected_market_date(now)
    if date.fromisoformat(value) != expected:
        raise RuntimeError(f"Data de mercado {value} diverge do D-1 B3 esperado: {expected}.")
    return expected


def validate_snapshot(data_root: Path, market_date: date) -> Path:
    """Exige nomes exatos, CRC válido e datas internas dos XMLs; nunca extrai."""
    folder = Path(data_root) / market_date.isoformat()
    for prefix in PREFIXES:
        path = folder / f"{prefix}{market_date:%y%m%d}.zip"
        if not path.is_file():
            raise FileNotFoundError(f"Fotografia D-1 incompleta: {path}")
        with ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"ZIP corrompido: {path}")
            members = [m for m in archive.infolist() if m.filename.lower().endswith(".xml")]
            if not members:
                raise RuntimeError(f"ZIP sem XML: {path}")
            # Mesma publicação escolhida pelos leitores do notebook 01.
            for member in [max(members, key=lambda item: item.date_time)]:
                observed = set()
                with archive.open(member) as source:
                    if prefix == "PR":
                        # PR inclui outros mercados; o pricer consome somente IBOV.
                        for _, record in etree.iterparse(source, events=("end",),
                                                         tag="{*}PricRpt", resolve_entities=False,
                                                         no_network=True):
                            ticker = record.findtext("{*}SctyId/{*}TckrSymb", default="")
                            if ticker.startswith("IBOV"):
                                observed.add(record.findtext("{*}TradDt/{*}Dt", default=""))
                            record.clear()
                            while record.getprevious() is not None:
                                del record.getparent()[0]
                        if observed != {market_date.isoformat()}:
                            raise RuntimeError(f"Data dos preços IBOV incompatível em {path.name}: {sorted(observed)}")
                        continue
                    for _, element in etree.iterparse(source, events=("end",),
                                                     resolve_entities=False, no_network=True):
                        tag = etree.QName(element).localname
                        parent = element.getparent()
                        expected_parent = "RptDtAndTm" if prefix == "IN" else "TradDt"
                        if tag in {"Dt", "DtTm"} and parent is not None and etree.QName(parent).localname == expected_parent:
                            observed.add((element.text or "")[:10])
                        element.clear()
                        if parent is not None:
                            while element.getprevious() is not None:
                                del parent[0]
                if observed != {market_date.isoformat()}:
                    raise RuntimeError(f"Data XML incompatível em {path.name}/{member.filename}: {sorted(observed)}")
    return folder
