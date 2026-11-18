"""D-1 de negociação por mercado (B3, NYSE, ...), independente das pastas disponíveis em disco.

O D-1 é sempre definido no fuso horário da bolsa: a data de execução D é a data local da bolsa
no instante da execução, e o D-1 é a última sessão estritamente anterior a D. Assim, um job que
roda às 00:30 de São Paulo (23:30 do dia anterior em Nova York) enxerga D-1 diferentes para B3 e
NYSE, como deve ser.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from zipfile import ZipFile

CALENDAR_NAME = "BVMF"
CALENDAR_VERSION = "4.13.2"
SAO_PAULO = ZoneInfo("America/Sao_Paulo")
NEW_YORK = ZoneInfo("America/New_York")
PREFIXES = ("IN", "IR", "PR", "SPRD")


@dataclass(frozen=True)
class ExchangeCalendarSpec:
    """Calendário de uma bolsa: código no exchange_calendars e fuso horário local."""
    code: str
    calendar_name: str
    timezone: ZoneInfo


EXCHANGES = {
    "B3": ExchangeCalendarSpec("B3", "BVMF", SAO_PAULO),
    "NYSE": ExchangeCalendarSpec("NYSE", "XNYS", NEW_YORK),
}


def exchange_spec(market: str) -> ExchangeCalendarSpec:
    try:
        return EXCHANGES[market]
    except KeyError:
        raise ValueError(f"Mercado sem calendário configurado: {market!r}. Opções: {sorted(EXCHANGES)}") from None


def execution_date(now: datetime | date | None = None, tz: ZoneInfo = SAO_PAULO) -> date:
    """Data D no fuso `tz`. Datas puras (date) já são consideradas locais da bolsa."""
    if now is None:
        return datetime.now(tz).date()
    if isinstance(now, datetime):
        if now.tzinfo is None:
            raise ValueError("Informe datetime com fuso horário explícito.")
        return now.astimezone(tz).date()
    if isinstance(now, date):
        return now
    raise TypeError("Data de execução inválida.")


def exchange_calendar(market: str = "B3"):
    from importlib.metadata import version
    import exchange_calendars

    if version("exchange-calendars") != CALENDAR_VERSION:
        raise RuntimeError(f"Configure exchange-calendars=={CALENDAR_VERSION} no ambiente.")
    return exchange_calendars.get_calendar(exchange_spec(market).calendar_name)


def b3_calendar():
    return exchange_calendar("B3")


def expected_market_date(now: datetime | date | None = None, market: str = "B3", calendar=None) -> date:
    """Última sessão estritamente anterior a D (inclusive se D não for sessão), no fuso da bolsa.

    `calendar` permite injetar um calendário (objeto com `date_to_session`) em testes.
    """
    spec = exchange_spec(market)
    yesterday = execution_date(now, spec.timezone) - timedelta(days=1)
    cal = calendar if calendar is not None else exchange_calendar(market)
    # Fora da cobertura do calendário, o provedor levanta erro; não há fallback.
    return cal.date_to_session(yesterday.isoformat(), direction="previous").date()


def require_market_date(value: str, now: datetime | date | None = None, market: str = "B3",
                        calendar=None) -> date:
    expected = expected_market_date(now, market, calendar)
    if date.fromisoformat(value) != expected:
        raise RuntimeError(f"Data de mercado {value} diverge do D-1 {market} esperado: {expected}.")
    return expected


def validate_snapshot(data_root: Path, market_date: date) -> Path:
    """Exige nomes exatos, CRC válido e datas internas dos XMLs; nunca extrai."""
    from lxml import etree  # import tardio: só a validação da fotografia B3 precisa de lxml

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
