"""Calendário de dias úteis ANBIMA (feriados nacionais) e contagem de prazos.

A renda fixa pública usa o calendário de feriados nacionais da ANBIMA, que não é
igual ao calendário de pregões da B3 (a B3 fecha, por exemplo, em 24/12 e 31/12).
O calendário é calculado por regra, sem dependência externa:

- feriados fixos: 1/1, 21/4, 1/5, 7/9, 12/10, 2/11, 15/11, 25/12;
- 20/11 (Consciência Negra), feriado nacional a partir de 2024 (Lei 14.759/2023);
- móveis: Carnaval (segunda e terça), Sexta-feira da Paixão e Corpus Christi.

Convenção de prazo: ``dias_uteis(d0, d1)`` conta os dias úteis em [d0, d1),
ou seja, de d0 (inclusive) até d1 (exclusive). Se d1 cair num feriado, o resultado
é o mesmo que contar até o próximo dia útil, que é a regra de pagamento dos títulos.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np

ANO_MIN = 1990
ANO_MAX = 2099
DU_ANO = 252


def pascoa(ano: int) -> date:
    """Domingo de Páscoa (algoritmo de Meeus/Jones/Butcher, calendário gregoriano)."""
    a = ano % 19
    b, c = divmod(ano, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes, dia = divmod(h + l - 7 * m + 114, 31)
    return date(ano, mes, dia + 1)


def feriados_nacionais(ano: int) -> list[date]:
    """Feriados nacionais ANBIMA de um ano, em ordem."""
    p = pascoa(ano)
    datas = [
        date(ano, 1, 1),
        p - timedelta(days=48),  # Carnaval (segunda)
        p - timedelta(days=47),  # Carnaval (terça)
        p - timedelta(days=2),  # Sexta-feira da Paixão
        date(ano, 4, 21),
        date(ano, 5, 1),
        p + timedelta(days=60),  # Corpus Christi
        date(ano, 9, 7),
        date(ano, 10, 12),
        date(ano, 11, 2),
        date(ano, 11, 15),
        date(ano, 12, 25),
    ]
    if ano >= 2024:
        datas.append(date(ano, 11, 20))
    return sorted(set(datas))


@lru_cache(maxsize=1)
def _feriados_np() -> np.ndarray:
    todas = [d for ano in range(ANO_MIN, ANO_MAX + 1) for d in feriados_nacionais(ano)]
    return np.array(todas, dtype="datetime64[D]")


@lru_cache(maxsize=1)
def _calendario() -> np.busdaycalendar:
    return np.busdaycalendar(weekmask="1111100", holidays=_feriados_np())


def _checa(d: date) -> None:
    if not (ANO_MIN <= d.year <= ANO_MAX):
        raise ValueError(f"Data {d} fora do calendário ({ANO_MIN}-{ANO_MAX}).")


def eh_dia_util(d: date) -> bool:
    _checa(d)
    return bool(np.is_busday(np.datetime64(d, "D"), busdaycal=_calendario()))


def dias_uteis(d0: date, d1: date) -> int:
    """Dias úteis em [d0, d1). Negativo se d1 < d0."""
    _checa(d0)
    _checa(d1)
    return int(np.busday_count(np.datetime64(d0, "D"), np.datetime64(d1, "D"), busdaycal=_calendario()))


def dias_corridos(d0: date, d1: date) -> int:
    return (d1 - d0).days


def soma_dias_uteis(d: date, n: int) -> date:
    """Data n dias úteis depois de d (antes, se n < 0). d é ajustada ao próximo dia útil."""
    _checa(d)
    r = np.busday_offset(np.datetime64(d, "D"), n, roll="forward", busdaycal=_calendario())
    return r.astype(object)


def proximo_dia_util(d: date) -> date:
    """d, se for dia útil; senão o próximo dia útil."""
    return soma_dias_uteis(d, 0)


def dia_util_anterior(d: date) -> date:
    """Último dia útil estritamente antes de d."""
    _checa(d)
    r = np.busday_offset(np.datetime64(d, "D") - 1, 0, roll="backward", busdaycal=_calendario())
    return r.astype(object)


def soma_meses(d: date, meses: int) -> date:
    """Soma meses mantendo o dia (limitado ao fim do mês)."""
    total = d.year * 12 + (d.month - 1) + meses
    ano, mes = divmod(total, 12)
    mes += 1
    ultimo = (date(ano + (mes == 12), mes % 12 + 1, 1) - timedelta(days=1)).day
    return date(ano, mes, min(d.day, ultimo))
