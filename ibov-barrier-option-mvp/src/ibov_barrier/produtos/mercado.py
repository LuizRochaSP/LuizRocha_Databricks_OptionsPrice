"""Estado de mercado da interface de produtos (etapa A1 do plano de COE).

O `EstadoMercado` reúne tudo de que um produto precisa para se avaliar:

- `ativos`: por subjacente, spot, vol implícita (número ou função do strike e do prazo), dividendos contínuos,
  moeda e um deslocamento aditivo de vol;
- `curvas`: por moeda, a taxa contínua r(T) usada no drift e no desconto (`CurvaPlana` ou `CurvaDI`);
- `cambios` e `correlacoes`: câmbio, vol do câmbio e correlação ativo x câmbio, para o ajuste Quanto;
- `spread_emissor_bp`: spread de funding do emissor, usado pela perna de título (etapa A2);
- `tempo`: tempo decorrido, em anos, desde a data de referência dos produtos (theta e ciclo de vida);
- `data`: a data de mercado, só para registro.

**Cenários vetoriais.** O spot, o deslocamento de vol, o câmbio, o spread e a taxa da `CurvaPlana` aceitam
vetores 1-D de mesmo tamanho n (ou tamanho 1). Assim um produto avalia n cenários de uma vez: os de fórmula
fechada pelo numpy, os de Monte Carlo um cenário por vez (`cenarios()`), sempre com a mesma semente.

**Convenções.**
- Prazos em anos, na convenção do ativo: DU/252 no IBOV, ACT/365 no SPX (as mesmas dos notebooks 01 e 07).
- Taxas contínuas. A `CurvaDI` converte a curva DI pré (taxa efetiva, DU/252) em r contínua pela mesma regra do
  `MarketCurves.r` do notebook 01: r = ln(1 + taxa efetiva).
- Vol "sticky strike": uma superfície passada como função é lida no strike absoluto; o choque de spot não a move.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Protocol, Union, runtime_checkable

import numpy as np

Numero = Union[float, np.ndarray]

DU_ANO = 252


def _tamanho(x) -> int:
    dim = np.ndim(x)
    if dim == 0:
        return 1
    if dim != 1:
        raise ValueError("cenários devem ser vetores 1-D")
    tamanho = int(np.size(x))
    if tamanho == 0:
        raise ValueError("vetor de cenários vazio")
    return tamanho


def _vetorial(x) -> bool:
    return np.ndim(x) >= 1


def _indice(x, i: int):
    """Elemento i de um campo vetorial (vetor de tamanho 1 vale para todos os cenários)."""
    if not _vetorial(x):
        return x
    a = np.asarray(x, dtype=float)
    return float(a[0] if a.size == 1 else a[i])


def _como_numero(x) -> Numero:
    """Escalar vira float; vetor vira ndarray de float (cópia)."""
    if _vetorial(x):
        return np.array(x, dtype=float)
    return float(x)


# ----------------------------------------------------------------------------------------------- curvas

@runtime_checkable
class Curva(Protocol):
    """Curva de juros de uma moeda, vista como taxa contínua ao ano."""

    def taxa_em(self, prazo: float) -> Numero: ...

    def fator_desconto(self, prazo: float) -> Numero: ...

    def deslocar(self, bp: float) -> "Curva": ...


@dataclass(frozen=True)
class CurvaPlana:
    """Taxa contínua constante. `deslocar(bp)` soma bp à taxa contínua."""

    taxa: Numero

    def __post_init__(self):
        object.__setattr__(self, "taxa", _como_numero(self.taxa))
        if not np.all(np.isfinite(self.taxa)):
            raise ValueError("taxa deve ser finita")

    def taxa_em(self, prazo: float) -> Numero:
        return self.taxa

    def fator_desconto(self, prazo: float) -> Numero:
        return np.exp(-np.asarray(self.taxa) * prazo) if _vetorial(self.taxa) else float(np.exp(-self.taxa * prazo))

    def deslocar(self, bp: float) -> "CurvaPlana":
        return CurvaPlana(self.taxa + bp * 1e-4)

    def _tamanho(self) -> int:
        return _tamanho(self.taxa)

    def _cenario(self, i: int) -> "CurvaPlana":
        return CurvaPlana(_indice(self.taxa, i))


@dataclass(frozen=True)
class CurvaDI:
    """Curva DI pré (`di_curve.DiscountCurve`: taxas efetivas, DU/252) lida como taxa contínua.

    O prazo em anos vira DU por `prazo * 252`. `deslocar(bp)` e `deslocar_vertice(du, bp)` deslocam a taxa
    EFETIVA dos vértices (`bump_parallel` e `bump_key`), como no DV01 da curva.

    O objeto `curva` só precisa de `rate(du)`, `log_df(du)`, `bump_parallel(bp)` e `bump_key(du, bp)`; o módulo
    não importa `di_curve` (que exige `exchange_calendars`).
    """

    curva: object

    def taxa_em(self, prazo: float) -> float:
        du = max(float(prazo), 1.0 / DU_ANO) * DU_ANO
        return float(np.log1p(self.curva.rate(du)))

    def fator_desconto(self, prazo: float) -> float:
        if prazo <= 0:
            return 1.0
        return float(np.exp(self.curva.log_df(prazo * DU_ANO)))

    def deslocar(self, bp: float) -> "CurvaDI":
        return CurvaDI(self.curva.bump_parallel(bp))

    def deslocar_vertice(self, du: float, bp: float) -> "CurvaDI":
        return CurvaDI(self.curva.bump_key(du, bp))

    @property
    def vertices(self) -> tuple[float, ...]:
        """Vértices da curva, em DU (`tenors_bd` da `DiscountCurve`)."""
        return tuple(float(v) for v in self.curva.tenors_bd)

    def _tamanho(self) -> int:
        return 1

    def _cenario(self, i: int) -> "CurvaDI":
        return self


def _tamanho_curva(curva) -> int:
    return curva._tamanho() if hasattr(curva, "_tamanho") else 1


def _cenario_curva(curva, i: int):
    return curva._cenario(i) if hasattr(curva, "_cenario") else curva


# ------------------------------------------------------------------------------------- ativos e câmbio

@dataclass(frozen=True)
class MercadoAtivo:
    """Mercado de um subjacente.

    - `vol`: número (vol plana) ou função `vol(strike, prazo)` que devolve a vol implícita.
    - `dividendos`: número ou função `q(prazo)`, contínuo ao ano.
    - `choque_vol`: deslocamento aditivo da vol (0,01 = 1 ponto percentual), escalar ou vetor de cenários.
    """

    spot: Numero
    vol: float | Callable[[float, float], float]
    dividendos: float | Callable[[float], float] = 0.0
    moeda: str = "BRL"
    choque_vol: Numero = 0.0

    def __post_init__(self):
        object.__setattr__(self, "spot", _como_numero(self.spot))
        object.__setattr__(self, "choque_vol", _como_numero(self.choque_vol))
        spot = np.asarray(self.spot)
        if not np.all(np.isfinite(spot)) or np.any(spot <= 0):
            raise ValueError("spot deve ser positivo e finito")
        if not callable(self.vol) and (not np.isfinite(self.vol) or self.vol < 0):
            raise ValueError("vol deve ser não negativa e finita")

    def vol_em(self, strike: float, prazo: float) -> Numero:
        base = self.vol(strike, prazo) if callable(self.vol) else self.vol
        vol = base + self.choque_vol
        if np.any(np.asarray(vol) < 0):
            raise ValueError("vol chocada ficou negativa")
        return vol

    def dividendos_em(self, prazo: float) -> float:
        return float(self.dividendos(prazo)) if callable(self.dividendos) else float(self.dividendos)

    def _tamanho(self) -> int:
        return max(_tamanho(self.spot), _tamanho(self.choque_vol))

    def _cenario(self, i: int) -> "MercadoAtivo":
        return replace(self, spot=_indice(self.spot, i), choque_vol=_indice(self.choque_vol, i))


@dataclass(frozen=True)
class MercadoCambio:
    """Câmbio `base`/`cotacao` (USDBRL: BRL por 1 USD) e a sua vol."""

    spot: Numero
    vol: float
    base: str = "USD"
    cotacao: str = "BRL"

    def __post_init__(self):
        object.__setattr__(self, "spot", _como_numero(self.spot))
        if np.any(np.asarray(self.spot) <= 0) or not np.isfinite(self.vol) or self.vol < 0:
            raise ValueError("câmbio deve ser positivo e a vol não negativa")

    def _tamanho(self) -> int:
        return _tamanho(self.spot)

    def _cenario(self, i: int) -> "MercadoCambio":
        return replace(self, spot=_indice(self.spot, i))


# ---------------------------------------------------------------------------------------------- estado

@dataclass(frozen=True)
class EstadoMercado:
    """Estado de mercado para `Produto.valor`. Imutável: os métodos `com_*` devolvem um estado novo."""

    ativos: Mapping[str, MercadoAtivo]
    curvas: Mapping[str, Curva]
    cambios: Mapping[str, MercadoCambio] = field(default_factory=dict)
    correlacoes: Mapping[tuple[str, str], float] = field(default_factory=dict)
    spread_emissor_bp: Numero = 0.0
    tempo: float = 0.0
    data: date | None = None

    def __post_init__(self):
        object.__setattr__(self, "ativos", dict(self.ativos))
        object.__setattr__(self, "curvas", dict(self.curvas))
        object.__setattr__(self, "cambios", dict(self.cambios))
        object.__setattr__(self, "correlacoes", dict(self.correlacoes))
        object.__setattr__(self, "spread_emissor_bp", _como_numero(self.spread_emissor_bp))
        for par, rho in self.correlacoes.items():
            if not -1.0 <= rho <= 1.0:
                raise ValueError(f"correlação {par} fora de [-1, 1]")
        self.n_cenarios  # valida os tamanhos

    @classmethod
    def simples(cls, spot: Numero, taxa: Numero, vol: Numero, dividendos: float = 0.0, *, ativo: str = "IBOV",
                moeda: str = "BRL", tempo: float = 0.0) -> "EstadoMercado":
        """Um ativo com taxa, dividendos e vol planos (o `MarketData` do motor)."""
        if _vetorial(vol):
            mercado = MercadoAtivo(spot=spot, vol=0.0, dividendos=dividendos, moeda=moeda, choque_vol=vol)
        else:
            mercado = MercadoAtivo(spot=spot, vol=float(vol), dividendos=dividendos, moeda=moeda)
        return cls(ativos={ativo: mercado}, curvas={moeda: CurvaPlana(taxa)}, tempo=tempo)

    # -------------------------------------------------------------------------------- cenários

    def _tamanhos(self) -> list[int]:
        tamanhos = [a._tamanho() for a in self.ativos.values()]
        tamanhos += [_tamanho_curva(c) for c in self.curvas.values()]
        tamanhos += [c._tamanho() for c in self.cambios.values()]
        tamanhos.append(_tamanho(self.spread_emissor_bp))
        return tamanhos

    @property
    def n_cenarios(self) -> int:
        tamanhos = set(self._tamanhos()) - {1}
        if len(tamanhos) > 1:
            raise ValueError(f"vetores de cenários com tamanhos diferentes: {sorted(tamanhos)}")
        return tamanhos.pop() if tamanhos else 1

    @property
    def vetorial(self) -> bool:
        """True se algum campo é vetor (mesmo de tamanho 1): o valor sai como ndarray."""
        campos = [a.spot for a in self.ativos.values()] + [a.choque_vol for a in self.ativos.values()]
        campos += [c.spot for c in self.cambios.values()] + [self.spread_emissor_bp]
        campos += [c.taxa for c in self.curvas.values() if isinstance(c, CurvaPlana)]
        return any(_vetorial(x) for x in campos)

    def cenario(self, i: int) -> "EstadoMercado":
        """O cenário i como estado escalar."""
        n = self.n_cenarios
        if not 0 <= i < n:
            raise IndexError(f"cenário {i} fora de 0..{n - 1}")
        return replace(
            self,
            ativos={k: a._cenario(i) for k, a in self.ativos.items()},
            curvas={k: _cenario_curva(c, i) for k, c in self.curvas.items()},
            cambios={k: c._cenario(i) for k, c in self.cambios.items()},
            spread_emissor_bp=_indice(self.spread_emissor_bp, i),
        )

    def cenarios(self) -> Iterator["EstadoMercado"]:
        for i in range(self.n_cenarios):
            yield self.cenario(i)

    # -------------------------------------------------------------------------------- leitura

    def ativo(self, nome: str) -> MercadoAtivo:
        try:
            return self.ativos[nome]
        except KeyError:
            raise KeyError(f"ativo '{nome}' não está no estado de mercado") from None

    def curva(self, moeda: str) -> Curva:
        try:
            return self.curvas[moeda]
        except KeyError:
            raise KeyError(f"curva de juros em '{moeda}' não está no estado de mercado") from None

    def cambio(self, par: str) -> MercadoCambio:
        try:
            return self.cambios[par]
        except KeyError:
            raise KeyError(f"câmbio '{par}' não está no estado de mercado") from None

    def correlacao(self, a: str, b: str) -> float:
        for chave in ((a, b), (b, a)):
            if chave in self.correlacoes:
                return float(self.correlacoes[chave])
        raise KeyError(f"correlação entre '{a}' e '{b}' não está no estado de mercado")

    # -------------------------------------------------------------------------------- choques

    def com_spot(self, ativo: str, spot: Numero) -> "EstadoMercado":
        return replace(self, ativos={**self.ativos, ativo: replace(self.ativo(ativo), spot=spot)})

    def com_choque_vol(self, ativo: str, choque: Numero) -> "EstadoMercado":
        """Troca o deslocamento de vol do ativo (absoluto, não acumula)."""
        return replace(self, ativos={**self.ativos, ativo: replace(self.ativo(ativo), choque_vol=choque)})

    def com_curva(self, moeda: str, curva: Curva) -> "EstadoMercado":
        return replace(self, curvas={**self.curvas, moeda: curva})

    def com_cambio(self, par: str, spot: Numero) -> "EstadoMercado":
        return replace(self, cambios={**self.cambios, par: replace(self.cambio(par), spot=spot)})

    def com_spread(self, spread_bp: Numero) -> "EstadoMercado":
        return replace(self, spread_emissor_bp=spread_bp)

    def com_tempo(self, tempo: float) -> "EstadoMercado":
        return replace(self, tempo=float(tempo))

    def avancar(self, dt: float) -> "EstadoMercado":
        return self.com_tempo(self.tempo + dt)
