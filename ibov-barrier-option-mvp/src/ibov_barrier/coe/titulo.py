"""Perna de título do COE (etapa A2 do plano): o principal protegido e os cupons fixos, descontados pela curva de
funding do emissor.

**Curva do emissor.** Taxa pré do emissor = taxa DI pré do prazo + spread do emissor, em taxa efetiva anual,
DU/252, capitalização exponencial:

    DF_emissor(T) = (1 + r_pre(T) + s)^(-T),   r_pre(T) = DF_pre(T)^(-1/T) - 1,

com T = DU/252, r_pre a taxa zero efetiva da curva DI no prazo e s o spread em `EstadoMercado.spread_emissor_bp`.
Com spread zero, o título é uma LTN na mesma curva.

**Spread em percentual do CDI.** O funding do emissor costuma ser cotado em % do CDI (CDB, LF). Um título que rende
p% do CDI acumula, dia útil a dia útil, (1 + p · cdi_d), com cdi_d a taxa diária implícita na curva pré:
cdi_d = DF(d-1)/DF(d) - 1. `spread_de_percentual_cdi` e `percentual_cdi_de_spread` convertem entre as duas formas
no prazo do título. A conversão depende do prazo: um mesmo % do CDI vale spreads diferentes em prazos diferentes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ibov_barrier.produtos.base import Composicao, Fluxo, Observacao, saida
from ibov_barrier.produtos.mercado import DU_ANO, Curva, EstadoMercado, Numero


def fator_desconto_emissor(curva: Curva, prazo: float, spread_bp: Numero) -> Numero:
    """DF pela curva pré + spread (taxa efetiva, DU/252). Prazo zero ou negativo vale 1."""
    if prazo <= 0:
        return np.ones_like(np.asarray(spread_bp, dtype=float)) if np.ndim(spread_bp) else 1.0
    df_pre = np.asarray(curva.fator_desconto(prazo), dtype=float)
    taxa_pre = df_pre ** (-1.0 / prazo) - 1.0
    fator = (1.0 + taxa_pre + np.asarray(spread_bp, dtype=float) * 1e-4) ** (-prazo)
    return fator if np.ndim(fator) else float(fator)


def _du(prazo: float) -> int:
    du = int(round(prazo * DU_ANO))
    if du < 1:
        raise ValueError("prazo deve ter pelo menos 1 dia útil")
    return du


def fator_desconto_percentual_cdi(curva: Curva, prazo: float, percentual_cdi: float) -> float:
    """DF de um título que rende `percentual_cdi`% do CDI até o prazo (arredondado ao dia útil).

    O CDI de cada dia é o forward diário da curva pré: cdi_d = DF(d-1)/DF(d) - 1.
    """
    if not np.isfinite(percentual_cdi) or percentual_cdi < 0:
        raise ValueError("percentual_cdi deve ser não negativo")
    du = _du(prazo)
    dfs = np.array([float(np.asarray(curva.fator_desconto(d / DU_ANO))) for d in range(du + 1)])
    cdi_diario = dfs[:-1] / dfs[1:] - 1.0
    return float(np.exp(-np.sum(np.log1p(percentual_cdi / 100.0 * cdi_diario))))


def spread_de_percentual_cdi(curva: Curva, percentual_cdi: float, prazo: float) -> float:
    """Spread sobre a curva pré (bp, taxa efetiva) equivalente a `percentual_cdi`% do CDI no prazo."""
    du = _du(prazo)
    t = du / DU_ANO
    df_emissor = fator_desconto_percentual_cdi(curva, t, percentual_cdi)
    taxa_emissor = df_emissor ** (-1.0 / t) - 1.0
    taxa_pre = float(np.asarray(curva.fator_desconto(t))) ** (-1.0 / t) - 1.0
    return (taxa_emissor - taxa_pre) * 1e4


def percentual_cdi_de_spread(curva: Curva, spread_bp: float, prazo: float) -> float:
    """Percentual do CDI equivalente a um spread sobre a curva pré (inversa de `spread_de_percentual_cdi`)."""
    alvo = lambda p: spread_de_percentual_cdi(curva, p, prazo) - spread_bp  # noqa: E731
    baixo, alto = 0.0, 500.0
    if alvo(baixo) > 0 or alvo(alto) < 0:
        raise ValueError(f"spread de {spread_bp} bp fora da faixa de 0% a 500% do CDI")
    return float(brentq(alvo, baixo, alto, xtol=1e-12))


@dataclass(frozen=True)
class Titulo(Composicao):
    """Título do emissor: `principal` no `vencimento` e `cupons` fixos ((tempo, valor), ...), em `moeda`.

    Tempos em anos DU/252 a partir da data de referência. Desconto pela curva da moeda mais o spread do emissor do
    estado. Fluxos com tempo anterior a `estado.tempo` já foram pagos e saem do valor.
    """

    vencimento: float
    principal: float = 1.0
    cupons: tuple[tuple[float, float], ...] = ()
    moeda: str = "BRL"

    # Sinaliza às gregas que o valor depende do spread do emissor (CS01).
    depende_do_spread = True

    def __post_init__(self):
        if not np.isfinite(self.vencimento) or self.vencimento <= 0:
            raise ValueError("vencimento deve ser positivo")
        if not np.isfinite(self.principal) or self.principal < 0:
            raise ValueError("principal deve ser não negativo")
        cupons = tuple(sorted((float(t), float(v)) for t, v in self.cupons))
        for t, v in cupons:
            if not 0 < t <= self.vencimento or not np.isfinite(v):
                raise ValueError("cupons devem ter tempo em (0, vencimento] e valor finito")
        object.__setattr__(self, "cupons", cupons)

    def _fluxos_conhecidos(self) -> list[tuple[float, float]]:
        fluxos = list(self.cupons)
        if self.principal:
            fluxos.append((self.vencimento, self.principal))
        return fluxos

    def valor(self, estado: EstadoMercado) -> Numero:
        curva = estado.curva(self.moeda)
        total = 0.0
        for t, v in self._fluxos_conhecidos():
            prazo = t - estado.tempo
            if prazo < 0:
                continue
            total = total + v * np.asarray(fator_desconto_emissor(curva, prazo, estado.spread_emissor_bp))
        return saida(total, estado)

    def payoff(self, finais: Mapping[str, np.ndarray]) -> np.ndarray:
        """Pagamento no vencimento (principal e cupom da data). Cupons intermediários não cabem num payoff final."""
        if any(t < self.vencimento for t, _ in self.cupons):
            raise TypeError("título com cupons intermediários não tem payoff europeu único")
        no_vencimento = self.principal + sum(v for t, v in self.cupons if t == self.vencimento)
        forma = np.shape(next(iter(finais.values()))) if finais else ()
        return np.full(forma, no_vencimento, dtype=float)

    def subjacentes(self) -> tuple[str, ...]:
        return ()

    def fluxos(self) -> tuple[Fluxo, ...]:
        rotulos = [(t, v, "cupom fixo") for t, v in self.cupons]
        if self.principal:
            rotulos.append((self.vencimento, self.principal, "principal"))
        return tuple(Fluxo(t, v, d, self.moeda) for t, v, d in sorted(rotulos, key=lambda x: x[0]))

    def observacoes(self) -> tuple[Observacao, ...]:
        return ()

    def taxa_emissor(self, estado: EstadoMercado) -> float:
        """Taxa efetiva anual (DU/252) do emissor no vencimento: taxa DI pré + spread."""
        df = fator_desconto_emissor(estado.curva(self.moeda), self.vencimento - estado.tempo,
                                    estado.spread_emissor_bp)
        return float(np.asarray(df) ** (-1.0 / (self.vencimento - estado.tempo)) - 1.0)
