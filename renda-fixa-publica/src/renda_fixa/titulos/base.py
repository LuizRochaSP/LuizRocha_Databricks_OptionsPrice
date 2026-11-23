"""Interface comum dos títulos públicos federais.

Unidades:
- taxas de entrada e saída em % a.a. (como cotadas pela ANBIMA: 13.9123 = 13,9123% a.a.);
- fluxos em % do valor-base (cotação): 100 = valor-base. O valor-base é R$ 1.000 para
  LTN e NTN-F e o VNA para NTN-B, NTN-B Principal e LFT;
- prazos em dias úteis (DU), base 252, calendário ANBIMA.

Regras de PU (validadas contra o arquivo de mercado secundário da ANBIMA):
- LTN e NTN-F: PU = soma dos fluxos descontados em R$, truncado na 6ª casa;
- NTN-B, NTN-B Principal e LFT: cotação truncada na 4ª casa e PU = VNA x cotação / 100,
  truncado na 6ª casa.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from ..arredondamento import truncar
from ..calendario import DU_ANO, dias_uteis, soma_meses

if TYPE_CHECKING:  # pragma: no cover
    from ..curvas.curva import CurvaZero

BP = 0.01  # 1 basis point em % a.a.


@dataclass(frozen=True)
class Fluxo:
    data: date  # data nominal do evento (o pagamento é no próximo dia útil)
    du: int  # dias úteis da referência até o pagamento
    valor: float  # % do valor-base


@dataclass(frozen=True)
class Analiticos:
    pu: float  # R$ por título
    taxa: float  # % a.a.
    du: int  # prazo até o vencimento
    duration_macaulay: float  # anos (DU/252)
    duration_modificada: float  # anos
    dv01: float  # R$ por título para +1 bp (positivo: perda quando a taxa sobe)
    convexidade: float  # anos^2


class TituloPublico(ABC):
    """Título público com fluxos conhecidos em % do valor-base."""

    nome: str = ""
    indexador: str = ""  # "PRE", "IPCA" ou "SELIC"
    usa_vna: bool = False
    valor_base_fixo: float | None = 1000.0

    def __init__(self, vencimento: date):
        self.vencimento = vencimento

    def __repr__(self) -> str:
        return f"{self.nome}({self.vencimento.isoformat()})"

    # ---------- fluxos ----------
    def datas_cupom(self, ref: date) -> list[date]:
        """Datas de evento após ``ref`` (inclui o vencimento). Padrão: só o vencimento."""
        return [self.vencimento] if self.vencimento > ref else []

    def cupom(self) -> float:
        """Cupom por período em % do valor-base (0 para títulos sem cupom)."""
        return 0.0

    def fluxos(self, ref: date) -> list[Fluxo]:
        datas = self.datas_cupom(ref)
        c = self.cupom()
        out = []
        for i, d in enumerate(datas):
            valor = c + (100.0 if i == len(datas) - 1 else 0.0)
            out.append(Fluxo(d, dias_uteis(ref, d), valor))
        return out

    def _base(self, vna: float | None) -> float:
        if self.usa_vna:
            if vna is None:
                raise ValueError(f"{self.nome} precisa do VNA.")
            return float(vna)
        return float(self.valor_base_fixo)

    # ---------- preço a partir da taxa ----------
    def cotacao_bruta(self, taxa: float, ref: date) -> float:
        """Cotação (% do valor-base) sem truncamento."""
        t = taxa / 100.0
        return sum(f.valor / (1.0 + t) ** (f.du / DU_ANO) for f in self.fluxos(ref))

    def pu_bruto(self, taxa: float, ref: date, vna: float | None = None) -> float:
        """PU sem truncamentos (usado no solver e nas sensibilidades)."""
        return self._base(vna) * self.cotacao_bruta(taxa, ref) / 100.0

    @abstractmethod
    def pu(self, taxa: float, ref: date, vna: float | None = None) -> float:
        """PU pela convenção oficial (truncamentos do Tesouro/ANBIMA)."""

    # ---------- taxa a partir do preço ----------
    def taxa(self, pu: float, ref: date, vna: float | None = None, casas: int | None = 4) -> float:
        """Taxa (% a.a.) que reproduz o PU, por bisseção sobre o PU bruto."""
        if not self.fluxos(ref):
            raise ValueError(f"{self} vencido em {ref}.")
        lo, hi = -50.0, 500.0
        f_lo = self.pu_bruto(lo, ref, vna) - pu
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = self.pu_bruto(mid, ref, vna) - pu
            if (f_mid > 0) == (f_lo > 0):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        x = 0.5 * (lo + hi)
        if casas is None:
            return x
        # Com truncamento, várias taxas de ``casas`` decimais levam ao mesmo PU oficial
        # (títulos curtos). Devolve a mais próxima de x entre as que reproduzem o PU.
        passo = 10.0 ** -casas
        alvo = round(pu, 6)
        candidatos = [round(x, casas) + k * passo for k in range(-60, 61)]
        iguais = [c for c in candidatos if self.pu(round(c, casas), ref, vna) == alvo]
        if iguais:
            return round(min(iguais, key=lambda c: abs(c - x)), casas)
        return round(x, casas)

    # ---------- preço a partir de uma curva ----------
    def pu_curva(self, curva: "CurvaZero", ref: date, vna: float | None = None) -> float:
        """PU (sem truncamento) descontando cada fluxo pela curva zero."""
        cot = sum(f.valor * curva.fator_desconto(f.du) for f in self.fluxos(ref))
        return self._base(vna) * cot / 100.0

    # ---------- sensibilidades ----------
    def analiticos(self, taxa: float, ref: date, vna: float | None = None) -> Analiticos:
        t = taxa / 100.0
        fl = self.fluxos(ref)
        base = self._base(vna)
        vps = [f.valor / (1.0 + t) ** (f.du / DU_ANO) for f in fl]
        cot = sum(vps)
        mac = sum(v * f.du / DU_ANO for v, f in zip(vps, fl)) / cot
        conv = sum(v * (f.du / DU_ANO) * (f.du / DU_ANO + 1.0) for v, f in zip(vps, fl)) / (cot * (1.0 + t) ** 2)
        dv01 = self.pu_bruto(taxa, ref, vna) - self.pu_bruto(taxa + BP, ref, vna)
        return Analiticos(
            pu=self.pu(taxa, ref, vna),
            taxa=taxa,
            du=fl[-1].du,
            duration_macaulay=mac,
            duration_modificada=mac / (1.0 + t),
            dv01=dv01,
            convexidade=conv,
        )


class TituloComCupomSemestral(TituloPublico):
    """Cupons a cada 6 meses, contados para trás a partir do vencimento."""

    def datas_cupom(self, ref: date) -> list[date]:
        datas, k = [], 0
        while True:
            d = soma_meses(self.vencimento, -6 * k)
            if d <= ref:
                break
            datas.append(d)
            k += 1
        return datas[::-1]


def pu_por_cotacao(vna: float, cotacao_bruta: float) -> float:
    """Regra dos títulos com VNA: cotação truncada na 4ª casa, PU truncado na 6ª."""
    return truncar(vna * truncar(cotacao_bruta, 4) / 100.0, 6)
