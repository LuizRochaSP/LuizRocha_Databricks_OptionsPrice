"""O COE como composição (etapa A3 do plano de COE): termsheet, COE e decomposição do preço de emissão.

    COE = nominal × (proteção × título + pacote de opções)

**Termsheet.** `TermsheetBase` reúne os campos comuns a toda estrutura: subjacente, nível inicial, prazo, nominal,
proteção do capital, moeda e datas. Cada estrutura é uma subclasse que diz qual é o seu pacote de opções
(`opcoes`) e qual é o payoff dele no vencimento (`payoff_opcoes`). Esta etapa traz a primeira, `Participacao`:
capital protegido com participação na alta (call ou call spread) ou na baixa (put ou put spread), com ou sem cap.
É o primeiro caso real (decisão 1 da seção 8). Tubarão, cupom digital, intervalo, autocall, média e worst-of entram
na fase C como novas subclasses, sobre a mesma classe `COE`.

**Percentuais em fração.** 100% = 1.0. A validação recusa valores como `protecao=100`, que seriam 10.000%.

**Prazo.** `vencimento` em anos DU/252 desde a emissão, a convenção da curva DI e do IBOV.
`Participacao.por_datas(...)` conta os dias úteis no calendário da B3.

**Payoff da `Participacao` no vencimento**, por unidade de nominal, com R = S_T / S_0 e k o strike em fração do
nível inicial (1 = no dinheiro):

    alta:   proteção + participação × min(max(R − k, 0), (1 + cap) − k)
    baixa:  proteção + participação × min(max(k − R, 0), k − (1 − cap))

O cap é a alta (ou a queda) máxima do subjacente que conta para o investidor. Sem cap, o mínimo some.

**Desconto.** O título é descontado na curva do emissor (DI pré + spread, etapa A2). As opções valem o preço de
mercado do hedge, na curva livre de risco. É a visão do estruturador: o título é funding do emissor e as opções são
compradas no mercado.

**Decomposição do preço de emissão** (seção 2 do plano):

    preço de emissão = título + opções + custos + reservas + margem

Os custos e as reservas são entradas manuais até as decisões 2 e 3 da seção 8 e a etapa D4. A margem é o que sobra.
O solver da etapa A4 inverte a conta: fixa a margem e resolve a participação ou o cap.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np

from ibov_barrier.coe.titulo import Titulo
from ibov_barrier.produtos.adaptadores import CallSpread, Vanilla
from ibov_barrier.produtos.base import Carteira, Composicao, Fluxo, Observacao
from ibov_barrier.produtos.mercado import DU_ANO, EstadoMercado, Numero

_DIRECOES = {"alta", "baixa"}

# Limites de sanidade dos percentuais (em fração). Pegam o erro de digitar 100 no lugar de 1.0.
_PROTECAO_MAX = 1.5
_PARTICIPACAO_MAX = 10.0
_CAP_MAX = 10.0
_STRIKE_MAX = 3.0


def _pct(x: float) -> str:
    """0.8 -> '80%', 0.075 -> '7,5%' (vírgula decimal, como no resto do projeto)."""
    texto = f"{100.0 * x:.2f}".rstrip("0").rstrip(".")
    return texto.replace(".", ",") + "%"


def _finito_positivo(nome: str, valor: float) -> None:
    if not np.isfinite(valor) or valor <= 0:
        raise ValueError(f"{nome} deve ser positivo e finito")


def _fracao(nome: str, valor: float, minimo: float, maximo: float) -> None:
    if not np.isfinite(valor) or not minimo <= valor <= maximo:
        raise ValueError(f"{nome} = {valor} fora de [{minimo:g}, {maximo:g}] (percentuais em fração: 100% = 1.0)")


# ----------------------------------------------------------------------------------------------- termsheet

@dataclass(frozen=True, kw_only=True)
class TermsheetBase:
    """Campos comuns a toda estrutura de COE.

    - `ativo`: subjacente, com o nome usado no `EstadoMercado` ("IBOV").
    - `nivel_inicial`: nível de fixing do subjacente na emissão (S_0), em pontos.
    - `vencimento`: prazo em anos DU/252 desde a emissão.
    - `nominal`: valor emitido, em R$.
    - `protecao`: fração do nominal garantida no vencimento (1.0 = capital protegido).
    - `moeda`: moeda do título e da liquidação.
    - `data_emissao`, `data_vencimento`: opcionais, só para registro e validação da ordem.
    - `nome`: identificação livre do produto.
    """

    ativo: str
    nivel_inicial: float
    vencimento: float
    nominal: float = 1_000.0
    protecao: float = 1.0
    moeda: str = "BRL"
    data_emissao: date | None = None
    data_vencimento: date | None = None
    nome: str = ""

    def __post_init__(self):
        if not self.ativo:
            raise ValueError("ativo não informado")
        _finito_positivo("nivel_inicial", self.nivel_inicial)
        _finito_positivo("vencimento", self.vencimento)
        _finito_positivo("nominal", self.nominal)
        _fracao("protecao", self.protecao, 0.0, _PROTECAO_MAX)
        if (self.data_emissao is None) != (self.data_vencimento is None):
            raise ValueError("informe as duas datas (emissão e vencimento) ou nenhuma")
        if self.data_emissao is not None and self.data_vencimento <= self.data_emissao:
            raise ValueError("data_vencimento deve ser posterior à data_emissao")

    @classmethod
    def por_datas(cls, *, data_emissao: date, data_vencimento: date, **campos):
        """Termsheet com o prazo contado em dias úteis da B3 entre as datas (exige `exchange_calendars`)."""
        from ibov_barrier.di_curve.conventions import business_days

        if data_vencimento <= data_emissao:
            raise ValueError("data_vencimento deve ser posterior à data_emissao")
        du = business_days(data_emissao, data_vencimento)
        return cls(vencimento=du / DU_ANO, data_emissao=data_emissao, data_vencimento=data_vencimento, **campos)

    @property
    def dias_uteis(self) -> int:
        return int(round(self.vencimento * DU_ANO))

    # O que cada estrutura define.
    def opcoes(self) -> Carteira:
        """O pacote de opções em R$ (já multiplicado pela quantidade)."""
        raise NotImplementedError

    def payoff_opcoes(self, finais: Mapping[str, np.ndarray]) -> np.ndarray:
        """O que o pacote de opções paga no vencimento, em R$, escrito direto do termsheet."""
        raise NotImplementedError

    def nivel_observado(self) -> float | None:
        """Nível que a observação do vencimento registra (o strike, nas estruturas com strike)."""
        return None

    def descricao(self) -> str:
        return f"COE {self.ativo}: proteção {_pct(self.protecao)}, {self.dias_uteis} DU"


@dataclass(frozen=True, kw_only=True)
class Participacao(TermsheetBase):
    """Capital protegido com participação, com ou sem cap (seção 2 do plano, primeiras linhas da tabela).

    - `participacao`: fração da alta (ou da queda) do subjacente repassada ao investidor.
    - `cap`: alta (ou queda) máxima do subjacente que conta, em fração do nível inicial; `None` = sem cap.
    - `strike`: nível a partir do qual a participação começa, em fração do nível inicial (1.0 = no dinheiro).
    - `direcao`: "alta" (call ou call spread) ou "baixa" (put ou put spread).
    """

    participacao: float = 1.0
    cap: float | None = None
    strike: float = 1.0
    direcao: str = "alta"

    def __post_init__(self):
        super().__post_init__()
        if self.direcao not in _DIRECOES:
            raise ValueError("direcao deve ser 'alta' ou 'baixa'")
        _fracao("participacao", self.participacao, 0.0, _PARTICIPACAO_MAX)
        _fracao("strike", self.strike, 1e-9, _STRIKE_MAX)
        if self.cap is not None:
            maximo = _CAP_MAX if self.direcao == "alta" else 1.0 - 1e-9
            _fracao("cap", self.cap, 1e-9, maximo)
            if self.direcao == "alta" and 1.0 + self.cap <= self.strike:
                raise ValueError("o cap deve ficar acima do strike: 1 + cap > strike")
            if self.direcao == "baixa" and 1.0 - self.cap >= self.strike:
                raise ValueError("o piso da queda deve ficar abaixo do strike: 1 - cap < strike")

    @property
    def quantidade(self) -> float:
        """Número de opções (em unidades do subjacente) que o pacote tem: nominal × participação / S_0."""
        return self.nominal * self.participacao / self.nivel_inicial

    @property
    def nivel_strike(self) -> float:
        return self.strike * self.nivel_inicial

    @property
    def nivel_cap(self) -> float | None:
        """Nível do subjacente onde a participação para (S_0 × (1 ± cap)); `None` sem cap."""
        if self.cap is None:
            return None
        sinal = 1.0 if self.direcao == "alta" else -1.0
        return (1.0 + sinal * self.cap) * self.nivel_inicial

    @property
    def pagamento_maximo(self) -> float | None:
        """Maior pagamento no vencimento, em fração do nominal; `None` sem cap na alta."""
        if self.direcao == "alta":
            if self.cap is None:
                return None
            return self.protecao + self.participacao * (1.0 + self.cap - self.strike)
        piso = 0.0 if self.cap is None else 1.0 - self.cap
        return self.protecao + self.participacao * (self.strike - piso)

    def opcoes(self) -> Carteira:
        tipo = "call" if self.direcao == "alta" else "put"
        comprada = Vanilla(self.ativo, self.nivel_strike, self.vencimento, tipo)
        if self.cap is None:
            pacote = comprada
        elif self.direcao == "alta":
            pacote = CallSpread(self.ativo, self.nivel_strike, self.nivel_cap, self.vencimento)
        else:
            pacote = comprada - Vanilla(self.ativo, self.nivel_cap, self.vencimento, "put")
        return self.quantidade * pacote

    def payoff_opcoes(self, finais: Mapping[str, np.ndarray]) -> np.ndarray:
        retorno = np.asarray(finais[self.ativo], dtype=float) / self.nivel_inicial
        if self.direcao == "alta":
            ganho = np.maximum(retorno - self.strike, 0.0)
            if self.cap is not None:
                ganho = np.minimum(ganho, 1.0 + self.cap - self.strike)
        else:
            ganho = np.maximum(self.strike - retorno, 0.0)
            if self.cap is not None:
                ganho = np.minimum(ganho, self.strike - (1.0 - self.cap))
        return self.nominal * self.participacao * ganho

    def nivel_observado(self) -> float:
        return self.nivel_strike

    def descricao(self) -> str:
        partes = [f"COE {self.ativo}: proteção {_pct(self.protecao)}",
                  f"participação {_pct(self.participacao)} na {self.direcao}"]
        if self.cap is not None:
            partes.append(f"cap {_pct(self.cap)}")
        if self.strike != 1.0:
            partes.append(f"strike {_pct(self.strike)}")
        partes.append(f"{self.dias_uteis} DU")
        return ", ".join(partes)


# ------------------------------------------------------------------------------------------------- COE

@dataclass(frozen=True)
class Decomposicao:
    """Decomposição do preço de emissão, em R$. `percentuais()` dá as mesmas rubricas em fração do nominal.

    preço de emissão = título + opções + custos de distribuição + custos operacionais + reservas + margem
    """

    nominal: float
    preco_emissao: float
    titulo: float
    opcoes: float
    custo_distribuicao: float
    custo_operacional: float
    reservas: float
    taxa_emissor: float          # taxa efetiva anual do emissor no vencimento (DI pré + spread)
    spread_emissor_bp: float

    @property
    def custos(self) -> float:
        return self.custo_distribuicao + self.custo_operacional

    @property
    def valor_justo(self) -> float:
        """Título + opções: o valor do COE, o que custa entregar o payoff."""
        return self.titulo + self.opcoes

    @property
    def margem(self) -> float:
        return self.preco_emissao - self.valor_justo - self.custos - self.reservas

    @property
    def viavel(self) -> bool:
        return self.margem >= 0.0

    _RUBRICAS = (
        ("preco_emissao", "Preço de emissão"),
        ("titulo", "Título (proteção)"),
        ("opcoes", "Opções"),
        ("custo_distribuicao", "Custos de distribuição"),
        ("custo_operacional", "Custos operacionais"),
        ("reservas", "Reservas"),
        ("margem", "Margem"),
    )

    def linhas(self) -> list[tuple[str, float, float]]:
        """[(rubrica, R$, fração do nominal), ...] na ordem da cascata: preço de emissão primeiro, margem por último."""
        return [(rotulo, getattr(self, campo), getattr(self, campo) / self.nominal) for campo, rotulo in self._RUBRICAS]

    def percentuais(self) -> dict[str, float]:
        return {campo: getattr(self, campo) / self.nominal for campo, _ in self._RUBRICAS}

    def como_dict(self) -> dict[str, float]:
        """R$ e frações do nominal, para relatório e JSON: {"titulo": ..., "titulo_pct": ..., ...}."""
        saida = {"nominal": self.nominal, "valor_justo": self.valor_justo, "taxa_emissor": self.taxa_emissor,
                 "spread_emissor_bp": self.spread_emissor_bp}
        for campo, _ in self._RUBRICAS:
            saida[campo] = getattr(self, campo)
            saida[f"{campo}_pct"] = getattr(self, campo) / self.nominal
        return saida


@dataclass(frozen=True)
class COE(Composicao):
    """COE = nominal × (proteção × título + pacote de opções), montado a partir do `termsheet`.

    É um `Produto`: vale `titulo().valor + opcoes().valor` no `EstadoMercado`, soma com outros produtos numa
    `Carteira` (o livro) e tem gregas pela interface, inclusive o CS01 da perna de título.
    """

    termsheet: TermsheetBase

    # As gregas calculam o CS01: a perna de título depende do spread do emissor.
    depende_do_spread = True

    def __post_init__(self):
        if not isinstance(self.termsheet, TermsheetBase):
            raise TypeError("termsheet deve ser um TermsheetBase (por exemplo, Participacao)")

    @property
    def vencimento(self) -> float:
        return self.termsheet.vencimento

    def titulo(self) -> Titulo:
        """Perna de título: nominal × proteção no vencimento, descontado na curva do emissor."""
        ts = self.termsheet
        return Titulo(vencimento=ts.vencimento, principal=ts.nominal * ts.protecao, moeda=ts.moeda)

    def opcoes(self) -> Carteira:
        return self.termsheet.opcoes()

    def pernas(self) -> Carteira:
        """A composição: título + pacote de opções."""
        return self.titulo() + self.opcoes()

    def valor(self, estado: EstadoMercado) -> Numero:
        return self.pernas().valor(estado)

    def payoff(self, finais: Mapping[str, np.ndarray]) -> np.ndarray:
        """Pagamento no vencimento, em R$, escrito direto do termsheet (sem passar pelos adaptadores)."""
        ts = self.termsheet
        return ts.nominal * ts.protecao + ts.payoff_opcoes(finais)

    def subjacentes(self) -> tuple[str, ...]:
        return (self.termsheet.ativo,)

    def fluxos(self) -> tuple[Fluxo, ...]:
        ts = self.termsheet
        return (Fluxo(ts.vencimento, ts.nominal * ts.protecao, "principal protegido", ts.moeda),
                Fluxo(ts.vencimento, None, f"rendimento das opções ({ts.descricao()})", ts.moeda))

    def observacoes(self) -> tuple[Observacao, ...]:
        ts = self.termsheet
        return (Observacao(ts.vencimento, ts.vencimento, "vencimento", ts.nivel_observado(), ts.ativo),)

    def decompor(self, estado: EstadoMercado, *, preco_emissao: float = 1.0, custo_distribuicao: float = 0.0,
                 custo_operacional: float = 0.0, reservas: float = 0.0) -> Decomposicao:
        """Decomposição do preço de emissão no estado escalar `estado`.

        `preco_emissao`, `custo_distribuicao`, `custo_operacional` e `reservas` em fração do nominal
        (0.03 = 3% do nominal, pagos na emissão).
        """
        if estado.vetorial:
            raise ValueError("a decomposição pede um estado escalar (sem vetores de cenários)")
        _fracao("preco_emissao", preco_emissao, 1e-9, 2.0)
        for nome, valor in (("custo_distribuicao", custo_distribuicao), ("custo_operacional", custo_operacional),
                            ("reservas", reservas)):
            _fracao(nome, valor, 0.0, 1.0)
        ts = self.termsheet
        titulo = self.titulo()
        return Decomposicao(
            nominal=ts.nominal,
            preco_emissao=preco_emissao * ts.nominal,
            titulo=float(titulo.valor(estado)),
            opcoes=float(self.opcoes().valor(estado)),
            custo_distribuicao=custo_distribuicao * ts.nominal,
            custo_operacional=custo_operacional * ts.nominal,
            reservas=reservas * ts.nominal,
            taxa_emissor=titulo.taxa_emissor(estado),
            spread_emissor_bp=float(estado.spread_emissor_bp),
        )
