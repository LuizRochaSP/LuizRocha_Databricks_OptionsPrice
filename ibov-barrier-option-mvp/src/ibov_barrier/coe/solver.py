"""Solver de margem alvo (etapa A4 do plano de COE).

A decomposição (A3) dá a margem de um termsheet:

    margem = preço de emissão − título − opções − custos − reservas

O solver inverte a conta. Fixados a margem alvo, os custos e o resto do termsheet, ele resolve um parâmetro livre:

- **Parâmetro linear** (a participação, que só escala o pacote de opções; a proteção, que só escala o título): a
  margem é afim no parâmetro. Duas avaliações nas pontas do intervalo dão a solução direta, e uma terceira confere
  que a margem bate.
- **Os outros** (cap, strike; cupom e barreira na fase C): busca de raiz (Brent). Antes, o solver avalia a margem
  numa grade do intervalo e verifica que ela é monotônica. Sem isso, a raiz pode não ser única, e o solver recusa.

**Fora do alcance.** A margem possível no intervalo vai de `margem_minima` a `margem_maxima`:

- alvo acima da máxima: **inviável**. A estrutura não paga essa margem nem no extremo mais barato. Exemplo: proteção
  de 100% com juro baixo e prazo curto, em que o título sozinho custa quase o preço de emissão.
- alvo abaixo da mínima: **folga**. A estrutura deixa mais margem do que o pedido mesmo no extremo mais generoso
  para o investidor (por exemplo, sem cap): sobra espaço em outro parâmetro.

Nos dois casos, `Solucao.valor` é `None`, e o termsheet e a decomposição da solução são os do extremo mais próximo
do alvo, para a tela mostrar o mais perto que dá.

    sol = resolver(ts, estado, "participacao", margem_alvo=0.02, custo_distribuicao=0.03, reservas=0.005)
    sol.situacao, sol.valor, sol.mensagem

Percentuais em fração do nominal (0.02 = 2%), como no resto do pacote.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

from ibov_barrier.coe.estrutura import COE, Decomposicao, ParametroLivre, TermsheetBase, _fracao, _pct
from ibov_barrier.produtos.mercado import EstadoMercado

SITUACOES = ("resolvido", "inviavel", "folga")

# Tolerâncias em fração do nominal.
_TOL_MONOTONIA = 1e-12     # diferença de margem entre pontos vizinhos da grade tratada como zero
_TOL_ALCANCE = 1e-12       # alvo a menos disso da margem extrema ainda conta como dentro do alcance
_TOL_CONFERENCIA = 1e-8    # margem da solução contra o alvo
_PONTOS_PADRAO = 61


@dataclass(frozen=True)
class Solucao:
    """Resultado do solver.

    - `situacao`: "resolvido", "inviavel" (alvo acima da maior margem possível) ou "folga" (alvo abaixo da menor).
    - `valor`: o parâmetro resolvido; `None` fora do alcance.
    - `termsheet`, `decomposicao`: os da solução ou, fora do alcance, os do extremo mais próximo do alvo.
    - `margem_minima`, `margem_maxima`: alcance da margem no `intervalo`, em fração do nominal.
    - `metodo`: "direto" (parâmetro linear) ou "raiz" (Brent, depois da verificação de monotonicidade).
    """

    parametro: str
    rotulo: str
    situacao: str
    margem_alvo: float
    valor: float | None
    margem_minima: float
    margem_maxima: float
    intervalo: tuple[float, float]
    metodo: str
    termsheet: TermsheetBase
    decomposicao: Decomposicao
    mensagem: str

    @property
    def viavel(self) -> bool:
        return self.situacao == "resolvido"

    @property
    def coe(self) -> COE:
        return COE(self.termsheet)

    @property
    def margem(self) -> float:
        """Margem do termsheet da solução, em fração do nominal (igual ao alvo quando resolvido)."""
        return self.decomposicao.margem / self.decomposicao.nominal

    def como_dict(self) -> dict:
        """Para relatório e JSON."""
        return {"parametro": self.parametro, "situacao": self.situacao, "valor": self.valor,
                "margem_alvo": self.margem_alvo, "margem": self.margem, "margem_minima": self.margem_minima,
                "margem_maxima": self.margem_maxima, "intervalo": list(self.intervalo), "metodo": self.metodo,
                "termsheet": self.termsheet.descricao(), "mensagem": self.mensagem,
                "decomposicao": self.decomposicao.como_dict()}


def resolver(termsheet: TermsheetBase | COE, estado: EstadoMercado, parametro: str, *, margem_alvo: float,
             preco_emissao: float = 1.0, custo_distribuicao: float = 0.0, custo_operacional: float = 0.0,
             reservas: float = 0.0, intervalo: tuple[float, float] | None = None,
             pontos: int = _PONTOS_PADRAO) -> Solucao:
    """Resolve `parametro` do termsheet para que a margem seja `margem_alvo` (fração do nominal).

    - `parametro`: um dos `termsheet.parametros_livres()` ("participacao", "protecao", "cap" ou "strike" na
      `Participacao`). Os outros campos ficam como estão.
    - `preco_emissao`, custos e reservas: em fração do nominal, como em `COE.decompor`.
    - `intervalo`: troca o intervalo declarado pelo termsheet (por exemplo, para limitar o cap a 50%).
    - `pontos`: tamanho da grade da verificação de monotonicidade (busca de raiz).

    Levanta `ValueError` para parâmetro desconhecido, estado vetorial, margem fora de [−100%, 100%] ou margem não
    monotônica no intervalo. Margem fora do alcance não é erro: volta com `situacao` "inviavel" ou "folga".
    """
    ts = termsheet.termsheet if isinstance(termsheet, COE) else termsheet
    if not isinstance(ts, TermsheetBase):
        raise TypeError("termsheet deve ser um TermsheetBase ou um COE")
    if estado.vetorial:
        raise ValueError("o solver pede um estado escalar (sem vetores de cenários)")
    _fracao("margem_alvo", margem_alvo, -1.0, 1.0)
    livres = ts.parametros_livres()
    if parametro not in livres:
        raise ValueError(f"'{parametro}' não é parâmetro livre de {type(ts).__name__}; "
                         f"os livres são: {', '.join(livres) or 'nenhum'}")
    livre = livres[parametro]
    if intervalo is not None:
        livre = replace(livre, minimo=float(intervalo[0]), maximo=float(intervalo[1]))
    if pontos < 3:
        raise ValueError("pontos deve ser pelo menos 3")

    custos = dict(preco_emissao=preco_emissao, custo_distribuicao=custo_distribuicao,
                  custo_operacional=custo_operacional, reservas=reservas)
    COE(ts).decompor(estado, **custos)  # valida custos e estado antes de qualquer conta
    nominal = ts.nominal
    disponivel = (preco_emissao - custo_distribuicao - custo_operacional - reservas) * nominal

    def com(x: float) -> TermsheetBase:
        return replace(ts, **{parametro: float(x)})

    def margem(x: float) -> float:
        return (disponivel - float(COE(com(x)).valor(estado))) / nominal

    if livre.linear:
        return _direto(livre, margem, com, estado, custos, margem_alvo)
    return _raiz(livre, margem, com, estado, custos, margem_alvo, pontos)


# --------------------------------------------------------------------------------------------- métodos

def _direto(livre: ParametroLivre, margem, com, estado, custos, alvo) -> Solucao:
    lo, hi = livre.minimo, livre.maximo
    m_lo, m_hi = margem(lo), margem(hi)
    if abs(m_hi - m_lo) <= _TOL_MONOTONIA:
        raise ValueError(f"a margem não depende de {livre.rotulo} no intervalo [{lo:g}, {hi:g}]")
    fora = _fora_do_alcance(livre, np.array([lo, hi]), np.array([m_lo, m_hi]), alvo, com, estado, custos,
                            "direto")
    if fora is not None:
        return fora
    x = lo + (alvo - m_lo) * (hi - lo) / (m_hi - m_lo)
    x = min(max(x, lo), hi)
    if abs(margem(x) - alvo) > _TOL_CONFERENCIA:
        raise RuntimeError(f"{livre.nome} foi declarado linear, mas a margem não é afim nele")
    return _resolvida(livre, x, (m_lo, m_hi), com, estado, custos, alvo, "direto")


def _grade(livre: ParametroLivre, pontos: int) -> np.ndarray:
    lo, hi = livre.minimo, livre.maximo
    if livre.escala == "geometrica":
        largura = hi - lo
        grade = lo + np.concatenate(([0.0], np.geomspace(largura * 1e-6, largura, pontos - 1)))
    else:
        grade = np.linspace(lo, hi, pontos)
    grade[0], grade[-1] = lo, hi  # pontas exatas: as mensagens reconhecem o máximo
    return grade


def _raiz(livre: ParametroLivre, margem, com, estado, custos, alvo, pontos) -> Solucao:
    grade = _grade(livre, pontos)
    margens = np.array([margem(x) for x in grade])
    passos = np.diff(margens)
    if np.all(np.abs(passos) <= _TOL_MONOTONIA):
        raise ValueError(f"a margem não depende de {livre.rotulo} no intervalo [{livre.minimo:g}, {livre.maximo:g}]")
    if not (np.all(passos >= -_TOL_MONOTONIA) or np.all(passos <= _TOL_MONOTONIA)):
        raise ValueError(f"a margem não é monotônica em {livre.rotulo} no intervalo "
                         f"[{livre.minimo:g}, {livre.maximo:g}]: a raiz pode não ser única; restrinja o intervalo")
    fora = _fora_do_alcance(livre, grade, margens, alvo, com, estado, custos, "raiz")
    if fora is not None:
        return fora
    alcance = sorted((float(margens[0]), float(margens[-1])))
    alvo_dentro = min(max(alvo, alcance[0]), alcance[1])
    desvio = margens - alvo_dentro
    exatos = np.flatnonzero(desvio == 0.0)
    if exatos.size:
        x = float(grade[exatos[0]])
    else:
        i = int(np.flatnonzero(np.sign(desvio[:-1]) != np.sign(desvio[1:]))[0])
        x = brentq(lambda v: margem(v) - alvo_dentro, grade[i], grade[i + 1], xtol=1e-13, rtol=1e-12,
                   maxiter=200)
    return _resolvida(livre, x, alcance, com, estado, custos, alvo, "raiz")


# ------------------------------------------------------------------------------------------ resultados

def _resolvida(livre, x, alcance, com, estado, custos, alvo, metodo) -> Solucao:
    ts = com(x)
    d = COE(ts).decompor(estado, **custos)
    if abs(d.margem / d.nominal - alvo) > _TOL_CONFERENCIA:
        raise RuntimeError(f"a margem da solução ({d.margem / d.nominal}) não bate com o alvo ({alvo})")
    mensagem = f"{livre.rotulo.capitalize()} de {_pct(x)} dá a margem de {_pct(alvo)}."
    return Solucao(livre.nome, livre.rotulo, "resolvido", alvo, float(x), float(min(alcance)), float(max(alcance)),
                   (livre.minimo, livre.maximo), metodo, ts, d, mensagem)


def _fora_do_alcance(livre, xs, margens, alvo, com, estado, custos, metodo) -> Solucao | None:
    """Alvo fora do alcance da margem. Margem monotônica: os extremos estão nas pontas do intervalo (na cauda plana,
    como a do cap grande, a grade tem vários pontos com a mesma margem, e a ponta é o extremo que vale)."""
    if margens[-1] >= margens[0]:
        (x_min, m_min), (x_max, m_max) = (xs[0], margens[0]), (xs[-1], margens[-1])
    else:
        (x_min, m_min), (x_max, m_max) = (xs[-1], margens[-1]), (xs[0], margens[0])
    m_min, m_max = float(m_min), float(m_max)
    if alvo > m_max + _TOL_ALCANCE:
        situacao, x = "inviavel", float(x_max)
    elif alvo < m_min - _TOL_ALCANCE:
        situacao, x = "folga", float(x_min)
    else:
        return None
    ts = com(x)
    d = COE(ts).decompor(estado, **custos)
    nota = livre.nota_maximo if x == livre.maximo else ""
    onde = f"{livre.rotulo} de {_pct(x)}{nota}"
    if situacao == "inviavel":
        n = d.nominal
        mensagem = (f"Margem de {_pct(alvo)} inviável: a maior margem possível é {_pct(m_max)}, com {onde}. "
                    f"Nesse extremo, o título custa {_pct(d.titulo / n)} do nominal, as opções {_pct(d.opcoes / n)} "
                    f"e os custos e reservas {_pct((d.custos + d.reservas) / n)}.")
    else:
        mensagem = (f"Margem de {_pct(alvo)} abaixo da menor possível, que é {_pct(m_min)}, com {onde}. A estrutura "
                    f"deixa mais margem do que o pedido: dá para melhorar outro parâmetro para o investidor ou "
                    f"subir a margem alvo.")
    return Solucao(livre.nome, livre.rotulo, situacao, alvo, None, m_min, m_max, (livre.minimo, livre.maximo),
                   metodo, ts, d, mensagem)
