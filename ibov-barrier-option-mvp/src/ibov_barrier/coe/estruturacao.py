"""Uma rodada de estruturação (etapa A5 do plano de COE): o que o notebook 12 calcula, grava e reporta.

    est = estruturar(ts, estado, custo_distribuicao=0.03, reservas=0.005, parametro="participacao",
                     margem_alvo=0.02)
    print(est.texto(proveniencia=rodada.proveniencia()))

- A decomposição do termsheet como foi digitado.
- O solver, quando há parâmetro e margem alvo: o termsheet resolvido e a decomposição dele (ou, fora do alcance, a
  do extremo mais próximo, com a mensagem do solver).
- As gregas iniciais do COE final (o resolvido, se houver; senão o digitado), por perna, e o DV01 por vértice.
  Sinal do investidor: o emissor está vendido no COE.
- A vol lida em cada nível do pacote (strike e cap), com a regra da superfície.
- Avisos: prazo além da superfície ou do último vértice DI, vol pelo vizinho mais próximo, margem fora do alcance.

`texto()` é o relatório para enviar; `como_dict()` é o JSON que o notebook grava.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime

import numpy as np

from ibov_barrier.coe.estrutura import COE, Decomposicao, TermsheetBase, _pct
from ibov_barrier.coe.solver import Solucao, resolver
from ibov_barrier.coe.titulo import percentual_cdi_de_spread
from ibov_barrier.produtos.gregas import Gregas, dv01_por_vertice, gregas
from ibov_barrier.produtos.mercado import EstadoMercado

SCHEMA_ESTRUTURACAO = 1


def _br(x: float, casas: int = 2) -> str:
    """1234567.891 -> '1.234.567,89'."""
    return f"{x:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _brl(x: float) -> str:
    return ("-R$ " if x < 0 else "R$ ") + _br(abs(x))


@dataclass(frozen=True)
class NivelVol:
    """Vol lida num nível do pacote de opções."""

    rotulo: str
    nivel: float
    fracao: float     # do nível inicial
    vol: float
    regra: str        # "linear", "vizinho mais próximo" ou "plana"


@dataclass(frozen=True)
class Estruturacao:
    termsheet: TermsheetBase
    decomposicao: Decomposicao
    solucao: Solucao | None
    custos: dict
    gregas: dict[str, Gregas]
    dv01_vertices: dict[float, float]
    vols: tuple[NivelVol, ...]
    taxa_pre: float
    percentual_cdi: float | None
    avisos: tuple[str, ...]
    spot: float
    data_mercado: date | None

    @property
    def termsheet_final(self) -> TermsheetBase:
        """O resolvido, quando o solver resolveu; senão o digitado."""
        return self.solucao.termsheet if self.solucao is not None and self.solucao.viavel else self.termsheet

    @property
    def decomposicao_final(self) -> Decomposicao:
        return self.solucao.decomposicao if self.solucao is not None and self.solucao.viavel else self.decomposicao

    # ------------------------------------------------------------------------------------------ saída

    def como_dict(self) -> dict:
        """JSON do notebook 12 (tipos puros)."""
        def termsheet(ts: TermsheetBase) -> dict:
            dados = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(ts).items()}
            return dict(dados, tipo=type(ts).__name__, descricao=ts.descricao(), dias_uteis=ts.dias_uteis)

        d = self.decomposicao
        return {
            "schema_version": SCHEMA_ESTRUTURACAO,
            "market_date": self.data_mercado.isoformat() if self.data_mercado else None,
            "spot": self.spot,
            "termsheet": termsheet(self.termsheet),
            "custos": dict(self.custos),
            "mercado": {"spread_emissor_bp": d.spread_emissor_bp, "taxa_pre": self.taxa_pre,
                        "taxa_emissor": d.taxa_emissor, "percentual_cdi": self.percentual_cdi,
                        "vols": [asdict(v) for v in self.vols]},
            "decomposicao": d.como_dict(),
            "solver": None if self.solucao is None else self.solucao.como_dict(),
            "termsheet_final": termsheet(self.termsheet_final),
            "decomposicao_final": self.decomposicao_final.como_dict(),
            "gregas": {perna: g.como_dict() for perna, g in self.gregas.items()},
            "dv01_por_vertice": {str(int(du)): v for du, v in self.dv01_vertices.items()},
            "avisos": list(self.avisos),
        }

    def _cascata(self, d: Decomposicao) -> list[str]:
        largura = max(len(rotulo) for rotulo, _, _ in d.linhas())
        return [f"  {rotulo:<{largura}}  {_brl(reais):>20}  {_pct(fracao):>8}" for rotulo, reais, fracao in d.linhas()]

    def texto(self, *, proveniencia: dict | None = None, agora: datetime | None = None) -> str:
        """Relatório em texto para enviar (o padrão dos notebooks: abre com o título e fecha com FIM)."""
        ts, d = self.termsheet, self.decomposicao
        quando = f" | {agora:%Y-%m-%d %H:%M} (Brasília)" if agora else ""
        linhas = [f"RELATÓRIO COE ESTRUTURAÇÃO{quando}"]
        if proveniencia:
            p = proveniencia
            linhas.append(f"Rodada: {p['validation_file']} | {p['results_file']} | SHA-256 {p['results_sha256'][:12]}… "
                          f"| mercado {p['market_date']} | {p['validation_decision']} "
                          f"(WARN {p['validation_warn_count']})")
        cdi = "" if self.percentual_cdi is None else f" ({_br(self.percentual_cdi, 1)}% do CDI)"
        spread = _br(d.spread_emissor_bp, 1).removesuffix(",0")
        linhas.append(f"Mercado: {ts.ativo} {_br(self.spot)} | DI pré no prazo {_pct(self.taxa_pre)} a.a. | spread do "
                      f"emissor {spread} bp{cdi}, informado à mão (decisão 2 da seção 8) | "
                      f"taxa do emissor {_pct(d.taxa_emissor)} a.a.")
        if self.vols:
            linhas.append("Vol: " + " | ".join(f"{v.rotulo} {_pct(v.fracao)} ({_br(v.nivel)}) {_pct(v.vol)}, {v.regra}"
                                               for v in self.vols))
        c = self.custos
        linhas.append(f"Custos (% do nominal): distribuição {_pct(c['custo_distribuicao'])}, operacional "
                      f"{_pct(c['custo_operacional'])}, reservas {_pct(c['reservas'])}; preço de emissão "
                      f"{_pct(c['preco_emissao'])}")
        linhas += ["", f"Termsheet digitado: {ts.descricao()} | nominal {_brl(ts.nominal)}",
                   "Decomposição (R$ e % do nominal):", *self._cascata(d)]

        if self.solucao is not None:
            s = self.solucao
            linhas += ["", f"Solver: {s.rotulo} para margem de {_pct(s.margem_alvo)} -> {s.situacao} ({s.metodo}; a "
                           f"margem vai de {_pct(s.margem_minima)} a {_pct(s.margem_maxima)} no intervalo)",
                       f"  {s.mensagem}"]
            titulo = "Termsheet resolvido" if s.viavel else "Extremo mais próximo do alvo"
            linhas += [f"{titulo}: {s.termsheet.descricao()}", "Decomposição:", *self._cascata(s.decomposicao)]

        g = self.gregas["COE"]
        ativo = ts.ativo
        delta, gamma, vega = g.delta.get(ativo, 0.0), g.gamma.get(ativo, 0.0), g.vega.get(ativo, 0.0)
        um_pct = 0.01 * self.spot
        qual = "resolvido" if self.termsheet_final is not ts else "digitado"
        linhas += ["", f"Gregas iniciais do COE {qual} (sinal do investidor; o emissor está vendido):",
                   f"  Delta: {_brl(delta)} por ponto | {_brl(delta * um_pct)} por +1% do {ativo}",
                   f"  Gamma: P&L de 2ª ordem para ±1% do {ativo}: {_brl(0.5 * gamma * um_pct ** 2)}",
                   f"  Vega: {_brl(vega)} por +1 ponto de vol",
                   f"  DV01 (DI pré, paralelo): {_brl(sum(g.dv01.values()))} por +1 bp"]
        maiores = [kv for kv in sorted(self.dv01_vertices.items(), key=lambda kv: -abs(kv[1]))[:3]
                   if abs(kv[1]) >= 0.005]
        if maiores:
            linhas.append("  DV01 nos maiores vértices: " + ", ".join(f"{int(du)} DU {_brl(v)}" for du, v in maiores))
        if g.cs01 is not None:
            linhas.append(f"  CS01 (spread do emissor): {_brl(g.cs01)} por +1 bp")
        linhas.append(f"  Theta: {_brl(g.theta)} por dia útil")
        linhas.append("  Por perna (delta por +1% | vega | DV01 | CS01): " + " ; ".join(
            f"{perna} {_brl(x.delta.get(ativo, 0.0) * um_pct)} | {_brl(x.vega.get(ativo, 0.0))} | "
            f"{_brl(sum(x.dv01.values()))} | {_brl(x.cs01 or 0.0)}" for perna, x in self.gregas.items()
            if perna != "COE"))

        linhas += ["", "Avisos:"] + ([f"  - {a}" for a in self.avisos] or ["  (nenhum)"])
        linhas.append("FIM DO RELATÓRIO")
        return "\n".join(linhas)


# --------------------------------------------------------------------------------------------- montagem

def _vols(ts: TermsheetBase, estado: EstadoMercado) -> tuple[NivelVol, ...]:
    mercado = estado.ativo(ts.ativo)
    niveis = []
    for rotulo, atributo in (("strike", "nivel_strike"), ("cap", "nivel_cap")):
        nivel = getattr(ts, atributo, None)
        if nivel is None:
            continue
        if hasattr(mercado.vol, "com_regra"):
            vol, regra = mercado.vol.com_regra(nivel, ts.vencimento)
            vol += float(mercado.choque_vol)
        else:
            vol, regra = float(mercado.vol_em(nivel, ts.vencimento)), "plana"
        niveis.append(NivelVol(rotulo, float(nivel), float(nivel) / ts.nivel_inicial, float(vol), regra))
    return tuple(niveis)


def _avisos(ts: TermsheetBase, estado: EstadoMercado, vols, solucao) -> list[str]:
    avisos = []
    superficie = estado.ativo(ts.ativo).vol
    alem_da_superficie = hasattr(superficie, "prazo_maximo") and ts.vencimento > superficie.prazo_maximo + 1e-12
    if alem_da_superficie:
        avisos.append(f"O prazo do COE ({_br(ts.vencimento, 2)} ano) passa do último vencimento da superfície "
                      f"({_br(superficie.prazo_maximo, 2)} ano): a vol sai do vizinho mais próximo, sem estrutura a "
                      f"termo (reserva de modelo na etapa C5).")
    curva = estado.curva(ts.moeda)
    vertices = getattr(curva, "vertices", ())
    if vertices and ts.dias_uteis > max(vertices):
        avisos.append(f"O prazo do COE ({ts.dias_uteis} DU) passa do último vértice DI ({int(max(vertices))} DU): a "
                      f"taxa segue o forward do último segmento.")
    for v in vols:
        if v.regra == "vizinho mais próximo" and not alem_da_superficie:
            avisos.append(f"Vol do {v.rotulo} ({_pct(v.fracao)}) pelo vizinho mais próximo: o ponto cai fora do casco "
                          f"dos dados da superfície.")
    if solucao is not None and not solucao.viavel:
        avisos.append(f"Solver: {solucao.mensagem}")
    return avisos


def estruturar(termsheet: TermsheetBase, estado: EstadoMercado, *, preco_emissao: float = 1.0,
               custo_distribuicao: float = 0.0, custo_operacional: float = 0.0, reservas: float = 0.0,
               parametro: str | None = None, margem_alvo: float | None = None,
               avisos: tuple[str, ...] | list[str] = ()) -> Estruturacao:
    """Decomposição, solver (se `parametro` e `margem_alvo`), gregas iniciais, vols e avisos.

    Percentuais em fração do nominal. `avisos`: os da rodada (data de mercado, ressalvas), somados aos daqui.
    """
    if (parametro is None) != (margem_alvo is None):
        raise ValueError("informe parametro e margem_alvo juntos (ou nenhum dos dois)")
    custos = dict(preco_emissao=preco_emissao, custo_distribuicao=custo_distribuicao,
                  custo_operacional=custo_operacional, reservas=reservas)
    coe = COE(termsheet)
    decomposicao = coe.decompor(estado, **custos)
    solucao = None if parametro is None else resolver(termsheet, estado, parametro, margem_alvo=margem_alvo,
                                                      **custos)
    final = solucao.termsheet if solucao is not None and solucao.viavel else termsheet
    coe_final = COE(final)
    pernas = {"COE": coe_final, "Título": coe_final.titulo(), "Opções": coe_final.opcoes()}
    gregas_pernas = {nome: gregas(produto, estado) for nome, produto in pernas.items()}
    try:
        vertices = dv01_por_vertice(coe_final, estado, final.moeda)
    except TypeError:  # curva sem vértices (CurvaPlana)
        vertices = {}

    curva = estado.curva(termsheet.moeda)
    prazo = termsheet.vencimento
    taxa_pre = float(np.asarray(curva.fator_desconto(prazo))) ** (-1.0 / prazo) - 1.0
    try:
        cdi = percentual_cdi_de_spread(curva, float(estado.spread_emissor_bp), prazo)
    except ValueError:
        cdi = None
    vols = _vols(final, estado)
    return Estruturacao(termsheet=termsheet, decomposicao=decomposicao, solucao=solucao, custos=custos,
                        gregas=gregas_pernas, dv01_vertices=vertices, vols=vols, taxa_pre=taxa_pre,
                        percentual_cdi=cdi, avisos=tuple(list(avisos) + _avisos(final, estado, vols, solucao)),
                        spot=float(estado.ativo(termsheet.ativo).spot), data_mercado=estado.data)
