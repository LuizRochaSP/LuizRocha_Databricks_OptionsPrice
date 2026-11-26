"""COE: perna de título, composição, solver e catálogo de estruturas (fase A do plano de COE).

Usa a interface de `ibov_barrier.produtos`: a perna de título é um `Produto` e se soma às opções numa `Carteira`.

    ts = Participacao(ativo="IBOV", nivel_inicial=130_000, vencimento=1.0, nominal=1_000_000,
                      protecao=1.0, participacao=0.8, cap=0.2)
    coe = COE(ts)
    coe.decompor(estado, custo_distribuicao=0.03).linhas()
    resolver(ts, estado, "participacao", margem_alvo=0.02, custo_distribuicao=0.03).valor
"""
from ibov_barrier.coe.estruturacao import Estruturacao, estruturar
from ibov_barrier.coe.estrutura import COE, Decomposicao, ParametroLivre, Participacao, TermsheetBase
from ibov_barrier.coe.solver import Solucao, resolver
from ibov_barrier.coe.titulo import (
    Titulo,
    fator_desconto_emissor,
    fator_desconto_percentual_cdi,
    percentual_cdi_de_spread,
    spread_de_percentual_cdi,
)

__all__ = [
    "COE", "Decomposicao", "Estruturacao", "ParametroLivre", "Participacao", "Solucao", "TermsheetBase", "Titulo",
    "fator_desconto_emissor", "fator_desconto_percentual_cdi", "estruturar", "percentual_cdi_de_spread",
    "resolver", "spread_de_percentual_cdi",
]
