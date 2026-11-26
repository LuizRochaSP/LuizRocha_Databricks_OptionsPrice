"""COE: perna de título, composição, solver e catálogo de estruturas (fase A do plano de COE).

Usa a interface de `ibov_barrier.produtos`: a perna de título é um `Produto` e se soma às opções numa `Carteira`.

    ts = Participacao(ativo="IBOV", nivel_inicial=130_000, vencimento=1.0, nominal=1_000_000,
                      protecao=1.0, participacao=0.8, cap=0.2)
    coe = COE(ts)
    coe.decompor(estado, custo_distribuicao=0.03).linhas()
"""
from ibov_barrier.coe.estrutura import COE, Decomposicao, Participacao, TermsheetBase
from ibov_barrier.coe.titulo import (
    Titulo,
    fator_desconto_emissor,
    fator_desconto_percentual_cdi,
    percentual_cdi_de_spread,
    spread_de_percentual_cdi,
)

__all__ = [
    "COE", "Decomposicao", "Participacao", "TermsheetBase", "Titulo", "fator_desconto_emissor",
    "fator_desconto_percentual_cdi", "percentual_cdi_de_spread", "spread_de_percentual_cdi",
]
