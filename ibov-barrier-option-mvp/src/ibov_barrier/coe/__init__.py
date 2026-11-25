"""COE: perna de título, composição, solver e catálogo de estruturas (fase A do plano de COE).

Usa a interface de `ibov_barrier.produtos`: a perna de título é um `Produto` e se soma às opções numa `Carteira`.
"""
from ibov_barrier.coe.titulo import (
    Titulo,
    fator_desconto_emissor,
    fator_desconto_percentual_cdi,
    percentual_cdi_de_spread,
    spread_de_percentual_cdi,
)

__all__ = [
    "Titulo", "fator_desconto_emissor", "fator_desconto_percentual_cdi", "percentual_cdi_de_spread",
    "spread_de_percentual_cdi",
]
