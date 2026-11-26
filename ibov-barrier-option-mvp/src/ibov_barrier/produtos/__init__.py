"""Interface de produtos: estado de mercado, protocolo `Produto`, adaptadores dos pricers e gregas genéricas.

Etapa A1 do plano de COE (`PLANO_COE_PRICING_RISCO.md`). Os pricers do motor não mudam: cada adaptador
reproduz o número do pricer original. `rodada` (etapa A5) monta o `EstadoMercado` de uma rodada aprovada do
pipeline.
"""
from ibov_barrier.produtos.adaptadores import (
    BarreiraSimples,
    CallSpread,
    Digital,
    NoTouch,
    OneTouch,
    Quanto,
    Vanilla,
    black_scholes,
    digital_cash,
)
from ibov_barrier.produtos.base import Carteira, Composicao, Fluxo, Observacao, Produto
from ibov_barrier.produtos.gregas import Gregas, dv01_por_vertice, gregas
from ibov_barrier.produtos.mercado import (
    Curva,
    CurvaDI,
    CurvaPlana,
    EstadoMercado,
    MercadoAtivo,
    MercadoCambio,
)
from ibov_barrier.produtos.rodada import (
    Rodada,
    SuperficieVol,
    conferir_referencia,
    estado_ibov,
    ultima_rodada_aprovada,
)

__all__ = [
    "BarreiraSimples", "CallSpread", "Carteira", "Composicao", "Curva", "CurvaDI", "CurvaPlana", "Digital",
    "EstadoMercado", "Fluxo", "Gregas", "MercadoAtivo", "MercadoCambio", "NoTouch", "Observacao", "OneTouch",
    "Produto", "Quanto", "Rodada", "SuperficieVol", "Vanilla", "black_scholes", "conferir_referencia",
    "digital_cash", "dv01_por_vertice", "estado_ibov", "gregas", "ultima_rodada_aprovada",
]
