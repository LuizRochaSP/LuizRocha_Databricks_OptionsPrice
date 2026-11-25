"""Interface de produtos: estado de mercado, protocolo `Produto`, adaptadores dos pricers e gregas genéricas.

Etapa A1 do plano de COE (`PLANO_COE_PRICING_RISCO.md`). Os pricers do motor não mudam: cada adaptador
reproduz o número do pricer original.
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
from ibov_barrier.produtos.gregas import Gregas, gregas
from ibov_barrier.produtos.mercado import (
    Curva,
    CurvaDI,
    CurvaPlana,
    EstadoMercado,
    MercadoAtivo,
    MercadoCambio,
)

__all__ = [
    "BarreiraSimples", "CallSpread", "Carteira", "Composicao", "Curva", "CurvaDI", "CurvaPlana", "Digital",
    "EstadoMercado", "Fluxo", "Gregas", "MercadoAtivo", "MercadoCambio", "NoTouch", "Observacao", "OneTouch",
    "Produto", "Quanto", "Vanilla", "black_scholes", "digital_cash", "gregas",
]
