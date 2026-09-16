# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# CÉLULA 1 — Instala o projeto Python localizado na raiz do repositório Git.

%pip install --force-reinstall --no-deps ..

# COMMAND ----------

# CÉLULA 2 — Reinicia o Python para carregar corretamente o pacote instalado.

dbutils.library.restartPython()

# COMMAND ----------

# CÉLULA 3 — Importa as bibliotecas e define mercado e contrato da opção.

import pandas as pd

from ibov_barrier.pricing import (
    BarrierContract,
    MarketData,
    greeks,
    price_down_and_out_call
)

# Dados de mercado
market = MarketData(
    spot=150_000,
    rate=0.12,
    dividend_yield=0.04,
    volatility=0.22
)

# Características da opção
contract = BarrierContract(
    strike=155_000,
    barrier=120_000,
    maturity=0.50
)

print("Parâmetros carregados com sucesso.")

# COMMAND ----------

# CÉLULA 4 — Precifica a call down-and-out por Monte Carlo.

result = price_down_and_out_call(
    market,
    contract,
    paths=50_000,
    steps=63,
    seed=42
)

pricing_summary = pd.DataFrame({
    "Métrica": [
        "Preço da opção com barreira",
        "Preço da call vanilla",
        "Erro-padrão",
        "Limite inferior do IC 95%",
        "Limite superior do IC 95%",
        "Probabilidade de knock-out"
    ],
    "Resultado": [
        result.price,
        result.vanilla_price,
        result.standard_error,
        result.ci_low,
        result.ci_high,
        result.knock_out_probability
    ]
})

display(pricing_summary)

# COMMAND ----------

# CÉLULA 5 — Calcula Delta, Gamma, Vega, Rho e Theta da opção.

greeks_result = greeks(
    market,
    contract,
    paths=200_000,
    steps=126,
    seed=42
)

greeks_table = pd.DataFrame({
    "Grega": greeks_result.keys(),
    "Valor": greeks_result.values()
})

display(greeks_table)

# COMMAND ----------

# CÉLULA 6 — Simula preços para diferentes níveis do Ibovespa e volatilidade.

spot_shocks = [
    -0.20,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.20
]

vol_shocks = [
    -0.05,
    -0.025,
    0.00,
    0.025,
    0.05
]

scenario_results = []

for spot_shock in spot_shocks:
    for vol_shock in vol_shocks:

        scenario_market = MarketData(
            spot=market.spot * (1 + spot_shock),
            rate=market.rate,
            dividend_yield=market.dividend_yield,
            volatility=market.volatility + vol_shock
        )

        scenario_result = price_down_and_out_call(
            scenario_market,
            contract,
            paths=20_000,
            steps=63,
            seed=42
        )

        scenario_results.append({
            "choque_ibovespa": f"{spot_shock:+.0%}",
            "volatilidade": f"{scenario_market.volatility:.1%}",
            "spot": scenario_market.spot,
            "preco_opcao": scenario_result.price,
            "probabilidade_knock_out":
                scenario_result.knock_out_probability
        })

scenarios_df = pd.DataFrame(scenario_results)

print(f"{len(scenarios_df)} cenários calculados.")

# COMMAND ----------

# CÉLULA 7 — Organiza os preços em uma matriz de spot versus volatilidade.

shock_order = [
    "-20%",
    "-10%",
    "-5%",
    "+0%",
    "+5%",
    "+10%",
    "+20%"
]

vol_order = [
    "17.0%",
    "19.5%",
    "22.0%",
    "24.5%",
    "27.0%"
]

price_table = scenarios_df.pivot(
    index="choque_ibovespa",
    columns="volatilidade",
    values="preco_opcao"
)

price_table = price_table.reindex(
    index=shock_order,
    columns=vol_order
)

price_table.index.name = "Choque Ibovespa"

display(
    price_table
    .round(2)
    .reset_index()
)

# COMMAND ----------

# CÉLULA 8 — Calcula o P&L em pontos usando o cenário central como referência.

reference_price = price_table.loc[
    "+0%",
    "22.0%"
]

pnl_table = price_table - reference_price

pnl_table.index.name = "Choque Ibovespa"

print(
    f"Preço de referência: "
    f"{reference_price:,.2f} pontos"
)

display(
    pnl_table
    .round(2)
    .reset_index()
)

# COMMAND ----------

# CÉLULA 9 — Mostra a probabilidade percentual de a barreira ser atingida.

knockout_table = scenarios_df.pivot(
    index="choque_ibovespa",
    columns="volatilidade",
    values="probabilidade_knock_out"
)

knockout_table = knockout_table.reindex(
    index=shock_order,
    columns=vol_order
)

knockout_table = knockout_table * 100

knockout_table.index.name = "Choque Ibovespa"

display(
    knockout_table
    .round(2)
    .reset_index()
)

# COMMAND ----------

# CÉLULA 10 — Converte preço e P&L de pontos para reais.

quantidade = 1_000

# Referência das opções padronizadas de Ibovespa:
# cada ponto equivale a R$ 0,01 por contrato.
valor_por_ponto = 0.01

# Use 1 para posição comprada e -1 para posição vendida.
lado_posicao = 1

valor_financeiro_posicao = (
    reference_price
    * quantidade
    * valor_por_ponto
    * lado_posicao
)

pnl_financeiro = (
    pnl_table
    * quantidade
    * valor_por_ponto
    * lado_posicao
)

pnl_financeiro.index.name = "Choque Ibovespa"

print(
    f"Valor inicial da posição: "
    f"R$ {valor_financeiro_posicao:,.2f}"
)

display(
    pnl_financeiro
    .round(2)
    .reset_index()
)

# COMMAND ----------

# CÉLULA 11 — Apresenta um resumo executivo da posição avaliada.

lado_descricao = (
    "Comprada"
    if lado_posicao == 1
    else "Vendida"
)

position_summary = pd.DataFrame({
    "Campo": [
        "Tipo da opção",
        "Lado da posição",
        "Quantidade de contratos",
        "Valor por ponto",
        "Preço unitário em pontos",
        "Valor financeiro da posição",
        "Strike",
        "Barreira",
        "Vencimento"
    ],
    "Valor": [
        "Call down-and-out",
        lado_descricao,
        f"{quantidade:,}".replace(",", "."),
        f"R$ {valor_por_ponto:.2f}".replace(".", ","),
        f"{reference_price:,.2f}",
        f"R$ {valor_financeiro_posicao:,.2f}",
        f"{contract.strike:,.0f} pontos",
        f"{contract.barrier:,.0f} pontos",
        f"{contract.maturity:.2f} ano"
    ]
})

display(position_summary)