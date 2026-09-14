# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Opção com Barreira no Ibovespa
# MAGIC MVP: call down-and-out, Monte Carlo e Brownian Bridge.

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Precificação de uma Opção com Barreira no Ibovespa
# MAGIC
# MAGIC ## 1. O que é uma opção com barreira?
# MAGIC
# MAGIC Uma opção com barreira é um derivativo cujo valor depende não apenas do preço do ativo no vencimento, mas também do caminho percorrido por ele durante a vida do contrato.
# MAGIC
# MAGIC Neste projeto, precificamos uma **call down-and-out sem rebate** sobre o Ibovespa:
# MAGIC
# MAGIC - **Call:** proporciona exposição à alta do Ibovespa.
# MAGIC - **Down:** a barreira está abaixo do nível inicial do índice.
# MAGIC - **Out:** a opção deixa de existir se a barreira for atingida.
# MAGIC - **Sem rebate:** não existe pagamento compensatório caso a opção seja extinta.
# MAGIC
# MAGIC A opção paga no vencimento:
# MAGIC
# MAGIC $$\\text{Payoff}=\\max(S_T-K,0)\\cdot\\mathbf{1}_{\\{\\min_{0\\leq t\\leq T}S_t>H\\}}$$
# MAGIC
# MAGIC
# MAGIC em que:
# MAGIC
# MAGIC - `S_T` é o nível do Ibovespa no vencimento;
# MAGIC - `K` é o strike ou preço de exercício;
# MAGIC - `H` é a barreira inferior;
# MAGIC - `T` é o prazo até o vencimento;
# MAGIC - $$\\mathbf{1}_{\\{\\min_{0\\leq t\\leq T}S_t>H\\}}$$ o termo indicador assume valor 1 se a barreira não tiver sido atingida e valor 0 se a barreira tiver sido atingida.
# MAGIC
# MAGIC
# MAGIC Portanto, mesmo que o Ibovespa termine acima do strike, o payoff será zero caso o índice tenha tocado a barreira durante a vigência da opção.
# MAGIC
# MAGIC **Observação__** *Uma possível leitura mais simples é: paga-se o valor positivo da call apenas quando o menor nível atingido pelo Ibovespa durante o contrato permaneceu acima da barreira.*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Como a opção funciona neste exemplo?
# MAGIC
# MAGIC O contrato analisado possui os seguintes parâmetros:
# MAGIC
# MAGIC | Parâmetro | Valor |
# MAGIC |---|---:|
# MAGIC | Ibovespa inicial | 150.000 pontos |
# MAGIC | Strike | 155.000 pontos |
# MAGIC | Barreira inferior | 120.000 pontos |
# MAGIC | Prazo | 6 meses |
# MAGIC | Volatilidade | 22% a.a. |
# MAGIC | Taxa livre de risco | 12% a.a. |
# MAGIC | Dividend yield ou carry | 0% a.a. |
# MAGIC | Rebate | Zero |
# MAGIC
# MAGIC O modelo simula diferentes trajetórias possíveis para o Ibovespa. Para cada trajetória:
# MAGIC
# MAGIC 1. verifica se a barreira inferior foi atingida;
# MAGIC 2. extingue a opção quando ocorre o knock-out;
# MAGIC 3. calcula o payoff das trajetórias sobreviventes;
# MAGIC 4. desconta os pagamentos pela taxa livre de risco;
# MAGIC 5. calcula a média dos valores presentes.
# MAGIC
# MAGIC O preço estimado por Monte Carlo pode ser representado por:
# MAGIC
# MAGIC \[
# MAGIC V_0
# MAGIC =
# MAGIC e^{-rT}
# MAGIC \mathbb{E}^{\mathbb{Q}}
# MAGIC \left[
# MAGIC \max(S_T-K,0)
# MAGIC \mathbf{1}_{\left\{\min_{0\leq t\leq T}S_t>H\right\}}
# MAGIC \right]
# MAGIC \]
# MAGIC
# MAGIC em que:
# MAGIC
# MAGIC - \(V_0\) é o preço atual da opção;
# MAGIC - \(r\) é a taxa livre de risco;
# MAGIC - \(\mathbb{Q}\) representa a medida neutra ao risco;
# MAGIC - \(\mathbb{E}^{\mathbb{Q}}\) representa o valor esperado sob essa medida.
# MAGIC
# MAGIC Além do preço, o projeto calcula:
# MAGIC
# MAGIC - intervalo de confiança da simulação;
# MAGIC - probabilidade de knock-out;
# MAGIC - P&L sob diferentes cenários;
# MAGIC - Delta;
# MAGIC - Gamma;
# MAGIC - Vega;
# MAGIC - Rho;
# MAGIC - Theta.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Dinâmica utilizada para o Ibovespa
# MAGIC
# MAGIC O modelo assume que o Ibovespa segue um movimento browniano geométrico sob a medida neutra ao risco:
# MAGIC
# MAGIC \[
# MAGIC dS_t
# MAGIC =
# MAGIC (r-q)S_t\,dt
# MAGIC +
# MAGIC \sigma S_t\,dW_t
# MAGIC \]
# MAGIC
# MAGIC em que:
# MAGIC
# MAGIC - \(S_t\) é o nível do índice no instante \(t\);
# MAGIC - \(r\) é a taxa livre de risco;
# MAGIC - \(q\) representa o dividend yield ou custo de carregamento;
# MAGIC - \(\sigma\) é a volatilidade;
# MAGIC - \(W_t\) é um movimento browniano.
# MAGIC
# MAGIC A simulação discretizada utiliza:
# MAGIC
# MAGIC \[
# MAGIC S_{t+\Delta t}
# MAGIC =
# MAGIC S_t
# MAGIC \exp
# MAGIC \left[
# MAGIC \left(r-q-\frac{1}{2}\sigma^2\right)\Delta t
# MAGIC +
# MAGIC \sigma\sqrt{\Delta t}\,Z
# MAGIC \right]
# MAGIC \]
# MAGIC
# MAGIC com:
# MAGIC
# MAGIC \[
# MAGIC Z\sim N(0,1)
# MAGIC \]
# MAGIC
# MAGIC O projeto também utiliza uma correção por **Brownian Bridge** para estimar a possibilidade de o índice atravessar a barreira entre duas datas simuladas.
# MAGIC
# MAGIC Uma call vanilla de Black–Scholes é empregada como variável de controle para reduzir a variância da estimativa de Monte Carlo.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Limitações e riscos da opção
# MAGIC
# MAGIC A call down-and-out normalmente custa menos que uma call vanilla porque pode ser extinta antes do vencimento. Essa redução de preço vem acompanhada de riscos adicionais.
# MAGIC
# MAGIC ### Risco de knock-out
# MAGIC
# MAGIC Uma queda temporária do Ibovespa pode extinguir definitivamente a opção, mesmo que o índice se recupere e termine acima do strike.
# MAGIC
# MAGIC ### Dependência da trajetória
# MAGIC
# MAGIC O valor da opção não depende apenas dos preços inicial e final. Todo o caminho percorrido pelo Ibovespa durante o contrato é relevante.
# MAGIC
# MAGIC ### Risco de gap
# MAGIC
# MAGIC O índice pode atravessar a barreira de forma abrupta, especialmente durante crises, divulgação de notícias ou abertura do mercado.
# MAGIC
# MAGIC ### Hedge instável
# MAGIC
# MAGIC Delta e Gamma podem mudar intensamente quando o Ibovespa se aproxima da barreira. Isso dificulta o hedge e pode exigir rebalanceamentos frequentes.
# MAGIC
# MAGIC ### Exposição à volatilidade
# MAGIC
# MAGIC Uma volatilidade maior aumenta a possibilidade de valorização do índice, mas também aumenta a probabilidade de atingir a barreira inferior. Por isso, o comportamento do Vega pode ser complexo perto da barreira.
# MAGIC
# MAGIC ### Liquidez
# MAGIC
# MAGIC Opções exóticas podem possuir mercado secundário limitado, dificultando a saída ou a transferência da posição.
# MAGIC
# MAGIC ### Risco de contraparte
# MAGIC
# MAGIC Quando a opção é negociada no mercado de balcão, existe o risco de a contraparte não cumprir suas obrigações financeiras.
# MAGIC
# MAGIC ### Especificação contratual
# MAGIC
# MAGIC O contrato deve definir claramente:
# MAGIC
# MAGIC - nível da barreira;
# MAGIC - fonte oficial do índice;
# MAGIC - horário de observação;
# MAGIC - frequência de monitoramento;
# MAGIC - tratamento de feriados;
# MAGIC - eventos de mercado;
# MAGIC - existência ou não de rebate.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Limitações do modelo na vida real
# MAGIC
# MAGIC Este projeto possui finalidade educacional e utiliza hipóteses simplificadoras.
# MAGIC
# MAGIC ### Volatilidade constante
# MAGIC
# MAGIC O modelo considera uma única volatilidade durante todo o contrato. Na prática, existe uma superfície de volatilidade que varia conforme strike e vencimento.
# MAGIC
# MAGIC ### Taxa de juros constante
# MAGIC
# MAGIC A taxa livre de risco é considerada constante. Uma implementação profissional utilizaria uma curva completa de juros e fatores de desconto para cada prazo.
# MAGIC
# MAGIC ### Dividend yield constante
# MAGIC
# MAGIC O dividend yield ou carry é simplificado. Na prática, deve ser estimado a partir dos componentes do índice, dos dividendos esperados e dos preços dos contratos futuros.
# MAGIC
# MAGIC ### Ausência de saltos
# MAGIC
# MAGIC O movimento browniano geométrico produz trajetórias contínuas. Mercados reais apresentam gaps e saltos que podem ser especialmente importantes para opções com barreira.
# MAGIC
# MAGIC ### Distribuição lognormal
# MAGIC
# MAGIC O modelo não representa completamente assimetria, curtose, caudas pesadas e ou mudanças de regime observadas nos retornos financeiros.
# MAGIC
# MAGIC ### Parâmetros não calibrados
# MAGIC
# MAGIC Os valores atuais de juros, volatilidade e carry são ilustrativos. Eles ainda não foram calibrados com dados reais de mercado da B3.
# MAGIC
# MAGIC ### Custos não considerados
# MAGIC
# MAGIC O modelo não inclui:
# MAGIC
# MAGIC - bid-ask spread;
# MAGIC - corretagem;
# MAGIC - tributos;
# MAGIC - custos de hedge;
# MAGIC - impacto de mercado;
# MAGIC - custos de funding;
# MAGIC - custos de liquidez.
# MAGIC
# MAGIC ### Risco de contraparte e ajustes de valor
# MAGIC
# MAGIC Não são considerados ajustes como CVA, DVA, FVA, MVA ou outros componentes associados ao risco de contraparte e ao custo de financiamento.
# MAGIC
# MAGIC ### Ruído de Monte Carlo
# MAGIC
# MAGIC O resultado é uma estimativa estatística. Mesmo com técnicas de redução de variância, existe erro amostral, representado pelo erro-padrão e pelo intervalo de confiança.
# MAGIC
# MAGIC ### Aproximação das gregas
# MAGIC
# MAGIC As gregas são calculadas por diferenças finitas. Portanto, dependem do tamanho dos choques utilizados e também podem apresentar ruído de Monte Carlo.
# MAGIC
# MAGIC ### Monitoramento da barreira
# MAGIC
# MAGIC O Brownian Bridge reduz o risco de não identificar cruzamentos entre duas datas simuladas, mas sua validade depende das hipóteses de continuidade e normalidade do modelo.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 6. Evoluções necessárias para uma aplicação profissional
# MAGIC
# MAGIC Para utilização real em uma área de risco de mercado ou precificação, seriam necessárias as seguintes evoluções:
# MAGIC
# MAGIC 1. calibrar a curva de juros DI;
# MAGIC 2. estimar o carry por meio do futuro de Ibovespa;
# MAGIC 3. construir uma superfície de volatilidade implícita;
# MAGIC 4. comparar o Monte Carlo com uma solução analítica independente;
# MAGIC 5. considerar volatilidade local, volatilidade estocástica ou modelos com saltos;
# MAGIC 6. realizar testes de convergência e estabilidade numérica;
# MAGIC 7. implementar backtesting e validação independente;
# MAGIC 8. incorporar custos de hedge, liquidez e contraparte;
# MAGIC 9. criar cenários históricos e hipotéticos de estresse;
# MAGIC 10. registrar versões, parâmetros, resultados e trilhas de auditoria.
# MAGIC
# MAGIC O objetivo deste notebook é apresentar uma primeira implementação transparente e reproduzível de uma opção com barreira, integrando precificação, sensibilidades e análise de cenários no ambiente Databricks.
# MAGIC

# COMMAND ----------

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