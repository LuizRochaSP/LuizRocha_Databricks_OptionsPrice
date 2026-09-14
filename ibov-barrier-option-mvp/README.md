# Precificação de Opção com Barreira no Ibovespa

Projeto quantitativo em Python, preparado para execução local e em Databricks, para precificar uma **call down-and-out sem rebate** sobre o Ibovespa.

## Objetivo

Estimar o valor justo e os principais riscos de uma opção que deixa de existir quando o índice toca uma barreira inferior. O motor usa Monte Carlo sob medida neutra ao risco, uma call Black–Scholes como variável de controle para redução de variância e duas formas de monitoramento:

- `discrete`: verifica a barreira somente nas datas simuladas;
- `brownian_bridge`: estima o toque entre duas datas consecutivas e reduz o viés de discretização.

O ativo segue

$$
dS_t=(r-q)S_t\,dt+\sigma S_t\,dW_t,
$$

e o payoff é

$$
e^{-rT}(S_T-K)^+\mathbf{1}_{\{\min_{0\leq t\leq T}S_t>H\}}.
$$

`q` representa o dividend yield/custo de carregamento do índice. Para uso real, juros, dividendos e volatilidade devem ser calibrados às curvas e à superfície de mercado.

## Contrato-base

| Parâmetro | Valor |
|---|---:|
| Spot | 150.000 pontos |
| Strike | 155.000 pontos |
| Barreira | 120.000 pontos |
| Vencimento | 0,5 ano |
| Volatilidade | 22% a.a. |
| Taxa livre de risco | 12% a.a. |
| Dividend yield/carry | 0% a.a. |
| Passos | 126 |
| Simulações | 500.000 |

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python -m ibov_barrier.cli
python -m unittest discover -s tests -v
```

Para uma execução rápida:

```bash
python -m ibov_barrier.cli --paths 100000 --seed 42
```

## Estrutura

```text
src/ibov_barrier/pricing.py       motor de precificação e gregas
src/ibov_barrier/cli.py           exemplo executável
databricks/01_barrier_pricing.py  notebook importável no Databricks
tests/test_pricing.py              testes de consistência financeira (unittest)
```

## Validações implementadas

- a opção com barreira não pode valer mais que a call vanilla;
- barreira já violada implica preço zero;
- barreira muito baixa converge para a call vanilla;
- o preço aumenta com o spot e diminui quando a barreira sobe;
- resultados são reproduzíveis com semente fixa.

## Próximas etapas

1. Calibrar a curva DI e o carry implícito com futuro de Ibovespa.
2. Calibrar volatilidade pela superfície de opções.
3. Acrescentar fórmula fechada de Reiner–Rubinstein como benchmark.
4. Implementar rebate, knock-in, up-and-out e paridade in/out.
5. Gerar P&L explain, cenários de estresse e tabelas Delta para Databricks.
