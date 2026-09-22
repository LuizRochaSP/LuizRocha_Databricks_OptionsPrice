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

## Estrutura do Projeto

Os notebooks estão organizados em duas subpastas dentro de `databricks/`:

```text
databricks/
├── rotinas/                        # fluxo operacional diário
│   ├── 00_coleta_dados_b3.py
│   ├── 01_main_option_pricer.py
│   ├── 02_validacao_pos_pricing.py
│   └── 03_analise_risco.py
└── cadernos/                       # material explicativo (não operacional)
    ├── caderno_equacoes_completo.py
    ├── caderno_equacoes_barrier_pricing.py
    ├── caderno_equacoes_curvas_mercado.py
    └── caderno_equacoes_risco.py
```

### Fluxo operacional diário (`databricks/rotinas/`)

Notebooks para execução operacional (ordem de execução):

1. **`rotinas/00_coleta_dados_b3.py`** — Download automático dos arquivos diários da B3 (IN, IR, PR, SPRD)
2. **`rotinas/01_main_option_pricer.py`** — Pipeline principal: leitura de dados B3, construção de curvas de mercado, superfície de volatilidade implícita e geração do resultado diário (`RESULTS_superficieVol_YYYYMMDD_HHMMSS.csv`)
3. **`rotinas/02_validacao_pos_pricing.py`** — Validação de qualidade dos dados e resultados do pricer. Exige execução do `rotinas/01_main_option_pricer` no mesmo dia.
4. **`rotinas/03_analise_risco.py`** — Calcula gregas, DV01, convexidade de taxa, VaR e Expected Shortfall somente para um resultado aprovado pelo notebook 02.

### Material explicativo e de desenvolvimento (`databricks/cadernos/`)

Notebooks de referência (não precisam ser executados no fluxo operacional):

- **`cadernos/caderno_equacoes_completo.py`** — Referência completa de equações e fundamentos teóricos
- **`cadernos/caderno_equacoes_barrier_pricing.py`** — Monte Carlo para opções com barreira, variáveis de controle, Brownian Bridge
- **`cadernos/caderno_equacoes_curvas_mercado.py`** — Construção de curvas DI e dividend yield implícito
- **`cadernos/caderno_equacoes_risco.py`** — Convenções e equações de gregas, DV01, convexidade de taxa, VaR e Expected Shortfall

### Saída do pipeline

O notebook `01_main_option_pricer` gera e salva automaticamente o arquivo:

```
/Workspace/Users/<username>/ibov-barrier-results/RESULTS_superficieVol_YYYYMMDD_HHMMSS.csv
```

Este arquivo contém:
- Metadata da execução (spot, target strike/maturity, taxas r/q, volatilidade interpolada)
- Catálogo de opções, preços, curvas, filtros aplicados, volatilidades implícitas e pontos da superfície
- Resultado da call vanilla e da down-and-out, desconto da barreira, probabilidade de knock-out, erro-padrão e IC 95%
- Diagnósticos da simulação (caminhos, passos, seed, monitoramento, timestamp e data de mercado)

O notebook `02_validacao_pos_pricing`:
- Lê automaticamente o resultado mais recente
- Valida que foi gerado **hoje**
- Verifica integridade dos dados, curvas, preços e superfície
- Reconcilia `metadata`, `pricing_result` e `simulation_diagnostics`
- Repete deterministicamente a precificação com o mesmo motor e a mesma seed
- Consolida controles em `PASS`, `WARN` e `FAIL` e interrompe a execução quando houver qualquer `FAIL`
- Interrompe a execução se o `01_main_option_pricer` não foi executado no dia corrente
- Persiste `VALIDATION_superficieVol_YYYYMMDD_HHMMSS.json`, com decisão, contagens de controles e SHA-256 do `RESULTS` aprovado

O notebook `03_analise_risco`:
- exige `FAIL=0` e decisão `APROVADO` ou `APROVADO COM RESSALVAS`;
- verifica se a validação corresponde ao mesmo `RESULTS` por nome e SHA-256;
- calcula Delta, Gamma, Vega, Theta, Rho, DV01 e convexidade de taxa;
- calcula VaR e Expected Shortfall model-based por aproximação delta-gamma do risco de spot;
- carrega e exibe as ressalvas de qualidade produzidas pelo notebook 02;
- persiste `RISK_superficieVol_YYYYMMDD_HHMMSS.json` fora da Git Folder.

> O VaR/ES inicial usa choques lognormais de spot e aproximação delta-gamma. Não é VaR
> histórico e ainda não inclui choques conjuntos de volatilidade e curva. Cenários e
> stress testing serão implementados em uma etapa posterior.

## Validações implementadas

- a opção com barreira não pode valer mais que a call vanilla;
- barreira já violada implica preço zero;
- barreira muito baixa converge para a call vanilla;
- o preço aumenta com o spot e diminui quando a barreira sobe;
- resultados são reproduzíveis com semente fixa.
- parâmetros não finitos e arrays aleatórios inválidos são rejeitados pelo motor;
- metadata, resultado de pricing e diagnóstico da simulação devem ser internamente consistentes;
- a reexecução determinística deve reproduzir preço, erro-padrão, probabilidade de knock-out e IC.

> A reexecução no notebook de validação comprova integração, persistência e
> reprodutibilidade. Ela não substitui uma validação independente do modelo. Um benchmark
> analítico de Reiner–Rubinstein permanece como evolução planejada.

## Próximas etapas

1. ✅ Calibrar a curva DI e o carry implícito com futuro de Ibovespa
2. ✅ Calibrar volatilidade pela superfície de opções
3. Acrescentar fórmula fechada de Reiner–Rubinstein como benchmark
4. Implementar rebate, knock-in, up-and-out e paridade in/out
5. Gerar P&L explain, cenários de estresse e tabelas Delta para Databricks
