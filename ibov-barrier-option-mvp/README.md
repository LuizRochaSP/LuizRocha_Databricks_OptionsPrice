# Precificação de Opção com Barreira no Ibovespa

Projeto quantitativo em Python, preparado para execução local e em Databricks, para precificar uma call down-and-out sem rebate sobre o Ibovespa. Mais precisamente, um pipeline end-to-end que consome dados públicos da B3, constrói superfície de volatilidade implícita, precifica uma opção com barreira por Monte Carlo com variável de controle e monitoramento Brownian Bridge, valida a qualidade dos dados e do modelo com controles auditáveis, e calcula métricas de risco de mercado (gregas, VaR, ES) — tudo reprodutível, testado e versionado. Cada execução estabelece uma cadeia de confiança criptográfica via SHA-256 (função de hash determinística que gera uma impressão digital única de 256 bits para cada arquivo, permitindo verificar sua integridade sem acessar o conteúdo) entre os artefatos de preço, validação e risco, garantindo rastreabilidade total. A data de mercado é selecionada automaticamente como D-1 via calendário de feriados da B3, sem parâmetros manuais. O projeto inclui 75 testes unitários cobrindo domínio, reprodutibilidade e casos extremos, com replay determinístico do motor de precificação para auditoria.

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
│   ├── 03_analise_risco.py
│   └── 04_cenarios_stress.py
├── cadernos/                       # material explicativo (não operacional)
│   ├── caderno_equacoes_completo.py
│   ├── caderno_equacoes_barrier_pricing.py
│   ├── caderno_equacoes_curvas_mercado.py
│   └── caderno_equacoes_risco.py
└── templates/                      # templates de entrada
    └── stress_scenarios.xlsx       # cenários de stress test (1 aba por cenário)
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

5. **`rotinas/04_cenarios_stress.py`** — Cenários e Stress Test:
   - Consome um conjunto coerente de RESULTS, VALIDATION e RISK aprovados
   - Verifica proveniência: schema, `FAIL=0`, `approved_for_risk`, SHA-256 e referências entre arquivos
   - Lê cenários de uma planilha Excel (`templates/stress_scenarios.xlsx`), uma aba por cenário
   - Cada aba define choques de spot (%), volatilidade (abs.) e taxa (bp), lado da posição (`COMPRADO`/`VENDIDO`) e `QtdPontos`
   - Reutiliza o motor de Monte Carlo existente com números aleatórios comuns entre base e cenário
   - Se o spot estressado atinge ou cruza a barreira, a opção é zerada (knock-out sem rebate)
   - Calcula P&L em pontos do índice: `P&L = (preço_estressado - preço_base) * QtdPontos` (comprado) ou sinal oposto (vendido)
   - Aba com entradas inválidas registra mensagem na própria aba sem interromper os demais cenários
   - Gera `STRESS_YYYYMMDD_HHMMSS.xlsx` (abas dos cenários + aba `RESUMO`) e `STRESS_YYYYMMDD_HHMMSS.json` (auditoria completa)
   - O Excel de entrada é preservado; não há sobrescrita de arquivos anteriores

### Módulo de stress test (`src/ibov_barrier/stress.py`)

Motor reutilizável para reprecificação sob choques:

- `ScenarioInput` — define nome, choques (spot %, vol abs., rate bp), lado e `QtdPontos`
- `SimulationParams` — replay dos parâmetros de Monte Carlo da execução base
- `verify_provenance` — verifica a cadeia de confiança RESULTS -> VALIDATION -> RISK (schema, SHA-256, `FAIL=0`, `approved_for_risk`)
- `run_scenario` — aplica choques, reprecifica com CRN e calcula P&L em pontos do índice
- `run_all_scenarios` — executa todos os cenários com números aleatórios comuns

> O VaR/ES inicial usa choques lognormais de spot e aproximação delta-gamma. Não é VaR
> histórico e não inclui choques conjuntos de volatilidade e curva.
>
> Cenários e stress testing são implementados no módulo 04 (ver abaixo).

## Validações implementadas

### Referência operacional D−1 B3

Os notebooks 00–03 usam `ibov_barrier.market_date`: a referência é a última
sessão B3/BVMF estritamente anterior à data de execução em America/Sao_Paulo.
Pastas de D0 ou futuras são ignoradas. Não há fallback para uma fotografia antiga.
Exemplo: execução em 22/09/2026 usa mercado de 21/09/2026; os nomes RESULTS,
VALIDATION e RISK mantêm o timestamp da execução, não a data de mercado.

Dependência fixada: `exchange-calendars==4.13.2` (calendário BVMF). Os notebooks
declaram essa dependência no ambiente Databricks; se necessário, configure-a nas
dependências do Serverless antes de executar. Sem o calendário ou fora de sua
cobertura, a rotina deve parar. Não substitua por calendário bancário ou weekdays.
Revisar periodicamente contra os comunicados oficiais da B3, especialmente
feriados extraordinários e mudanças anuais. Fonte de conferência para 2026:
[Ofício B3 054/2025-VNC](https://www.b3.com.br/data/files/21/F3/6B/17/6FAEA9105B12E5A9AC094EA8/CL%20054-2025-VNC%20CALENDARIO%20DE%20FERIADOS%20EM%202026%20E%20FUNCIONAMENTO%20DA%20B3%20EM%2018022026%20QUARTAFEIRA%20DE%20CINZAS_EN.pdf).

No 00, START_DATE e END_DATE iguais a None selecionam D−1 automaticamente.
Intervalos explícitos continuam disponíveis para coleta histórica, mas não podem
ultrapassar D−1. Datas sem sessão são puladas; falha em sessão esperada interrompe.
No 01, MARKET_DATE explícita somente é aceita se coincidir com D−1; None é o padrão.
O 02 exige RESULT produzido hoje e mercado D−1, em vez da maior pasta disponível.
O 03 reconfirma D−1 e a igualdade da data de mercado entre RESULTS e VALIDATION.

Antes de precificar/validar, exige-se cada ZIP com nome exato, integridade CRC e
publicação XML selecionada com data de referência compatível (`RptDtAndTm` para
IN, `TradDt` para IR/PR/SPRD). No PR verificam-se os registros IBOV consumidos
pelo pricer: o arquivo também contém outros mercados, que podem ter datas distintas.
A publicação selecionada é a de maior timestamp interno do ZIP, como no 01.
Datas de vencimento/criação não são usadas como data de mercado.
Dados ausentes, corrompidos ou incompatíveis bloqueiam sem alterar arquivos.

Verificação local: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v`.

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
