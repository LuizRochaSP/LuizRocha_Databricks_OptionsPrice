# Precificação e Risco de Opções Exóticas: Ibovespa e S&P 500

> **Situação em 24/11/2026: em produção no Databricks.**
> - **Ativos:** o Ibovespa, com dados da B3, e o S&P 500, com dados da ThetaData. O S&P 500 liquida em US$ ou em
>   R$ (Quanto/Compo).
> - **Rodada diária:** todo dia útil, dois jobs coletam o pregão anterior (D-1), montam a superfície de
>   volatilidade, precificam o contrato de referência, validam o resultado e calculam o risco.
> - **Dashboard:** um Databricks App mostra o resultado de cada ativo e traz o Structuring Desk, com 30 estratégias.
> - **Feature Freeze:** terminou em 23/11/2026, quando o motor multiativos entrou no `main`.

## Resumo

O projeto é uma esteira completa de precificação e risco de opções exóticas sobre índices de ações, escrita em
Python e operada no Databricks. Ela faz seis coisas:

1. **Coleta** os dados de mercado do pregão anterior (D-1) de cada bolsa: os arquivos públicos da B3 para o Ibovespa
   e a cadeia de opções SPX/SPXW da ThetaData para o S&P 500.
2. **Monta a superfície de volatilidade implícita**, com a curva de juros r(T) e a de dividendos q(T) tiradas do
   próprio mercado: DI1 e futuro de Ibovespa no Brasil, paridade put-call nos Estados Unidos.
3. **Precifica** um contrato de referência, uma call down-and-out. O cálculo é por Monte Carlo com ponte browniana
   e variável de controle, conferido contra a fórmula fechada de Reiner-Rubinstein.
4. **Valida** dados, superfície e preço com controles PASS, WARN e FAIL. Com qualquer FAIL, a esteira para.
5. **Mede o risco:** gregas, DV01, convexidade de taxa, VaR e Expected Shortfall (99%, 1 dia). Os cenários de
   stress rodam sob demanda.
6. **Mostra o resultado** num dashboard com uma página por ativo. O Structuring Desk precifica na hora 30
   estratégias (barreiras simples e duplas, estruturas combinadas, digitais, Parisian, Step e Soft); no S&P 500,
   em US$ ou em R$.

Cada execução deixa três arquivos encadeados por SHA-256 (RESULTS → VALIDATION → RISK). A data de mercado é
sempre o D-1 do calendário da bolsa, sem parâmetro manual.

### Em números

| Item | Valor |
| :--- | :--- |
| Ativos | 2: Ibovespa e S&P 500 (este também liquidado em R$: Quanto, Compo e Convertida) |
| Estratégias no Structuring Desk | 30, mais as variantes Up e o rebate configurável |
| Notebooks operacionais | 11 (`00` a `11`; não há `05`) |
| Jobs diários | 2 agendados, mais 1 disparado pelo botão do dashboard |
| Monte Carlo de produção | 200.000 caminhos × 252 passos, semente fixa |
| Benchmark analítico | erro do Monte Carlo contra Reiner-Rubinstein abaixo de 0,02% (S&P 500 em 23/11/2026: 0,0035%) |
| Histórico do S&P 500 | 80 pregões, de 03/08 a 23/11/2026 |
| Testes unitários | 309, todos verdes em 24/11/2026 |

## 1. Ativos cobertos

| | Ibovespa | S&P 500 |
| :--- | :--- | :--- |
| Fonte | B3: arquivos públicos IN, IR, PR e SPRD | ThetaData: cadeia EOD de SPX e SPXW |
| Calendário e D-1 | B3, America/Sao_Paulo | NYSE, America/New_York |
| Prazo | T da superfície em dias corridos/365; r e q no prazo em dias úteis/252 | ACT/365 com hora: vencimento AM (09:30 ET) ou PM (16:00 ET) |
| Juros r(T) | curva DI1 do SPRD, flat forward | paridade put-call, com uma reta r(T) robusta |
| Dividendos q(T) | futuro de Ibovespa (IND) | paridade put-call: q = r − ln(F/S)/T |
| Spot | fechamento do índice (IR) | implícito pela paridade (padrão) ou fechamento oficial |
| Preço de cada opção | mid ou último negócio (book raso) | sempre o mid |
| Contrato de referência | call down-and-out: K 155.000, H 140.000, 6 meses | call down-and-out: K 95% e H 85% do spot, 6 meses |
| Etapas | `00` → `01` → `02` → `03` | `07` → `11` → `03` (`ativo=spx`); `07` → `09` |
| Arquivos | `*_superficieVol_<carimbo>` | `*_spx_superficie_<carimbo>` e `RESULTS_quanto_<carimbo>` |

A camada multiativos (`src/ibov_barrier/markets/`) separa o que é de cada mercado do que é comum. Para acrescentar
um índice basta escrever o adaptador de dados dele; a superfície, o motor, o risco e o Desk são os mesmos. Detalhes
em [`ARQUITETURA_MULTIATIVOS.md`](ARQUITETURA_MULTIATIVOS.md).

## 2. Rodada diária

```text
IBOV     (job "Pipeline Diário B3 — Barrier Option", segunda a sexta, 08:00)
         00 coleta B3 ──> 01 superfície e preço ──> 02 validação ──> 03 risco

S&P 500  (job "opcoes-pipeline-spx", terça a sábado, 10:00)
         07 superfície e preço ──> 11 validação ──> 03 risco (ativo=spx)
                                └─> 09 Quanto/Compo e vol local
```

| Job | Agenda (Brasília) | Etapas | Papel |
| :--- | :--- | :--- | :--- |
| `Pipeline Diário B3 — Barrier Option` | segunda a sexta, 08:00 | `00` → `01` → `02` → `03` | rodada diária do Ibovespa |
| `opcoes-pipeline-spx` | terça a sábado, 10:00 | `07` → `11` → `03`; `07` → `09` | rodada diária do S&P 500. Roda de terça a sábado porque processa o pregão de Nova York do dia anterior |
| `opcoes-pipeline-ibov` | sem agenda | `00` → `01` → `02` → `03` | botão "Rodar pipeline" do dashboard |

Os jobs rodam em serverless, a partir do Git folder de produção (`main`). O serverless repete uma vez a tarefa que
falha. As tarefas do S&P 500 têm timeout de 90 minutos.

## 3. Notebooks (`databricks/rotinas/`)

| Notebook | O que faz | Grava | Onde roda |
| :--- | :--- | :--- | :--- |
| `00_coleta_dados_b3` | Baixa IN, IR, PR e SPRD do D-1 (ou de um intervalo). Confere cada ZIP (nome, CRC e data interna) e grava de forma atômica, sem deixar arquivo pela metade | `data/AAAA-MM-DD/` | job do IBOV |
| `01_main_option_pricer` | Lê o cadastro e os preços, monta r(T) e q(T), filtra, inverte Black-Scholes, monta smiles e superfície e precifica a call down-and-out de referência | `RESULTS_superficieVol_*` | job do IBOV |
| `02_validacao_pos_pricing` | Controles por bloco: integridade, cadastro e liquidez, curvas, não arbitragem, paridade, cobertura da superfície e replay determinístico do Monte Carlo. Decide APROVADO, APROVADO COM RESSALVAS ou REPROVADO | `VALIDATION_superficieVol_*` (só sem FAIL) | job do IBOV |
| `03_analise_risco` | Exige validação aprovada do mesmo RESULTS (nome e SHA-256). Calcula gregas, DV01, convexidade de taxa, VaR e ES delta-gamma (100.000 cenários, 99%, 1 dia). Widget `ativo` = `ibov` ou `spx` | `RISK_*` | jobs do IBOV e do S&P 500 |
| `04_cenarios_stress` | Lê os cenários de `templates/stress_scenarios.xlsx` (choques de spot, vol e taxa; lado e quantidade), reprecifica com os mesmos números aleatórios e calcula o P&L | `STRESS_*.xlsx` e `.json` | manual |
| `06_coleta_anbima` | Protótipo antigo de renda fixa (ETTJ da ANBIMA, LTN, LFT, NTN-B). A renda fixa nova fica no branch `feature/renda-fixa-publica` | — | manual |
| `07_spx_superficie_d1` | Cadeia EOD de SPX/SPXW, limpeza, paridade (forward, desconto, spot, r e q), vol implícita e superfície. Precifica a call down-and-out de referência e guarda a cadeia do dia | `RESULTS_spx_superficie_*` e `spx_dados/brutos/` | job do S&P 500 |
| `08_spx_historico_taxa` | Refaz a fotografia do 07 em todos os pregões desde 03/08/2026. Compara o método antigo de taxa (`atm`) com o novo (`smooth`) e com o box spread | `spx_dados/resumo_diario.parquet` | manual |
| `09_quanto_compo_spx` | S&P 500 pago em R$: Quanto, Compo e Convertida, com e sem barreira, e vol local de Dupire. Usa r BRL da DI1 (ou da ANBIMA), PTAX e vol do câmbio e correlação semanais | `RESULTS_quanto_*` | job do S&P 500 |
| `10_validacao_adaptador_ibov` | Confere item a item que o adaptador IBOV do motor multiativos reproduz o 01 (só leitura) | — | manual |
| `11_spx_validacao` | O 02 do S&P 500: integridade, superfície, forwards e carry, preço (Monte Carlo × Reiner-Rubinstein e replay) | `VALIDATION_spx_superficie_*` (só sem FAIL) | job do S&P 500 |

Os notebooks de `databricks/cadernos/` explicam as equações (barreiras, curvas e risco) e não fazem parte da rodada
diária. `FX__Charts.ipynb` e `test_fixed_income.py` são auxiliares.

## 4. Motor de precificação (`src/ibov_barrier/`)

### 4.1 Monte Carlo base

O ativo segue

$$
dS_t=(r-q)S_t\,dt+\sigma S_t\,dW_t,
$$

e o payoff do contrato de referência (call down-and-out) é

$$
e^{-rT}(S_T-K)^+\mathbf{1}_{\{\min_{0\leq t\leq T}S_t>H\}}.
$$

- **Produção:** 200.000 caminhos × 252 passos (um por dia útil), semente 42.
- **Monitoramento:** `brownian_bridge` estima o toque entre dois passos e tira o viés da discretização. `discrete`
  olha só nas datas simuladas.
- **Variável de controle:** a call vanilla de Black-Scholes, que reduz a variância.
- **Gregas:** diferenças finitas centrais, com os mesmos números aleatórios.

### 4.2 Famílias de produtos

| Família | Produtos | Benchmark independente |
| :--- | :--- | :--- |
| A: barreira simples | as 8 combinações Call/Put × Down/Up × In/Out, com rebate | Reiner-Rubinstein |
| B: dupla barreira | Double Knock-Out e Knock-In (com rebate); KI/KO com 4 convenções de ordem | Ikeda-Kunitomo, paridade e densidades do primeiro toque |
| C: estruturas | Straddle, Strangle, Risk Reversal, Collar, Call e Put Spread, Forward (Down, Up e Knock-In) | composição das barreiras simples |
| D: digitais | One-Touch e No-Touch, Double One-Touch e No-Touch, Range Accrual | princípio da reflexão e séries de Fourier |
| Dependentes da trajetória | Parisian e ParAsian, Step, Soft (janela) | cadeia de Markov e quadratura por segmentos |

O inventário completo, com as convenções e as identidades testadas, está em
[`INVENTARIO_MODELOS.md`](INVENTARIO_MODELOS.md).

### 4.3 S&P 500 em reais (`pricing_quanto.py`)

| Produto | Payoff em R$ | Como precifica |
| :--- | :--- | :--- |
| Quanto | `X_fix · max(S_T − K, 0)` | drift do S&P 500 ajustado por −ρ·σ_S·σ_X e desconto em r BRL |
| Compo | `max(S_T·X_T − K_BRL, 0)` | Black sobre S·X com a vol combinada |
| Convertida | `X_T · max(S_T − K, 0)` | `X_0` vezes o preço em US$ (referência) |

Qualquer pricer do motor vira Quanto (a barreira é observada no próprio S&P 500). Um Monte Carlo de dois fatores
confere as fórmulas e cobre a Compo com barreira, que não tem fórmula fechada. A vol do câmbio e a correlação são
históricas, com retornos semanais: a PTAX fecha às 13:00 de Brasília e o S&P 500 às 16:00 de Nova York, e com
retornos diários essa diferença de horário puxa a correlação para zero.

### 4.4 Vol local (`local_vol.py`)

SVI por vencimento na variância total, sem arbitragem de calendário, e vol local de Dupire na forma de Gatheral, com
limites de 3% a 150%. O Monte Carlo com vol local reprecifica o smile do dia e é o preço de referência para as
barreiras no notebook 09.

### 4.5 Outros módulos

| Módulo | Função |
| :--- | :--- |
| `markets/` | contrato comum (`MarketSpec`, `MarketSnapshot`, `MarketAdapter`), adaptadores IBOV e SPX, limpeza, paridade, núcleo de vol idêntico ao do notebook 01, contrato de referência e validação do S&P 500 |
| `market_date.py` | D-1 por bolsa (B3, NYSE), no fuso de cada uma |
| `b3_arquivos.py` | integridade dos ZIPs da B3 e gravação atômica |
| `di_curve/` | curva DI1 (flat forward, spline, Nelson-Siegel-Svensson), swap DI × Pré, DV01, key-rate, PCA, conferência com a ANBIMA |
| `risk.py`, `stress.py` | VaR e ES delta-gamma; reprecificação em cenários com cadeia de proveniência |
| `results_bundle.py` | formato único dos arquivos RESULTS, lido pelo dashboard |
| `markets/credentials.py` | onde ler a chave da ThetaData (nunca no código nem no repositório) |
| `fixed_income/` | protótipo antigo de LTN, LFT e NTN-B (a versão nova fica no branch de renda fixa) |

## 5. Dashboard (Databricks App)

O app se chama `dashboard-renda-fixa` por razões históricas, mas é o app de **opções**. Roda Streamlit 1.38 no
Databricks Apps e lê os arquivos da pasta de resultados pelo SDK do Databricks.

- **Barra lateral:**
  - o combo de instrumento (Ibovespa ou S&P 500), que também fica na URL (`?ativo=ibov` ou `?ativo=spx`);
  - a lista de execuções, cada uma marcada como completa, sem risco ou não validada. O app abre na completa mais
    recente;
  - o botão do pipeline de cada ativo, com o andamento em horário de Brasília.
- **Página do Ibovespa (6 abas):** Resumo do Gestor, Superfície e Volatilidade, Curvas de Mercado, Motor de
  Precificação, Controle e Ingestão, Structuring Desk.
- **Página do S&P 500 (8 abas):** Resumo do Gestor, Superfície e Smiles, Taxa e Dividendos, Motor de Precificação,
  Controle e Ingestão, Quanto / Compo (BRL), Vol Local, Structuring Desk.
- **Structuring Desk:**
  - as 30 estratégias, com diagrama de payoff e, quando existe, o benchmark analítico;
  - com "Usar dados de mercado do dia", o Desk lê spot, r(T), q(T) e a vol no strike da última execução, com as
    mesmas regras do pipeline;
  - no S&P 500, também em R$ (Quanto).
- **Números:** no padrão brasileiro (ponto no milhar, vírgula no decimal).

Detalhes em [`dashboard/README.md`](dashboard/README.md).

## 6. Arquivos de resultado

Pasta `/Workspace/Users/<usuário>/ibov-barrier-results/`:

| Arquivo | Notebook | Conteúdo |
| :--- | :--- | :--- |
| `RESULTS_superficieVol_<AAAAMMDD_HHMMSS>.csv` | 01 | metadados, catálogo, preços, curvas, filtros, vols, superfície, vanilla e down-and-out, P(knock-out), erro-padrão, IC 95%, diagnóstico da simulação |
| `VALIDATION_superficieVol_<carimbo>.json` | 02 | decisão, contagem dos controles e SHA-256 do RESULTS aprovado |
| `RISK_superficieVol_<carimbo>.json` | 03 | gregas, DV01, convexidade, VaR, ES e as ressalvas da validação |
| `RESULTS_`, `VALIDATION_` e `RISK_spx_superficie_<carimbo>` | 07, 11 e 03 | o mesmo conjunto para o S&P 500 |
| `RESULTS_quanto_<carimbo>.csv` | 09 | Quanto, Compo, Convertida e vol local |
| `STRESS_<carimbo>.xlsx` e `.json` | 04 | P&L por cenário e auditoria |

Outras pastas:
- `data/AAAA-MM-DD/`, dentro do Git folder: os ZIPs da B3, fora do git;
- `/Workspace/Users/<usuário>/spx_dados/`:
  - `brutos/`: a cadeia de cada dia, em parquet;
  - `resumo_diario.parquet`: o histórico do 08;
  - `ptax.parquet`: o cache do 09.

## 7. Datas e cadeia de confiança

### D-1 por bolsa

Os notebooks usam `ibov_barrier.market_date`. A referência é a última sessão da bolsa estritamente anterior à data
de execução, no fuso da própria bolsa: B3 em America/Sao_Paulo, NYSE em America/New_York.

- Pastas de D0 ou de datas futuras são ignoradas. Não há fallback para uma fotografia antiga.
- **Exemplo:** uma execução em 22/09/2026 usa o mercado de 21/09/2026.
- **Nomes dos arquivos:** RESULTS, VALIDATION e RISK levam o carimbo da execução, não a data de mercado.
- **Fusos:** entre 00:00 de São Paulo e 00:00 de Nova York, os dois mercados enxergam D diferentes. O código trata
  isso.
- **Dependência fixada:** `exchange-calendars==4.13.2`. Sem o calendário, ou fora da cobertura dele, a rotina para.
  Não substitua por calendário bancário nem por dias da semana.
- **Feriados da B3:** revise periodicamente contra os comunicados oficiais. Fonte de conferência para 2026:
  [Ofício B3 054/2025-VNC](https://www.b3.com.br/data/files/21/F3/6B/17/6FAEA9105B12E5A9AC094EA8/CL%20054-2025-VNC%20CALENDARIO%20DE%20FERIADOS%20EM%202026%20E%20FUNCIONAMENTO%20DA%20B3%20EM%2018022026%20QUARTAFEIRA%20DE%20CINZAS_EN.pdf).

O que cada notebook faz com a data:

| Notebook | Regra |
| :--- | :--- |
| 00 | `START_DATE` e `END_DATE` iguais a `None` selecionam o D-1. Um intervalo explícito, para coleta histórica, não passa do D-1. Datas sem sessão são puladas; falha numa sessão esperada interrompe |
| 01 | `MARKET_DATE` explícita só é aceita se for o D-1 |
| 02 | exige RESULTS gerado hoje, com mercado D-1 |
| 03 | reconfirma o D-1 e a mesma data de mercado em RESULTS e VALIDATION |

### Integridade dos dados da B3

- Antes de precificar ou validar, cada ZIP precisa ter o nome exato, CRC íntegro e data interna compatível:
  `RptDtAndTm` no IN, `TradDt` no IR, PR e SPRD.
- No PR, a data é conferida nos registros do Ibovespa que o pricer usa, porque o arquivo traz outros mercados, que
  podem ter datas diferentes.
- Vale a publicação com o maior carimbo interno do ZIP.
- Dados ausentes, corrompidos ou incompatíveis bloqueiam a rodada sem alterar arquivos.
- O 00 baixa de novo a data que tiver ZIP faltando ou corrompido, e grava cada arquivo de forma atômica.

### Cadeia de confiança

| Etapa | O que garante |
| :--- | :--- |
| VALIDATION | só é gravado sem FAIL e leva o SHA-256 do RESULTS |
| RISK | recusa um RESULTS diferente do que foi validado |
| STRESS | confere RESULTS → VALIDATION → RISK antes de reprecificar |
| Replay | o 02 e o 11 refazem a precificação com a mesma semente e exigem o mesmo preço, erro-padrão, P(knock-out) e IC |

## 8. Validações do modelo

Propriedades verificadas nos testes e na rodada diária:

- a opção com barreira não vale mais que a call vanilla;
- barreira já violada implica preço zero;
- barreira muito baixa converge para a call vanilla;
- o preço sobe com o spot e cai quando a barreira sobe;
- Knock-In + Knock-Out = vanilla (paridade exata, com os mesmos caminhos);
- parâmetros não finitos e entradas aleatórias inválidas são rejeitados;
- metadados, preço e diagnóstico da simulação são consistentes entre si;
- o Monte Carlo bate com a fórmula fechada de Reiner-Rubinstein (erro abaixo de 0,02%) nos dois regimes, K ≥ H e
  K < H.

No S&P 500, o 11 também confere:
- r, q, forward e vol do contrato, reproduzidos a partir da superfície;
- a cobertura da superfície e os cortes por arbitragem;
- a distância entre o spot da paridade e o fechamento oficial;
- a distância entre o r implícito e a SOFR.

A reexecução comprova integração, persistência e reprodutibilidade. Ela não substitui uma validação independente
do modelo, papel que cabe aos benchmarks analíticos.

## 9. Testes

Verificação local:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

Em 24/11/2026: **309 testes, todos verdes**. Os testes de calendário, de curva DI e do adaptador IBOV precisam de
`exchange-calendars` e `lxml`, que estão no `pyproject.toml`. Os do dashboard precisam de Streamlit e Plotly
(`dashboard/requirements.txt`). Quando falta uma dessas bibliotecas, esses testes são pulados ou não carregam.

Na branch `feature/coe-pricing-risco` há mais 57 testes, em `test_produtos.py` (interface de produtos do COE).
O total vai a 366.

| Módulo de teste | Testes | O que cobre |
| :--- | ---: | :--- |
| `test_produtos.py` | 57 | interface de produtos do COE: adaptadores no mesmo número dos pricers, cenários vetoriais, carteira, Quanto e gregas por diferenças finitas |
| `test_pricing.py` | 40 | motor Monte Carlo, validação das entradas, reprodutibilidade, gregas |
| `test_markets_spx.py` | 30 | adaptador do S&P 500: normalização da ThetaData, prazos AM/PM, núcleo de vol idêntico ao do notebook 01, paridade, limpeza com problemas injetados |
| `test_extensions.py` | 23 | rebate, binários Down, estruturas Knock-In, Range Accrual, Double Touch |
| `test_double_barriers.py` | 19 | família B: DKO, DKI e as 4 convenções KI/KO contra benchmarks analíticos |
| `test_dashboard_multiativos.py` | 18 | contrato entre os notebooks 07 e 09 (gravação) e as abas do S&P 500 (leitura e desenho) |
| `test_di_curve/` | 17 | curva DI, convenções, curvas de mercado e swap |
| `test_market_date.py` | 16 | D-1 pelo calendário da B3 e integridade da fotografia de mercado |
| `test_quanto.py` | 15 | Quanto, Compo e Convertida; Monte Carlo de dois fatores contra as fórmulas fechadas |
| `test_path_barriers.py` | 14 | Parisian, Step e Soft contra quadraturas semi-analíticas |
| `test_fx_inputs.py` | 14 | PTAX, conversão de taxas, ETTJ da ANBIMA, SPRD e fechamentos da ThetaData |
| `test_dashboard_pipeline.py` | 13 | execução completa (RESULTS → VALIDATION → RISK), disparo do job e números no padrão brasileiro |
| `test_stress.py` | 12 | cenários de stress |
| `test_market_date_multi.py` | 11 | D-1 por bolsa, fusos de São Paulo e Nova York, feriados da NYSE |
| `test_local_vol.py` | 10 | SVI, Dupire e Monte Carlo com vol local |
| `test_dashboard_desk.py` | 10 | Structuring Desk com os dados de mercado do dia, nos dois ativos |
| `test_spx_referencia.py` | 9 | contrato de referência do S&P 500, cadeia do dia em disco e validação do notebook 11 |
| `test_rebate_accrual.py` | 8 | rebate nas funções Knock-Out originais e taxa de accrual |
| `test_risk.py` | 7 | VaR, ES e métricas de risco |
| `test_markets_ibov.py` | 6 | adaptador IBOV contra as próprias células do notebook 01 |
| `test_credentials.py` | 6 | chave da ThetaData: ordem das fontes, nunca exposta em log |
| `test_b3_arquivos.py` | 5 | ZIP da B3 truncado não passa por "já baixado" |
| `test_dashboard_app.py` | 3 | o app inteiro no AppTest do Streamlit, com uma pasta de resultados sintética |
| `test_analytical.py` | 2 | Reiner-Rubinstein |
| `test_barrier_types.py` | 1 | paridade e tipos de barreira |

## 10. Como executar

### Local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python -m unittest discover -s tests -v
python -m ibov_barrier.cli --paths 100000 --seed 42
```

O CLI precifica um contrato de exemplo, com 500.000 caminhos × 126 passos por padrão:

| Parâmetro | Valor |
|---|---:|
| Spot | 150.000 pontos |
| Strike | 155.000 pontos |
| Barreira | 120.000 pontos |
| Vencimento | 0,5 ano |
| Volatilidade | 22% a.a. |
| Taxa livre de risco | 12% a.a. |
| Dividend yield | 0% a.a. |

O S&P 500 local exige Python 3.12 e a biblioteca `thetadata`. A chave é gravada fora do repositório e do OneDrive
com `python tools/set_thetadata_key.py`, que nunca mostra a chave na tela.

### Databricks

- **Git folder de produção:** `LuizRocha_Databricks_OptionsPrice`, no `main`. Atualiza por Pull (fast-forward).
- **Jobs:** os da seção 2, em serverless.
- **Notebooks do S&P 500 (07, 08, 09 e 11):** precisam do cabeçalho de ambiente serverless
  `environment_version = "6"` (Python 3.12). Sem ele, o serverless usa Python 3.11 e a `thetadata` não instala.
- **Chave da ThetaData:** fica no Secret `quantita/thetadata_api_key` e nunca aparece em notebook.
- **App:** deploy a partir de `ibov-barrier-option-mvp/dashboard` no Git folder de produção. O `app.yaml` define a
  pasta de resultados e o nome dos dois jobs. O service principal do app precisa de CAN_MANAGE_RUN nos jobs do
  botão.

## 11. Limitações conhecidas

### Dividend yield implícito do Ibovespa abaixo do histórico

O q(T) inferido do futuro de Ibovespa (IND) por F = S·exp((r−q)·T) ficou em ~0,8% a.a. em 23 a 25/09/2026,
estável, com desvio-padrão de 5,87 bps. O dividend yield histórico (série NEFIN/USP) está em 5% a 7% a.a.

Três fontes confirmam que o subjacente do IND é o Ibovespa de preço, não o de retorno total:
- a especificação da B3;
- o campo `AsstDesc` do arquivo IN;
- o campo `Desc` do arquivo IN.

A fórmula F = S·(1+r_DI)^(DU/252)·e^(−q·DU/252) é a convenção de mesa, e `di_curve/market_curves.py` segue essa
convenção. O q baixo reflete o que o mercado precifica, não um erro de cálculo.

Hipóteses:
1. dividendos concentrados em abril e maio e em agosto a novembro;
2. expectativa de corte de proventos;
3. prêmio de liquidez ou carry no IND;
4. uma base de desconto diferente do DI1.

Impacto: o q baixo eleva o forward, o preço da call e a probabilidade de knock-out. Trocar só a contagem de dias
mudou o preço da barreira em +1,32%.

### Demais limitações

- **Juros do S&P 500 acima da SOFR:**
  - em 78 pregões (03/08 a 19/11/2026), o r implícito na paridade ficou de +75 a +115 bps acima da SOFR overnight,
    crescendo com o prazo;
  - o resultado é igual nos três métodos (10 pares do ATM, regressão ampla e box spread). É característica do
    mercado de opções, e o pipeline usa o r implícito, coerente com os preços.
- **Spot do S&P 500:** o EOD das opções sai por volta das 17:15 ET, depois do fechamento do índice (16:00). Por isso
  o padrão é o spot da paridade, que deixa o q estável. O fechamento oficial fica registrado no RESULTS.
- **Cortes por arbitragem no S&P 500:** em 23/11/2026, 19,3% dos pontos caíram nos filtros de vertical e butterfly,
  acima do limite de 15% do 11. O resultado foi um WARN, e o critério está em avaliação.
- **Dias de baixa cobertura:** são marcados, não descartados. Exemplos: 19 a 30/10 e 05/11/2026 no S&P 500.
- **VaR e ES:** são delta-gamma, com choques lognormais de spot. Não são históricos e não incluem choques conjuntos
  de vol e de curva.
- **Exercício:** o motor só aceita opções europeias. ETFs como SPY e QQQ ficam de fora.
- **Armazenamento:** os resultados são arquivos no workspace. As tabelas Delta por mercado ainda não foram feitas.

## 12. Histórico e próximos passos

| Data | Marco |
| :--- | :--- |
| set/2026 | rodada diária do Ibovespa em produção (coleta da B3, superfície, Monte Carlo com ponte browniana, validação e risco), com histórico da B3 desde 03/08/2026 |
| até 18/11/2026 | roadmap de 10 modelos concluído (famílias A a D, Parisian, Step e Soft) e Feature Freeze |
| 23/11/2026 | motor multiativos no `main`: S&P 500, Quanto/Compo e vol local |
| 24/11/2026 | dashboard com a página do S&P 500 e o Desk multiativo em produção; job diário do S&P 500 ativo |

Próximos passos:
1. versão nova do [`WORKING_PAPER.md`](WORKING_PAPER.md), hoje com o texto de 18/11/2026;
2. estudo de um framework de estruturados (COE): bond do emissor, pacote de opções e hedge;
3. tabelas Delta por mercado;
4. um único job agendado para o Ibovespa (hoje são dois: o agendado e o do botão);
5. revisão do limite de cortes por arbitragem do 11.

## 13. Projeto irmão: renda fixa pública

A renda fixa pública brasileira é um projeto separado:
- **Branch e pasta:** `feature/renda-fixa-publica`, pasta `renda-fixa-publica/`;
- **App:** `renda-fixa-publica`;
- **Conteúdo:** títulos públicos (LTN, NTN-F, NTN-B, LFT e NTN-C), DI1/DAP e swaps, com o PU conferido contra a
  ANBIMA.

Ele não importa nada deste pacote. O app de opções e o de renda fixa continuam separados de propósito.

## Documentos

| Documento | Conteúdo |
| :--- | :--- |
| [`INVENTARIO_MODELOS.md`](INVENTARIO_MODELOS.md) | todos os produtos, convenções e benchmarks |
| [`ARQUITETURA_MULTIATIVOS.md`](ARQUITETURA_MULTIATIVOS.md) | contrato entre mercados, regras do S&P 500, limpeza, Quanto e vol local |
| [`dashboard/README.md`](dashboard/README.md) | abas, Structuring Desk, pipeline pelo app e deploy |
| [`WORKING_PAPER.md`](WORKING_PAPER.md) | texto acadêmico do projeto (versão de 18/11/2026) |
| [`PLANO_COE_PRICING_RISCO.md`](PLANO_COE_PRICING_RISCO.md) | plano da ferramenta de precificação, estruturação e risco de COE, na visão do estruturador (branch `feature/coe-pricing-risco`) |
| `databricks/cadernos/` | equações de barreiras, curvas e risco |
