# Dashboard Renda Fixa — Databricks Apps

Dashboard Streamlit para visualizar os resultados do projeto de precificação de opção com barreira sobre Ibovespa. Roda como **Databricks App** — a solução oficial de hospedagem de aplicações web do Databricks.

## Arquitetura

- **Framework:** Streamlit
- **Hospedagem:** Databricks Apps (container dedicado)
- **Acesso a dados:** Databricks SDK (`WorkspaceClient.workspace.download`)
- **Fontes (pasta `ibov-barrier-results`):**
  - `RESULTS_superficieVol_*.csv` (notebook 01, IBOV)
  - `RESULTS_spx_superficie_*.csv` (notebook 07, S&P 500)
  - `RESULTS_quanto_*.csv` (notebook 09, Quanto/Compo e vol local)
- **URL:** HTTPS permanente fornecida pelo Databricks

## Estrutura

```
dashboard/
├── app.py              # código do dashboard (barra lateral, abas do IBOV, página do S&P 500)
├── desk.py             # Structuring Desk (IBOV e S&P 500, US$ ou Quanto em R$)
├── mercado.py          # spot, r(T), q(T) e vol no strike de cada ativo para o Desk
├── multiativos.py      # abas do S&P 500 (notebooks 07 e 09)
├── paineis.py          # painéis comuns aos dois ativos (pricer, risco, histórico, motor, checagens)
├── pipeline.py         # completude RESULTS -> VALIDATION -> RISK e job do pipeline
├── formato_br.py       # números no padrão brasileiro
├── app.yaml            # configuração do Databricks App
├── requirements.txt    # dependências
└── README.md
```

## Como Fazer Deploy

### Pré-requisitos

- Workspace Databricks com Apps habilitado
- Permissão para criar Apps
- Código commitado na branch `feature/streamlit-dashboard`

### Passo a Passo

1. No Databricks: menu lateral → **Compute** → **Apps**
2. Clique em **"Create app"**
3. Preencha:
   - **Name:** `dashboard-renda-fixa`
   - **Source:** Git repository
   - **Repo:** `LuizRochaSP/LuizRocha_Databricks_OptionsPrice`
   - **Branch:** `feature/streamlit-dashboard`
   - **Path:** `ibov-barrier-option-mvp/dashboard`
4. Clique em **"Create"**

Databricks Apps vai ler o `app.yaml`, instalar as deps do `requirements.txt`, rodar o `app.py` e gerar uma URL HTTPS.

## Abas do Dashboard

O combo **Tipo de Instrumento** da barra lateral escolhe o ativo, e só as abas dele rodam (o Streamlit
executa todas as abas a cada clique, então o outro ativo não é carregado nem desenhado). A URL guarda a
escolha: `?ativo=ibov` ou `?ativo=spx`.

**Opção com Barreira (IBOV)** — arquivos do notebook 01 (com 02 e 03):

1. **Resumo do Gestor** — KPIs do pricer (data de mercado, spot, preço da barreira, probabilidade de knock-out)
2. **Superfície e Volatilidade** — superfície 3D e smile por vencimento
3. **Curvas de Mercado** — curva DI e dividend yield q(T)
4. **Motor de Precificação** — diagnóstico e visualização do Monte Carlo (caminhos, convergência, payoff)
5. **Controle e Ingestão** — checks de validação e qualidade dos dados
6. **Structuring Desk** — simulador com 30 estratégias (ver abaixo)

**Opção com Barreira (S&P 500)** — arquivos dos notebooks 07, 11, 03 (`ativo=spx`) e 09:

1. **Resumo do Gestor** — Resumo do Pricer da call down-and-out de referência (strike 95%, barreira 85%, 6 meses: vanilla, down-and-out, Reiner-Rubinstein, P(knock-out)), mercado do dia (spot e de onde veio, vencimentos, pontos, vol ATM 1m/12m, r e q de 1 ano, SOFR, avisos), Risco & Gregas e histórico de execuções
2. **Superfície e Smiles** — superfície 3D, smiles nos prazos de referência, vol ATM por prazo, cortes de arbitragem e funil de limpeza
3. **Taxa e Dividendos** — r(T) e q(T) implícitos da paridade put-call e a tabela por vencimento
4. **Motor de Precificação** — o mesmo painel do IBOV (simulação, convergência, payoffs)
5. **Controle e Ingestão** — checagens do preço, os controles do notebook 11 e exportação
6. **Quanto / Compo (BRL)** — insumos (PTAX, r BRL, vol do câmbio, correlação), preços em BRL e em % do nocional, correlação móvel e sensibilidade a rho e à vol do câmbio (notebook 09)
7. **Vol Local** — preço com barreira com vol constante x vol local de Dupire, smile implícito x local, onde a vol local é cortada e ajuste SVI (notebook 09)
8. **Structuring Desk** — o mesmo simulador, em pontos do S&P 500 e US$, ou em R$ (Quanto)

Os painéis comuns aos dois ativos (Resumo do Pricer, Risco & Gregas, histórico, Motor e checagens) ficam em
`paineis.py`; as abas próprias do S&P 500, em `multiativos.py`. Sem arquivos do 07 ou do 09, elas mostram qual notebook rodar e o Desk funciona com valores de reserva.

### Structuring Desk (`desk.py` e `mercado.py`)

- **Valores iniciais** de spot, r, q e vol: do arquivo do ativo (01 para o IBOV, 07 para o S&P 500), no prazo padrão de 6 meses (vol ATM). Sem arquivo, valores de reserva fixos. O q, antes fixo em 2%, agora é um controle.
- **Usar dados de mercado do dia**: spot do arquivo, r(T) e q(T) no prazo escolhido e vol da superfície no strike (vol ATM nas estruturas sem strike). Regras iguais às do pipeline: vol linear em ln(K/F) x prazo com vizinho mais próximo fora dos dados (notebook 01); r do IBOV em flat-forward sobre a curva DI1; r e q do SPX da paridade, lineares entre vencimentos. No contrato de referência, o Desk reproduz o r, o q e a vol do 01 e a vol no strike do 09.
- **Liquidação em R$ (Quanto)**, só no S&P 500: drift ajustado por −rho·sigma_S·sigma_X, desconto em r BRL e preço em pontos vezes o câmbio fixado; payout, rebate e notional são digitados em R$. Câmbio, r BRL, vol do câmbio e correlação vêm da última execução do notebook 09.
- Com os mesmos parâmetros, o Desk do IBOV dá exatamente os preços de antes nas 30 estratégias.

## Configuração

A env var `RESULTS_BASE_PATH` aponta para a pasta dos CSVs no Workspace. Já configurada em `app.yaml`.

Para apontar para outra pasta, edite o `app.yaml` e faça redeploy.

## Diferenciais vs. dbtunnel

| | dbtunnel | Databricks Apps |
|---|---|---|
| Hospedagem | Container efêmero | Container dedicado |
| URL | Proxy temporário | HTTPS permanente |
| Hibernação | Sim | Não |
| Deploy | Runtime | UI + versionamento Git |

## Execução completa e pipeline

Cada execução só está completa com os três arquivos: `RESULTS_*`, `VALIDATION_*` (gravado só com
FAIL=0) e `RISK_*`. A barra lateral marca cada execução como completa, sem risco ou não validada, e o
app abre na completa mais recente. Quando falta informação, o app diz qual notebook não rodou e
oferece o botão do pipeline do ativo, que dispara o job e mostra o andamento até terminar. Não
dispara uma segunda rodada enquanto houver uma ativa.

| Ativo | Arquivos | Job (`app.yaml`) | Etapas |
|---|---|---|---|
| IBOV | `*_superficieVol_<ts>` | `opcoes-pipeline-ibov` (`PIPELINE_JOB_NAME`) | 00 -> 01 -> 02 -> 03 |
| S&P 500 | `*_spx_superficie_<ts>` | `opcoes-pipeline-spx` (`PIPELINE_JOB_NAME_SPX`) | 07 -> 11 -> 03 (`ativo=spx`); 07 -> 09 |

O service principal do app precisa de CAN_MANAGE_RUN nos dois jobs.

O combo de instrumento lê `?ativo=` da URL só na primeira carga da sessão: no Streamlit 1.38 (o do
Databricks Apps) um `index` recalculado a cada execução muda a identidade do widget e o clique se perdia.

Números no padrão brasileiro (ponto no milhar, vírgula no decimal): `formato_br.instalar(st)`.
