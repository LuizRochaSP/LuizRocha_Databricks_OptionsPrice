# Dashboard Renda Fixa — Databricks Apps

Dashboard Streamlit para visualizar os resultados do projeto de precificação de opção com barreira sobre Ibovespa. Roda como **Databricks App** — a solução oficial de hospedagem de aplicações web do Databricks.

## Arquitetura

- **Framework:** Streamlit
- **Hospedagem:** Databricks Apps (container dedicado)
- **Acesso a dados:** Databricks SDK (`WorkspaceClient.workspace.download`)
- **Fonte:** `/ibov-barrier-results/RESULTS_superficieVol_*.csv`
- **URL:** HTTPS permanente fornecida pelo Databricks

## Estrutura

```
dashboard/
├── app.py              # código do dashboard
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

## Seções do Dashboard

1. Resumo do Pricer
2. Curva de Juros (DI)
3. Dividend Yield q(T)
4. Superfície de Volatilidade (3D)
4b. Smile de Volatilidade por Vencimento
5. Diagnóstico Monte Carlo
5b. Visualização Monte Carlo (caminhos, convergência, payoff)
6. Validação e Qualidade (6 checks)
7. Exportação CSV
8. Histórico comparativo
9. Dados Brutos

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
