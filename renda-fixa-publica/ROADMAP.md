# Roadmap: Motor de Renda Fixa Pública Brasil

> Projeto independente em `renda-fixa-publica/`, branch `feature/renda-fixa-publica`.
> O protótipo antigo (`ibov-barrier-option-mvp/src/ibov_barrier/fixed_income/`) fica intocado; este motor
> o substitui sem depender dele.

## Situação em 23/11/2026

| # | Fase | Situação | Aceite |
| :--- | :--- | :--- | :--- |
| 1 | Títulos (LTN, NTN-F, NTN-B, NTN-B Principal, LFT) | Pronto | PU idêntico à ANBIMA em 49/49 títulos, dois dias reais |
| 2 | Dados (ANBIMA, BCB, Tesouro Direto) | Pronto | Cache, retries, falha explícita (sem mock) |
| 3 | Curvas (bootstrap pré e real, NSS, inflação implícita, z-spread) | Pronto | Bootstrap reprecifica os títulos; NSS dentro de 11 bps |
| 4 | Derivativos (DI1, DAP, FRA, swaps DI x Pré e DI x IPCA) | Pronto | Taxa par zera o swap; MtM de swap em andamento com CDI real |
| 5 | Risco (DV01, key-rate, cenários, carry, hedge, VaR, PCA) | Pronto | Soma dos key-rates = DV01 paralelo; hedge zera o DV01 |
| 6 | Notebooks 01-03 e dashboard | Pronto localmente | Rodam de ponta a ponta com dados reais; falta rodar no Databricks |
| 7 | NTN-C (IGP-M) | Pronto | PU idêntico à ANBIMA; cupom de 12% conferido no Tesouro Direto; VNA conferido com o IGP-M |
| 8 | Curva de ajustes de DI1 da B3 x pré dos títulos | Pronto | PU de ajuste reproduzido em 317/317 contratos; spread por contrato |

## Problemas do protótipo antigo, resolvidos aqui
1. LFT devolvia o VNA capitalizado; agora PU = VNA x cotação, com a taxa sendo o ágio/deságio.
2. NTN-B com cupons a cada 126 DU; agora nas datas reais (15/mai-15/nov ou 15/fev-15/ago).
3. Sem truncamentos; agora com as regras do Tesouro e da ANBIMA, validadas.
4. Sem testes; agora 52 testes `unittest`.
5. Mock silencioso de curva; agora dado indisponível levanta `DadoIndisponivel`.

## Próximos passos
- Rodar os notebooks 01-03 no Databricks e agendar o 01 diariamente (acumula o histórico oficial da ANBIMA).
- Validar a cada novo dia: o notebook 02 mostra quantos PUs batem com a ANBIMA.
- Opcional: DAP da B3 contra a curva real das NTN-B (os ajustes já são baixados), histórico do spread
  DI1 x títulos, projeção oficial de IPCA e IGP-M da ANBIMA, opções de IDI e de DI1, atribuição de P&L e
  crédito privado.
- Deploy do dashboard como app separado no Databricks Apps, quando o usuário decidir.
