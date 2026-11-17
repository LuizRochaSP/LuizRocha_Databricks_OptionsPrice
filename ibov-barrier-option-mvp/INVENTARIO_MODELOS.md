# Inventário de Modelos Implementados - Quantitative Pricing & Risk Dashboard

## Família A — Single Barrier Options (100% Completa)
Temos as 8 variações clássicas de barreira simples, combinando Call/Put, Up/Down e In/Out.

| Tipo | Call | Put | Status |
| :--- | :--- | :--- | :--- |
| **Down-and-Out** | DOC | DOP | Implementado |
| **Up-and-Out** | UOC | UOP | Implementado |
| **Down-and-In** | DIC | DIP | Implementado |
| **Up-and-In** | UIC | UIP | Implementado |

## Família B — Double Barrier Options (Não Implementada)
Devido à complexidade matemática de monitorar duas barreiras simultaneamente no Monte Carlo, o motor atual não possui essas estruturas.

| Estrutura | Status |
| :--- | :--- |
| Double Knock-Out (DKO) | Pendente |
| Double Knock-In (DKI) | Pendente |
| Knock-In / Knock-Out (KIKO) | Pendente |
| Knock-Out / Knock-In (KOKI) | Pendente |

## Família C — Barrier-Embedded Structures (100% Completa nas variantes Down)
Temos as 7 estruturas combinadas mais usadas em mesa de tesouraria, todas implementadas com barreira inferior (Down).

| Estrutura | Status | Descrição |
| :--- | :--- | :--- |
| **Knock-Out Straddle** | Implementado | Call DOC + Put DOC (Aposta na volatilidade) |
| **Barrier Strangle** | Implementado | Call DOC + Put DOC (Strikes diferentes) |
| **Barrier Risk Reversal** | Implementado | Compra Call DOC - Vende Put DOC (Direcional) |
| **Barrier Collar** | Implementado | Compra Put DOC - Vende Call DOC (Proteção barata) |
| **Barrier Call Spread** | Implementado | Compra Call DOC - Vende Call DOC (Limita ganho e custo) |
| **Barrier Put Spread** | Implementado | Compra Put DOC - Vende Put DOC (Limita ganho e custo) |
| **Knock-Out Forward** | Implementado | Forward sintético que morre no crash |

## Família D — Range-Dependent e Digitais (Parcial)
Temos os binários de toque para cima (Up). As variantes Down e os modelos de faixa não foram implementados.

| Estrutura | Status | Descrição |
| :--- | :--- | :--- |
| **One-Touch Up** | Implementado | Paga um valor fixo se bater na barreira superior |
| **No-Touch Up** | Implementado | Paga um valor fixo se NUNCA bater na barreira superior |
| Range Accrual | Pendente | Pagamento proporcional aos dias na faixa |
| Double One-Touch | Pendente | Pagamento se tocar qualquer uma das duas |
| Double No-Touch | Pendente | Pagamento se não tocar nenhuma das duas |

## Modelos Adicionais (Exóticos)
Nenhum dos modelos avançados foi implementado, pois exigem mudanças profundas no motor de simulação.

| Modelo | Status |
| :--- | :--- |
| Parisian Barriers | Pendente |
| Step Barriers | Pendente |
| Soft Barriers | Pendente |

## Resumo Executivo
- Total de modelos prontos: 17 estratégias expostas no Structuring Desk.
- Cobertura: Família A completa, Família C completa (variante Down) e parte da Família D.
- O que ficou de fora: Rebate, Double Barriers, opções "Down" dos binários e os modelos exóticos de terceira geração.
