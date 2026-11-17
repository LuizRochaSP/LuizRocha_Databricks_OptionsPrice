# Inventário de Modelos Implementados - Quantitative Pricing & Risk Dashboard

## Família A — Single Barrier Options (100% Completa, com Rebate)
Temos as 8 variações clássicas de barreira simples, combinando Call/Put, Up/Down e In/Out.
Todas aceitam **rebate** pago no vencimento via `price_single_barrier` (`pricing_extensions.py`):
no Knock-Out o rebate é pago se a barreira for tocada; no Knock-In, se ela nunca for tocada.
Com rebate zero, o precificador genérico reproduz exatamente os precificadores dedicados (teste de regressão).

| Tipo | Call | Put | Status |
| :--- | :--- | :--- | :--- |
| **Down-and-Out** | DOC | DOP | Implementado (com rebate) |
| **Up-and-Out** | UOC | UOP | Implementado (com rebate) |
| **Down-and-In** | DIC | DIP | Implementado (com rebate) |
| **Up-and-In** | UIC | UIP | Implementado (com rebate) |

## Família B — Double Barrier Options (Não Implementada)
Devido à complexidade de monitorar duas barreiras com payoff de opção (Knock-In/Knock-Out cruzados), o motor ainda não possui essas estruturas.

| Estrutura | Status |
| :--- | :--- |
| Double Knock-Out (DKO) | Pendente |
| Double Knock-In (DKI) | Pendente |
| Knock-In / Knock-Out (KIKO) | Pendente |
| Knock-Out / Knock-In (KOKI) | Pendente |

## Família C — Barrier-Embedded Structures (Completa: Down, Up e Knock-In)
Estruturas combinadas mais usadas em mesa de tesouraria. Collar, Spreads e Strangle aceitam
`direction="down"` (padrão) ou `direction="up"`; Straddle e Forward têm versões Knock-Out e Knock-In.

| Estrutura | Down | Up | Descrição |
| :--- | :--- | :--- | :--- |
| **Knock-Out Straddle** | Implementado | — | Call KO + Put KO, mesmo strike (aposta na volatilidade) |
| **Knock-In Straddle** | Implementado | Implementado (biblioteca) | Call KI + Put KI: só nasce se tocar a barreira |
| **Barrier Strangle** | Implementado | Implementado | Call KO + Put KO (strikes diferentes) |
| **Barrier Risk Reversal** | Implementado | — | Compra Call KO − Vende Put KO (direcional) |
| **Barrier Collar** | Implementado | Implementado | Compra Put KO − Vende Call KO (proteção barata) |
| **Barrier Call Spread** | Implementado | Implementado | Compra Call KO − Vende Call KO |
| **Barrier Put Spread** | Implementado | Implementado | Compra Put KO − Vende Put KO |
| **Knock-Out Forward** | Implementado | — | Forward sintético que morre no crash |
| **Knock-In Forward** | Implementado | Implementado (biblioteca) | Forward sintético que só nasce no crash (KI + KO Forward = forward vanilla) |

## Família D — Range-Dependent e Digitais (Completa nas variantes de barreira simples e dupla)
Binários pagos no vencimento. Cada produto tem benchmark analítico exibido no Structuring Desk.

| Estrutura | Status | Benchmark analítico |
| :--- | :--- | :--- |
| **One-Touch Up / Down** | Implementado | Primeira passagem (princípio da reflexão) |
| **No-Touch Up / Down** | Implementado | Complemento exato do One-Touch |
| **Double One-Touch** | Implementado | Complemento do Double No-Touch |
| **Double No-Touch** | Implementado | Série de Fourier (Browniano com drift entre duas barreiras absorventes) |
| **Range Accrual** (túnel, Down, Up) | Implementado | Valor exato: média das probabilidades lognormais por fixing |

O Range Accrual usa 1 fixing por passo de simulação (no Desk: 1 por dia útil do prazo). Os binários
de dupla barreira usam Brownian Bridge com probabilidade de saída por passo p_L + p_U (os termos
omitidos exigem cruzar as duas barreiras no mesmo passo e são desprezíveis).

## Modelos Adicionais (Exóticos)
Ainda não implementados: exigem mudanças profundas no motor de simulação.

| Modelo | Status |
| :--- | :--- |
| Parisian Barriers | Pendente |
| Step Barriers | Pendente |
| Soft Barriers | Pendente |

## Resumo Executivo
- **24 estratégias** no seletor do Structuring Desk, mais as variantes Up de Collar, Call Spread, Put Spread e Strangle e o rebate configurável nas 8 barreiras simples.
- Cobertura: Família A completa com rebate, Família C completa (Down, Up e Knock-In), Família D completa nos binários de toque simples/duplos e Range Accrual.
- O que falta: Family B (barreiras duplas com payoff de opção) e os exóticos de terceira geração (Parisian, Step, Soft).
