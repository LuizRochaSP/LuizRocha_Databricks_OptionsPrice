# Inventário de Modelos Implementados - Quantitative Pricing & Risk Dashboard

## Família A — Single Barrier Options (100% Completa, com Rebate)
Temos as 8 variações clássicas de barreira simples, combinando Call/Put, Up/Down e In/Out.
Todas aceitam **rebate** pago no vencimento via `price_single_barrier` (`pricing_extensions.py`):
no Knock-Out o rebate é pago se a barreira for tocada; no Knock-In, se ela nunca for tocada.
Com rebate zero, o precificador genérico reproduz exatamente os precificadores dedicados (teste de regressão).
As 4 funções Knock-Out originais (`price_down_and_out_call`, `price_up_and_out_call`, `price_down_and_out_put`,
`price_up_and_out_put`) também aceitam `rebate` e delegam ao motor genérico, com resultado idêntico.

| Tipo | Call | Put | Status |
| :--- | :--- | :--- | :--- |
| **Down-and-Out** | DOC | DOP | Implementado (com rebate) |
| **Up-and-Out** | UOC | UOP | Implementado (com rebate) |
| **Down-and-In** | DIC | DIP | Implementado (com rebate) |
| **Up-and-In** | UIC | UIP | Implementado (com rebate) |

## Família B — Double Barrier Options (100% Completa)
Teto e chão monitorados simultaneamente (`pricing_double.py`). O motor registra, por caminho e por passo,
o toque em cada barreira (Brownian Bridge com um único uniforme particionado em p_L e p_U), o que permite
precificar estruturas que dependem da **ordem** dos toques. Call e Put; DKO/DKI aceitam rebate.

| Estrutura | Status | Benchmark analítico |
| :--- | :--- | :--- |
| **Double Knock-Out (DKO)** | Implementado (com rebate) | Série de Ikeda-Kunitomo (1992) |
| **Double Knock-In (DKI)** | Implementado (com rebate) | Paridade exata DKO + DKI = vanilla |
| **Dupla barreira KI/KO** | Implementado (4 convenções) | Ver tabela abaixo |

As estruturas com uma barreira Knock-In e outra Knock-Out (em lados opostos do spot) não seguem uma
única convenção de mercado: o que acontece depende da **ordem** dos toques. Por isso o precificador
`price_double_barrier_option` recebe o parâmetro `convention` (no Desk: seletor "Convenção de Ordem KI/KO"):

| `convention` | Evento mestre | Comportamento | Benchmark analítico |
| :--- | :--- | :--- | :--- |
| `KI_first` (KIKO) | Knock-In | Só nasce se tocar o KI; o KO mata a qualquer momento, antes ou depois da ativação | Identidade exata: KO simples(H_KO) − DKO(L, U) |
| `KO_after_KI` (KOKI) | Knock-In | Só nasce se tocar o KI; o KO só é monitorado após a ativação | Markov forte: densidade do 1º toque em H_KI × Reiner-Rubinstein |
| `KO_first` | Knock-Out | Nasce viva como Knock-Out; se o KI vier primeiro o KO deixa de existir (vira vanilla); se o KO vier primeiro, morre | Densidade de saída do túnel pelo KO (série de Fourier) × vanilla |
| `sequential` | Ambos | Só paga se tocar o KI e depois o KO (duplo toque sequencial) | Identidade exata: Knock-In simples − KOKI |

Propriedades verificadas nos testes: KIKO ≤ KOKI ≤ Knock-In simples; KO simples < KO_first < vanilla;
KOKI + Sequential = Knock-In simples; as densidades de saída integram exatamente 1 − P(Double No-Touch).
Toques nas duas barreiras no mesmo passo de simulação contam como KO (convenção conservadora).

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
| **Range Accrual** (túnel, Down, Up) | Implementado (com taxa de accrual) | Valor exato: média das probabilidades lognormais por fixing |

O Range Accrual usa 1 fixing por passo de simulação (no Desk: 1 por dia útil do prazo). Com `accrual_rate`
(% a.a.) paga o cupom de mercado Notional × taxa × T × fração dos fixings na faixa; sem ela, Notional × fração. Os binários
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
- **27 estratégias** no seletor do Structuring Desk, mais as variantes Up de Collar, Call Spread, Put Spread e Strangle e o rebate configurável nas 8 barreiras simples.
- Cobertura: Família A completa com rebate, Família B completa (DKO, DKI e dupla barreira KI/KO com 4 convenções de ordem), Família C completa (Down, Up e Knock-In), Família D completa nos binários de toque simples/duplos e Range Accrual.
- O que falta: os exóticos de terceira geração (Parisian, Step, Soft).
