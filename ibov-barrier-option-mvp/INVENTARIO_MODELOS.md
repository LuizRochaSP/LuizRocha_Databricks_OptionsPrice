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

## Modelos Adicionais (Exóticos dependentes da trajetória) — Completos
Módulo `pricing_path_barriers.py`. Todos aceitam Call/Put, barreira Down/Up, Knock-Out/Knock-In e rebate
pago no vencimento. Cada um tem um benchmark semi-analítico independente do Monte Carlo.

| Modelo | Regra do evento de barreira | Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Parisian** | `window` fechamentos CONSECUTIVOS além da barreira (1 fechamento por dia útil) | Cadeia de Markov exata do monitoramento discreto (log-preço × contador de dias) | Implementado |
| **ParAsian** (`cumulative=True`) | `window` fechamentos no TOTAL além da barreira | Mesma cadeia, contador sem reset | Implementado |
| **Step** | Nível muda em datas pré-definidas: `schedule = [(0, H1), (t1, H2), ...]` | Densidade de transição morta na barreira (reflexão com drift) encadeada por degrau | Implementado |
| **Soft (janela)** | Barreira monitorada só dentro de `windows = [(t_ini, t_fim), ...]` | Mesma quadratura por segmentos (barreira ligada/desligada) | Implementado |

Identidades verificadas nos testes: Parisian com `window=1` = barreira simples discreta (mesmos caminhos);
Step com um único nível e Soft com janela [0, T] = barreira simples com Brownian Bridge (mesmos caminhos);
quadratura por segmentos com um nível = Reiner-Rubinstein (erro < 0,01%); Knock-In + Knock-Out = vanilla.
Step e Soft usam Brownian Bridge com um nível de barreira por passo; a data de mudança/janela é aplicada pelo
ponto médio de cada passo (o Desk arredonda as datas para o dia útil mais próximo).
Obs.: na literatura (Hart-Ross, Haug) "soft barrier" também designa o knock-out proporcional dentro de uma
faixa; o projeto segue a definição do roadmap (monitoramento em janelas, ou partial-time barrier).

## Resumo Executivo
- **30 estratégias** no seletor do Structuring Desk, mais as variantes Up de Collar, Call Spread, Put Spread e Strangle e o rebate configurável nas 8 barreiras simples.
- Cobertura: Família A completa com rebate, Família B completa (DKO, DKI e dupla barreira KI/KO com 4 convenções de ordem), Família C completa (Down, Up e Knock-In), Família D completa nos binários de toque simples/duplos e Range Accrual, e os exóticos dependentes da trajetória (Parisian/ParAsian, Step e Soft) com Call/Put, Down/Up e Out/In.
- Roadmap de 10 lacunas concluído. Extensões naturais: rebate pago no toque e Parisian com monitoramento contínuo.
