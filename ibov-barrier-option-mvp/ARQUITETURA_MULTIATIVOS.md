# Arquitetura Multi-Ativos

> Documento de referência para levar o projeto de "pricer de Ibovespa" a um motor multi-ativos.
> Desenvolvido na branch `feature/sp500` e incorporado ao `main` em 23/11/2026 (fim do Feature Freeze).

## 1. Objetivo e princípio

Adicionar um índice (S&P 500, Nasdaq-100, Nikkei 225, KOSPI 200, Euro Stoxx 50, DAX...) deve exigir **apenas**
a camada de coleta e tratamento de dados desse mercado. Todo o resto é comum:

```
 Fornecedor de dados        Adaptador do mercado (específico)            Núcleo comum (agnóstico)
 ─────────────────────      ──────────────────────────────────           ─────────────────────────────────────
 B3 (XML IN/IR/PR/SPRD) ─┐  normalização -> cadeia padrão                superfície de volatilidade (VolSurface)
 ThetaData (EOD SPX)    ─┼─> calendário e D-1 no fuso da bolsa       ─>  motor Monte Carlo / analítico (30 estratégias)
 (futuros: JPX, KRX,    ─┘  limpeza de cotações                          risco (VaR/ES, gregas), stress
  Eurex...)                 forward, r(T), q(T)  ->  MarketSnapshot      Structuring Desk (Streamlit)
```

A fronteira é o `MarketSnapshot` (`src/ibov_barrier/markets/base.py`). Nada à direita dela sabe qual é o mercado,
exceto pelos metadados de exibição (moeda, multiplicador) que viajam no `MarketSpec`.

## 2. O contrato

### 2.1 `MarketSpec`: metadados do mercado

| Campo | IBOV | SPX | Uso |
| :--- | :--- | :--- | :--- |
| `currency` | BRL | USD | Desk, relatórios, Quanto |
| `exchange` | B3 | NYSE | chave do calendário e do fuso em `market_date.EXCHANGES` |
| `timezone` | America/Sao_Paulo | America/New_York | definição do D-1 e prazos |
| `close_time` | 18:00 | 16:00 | instante de avaliação |
| `exercise` | european | european | o motor só aceita europeias |
| `multiplier` | 1 | 100 | valor financeiro por ponto |
| `day_count` | BUS/252 | ACT/365 | prazo T usado na superfície |
| `rate_index` | DI1 | SOFR | curva livre de risco de referência |
| `roots` | IBOV | SPX, SPXW | séries de opções coletadas |
| `settlement_times` | - | AM 09:30, PM 16:00 | vencimento intradiário |

### 2.2 Cadeia padronizada (`CHAIN_COLUMNS`)

Uma linha por contrato: `root, expiration, strike, option_type (CALL/PUT), bid, ask, last, volume,
open_interest, settlement (AM/PM)`. Preços em pontos do índice. `validate_chain` confere colunas e tipos e
falha alto em vez de corrigir silenciosamente.

### 2.3 `MarketSnapshot`: o que o núcleo recebe

| Campo | Conteúdo |
| :--- | :--- |
| `market_date` | D-1 da bolsa (seção 3) |
| `spot`, `spot_source` | fechamento do índice; "fornecedor" ou "implícito (paridade)" |
| `forwards` | `ForwardPoint` por vencimento: forward, fator de desconto, nº de pares, resíduo |
| `term_structure` | r(T) e q(T) contínuos, flat forward entre vértices |
| `options` | cadeia limpa com `T, r, q, forward, log_moneyness, market_price, implied_vol` |
| `report` | `CleaningReport`: contratos restantes após cada etapa e avisos |

### 2.4 `MarketAdapter`

```python
class MarketAdapter(Protocol):
    spec: MarketSpec
    def market_date(self, now=None) -> date: ...
    def load_snapshot(self, market_date: date) -> MarketSnapshot: ...
```

O acesso ao fornecedor é injetado no adaptador (`fetch_chain(root, data) -> DataFrame`). Os testes usam dados
sintéticos sem rede, e trocar de fornecedor muda só a normalização.

## 3. Datas, fusos e D-1 (`market_date.py`)

**Regra:** o D-1 é definido **no fuso da bolsa**. A data de execução D é a data local da bolsa no instante da
execução; o D-1 é a última sessão do calendário dessa bolsa estritamente anterior a D.

- `expected_market_date(now, market="B3" | "NYSE", calendar=None)`. O padrão continua sendo B3, então todo
  o código do IBOV segue idêntico.
- Datetimes sem fuso são rejeitados. Datas puras são tratadas como datas locais da bolsa.
- O Brasil não tem horário de verão desde 2019; Nova York tem. Entre 00:00 de São Paulo e 00:00 de Nova York
  (1 hora no verão americano, 2 horas no inverno), os dois mercados enxergam D diferentes. Exemplo: em
  22/09/2026 às 00:30 BRT, o D-1 da B3 é 21/09 e o da NYSE é 18/09.
- Os feriados vêm de cada bolsa: Independence Day, Thanksgiving e Presidents Day na NYSE; Carnaval e Corpus
  Christi na B3. Em 16/02/2026 as duas fecham, por motivos diferentes.
- `lxml` passou a ser importado só dentro de `validate_snapshot` (a validação dos XMLs da B3). O cálculo do D-1
  não depende mais dele.
- **Agendamento sugerido no Databricks:** job do SPX às 08:00 BRT de D+0. O EOD da ThetaData sai por volta de
  17:15 ET de D-1, e o SOFR de D-1 é publicado às 08:00 ET de D+0 (só importa se a curva SOFR for usada; ver 4.3).

## 4. Adaptador SPX (`markets/spx.py`)

### 4.1 Séries e prazos

- **SPX** (mensal, 3ª sexta) liquida pela abertura do vencimento (AM, ~09:30 ET). **SPXW** (semanais e diárias,
  incluindo 3ª sexta) liquida pelo fechamento (PM, 16:00 ET).
- O prazo é ACT/365 com hora do dia: avaliação às 16:00 ET de D-1, vencimento às 09:30 ou 16:00 ET. No mesmo
  dia, um SPX AM vence 6,5 horas antes de um SPXW PM.
- Quando as duas séries vencem na mesma data, fica só a SPXW (PM, mais líquida e coerente com a avaliação no
  fechamento). Configurável em `CleaningConfig.preferred_roots`.

### 4.2 Forward e dividendos pela paridade put-call (`markets/parity.py`)

`C − P = DF·(F − K)`. A inclinação da reta de `C − P` em K dá `−DF`; com DF conhecido, o forward sai da
média de `K + (C−P)/DF` nos 10 strikes mais próximos do ATM com call e put cotadas.

- **Com curva externa** (`rate_curve`, SOFR/Treasuries): DF fixo pela curva.
- **Sem curva externa, método padrão `rate_method="smooth"`:**
  1. cada vencimento com T ≥ 3 meses (`min_T_rate`) mede a própria taxa pela regressão **ampla**, com todos
     os pares em ±10% do ATM (~60 strikes, contra ~50 pontos de faixa nos 10 pares);
  2. uma reta `r(T) = a + b·T` robusta (início Theil-Sen, descarte por MAD) é ajustada a essas taxas; as que
     ficam fora da curva são ignoradas no ajuste, com aviso;
  3. a curva vale para **todos** os vencimentos (constante fora da faixa de prazos ajustados). Abaixo de
     3 meses, `r = −ln(DF)/T` amplifica o ruído das cotações (em 17/11/2026 saíram taxas de −21% a +26% em
     vencimentos de 1 a 3 semanas), então esses prazos só usam a curva.
- **Método anterior `rate_method="atm"`:** regressão com os 10 pares do ATM em cada T ≥ 3 meses; prazos
  curtos usam a taxa do vencimento confiável mais curto. Fica disponível para comparação (notebook 08).
- **Vencimento incoerente:** resíduo acima de max(1,5; 10 × mediana do dia) descarta o vencimento, com aviso.
  O piso de 1,5 evita cortes marginais (2028-06-16 em 18/11/2026 saía com 1,004 contra limite 1,0); os casos
  ruins do histórico ficaram entre 5 e 29.
- **Taxa acima da SOFR:** em 78 pregões (03/08 a 19/11/2026), o r implícito ficou de +75 a +115 bps acima
  da SOFR overnight, crescendo com o prazo, igual nos três métodos (10 pares, regressão ampla, box spread).
  Comprar o box no ask ainda rendeu acima da SOFR em ~60% das observações de 6 a 18 meses. É característica
  do mercado de opções, não do método: o pipeline usa o r implícito, coerente com os preços.
- **Dia de baixa cobertura** (`markets/quality.py`): superfície com menos de 70% da mediana de pontos dos 20
  pregões anteriores não marcados é marcada (não descartada). Ex.: 19/10 a 30/10/2026 e 05/11/2026.
- `q(T) = r(T) − ln(F/S)/T` por vencimento. Não é preciso coletar dividendos.
- **Spot:** se o plano de dados não trouxer o fechamento do índice, ele é inferido do vencimento mais curto:
  `S = F·DF·exp(q·T)`, com q assumido de 1,3% a.a. O erro é da ordem de q·T, centésimos de % para uma ou duas
  semanas. Com o fechamento disponível, use `spot_source`.

### 4.3 Juros

A taxa implícita na paridade (taxa "box spread") fica acima do SOFR: em 17/11/2026 ficou entre 4,8% e 5,1%,
contra SOFR de 3,93%. Para a
superfície, o que importa é a coerência entre DF e F no mesmo vencimento, e a paridade garante isso. A curva
SOFR/Treasuries entra como `rate_curve` quando quisermos r(T) "oficial" no motor (por exemplo, no Quanto).

### 4.4 Normalização da ThetaData

API verificada em 18/11/2026 com a conta real (plano FREE, biblioteca `thetadata` 1.0.18, Python 3.12+), em
`ThetaDataSource`:

| Chamada | Retorno | Observações |
| :--- | :--- | :--- |
| `ThetaClient(api_key=..., dataframe_type="pandas")` | cliente | a biblioteca importa `python-dotenv` sem declarar a dependência: instalar junto |
| `option_history_eod(d, d, "SPXW", "*")` | ~20 mil linhas, 39 vencimentos | `symbol, expiration` (ISO), `strike` (dólares), `right` (CALL/PUT), `bid, ask, close, volume`, ... |
| `option_history_eod(d, d, "SPX", "*")` | ~10 mil linhas | mensais AM |
| `index_history_eod("SPX", d, d)` | `close` | fechamento oficial (7.340,53 em 17/11/2026) |
| `interest_rate_history_eod("SOFR", d, d)` | `rate` em % | 3,93 em 17/11/2026 |
| open interest | negado | exige plano pago; o filtro de OI fica desligado |

O EOD de D-1 é gerado por volta de 17:15-17:20 ET (coluna `created`). `normalize_thetadata_eod` também aceita o
formato da API REST v2 (strike em milésimos, C/P, YYYYMMDD). A série vem sempre da requisição, não da coluna
`symbol` do fornecedor.

**Chave da API** (`markets/credentials.py`): nunca fica no código nem no repositório. Ela é procurada nesta ordem:

| Ordem | Fonte | Uso |
| :--- | :--- | :--- |
| 1 | argumento `api_key=` | testes |
| 2 | variável `THETADATA_API_KEY` | sessão pontual |
| 3 | Databricks Secret `quantita/thetadata_api_key` | jobs e app no Databricks (só é consultado lá) |
| 4 | arquivo `THETADATA_KEY_FILE` ou `~/.thetadata/tdk.txt` | máquina local, fora do OneDrive e do git |

Para trocar a chave, nenhum código muda:

- **Local:** `python tools/set_thetadata_key.py`. Pede a chave sem exibi-la, grava com acesso só do usuário e
  se recusa a gravar em pasta sincronizada ou repositório. `--status` mostra de onde a chave está vindo
  (mascarada) e `--test` valida na API.
- **Databricks:** `databricks secrets put-secret quantita thetadata_api_key`.

O `.gitignore` bloqueia `thetadata_key/`, `.env` e `creds.txt` como segunda barreira.

### 4.5 Validação com dados reais (D-1 = 17/11/2026)

- 30.024 contratos recebidos (SPX + SPXW) e 8.223 pontos na superfície, em 45 vencimentos de 1 semana a
  19 meses. Os avisos de limpeza ficaram vazios.
- Volatilidade ATM: 18,3% em 3 meses e 18,7% em 1 ano. Skew: 25,0% em k = −0,10 e 27,9% em k = −0,20 / 6 meses.
- q(T) entre 0,4% e 1,0% a partir de 1 mês. No vencimento de 1 semana, q fica negativo (−2,8%) porque o
  fechamento do índice é às 16:00 ET e as opções de índice negociam até 16:15 ET. Na prática, o forward da
  paridade já embute o movimento desses 15 minutos. Isso não afeta o preço (o motor usa F e DF, coerentes
  entre si), mas desloca q nos prazos muito curtos.

## 5. Data Cleaning (`markets/cleaning.py`)

O preço de mercado é sempre o **mid**. No SPX, o último negócio costuma estar defasado do fechamento do índice
e não é usado como fallback, diferente do IBOV, onde o book é raso. Cada etapa fica registrada no
`CleaningReport`, no mesmo formato do `register_filter` do notebook IBOV.

| Estágio | Etapa | Padrão SPX | Por quê |
| :--- | :--- | :--- | :--- |
| 1. Cotações | Sem duplicatas, uma série por vencimento | SPXW preferida | AM/PM na mesma data distorcem a superfície |
| | Prazo | 7 a 730 dias | 0DTE e semanais curtíssimas: microestrutura e eventos |
| | Bid/ask presentes e não cruzados | `ask >= bid`, `ask > 0` | cotação inválida |
| | Bid mínimo | 0,05 (tick) | bid zero = sem mercado; preço só do lado vendedor |
| | Spread máximo | 25% do mid, ou até 0,20 em valor absoluto | ilíquidas; asas baratas têm spread relativo alto mas absoluto pequeno |
| | Posição em aberto/volume | desligado (0) | no SPX há strikes bem cotados sem negócio no dia |
| 2. Contra o forward | Limites de arbitragem | `max(DF(F−K),0) < C < DF·F` (puts análogo) | preços absurdos |
| | Só OTM | calls K ≥ F, puts K < F | ITM têm IV ruidosa (valor intrínseco domina) |
| | Moneyness | \|ln(K/F)\| ≤ 0,6 | asas extremas sem informação |
| | Vertical e butterfly | tolerância de 0,05 ponto | arbitragem estática; sai a cotação de maior spread relativo |
| 3. Smile | IV calculável e entre 3% e 200% | | inversão falhou ou valor fora do plausível |
| | Outliers | resíduo > 5 MAD da mediana móvel (janela 5) e > 2 pontos de vol | pontos isolados que entortam a interpolação |
| | Pontos por vencimento | ≥ 3 | vencimento sem smile |

Os parâmetros ficam em `CleaningConfig`. Outros índices reaproveitam o módulo com outro preset (tick, faixa de
prazo e spread próprios).

## 6. Núcleo comum (`markets/vol.py`)

Black-Scholes, limites de arbitragem, volatilidade implícita (brentq em [1e-4, 5], xtol 1e-10) e superfície
(interpolação linear em (ln(K/F), T), com fallback nearest fora do casco convexo) são **cópias exatas** das
seções 5 e 6 do notebook `01_main_option_pricer.py`, sem variáveis globais. O teste
`VolCoreMatchesIbovNotebook` compara 200 casos aleatórios com o código do notebook e exige igualdade exata.

## 7. Dados no Databricks (a implementar)

- Tabelas Delta com coluna `market` e particionadas por `market` e `market_date`:
  `options_clean`, `forwards`, `term_structure`, `vol_points`, `cleaning_report`, `pricing_results`.
- Um job de coleta por mercado (IBOV: notebook 00 atual; SPX: novo notebook) e um job comum de superfície e
  precificação, parametrizado por `market`.
- O app de produção continua lendo só o IBOV até a integração final (seção 10).

## 8. Quanto/Compo (`pricing_quanto.py`)

Opção sobre o S&P 500 paga em BRL. S = índice em USD, X = USDBRL (BRL por USD), rho = correlação dos
log-retornos de S e X. Sob a medida neutra ao risco em BRL, S tem drift `r_USD - q - rho sigma_S sigma_X` e
X tem drift `r_BRL - r_USD`.

| Produto | Payoff em BRL | Preço |
| :--- | :--- | :--- |
| Quanto | `X_fix * max(S_T - K, 0)`, K em USD | Black com forward `S e^{(r_USD - q - rho sigma_S sigma_X) T}`, vol sigma_S, desconto r_BRL |
| Compo | `max(S_T X_T - K_BRL, 0)`, K em BRL | Black sobre `S X`, drift `r_BRL - q`, vol `sqrt(sigma_S^2 + sigma_X^2 + 2 rho sigma_S sigma_X)` |
| Convertida | `X_T * max(S_T - K, 0)` | `X_0 * BS_USD`; não depende de rho nem de sigma_X (referência) |

- **Barreiras no Quanto:** como a barreira é observada no próprio S, `quanto_market_data` devolve um
  `MarketData` (rate = r_BRL, dividend_yield = r_BRL - drift Quanto) e qualquer pricer do motor vira
  Quanto; o preço é multiplicado por `X_fix`.
- **Monte Carlo de dois fatores** (`price_two_factor_mc`): S e X correlacionados, barreira knock-out em S
  (discreta ou ponte browniana) e o vanilla fechado como variável de controle. Valida as fórmulas e cobre a
  Compo com barreira, que não tem fórmula fechada.
- **Entradas:** sigma_S da superfície SPX (strike e prazo do contrato), r_USD e q do adaptador SPX, r_BRL da
  curva DI, sigma_X e rho históricos (`historical_fx_inputs`). O padrão é retorno semanal: o SPX fecha às
  17:00 ET e a PTAX às 13:00 BRT, e com retornos diários essa assincronia puxa a correlação para zero.
- **Notebook 09** (`09_quanto_compo_spx.py`): spot, r_USD, q e vol do SPX no vencimento listado mais próximo do
  prazo (cadeia em disco do notebook 08 ou ThetaData); r_BRL da curva DI1 do SPRD (notebook 00), com
  alternativa ANBIMA PRE (`pyettj`) e valor manual; PTAX do SGS 1; vol e correlação semanais em 2 anos
  (`markets/fx_inputs.py`). Mostra preços com e sem barreira, a referência analítica da DOC, a vol na
  barreira contra a vol no strike, e a sensibilidade a rho e à vol do câmbio, com relatório único no fim.
- **Ordem de grandeza** (S 7.000, K 7.100, T 0,75, r_USD 4,8%, q 0,9%, sigma_S 18%, X 5,40, r_BRL 13,5%,
  sigma_X 13%, rho -0,40): Quanto 2.585, Convertida 2.607, Compo (K = 7.100 x 5,40) 3.926 BRL por ponto.
  Com rho < 0 o ajuste do Quanto é positivo (+94 bps ao ano no exemplo).

## 9. Testes

| Arquivo | Testes | Cobre |
| :--- | ---: | :--- |
| `test_markets_spx.py` | 19 | normalização ThetaData, prazos AM/PM, núcleo idêntico ao notebook IBOV, paridade (F, DF, spot, r, q), limpeza com problemas injetados, recuperação exata do smile, cotações arredondadas ao tick |
| `test_quanto.py` | 15 | Quanto, Compo e Convertida: paridade, limites (rho = 0, sigma_X = 0), mapeamento para o motor, MC de dois fatores contra fórmula fechada e contra a DOC analítica (Quanto e Convertida com barreira), vol e correlação históricas |
| `test_markets_ibov.py` | 6 | adaptador IBOV contra as próprias células do notebook 01 sobre ZIPs sintéticos: leitura de IN/PR/IR, contagens por filtro, vol implícita por ticker e vol do alvo idênticas; contrato do `MarketSnapshot` |
| `test_local_vol.py` | 10 | SVI (ajuste e superfície), diagnóstico de corte por região, Dupire plano = vol constante, skew local mais inclinado que o implícito, MC com vol local reprecifica o smile e bate as fórmulas fechadas no caso plano, DOC mais barata com skew |
| `test_fx_inputs.py` | 13 | PTAX do SGS (parse, blocos, novas tentativas após 502, alternativa Olinda, feriados), conversão 252 DU -> contínua ACT/365, ETTJ em formatos variados, busca do SPRD, fechamentos da ThetaData |
| `test_credentials.py` | 6 | ordem das fontes da chave, limpeza do valor, chave nunca exposta em logs |
| `test_market_date_multi.py` | 11 | fusos SP×NY (verão e inverno americano), padrão B3 inalterado, feriados reais da NYSE e da B3 (estes rodam só com `exchange_calendars` instalado) |

A cadeia sintética é gerada por Black-Scholes com spot, juros, dividendos e smile conhecidos, e o pipeline
precisa recuperá-los. Problemas são injetados um a um (cotação cruzada, bid zero, spread largo, preço absurdo,
vertical negativa, butterfly), e cada um precisa sair na etapa certa.

## 10. Plano de fases

| Fase | Entrega | Status |
| :--- | :--- | :--- |
| 1 | Este documento: contrato, fusos, regras do SPX, limpeza | concluída |
| 2 | Pacote `markets/` (contrato, núcleo de vol, paridade, limpeza, adaptador SPX), `market_date` multi-fuso, testes sintéticos | concluída |
| 3 | API ThetaData verificada e ligada (`ThetaDataSource`); D-1 real de 17/11/2026 validado | concluída; falta acompanhar vários dias seguidos no Databricks |
| 4 | Adaptador IBOV sobre o notebook 01, com critério de aceite: mesmos números que o `main` para a mesma data | `markets/ibov.py` e notebook 10 prontos; teste local executa as células do notebook 01 e exige igualdade exata; falta rodar o 10 no Databricks com os XMLs reais |
| 5 | Tabelas Delta por mercado e jobs | a fazer |
| 6 | Quanto/Compo | motor, testes e notebook 09 rodados com dados reais (20/11/2026); vol local de Dupire, faixa de smile e política de rho adicionadas |
| 7 | Desk com seletor de mercado (moeda, multiplicador, calendário) | a fazer |

## 11. Integração final (concluída em 23/11/2026)

1. **Aceite:** suíte completa verde (250 testes); IBOV idêntico ao `main` (notebook 10: 22/22 itens);
   `market_date` igual ao do `main` em 12.937 comparações; notebooks 00 a 06 e 08 a 10 rodados no Databricks.
   O 07 não rodou no teste geral por falta de compute; a última execução bem-sucedida foi em 20/11/2026.
2. **Merge:** `feature/sp500` -> `main` por fast-forward (o `main` não tinha commits novos). A branch foi
   apagada depois do merge. Nada do que o dashboard lê mudou: notebooks 00 a 06 e `dashboard/` idênticos.
3. **Dashboard (branch `feature/dashboard-multiativos`):** o 07 grava `RESULTS_spx_superficie_*.csv` e o 09 grava
   `RESULTS_quanto_*.csv` em `ibov-barrier-results`, no formato do 01 (`ibov_barrier/results_bundle.py`). O app ganha
   as abas S&P 500, Quanto / Compo (BRL) e Vol Local (`dashboard/multiativos.py`; contrato testado em
   `tests/test_dashboard_multiativos.py`). Deploy só por decisão explícita. Rollback do app: redeploy de `baadf7d`.

## 12. Como adicionar um novo índice

1. `MarketSpec` (moeda, bolsa, fuso, fechamento, exercício, multiplicador, day count, séries, liquidação).
2. Bolsa em `market_date.EXCHANGES` (código do `exchange_calendars`: XJPX, XKRX, XEUR/XETR...).
3. Normalização do fornecedor para `CHAIN_COLUMNS`.
4. Preset de `CleaningConfig` (tick, prazos, spread).
5. Testes sintéticos no molde de `test_markets_spx.py`.

Particularidades já conhecidas:

| Índice | Atenção |
| :--- | :--- |
| Nasdaq-100 (NDX/NDXP) | mesma lógica AM/PM do SPX |
| Nikkei 225 | opções da JPX com liquidação pelo SQ (abertura); fuso Asia/Tokyo sem horário de verão |
| KOSPI 200 | multiplicador alto; fuso Asia/Seoul |
| Euro Stoxx 50 | Eurex, liquidação ao meio-dia (média 11:50-12:00 CET), juros em €STR |
| DAX | índice de retorno total: q ≈ 0 na paridade (bom teste de sanidade) |
| ETFs (SPY, QQQ) | exercício americano: fora do escopo do motor europeu |

### Vol local de Dupire (fase 6, complemento)

`src/ibov_barrier/local_vol.py`: SVI bruto por vencimento na variância total w(k, T) (fatias com RMSE acima de 3
pontos de vol ficam fora), w linear em T com máximo acumulado (sem arbitragem de calendário) e vol local pela
fórmula de Dupire em variância total (Gatheral), com piso no denominador e limites de 3% a 150%. O Monte Carlo
de dois fatores aceita `local_vol=`: S usa a vol local em k = ln(S/F_usd(t)) a cada passo, o ajuste Quanto
passa a ser -rho sigma_loc sigma_X e a variável de controle vira S_T X_T, cujo valor esperado é conhecido sob
qualquer vol. O notebook 09 mostra, para cada produto, o preço com barreira com vol plana em K, no ponto
médio K/H, na barreira e com vol local, além do vanilla com vol local como checagem da calibração.

Política de rho no notebook 09: widget `janela_rho` (histórico completo, 52 semanas ou manual) define rho e
vol do câmbio; o relatório mostra o preço da call em cada estimativa (janelas, diário e extremos da móvel de
26 semanas).

### Adaptador IBOV (fase 4)

`src/ibov_barrier/markets/ibov.py` é cópia fiel das seções 2 a 6 do notebook 01: mesmos leitores em
streaming, mesmo preço (mid ou último), T em dias corridos/365, r e q pelo prazo em DU, mesmos filtros na
mesma ordem, mesma inversão de Black-Scholes e mesmo alvo (DU = round(T*252), griddata linear com nearest de
reserva). `IBOVAdapter.load_snapshot` devolve o `MarketSnapshot` padrão; `adapter.last` guarda catálogo,
preços, curvas e a tabela filtrada para auditoria. O notebook 10 lê o `RESULTS_superficieVol_*.csv` salvo pelo
notebook 01 do `main`, roda o adaptador nos mesmos ZIPs e compara item a item (contagens iguais; vols por
ticker até 1e-9, limite do `to_json` do 01; alvo até 1e-12; preços MC com a mesma semente até 1e-8).

Diagnóstico da vol local (notebook 09, seção 4b): fração cortada pelos limites por período do prazo e por faixa
de strike/forward, com o motivo (abaixo do piso, acima do teto ou denominador de Dupire).
