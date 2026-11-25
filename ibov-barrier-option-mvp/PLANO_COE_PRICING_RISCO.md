# Plano de desenvolvimento: precificação, estruturação e risco de COE (visão do estruturador)

- **Branch:** `feature/coe-pricing-risco`
  - criada a partir do `main` em `c952c71`;
  - renomeada de `feature/tabelas-var-americano` em 25/11/2026.
- **Data:** 25/11/2026
- **Situação:** A1 (interface de produto e adaptadores) implementado em 25/11/2026. Próximo: A2 (perna de título).
  As decisões 1 e 6 da seção 8 foram tomadas em 25/11.
- **Substitui:** a primeira versão deste plano (`PLANO_TABELAS_VAR_AMERICANO.md`, commit `73dbc8b`). Aquela versão
  tratava só das tabelas Delta, do VaR e do exercício americano.

## Resumo

O objetivo é transformar o projeto de um precificador de opções com barreira numa ferramenta de estruturação e risco
de COE (Certificado de Operações Estruturadas).

O plano começa pela visão do **estruturador**. É quem:
- desenha o produto;
- define o preço de emissão;
- faz o hedge;
- carrega o risco até o vencimento.

**Ponto de partida.** O motor atual já precifica a maior parte do pacote de opções de um COE. Faltam:
- a perna de título;
- a composição do produto e o cálculo da margem;
- o hedge;
- o risco de um livro de produtos.

**As três limitações da primeira versão.** O VaR paramétrico, o exercício só europeu e os resultados em arquivos
(seção 10 do working paper) continuam no plano. Agora entram como partes do caminho, na ordem em que o COE precisa
delas.

---

## 1. O que a ferramenta faz para o estruturador

1. **Montar o produto** a partir de um termsheet: subjacente, prazo, nominal, proteção do capital, participação, cap,
   barreiras, cupons e datas de observação.
2. **Calcular o valor justo** e decompor o preço de emissão em título, opções, custos, reservas e margem.
3. **Resolver o parâmetro livre** (participação, cap, cupom ou barreira) que entrega a margem alvo.
4. **Desenhar o hedge** inicial com instrumentos listados.
5. **Simular o hedge** ao longo da vida do produto e medir a margem que de fato se realiza.
6. **Medir o risco do livro:** gregas por subjacente e por vértice, VaR, stress e explicação diária do P&L.
7. **Acompanhar o ciclo de vida:** observações, cupons, autocall, eventos de barreira e liquidação.

**Fora do escopo por enquanto:**
- a visão do investidor, que entra depois e reaproveita o mesmo motor (seção 10);
- o exercício americano de subjacentes, como ações da B3 e ETFs (fase G);
- registro, contabilização e documentos regulatórios do produto.

---

## 2. O COE visto pelo estruturador

O emissor recebe o preço de emissão (100% do nominal), assume a obrigação de pagar o payoff e faz o hedge. Em valor
presente:

$$
\underbrace{100\%}_{\text{preço de emissão}}
= \underbrace{PV_{\text{título}}}_{\text{proteção}}
+ \underbrace{PV_{\text{opções}}}_{\text{pacote}}
+ \underbrace{C}_{\text{custos}}
+ \underbrace{R}_{\text{reservas}}
+ \underbrace{M}_{\text{margem}}
$$

- **Título.** Valor presente da parte garantida, descontado pela curva de funding do emissor (curva DI pré mais o
  spread do emissor). Quanto maiores o juro e o spread, mais barato o título e mais sobra para as opções.
- **Opções.** O pacote que gera a participação, precificado pelo motor atual com a superfície de vol do dia.
- **Custos.** Remuneração da distribuição e custos operacionais.
- **Reservas.** Cobrem três riscos:
  - o bid/ask do hedge;
  - o risco de modelo (por exemplo, vol de prazo longo extrapolada);
  - o risco de descontinuidade em barreiras e digitais.
- **Margem.** O que sobra. O solver (etapa A4) inverte a conta: fixa a margem e resolve o parâmetro do produto.

**Estruturas cobertas pelo plano:**

| Estrutura | Pacote de opções | No motor hoje |
| :--- | :--- | :--- |
| Capital protegido com participação | título + call (ou put) com participação | sim |
| Capital protegido com cap | título + call spread | sim |
| Tubarão | título + call up-and-out com rebate | sim (barreiras com rebate) |
| Cupom digital | título + digital | sim (digital, one-touch) |
| Intervalo | título + double no-touch ou range accrual | sim |
| Autocall | cupons condicionais, resgate antecipado automático e barreira final | não |
| Média | participação sobre a média de observações | não |
| Worst-of / cesta | pior desempenho entre Ibovespa e S&P 500 em reais | não; há Monte Carlo de dois fatores só para o Quanto |
| S&P 500 em reais | qualquer estrutura acima com ajuste Quanto | sim (`pricing_quanto.py`) |

---

## 3. O que já existe e o que falta

| Bloco | Já existe | Falta |
| :--- | :--- | :--- |
| Opções | barreiras simples e duplas com rebate, digitais, one/no-touch, Parisian, step, soft, Quanto/Compo, vol local; Monte Carlo com ponte browniana e variável de controle | autocall, média, worst-of, vol de prazo longo |
| Juros | `di_curve`: curva, interpoladores, swap DI x pré, DV01 por vértice (`bump_key`), PCA. `fixed_income`: LTN, NTN-B, LFT. Curva ANBIMA no notebook 09. | perna de título com o spread do emissor |
| Produto | pricers soltos, cada um com a sua assinatura | interface comum, composição de pernas, termsheet |
| Estruturação | Structuring Desk com 30 estratégias | decomposição do preço de emissão, margem, solver |
| Hedge | — | instrumentos listados, razões de hedge, simulação do hedge, reservas |
| Risco | gregas, DV01, convexidade, VaR delta-gamma só no spot, stress com reprecificação | reprecificação completa, fatores conjuntos, juros por vértice, spread, explicação do P&L, livro, VaR histórico |
| Dados | arquivos por rodada, com SHA-256 | livro de produtos, marcação diária, histórico consultável (tabelas) |

---

## 4. Regras que valem para todas as etapas

**Git**
- O trabalho acontece nesta branch. Rode `git fetch` no início de cada sessão, porque o `main` também recebe commits
  feitos pelo workspace do Databricks.
- Correções do `main` entram aqui com `git merge main`.
- Cada marco (seção 7) vai para o `main` separadamente, em merges pequenos.

**Números de produção**
- Os pricers atuais não mudam. Cada um ganha um adaptador para a interface nova, com teste "mesmo número" contra a
  função original.
- Toda estrutura nova tem uma referência independente: fórmula fechada, identidade ou um Monte Carlo independente.
- Mudança de metodologia entra com widget e com o padrão antigo ligado até ser validada. Depois é declarada no WP.

**Testes**
- Comando: `PYTHONPATH=src python -m unittest` (unittest, sem pytest).
- Todo módulo novo tem o seu arquivo em `tests/`.

**Notebooks**
- Cabeçalho de ambiente serverless `"6"`.
- Primeira célula com `%pip install` de todas as bibliotecas importadas, seguida de `restartPython`.
- Relatório final em texto, para a Genie copiar.
- Arquivos gravados de forma atômica.
- Horários no fuso de Brasília.

**Dashboard**
- Módulos novos do motor usados pelo app são copiados para `dashboard/ibov_barrier/`.
- Testes na versão do Databricks Apps (Streamlit 1.38).
- Números no formato brasileiro (`formato_br.py`).
- Gráficos no padrão do projeto: fundo preto, texto branco, rampa quente, marcador e traço por série.
- Sem emojis.

**Produção**
- Nada vai para produção antes do merge no `main`.
- O Pull do Git folder de produção, o ajuste de jobs e o deploy do app são feitos por prompt para a Genie.

---

## 5. Arquitetura proposta

```text
src/ibov_barrier/
  produtos/   interface Produto, EstadoMercado, adaptadores dos pricers atuais (A1, pronto)
    mercado.py      EstadoMercado, MercadoAtivo, MercadoCambio, CurvaPlana, CurvaDI
    base.py         protocolo Produto, Fluxo, Observacao, Carteira (soma e peso de produtos)
    adaptadores.py  Vanilla, CallSpread, Digital, BarreiraSimples, OneTouch, NoTouch, Quanto
    gregas.py       gregas por diferenças finitas sobre a interface
  coe/        termsheet, perna de título, composição, solver, catálogo de estruturas
  hedge/      instrumentos listados, razões de hedge, simulação do hedge, reservas
  risco/      cenários, fatores, VaR/ES, explicação do P&L, livro
  tabelas.py  leitor do histórico dos arquivos (pandas puro)

databricks/rotinas/
  12_coe_estruturacao.py     termsheet -> decomposição, margem, solver, hedge inicial
  13_coe_simulacao_hedge.py  simulação do hedge ao longo da vida do produto
  14_coe_livro_risco.py      marcação, gregas, VaR e explicação do P&L do livro (diário)
  15_carga_tabelas.py        carga das tabelas Delta (fase F)

dashboard/
  coe.py                     página "Estruturação de COE"
```

- **Tudo fica no pacote `ibov_barrier`.** O COE usa os mesmos pricers, curvas e superfícies, e um pacote separado
  duplicaria o motor. Isso substitui a ideia de 24/11 de uma pasta irmã `estruturados/` (decisão 6 da seção 8,
  confirmada em 25/11).
- **`produtos/` é a peça central.** VaR, simulação de hedge, solver e Desk só conversam com a interface. Assim, cada
  estrutura nova ganha risco e hedge sem código extra.

---

## 6. Etapas

### Fase A: interface de produto e um COE mínimo de ponta a ponta

**Objetivo:** validar a arquitetura com o COE real mais simples antes de investir em risco e hedge.

**A1. `produtos/`: interface de produto** (pronto em 25/11/2026; 57 testes em `tests/test_produtos.py`)

O que foi decidido na implementação:
- **Prazo.** Cada produto tem `vencimento` em anos a partir da data de referência, na convenção do ativo (DU/252 no
  IBOV, ACT/365 no SPX). O estado tem `tempo`, o tempo já decorrido. O prazo restante é a diferença, e o theta é
  `avancar(dt)`.
- **Juros por moeda.** O estado guarda uma curva por moeda. A `CurvaDI` lê a `DiscountCurve` e converte a taxa
  efetiva em contínua pela regra do notebook 01, r = ln(1 + taxa). A `CurvaPlana` é o `MarketData` de hoje. Assim, o
  mesmo choque na curva DI move o título e as opções em BRL.
- **Vol.** Pode ser um número ou uma função do strike e do prazo, com deslocamento aditivo por ativo. A leitura é
  "sticky strike": o choque de spot não move a superfície. Opções com strike usam a vol no strike, e os toques usam a
  vol na barreira. O Desk usa a vol ATM nas estruturas sem strike.
- **Quanto.** Vale para qualquer adaptador via `quanto=Quanto(par, cambio_fixo)`. A conta é a mesma do
  `quanto_market_data`: drift com -rho sigma_S sigma_X e desconto em BRL. Payoffs em pontos (vanilla, barreira,
  rebate) são convertidos pelo câmbio fixo. Pagamentos em dinheiro (digital, toques) já estão em BRL.
- **Cenários vetoriais.** Spot, deslocamento de vol, câmbio, spread e `CurvaPlana` aceitam vetores. A fórmula fechada
  avalia tudo de uma vez pelo numpy. O Monte Carlo avalia um cenário por vez, sempre com a mesma semente.
- **Composição.** Produtos se somam e se multiplicam por número (`titulo + 0.8 * call_spread`), formando uma
  `Carteira`. Os produtos europeus expõem o `payoff`, que o teste de Monte Carlo do A3 usa.
- **Gregas.** Delta e gamma com choque relativo de 1% no spot, vega por 1 p.p., DV01 por +1 bp em cada curva e
  theta em `dt`. A grega do CS01 (spread do emissor) entra com o título, no A2.
- **Testes "mesmo número".** Vanilla e call spread contra o `black_scholes_call` (diferença de arredondamento,
  abaixo de 1e-9 do spot). Digital contra o `_bs_digital`. Barreiras simples (8 tipos, com e sem rebate), one-touch
  e no-touch dão exatamente o mesmo `PriceResult` do pricer. Quanto contra o `price_quanto_vanilla`. A barreira
  Quanto contra a conta do Desk.
- **Testes de referência independente.** Digital contra um call spread estreito. Delta da DOC em Monte Carlo
  contra Reiner-Rubinstein. Gregas da call contra as fórmulas fechadas.

Especificação original:
- `EstadoMercado` reúne:
  - spot por subjacente;
  - deslocamento de vol;
  - curva de desconto (`DiscountCurve`);
  - spread do emissor;
  - câmbio;
  - data.

  Aceita vetores, para avaliar milhares de cenários de uma vez.
- `Produto` é um protocolo com `valor(estado)`, `fluxos()`, `observacoes()` e `subjacentes()`.
- As gregas genéricas saem por diferenças finitas sobre a interface. Nos produtos de Monte Carlo, com os mesmos
  números aleatórios.
- Adaptadores para os pricers que os COEs usam: call e put, call spread, barreiras com rebate, digital, one-touch e
  Quanto.
- **Testes:** cada adaptador reproduz o pricer original no mesmo número, e a avaliação vetorizada é igual à escalar.

**A2. `coe/titulo.py`: perna de título**
- Valor presente do principal protegido, e dos cupons fixos se houver, pela curva DI pré mais o spread do emissor.
- O spread pode ser dado em pontos-base sobre a curva pré ou em percentual do CDI, com a conversão entre os dois.
- Convenção brasileira: dias úteis/252, capitalização exponencial.
- **Testes:**
  - com spread zero, o valor bate com o de uma LTN de mesmo prazo na mesma curva (`fixed_income/ltn.py`);
  - o valor cai quando o spread sobe;
  - a soma dos DV01 por vértice (`bump_key`) bate com o DV01 paralelo.

**A3. `coe/estrutura.py`: o COE como composição**
- `Termsheet` com os campos da seção 1 e validação de datas, níveis e percentuais.
- `COE` = nominal × (proteção × título + pacote de opções).
- O payoff no vencimento fica explícito. Ele é usado nos testes e, mais tarde, na visão do investidor.
- Decomposição do preço de emissão em título, opções, custos, reservas e margem, em % do nominal e em R$.
- **Testes:**
  - a soma das pernas é igual ao valor do COE;
  - o payoff simulado no vencimento, descontado, bate com o valor (Monte Carlo independente).

**A4. `coe/solver.py`: margem alvo**
- Fixada a margem, o solver resolve o parâmetro livre:
  - a participação, que é linear e tem solução direta;
  - cap, cupom ou barreira, por busca de raiz, depois de verificar que o parâmetro é monotônico.
- Avisa quando a margem é inviável. Exemplo: proteção de 100% com juro baixo e prazo curto.
- **Testes:** ida e volta (parâmetro → margem → parâmetro), monotonicidade e caso inviável.

**A5. Notebook `12_coe_estruturacao.py`**
- Widgets do termsheet.
- Lê a superfície e as curvas da última rodada aprovada (RESULTS e VALIDATION, com conferência do SHA-256).
- Relatório em texto com a decomposição, o parâmetro resolvido e as gregas iniciais.

**A6. Primeiro caso real** (proposta, a confirmar)
- **O caso:** capital protegido de 100% com participação no Ibovespa e cap, ou seja, título + call spread.
- **Por que este:** é o mais simples, tem fórmula fechada e exercita título, opções, margem e solver.
- **Problema conhecido:** o book de opções do Ibovespa é raso e curto, enquanto COEs costumam ter de 1 a 5 anos.
  Além do último vencimento confiável, a vol sai de extrapolação, com reserva de modelo (etapa C5). Por isso o
  primeiro caso:
  - tem prazo dentro da superfície confiável;
  - ou usa o S&P 500 em reais, que tem vencimentos longos.
- **Conferência:** a margem implícita deve ter a mesma ordem de grandeza da de COEs parecidos, segundo os DIEs
  públicos (Documento de Informações Essenciais).

**A7. Página "Estruturação de COE" no dashboard (mínima)**
- Formulário do termsheet.
- Decomposição em cascata: 100% = título + opções + custos + reservas + margem.
- Solver na tela.

**Pronto quando:** o primeiro caso roda de ponta a ponta no notebook e no app, os testes estão verdes e a
decomposição foi conferida.

### Fase B: risco por reprecificação sobre a interface

**B0. Leitor do histórico (`tabelas.py`)**
- Converte RESULTS, VALIDATION e RISK em tabelas pandas, com as chaves `run_id`, `market` e `market_date`.
- Serve para calibrar os fatores do B2.
- Não exige catálogo nem warehouse. Na fase F, as tabelas Delta só gravam o que ele já monta.

**B1. VaR e ES por reprecificação completa**
- `risco/cenarios.py` gera os cenários dos fatores.
- `risco/var.py` aplica os cenários a qualquer `Produto` ou carteira, com P&L = V(estado chocado) − V(estado base),
  sem aproximação de Taylor.
- **Como avaliar:**
  - produtos com fórmula fechada: avaliação direta e vetorizada;
  - produtos de Monte Carlo: grade e interpolação PCHIP, com os mesmos números aleatórios.
- **Toque da barreira dentro do horizonte.** A ponte browniana dá, para uma barreira de baixa,

  $$P(\text{toque} \mid S_0, S_1) = \exp\!\left(-\frac{2\,\ln(S_0/H)\,\ln(S_1/H)}{\sigma^2 h}\right).$$

  Cada cenário sorteia o toque com essa probabilidade. Se houver toque, a perda é o valor da opção menos o rebate.
- **Notebook 03:** ganha o widget `metodo_var`, com padrão `delta_gamma` até a validação. Os campos novos do RISK são
  aditivos.
- **Testes:**
  - **Vanilla com resposta exata.** A call é crescente no spot, então o VaR de 99% é $V_0 - V(S_{1\%})$. A simulação
    tem de bater com esse valor dentro do erro do quantil.
  - **Longe da barreira:** reprecificação e delta-gamma ficam próximas.
  - **ES ≥ VaR** e reprodutibilidade com a mesma seed.

**B2. Fatores de risco**

| Fator | Choque | Calibração inicial |
| :--- | :--- | :--- |
| Spot, por subjacente | log-retorno | Ibovespa: série longa do índice (fonte a definir). S&P 500: `spx_fechamentos.parquet` do notebook 09 |
| Vol | deslocamento paralelo, depois por prazo | vol ATM do histórico (curto); até haver histórico, um beta da vol contra o spot |
| Curva DI | choque por vértice (`bump_key`); depois PCA (`pca_decomposition`) | histórico da curva (B3/ANBIMA) |
| Spread do emissor | deslocamento em bp | entrada manual no início |
| Câmbio | log-retorno do USDBRL | PTAX (`ptax.parquet` do notebook 09) |

- **Covariância:** EWMA diária com λ = 0,94 (RiskMetrics).
- **Relatório:** VaR por fator, VaR conjunto e efeito da diversificação.

**B3. Gregas do COE**
- Delta, gamma, vega por prazo, theta, DV01 por vértice, sensibilidade ao spread do emissor (CS01) e delta de
  câmbio.
- Saem por diferenças finitas sobre a interface.
- **Teste:** a soma dos DV01 por vértice bate com o DV01 paralelo.

**B4. Explicação diária do P&L**

$$
\Delta V \approx \delta\,\Delta S + \tfrac{1}{2}\,\Gamma\,\Delta S^2 + \nu\,\Delta\sigma + \Theta\,\Delta t
+ \sum_i \mathrm{DV01}_i\,\Delta r_i + \mathrm{CS01}\,\Delta s + \delta_X\,\Delta X + \varepsilon
$$

- O resíduo $\varepsilon$ é um controle de qualidade. Um resíduo grande indica uma grega errada ou um fator de
  mercado fora do modelo.
- **Teste:** em movimentos pequenos, o resíduo tende a zero.

### Fase C: estruturas de COE no motor

**C1. Autocall (`coe/autocall.py`)**
- Datas de observação com gatilho de resgate (fixo ou decrescente) e cupom condicional com memória.
- Barreira final de proteção: abaixo dela, o investidor perde junto com o índice (uma put down-and-in vendida).
- Precificação por Monte Carlo com observação discreta.
- **Testes de casos-limite:**
  - gatilho inalcançável: sobra só o título com o payoff final;
  - gatilho zero: resgate na primeira observação, o que equivale a um título até essa data mais o cupom.

**C2. Média**
- Participação sobre a média aritmética das observações.
- A média geométrica, que tem fórmula fechada (Kemna e Vorst, 1990), entra como variável de controle.

**C3. Worst-of e cesta**
- Monte Carlo com N ativos correlacionados (Cholesky), generalizando o `price_two_factor_mc`.
- Correlações estimadas do histórico. O notebook 09 já estima a correlação entre o S&P 500 e o câmbio.
- **Testes:**
  - com correlação 1 e ativos idênticos, o worst-of vale o mesmo que o ativo único;
  - a call worst-of perde valor quando a correlação cai.

**C4. Estruturas que já têm pricer**
- Tubarão, cupom digital, intervalo e S&P 500 em reais.
- Cada uma entra no catálogo pelos adaptadores do A1, com teste "mesmo número".

**C5. Vol de prazo longo**
- Extrapolação da estrutura a termo da variância ATM além do último vencimento confiável, por exemplo com variância
  forward constante.
- A reserva de modelo é a diferença de preço entre extrapolações alternativas.

**C6. Catálogo de estruturas (`coe/catalogo.py`)**
- Cada estrutura é um modelo de termsheet mais o seu parâmetro livre para o solver.
- O notebook 12 e o app usam o catálogo.

**Pronto quando (cada estrutura):** referência independente verde, solver funcionando e entrada no catálogo.

### Fase D: hedge e simulação do hedge

**D1. Instrumentos de hedge (`hedge/instrumentos.py`)**

| Instrumento | Uso | Tamanho (a conferir na especificação oficial) |
| :--- | :--- | :--- |
| Ibovespa futuro (IND) e mini (WIN) | delta do Ibovespa | R$ 1,00 e R$ 0,20 por ponto |
| Dólar futuro (DOL) e mini (WDO) | delta de câmbio (S&P 500 em reais) | US$ 50.000 e US$ 10.000 |
| DI1 | DV01 por vértice da perna de título | PU de 100.000 pontos, R$ 1,00 por ponto |
| E-mini S&P 500 (ES) e Micro (MES) | delta do S&P 500 | US$ 50 e US$ 5 por ponto |
| Opções listadas (Ibovespa, SPX) | gamma e vega, quando houver liquidez | — |

- **Preços de ajuste:** verificar se os arquivos da B3 que o notebook 00 já baixa trazem IND, DOL e DI1. Se não
  trouxerem, criar uma coleta nova.

**D2. Razões de hedge (`hedge/razoes.py`)**
- Converte as gregas do B3 em contratos, com arredondamento e risco residual.
- Gera o ticket inicial do hedge.
- **Teste:** as gregas da carteira de hedge anulam as do produto dentro do erro de arredondamento.

**D3. Simulação do hedge (`hedge/simulacao.py`)**
- **Caminhos:**
  - na medida real (P), ao longo da vida do produto, com drift, vol e correlações históricas;
  - juros por deslocamentos da curva (PCA);
  - saltos e vol estocástica, só se a análise mostrar que fazem falta.
- **Rebalanceamento:**
  - diário, semanal ou por limite de delta;
  - custos de spread, corretagem e emolumentos;
  - ajuste diário dos futuros e custo de carregar a margem;
  - rolagem dos vencimentos.
- **Reprecificação em cada data:**
  - produtos com fórmula: direto;
  - produtos de Monte Carlo: uma aproximação por grade ou regressão em (t, S), calibrada antes da simulação, porque
    Monte Carlo dentro de Monte Carlo não cabe no tempo.
- **Saídas:**
  - distribuição do P&L do hedge;
  - margem realizada (margem inicial + P&L do hedge), com percentis;
  - contribuição de gamma e de vega;
  - sensibilidade à frequência de rebalanceamento e à vol realizada.
- **Testes:**
  - na medida neutra ao risco, sem custos, com rebalanceamento cada vez mais frequente, a média do P&L tende a zero
    e o desvio cai (replicação de Black-Scholes);
  - custos maiores pioram o P&L;
  - com vol realizada acima da implícita, quem está vendido em gamma perde.

**D4. Reservas (`hedge/reservas.py`)**
- Cobrem bid/ask, risco de modelo, liquidez e descontinuidades.
- Nas descontinuidades: deslocamento da barreira e overhedge das digitais (a digital vira um call spread estreito).
- O valor das reservas volta para a decomposição do A3 como o termo R.

**D5. Notebook 13 e seção no app**
- Histogramas da margem realizada e caminhos do P&L do hedge, no padrão de gráficos do projeto.

### Fase E: livro de COEs e risco diário

**E1. Livro**
- Produtos emitidos e o hedge de cada um.
- Marcação diária, gregas agregadas por subjacente e por vértice, e limites.
- Até a fase F, o livro fica em arquivos com as colunas e chaves das futuras tabelas, para a migração ser uma carga
  direta.

**E2. Ciclo de vida**
- Calendário de eventos: observações, cupons, autocall, barreira e vencimento.
- Liquidação e rolagem do hedge.

**E3. VaR do livro**
- Reprecificação com fatores conjuntos (fase B).
- Simulação histórica, pura e filtrada pela vol EWMA.
- Backtesting:
  - Kupiec: a proporção de exceções;
  - Christoffersen: a independência das exceções;
  - semáforo de Basileia (250 observações a 99%): verde de 0 a 4 exceções, amarelo de 5 a 9, vermelho de 10 em
    diante.
- Contribuições de Euler por produto: a média do P&L de cada produto nos cenários da cauda do livro.
- **Limite de histórico:** 250 pregões desde 03/08/2026 só existem por volta do fim de julho de 2027. Até lá, o
  backtest aparece com a ressalva de que o teste de Kupiec tem pouco poder.

**E4. Stress do livro**
- Cenários de queda forte da bolsa, choque de juros, choque de câmbio e alta de vol.
- Todos reprecificados, no mesmo espírito do notebook 04.

**E5. Notebook `14_coe_livro_risco.py`** (diário)

### Fase F: tabelas Delta

**Quando:** é a última fase de infraestrutura, mas precisa ficar pronta antes de o livro (fase E) rodar todo dia em
produção.

**Princípio:** os arquivos continuam sendo a fonte da verdade e a trilha de auditoria. As tabelas são uma carga
derivada, feita pelo notebook `15_carga_tabelas.py`:
- usa o leitor `tabelas.py` do B0;
- grava com `MERGE` pela chave, de modo que rodar de novo não duplica;
- confere o SHA-256 antes de carregar;
- usa clustering líquido em `(market, market_date)` no lugar de partição.

**Tabelas de mercado:** `runs`, `validation_checks`, `term_structure`, `vol_points`, `vol_surface`,
`cleaning_report`, `pricing_results`, `risk_metrics`, `quanto_results` e `spx_daily_summary`.

**Tabelas do COE:** `termsheets`, `positions`, `hedge_trades`, `mtm_daily`, `greeks_daily`, `pnl_explain`,
`hedge_simulations` e `var_backtest`.

**Passos:**
1. Decidir catálogo, schema, SQL warehouse e permissões.
2. Carregar o passado.
3. Acrescentar a tarefa de carga aos jobs. Boa hora para unificar os dois jobs do Ibovespa.
4. Trocar a fonte do dashboard pela variável `FONTE_DADOS` no `app.yaml`.

### Fase G: resgate antecipado e exercício americano (sob demanda)

- **Resgate a critério do emissor (COE "callable").** Longstaff-Schwartz no Monte Carlo: o emissor resgata quando o
  valor de continuar é maior que o preço de resgate. A validação é contra uma EDP ou uma árvore no caso de um fator.
- **Exercício americano de subjacentes** (ações da B3, ETFs): adiado. O que seria preciso está descrito na primeira
  versão deste plano (commit `73dbc8b`):
  - EDP e árvore binomial;
  - vol implícita invertida pelo modelo americano;
  - r e q de fonte externa, porque a paridade put-call não vale para opções americanas.

---

## 7. Marcos

| Marco | Conteúdo | Depende de | Pronto quando |
| :--- | :--- | :--- | :--- |
| M1 | Interface de produto e adaptadores (A1) | — | todos os adaptadores no mesmo número |
| M2 | COE mínimo: título, composição, solver, notebook 12 (A2 a A6) | M1 | primeiro caso de ponta a ponta, decomposição conferida |
| M3 | Página de estruturação no app (A7) | M2 | AppTest no Streamlit 1.38 sem erro |
| M4 | Leitor do histórico, VaR com reprecificação e gregas do COE (B0, B1, B3) | M1 | testes verdes e comparação com o delta-gamma apresentada |
| M5 | Fatores conjuntos e explicação do P&L (B2, B4) | M4 | resíduo pequeno no histórico disponível |
| M6 | Estruturas de COE (C1 a C6) | M1 | cada estrutura com referência independente verde |
| M7 | Hedge e simulação do hedge (D) | M2 e M4 | testes de replicação verdes |
| M8 | Livro e risco diário (E) | M5 e M7 | livro de teste marcado em rodadas seguidas |
| M9 | Tabelas Delta (F) | M8, antes da produção diária | contagens e números iguais aos dos arquivos |
| M10 | Resgate antecipado (G) | decisão de produto | validação contra EDP ou árvore |

M3 pode andar em paralelo com M4. M6 pode andar em paralelo com M4 e M5.

---

## 8. Decisões pendentes

1. **Primeiro caso real.** Decidido em 25/11: capital protegido de 100% com participação no Ibovespa e cap, num prazo
   dentro da superfície confiável. Depois, o mesmo produto no S&P 500 em reais.
2. **Spread de funding do emissor.** Começar com entrada manual. Mais tarde, uma curva de CDB ou LF do emissor.
3. **Custos de distribuição e política de reservas.**
4. **Instrumentos de hedge permitidos** (seção D1).
5. **Medida real da simulação do hedge:** drift, vol e modelo de juros.
6. **Onde fica o código do COE.** Decidido em 25/11: subpacotes dentro do `ibov_barrier` (seção 5).
7. **Catálogo, schema e SQL warehouse** da fase F, e a unificação dos jobs do Ibovespa.
8. **Quando trocar o padrão do VaR** do notebook 03 para a reprecificação.

---

## 9. Validação, em resumo

| O quê | Como |
| :--- | :--- |
| Adaptadores | mesmo número que o pricer original |
| Título e composição | LTN na mesma curva; soma das pernas; payoff simulado descontado |
| Estruturas novas | fórmula fechada, casos-limite e Monte Carlo independente |
| Margem | ordem de grandeza dos DIEs públicos de COEs parecidos |
| Hedge | replicação de Black-Scholes na medida neutra ao risco |
| Risco | quantil exato no vanilla; resíduo da explicação do P&L; backtesting do VaR |

---

## 10. Depois: a visão do investidor

Usa o mesmo motor, sem módulos novos de precificação:
- distribuição do retorno no vencimento, na medida real;
- chance de superar o CDI;
- custo embutido (custos, reservas e margem em % do nominal);
- comparação entre cenários.

---

## 11. Documentação a atualizar ao longo do caminho

| Documento | O que muda |
| :--- | :--- |
| `WORKING_PAPER.md` | 6.2 (VaR); 7.3 (arquivos e tabelas); 10, itens 4, 6 e 8; 11 (próximos passos); uma seção nova sobre COE |
| `README.md` | notebooks 12 a 15, página de COE, jobs, tabelas e documentos |
| `ARQUITETURA_MULTIATIVOS.md` | seção 7 com o esquema final das tabelas e o clustering no lugar da partição |
| `INVENTARIO_MODELOS.md` | estruturas de COE, perna de título, autocall, média, worst-of |
| `dashboard/README.md` | página de COE, fonte de dados (`FONTE_DADOS`) e warehouse |
