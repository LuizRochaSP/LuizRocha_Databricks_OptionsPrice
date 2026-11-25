# Plano de desenvolvimento: tabelas Delta, VaR com reprecificação e exercício americano

- **Branch:** `feature/tabelas-var-americano`, criada a partir do `main` em `c952c71`
- **Data:** 25/11/2026
- **Situação:** plano. Nada foi implementado ainda.

Este documento é o passo a passo para resolver três limitações declaradas na seção 10 do
[`WORKING_PAPER.md`](WORKING_PAPER.md):

| Limitação (WP, seção 10) | Como está hoje | Onde está no código |
| :--- | :--- | :--- |
| 8. Armazenamento | Cada rodada grava `RESULTS_*.csv`, `VALIDATION_*.json` e `RISK_*.json` na pasta `ibov-barrier-results`. O dashboard lista e baixa esses arquivos pelo SDK. | `results_bundle.py`; notebooks 01, 02, 03, 07, 09 e 11; `dashboard/app.py` |
| 4. VaR paramétrico | Delta-gamma com choque lognormal só no spot: 100.000 cenários, 99% e 1 dia útil, para um único contrato (a call down-and-out de referência). | `risk.py` (`delta_gamma_var_es`, `calculate_risk_metrics`); notebook 03 |
| 6. Exercício europeu | O motor e a camada de mercado supõem exercício europeu: a paridade put-call dá r, q e o forward, e a vol implícita inverte Black-Scholes. | `markets/parity.py`, `markets/vol.py`, `markets/ibov.py` (filtro "Somente opções europeias"). O campo `MarketSpec.exercise` já aceita `"american"`. |

---

## 1. Ordem e motivo

```text
Fase 1   Tabelas Delta ───────────────────────────────┐
                                                       ├──> 2c VaR histórico e backtesting ──> 2d VaR de carteira (COE)
Fase 2a  VaR com reprecificação ──> 2b Choques conjuntos ┘

Fase 3   Exercício americano (só quando entrar um produto americano)
```

- **As fases 1 e 2a andam em paralelo.** Uma não depende da outra.
- **2c depende da fase 1.** O VaR histórico e o backtesting precisam de histórico consultável de preços, superfícies
  e P&L.
- **A fase 3 fica por último.** As opções do Ibovespa e do S&P 500 são europeias, e o payoff do COE também é no
  vencimento. O exercício americano só importa com opções de ações da B3 ou com ETFs americanos (SPY, QQQ).
- **Encaixe com o COE** (próxima frente do projeto): o COE precisa das tabelas, de um VaR com reprecificação e de
  risco de carteira. Não precisa de exercício americano.

---

## 2. Regras que valem para todas as etapas

**Git**
- O trabalho acontece nesta branch. Rode `git fetch` no início de cada sessão, porque o `main` também recebe commits
  feitos pelo workspace do Databricks.
- Correções do `main` entram aqui com `git merge main`.
- Cada marco (seção 6) vai para o `main` separadamente, em merges pequenos, para a branch não se afastar do `main`.

**Números de produção**
- Nenhuma etapa muda um número de produção sem um teste "mesmo número" contra o código atual.
- Mudança de metodologia entra com widget e com o padrão antigo ligado até ser validada. Depois é declarada no WP,
  como foi feito com o spot do S&P 500.

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
- Todo módulo novo do motor usado pelo app é copiado para `dashboard/ibov_barrier/`.
- Testes na versão do Databricks Apps (Streamlit 1.38).
- Números no formato brasileiro (`formato_br.py`).
- Gráficos no padrão do projeto: fundo preto, texto branco, rampa quente, marcador e traço por série.
- Sem emojis.

**Produção**
- Nada vai para produção antes do merge no `main`.
- O Pull do Git folder de produção, o ajuste de jobs e o deploy do app são feitos por prompt para a Genie.

**Documentação**
- Ao fim de cada fase, atualizar o `README.md`, o `WORKING_PAPER.md` (seções 6, 7.3, 10 e 11) e o
  `ARQUITETURA_MULTIATIVOS.md`.

---

## 3. Fase 1: tabelas Delta

**Objetivo:** histórico consultável por SQL, sem perder a cadeia de confiança dos arquivos.

**Princípio:** os arquivos continuam sendo a fonte da verdade e a trilha de auditoria, com o SHA-256 do RESULTS
aprovado. As tabelas são uma **carga derivada** dos três arquivos de cada rodada. Consequências:

- os notebooks de produção (01, 02, 03, 07, 09 e 11) não mudam;
- se a carga falhar, a rodada não se perde: basta rodar a carga de novo;
- a carga do passado e a carga diária usam o mesmo código.

### Passo 1.0: decisões antes de começar

1. Nome do catálogo e do schema no Unity Catalog.
2. SQL warehouse para o app ler as tabelas: serverless, no menor tamanho e com parada automática. Este é o principal
   custo novo da fase.
3. Permissões:
   - a identidade que roda os jobs precisa de `USE CATALOG`, `USE SCHEMA`, `CREATE TABLE` e `MODIFY`;
   - o service principal do app `dashboard-renda-fixa` precisa de `SELECT` nas tabelas e `CAN USE` no warehouse.

### Passo 1.1: esquema

O esquema parte do desenho da seção 7 do `ARQUITETURA_MULTIATIVOS.md` (`options_clean`, `forwards`, `term_structure`,
`vol_points`, `cleaning_report`, `pricing_results`) e acrescenta rodada, validação e risco.

- Todas as tabelas têm `run_id`, `market` e `market_date`.
- O `run_id` vem do nome do arquivo, por exemplo `superficieVol_20261124_132739`.
- **Clustering líquido em `(market, market_date)` no lugar de partição.** O volume é pequeno, e partição física em
  tabela pequena só gera arquivos minúsculos. A seção 7 da arquitetura deve ser atualizada para refletir isso.

| Tabela | Chave | Origem | Conteúdo |
| :--- | :--- | :--- | :--- |
| `runs` | `run_id` | `metadata` do RESULTS, VALIDATION e RISK | mercado, tipo (`superficieVol`, `spx_superficie`, `quanto`), data de mercado, carimbo, nomes dos arquivos, SHA-256, `schema_version`, decisão, número de WARN e estado (completa, sem risco, não validada) |
| `validation_checks` | `run_id`, controle | VALIDATION | cada controle com status, valor e limite |
| `term_structure` | `run_id`, vencimento | `curve` (IBOV) e `forwards` (SPX) | T, r, q, forward e fator de desconto |
| `vol_points` | `run_id`, vencimento, strike, tipo | `iv_data` (IBOV) e `options` (SPX) | preço, log-moneyness e vol implícita |
| `vol_surface` | `run_id`, T, k | `surface_data` | grade da superfície |
| `cleaning_report` | `run_id`, etapa | `filtered` (IBOV) e `funnel` (SPX) | contagens por filtro |
| `pricing_results` | `run_id`, produto | `pricing_result` e `simulation_diagnostics` | contrato, dados de mercado, preço vanilla e com barreira, erro-padrão, IC, P(knock-out), caminhos, passos e seed |
| `risk_metrics` | `run_id`, método | RISK | gregas, DV01, convexidade, VaR, ES, horizonte, confiança e cenários |
| `quanto_results` | `run_id`, produto | RESULTS do 09 | r_BRL, PTAX, rho, sigma_X e preços |
| `spx_daily_summary` | `market_date`, vencimento | `spx_dados/resumo_diario.parquet` (notebook 08) | r, q e qualidade do S&P 500 desde 03/08/2026, anteriores aos RESULTS |

O mapeamento coluna a coluna de cada objeto do bundle é a primeira tarefa do passo 1.2.

### Passo 1.2: módulo `src/ibov_barrier/tabelas.py`

- Recebe o RESULTS (via `read_bundle`), o VALIDATION e o RISK de uma rodada e devolve `{tabela: DataFrame}` com tipos
  explícitos.
- É pandas puro, sem Spark, para ser testado localmente.
- Testes (`tests/test_tabelas.py`), com bundles sintéticos no formato do 01, do 07 e do 09:
  - esquema de cada tabela: colunas, tipos e chave única;
  - ida e volta: os valores das tabelas são idênticos aos dos arquivos;
  - rodada incompleta (sem RISK ou sem VALIDATION) entra com o estado certo;
  - SHA-256 divergente gera erro, não carga.

### Passo 1.3: notebook `12_carga_tabelas.py`

- Widgets:
  - `modo`: `rodada` (a última rodada de um mercado) ou `historico` (todas as rodadas da pasta);
  - `ativo`: `ibov`, `spx` ou `quanto`;
  - `catalogo` e `schema`.
- Antes de carregar, confere o SHA-256 do RESULTS contra o VALIDATION, com a mesma regra do notebook 03.
- Cria as tabelas com `CREATE TABLE IF NOT EXISTS`, com comentário em cada coluna.
- Grava com `MERGE INTO` pela chave, de modo que rodar duas vezes não duplica nada.
- Termina com um relatório em texto: linhas inseridas e atualizadas por tabela.

### Passo 1.4: carga do passado

- Rodar o 12 em modo `historico` sobre todos os arquivos de `ibov-barrier-results`: o IBOV desde setembro de 2026 e
  o S&P 500 e o Quanto desde novembro de 2026.
- Carregar o painel do notebook 08 em `spx_daily_summary`.
- Os RESULTS são o histórico do IBOV. Os ZIPs brutos da B3 no Git folder de produção não guardam todo o período desde
  agosto, porque a pasta `data/` foi perdida e recriada em 24/11/2026.
- **Pronto quando:** o número de rodadas completas em `runs` é igual ao número de rodadas completas que o dashboard
  lista hoje, e os números conferem com os arquivos numa amostra de rodadas.

### Passo 1.5: jobs

- Acrescentar a tarefa `12_carga_tabelas`:
  - no fim do job agendado do IBOV ("Pipeline Diário B3 — Barrier Option") e do `opcoes-pipeline-ibov`, depois do 03;
  - no `opcoes-pipeline-spx`, depois do 03 e do 09.
- Boa hora para resolver o item 4 da seção 11 do WP: levar a agenda das 08:00 para o `opcoes-pipeline-ibov`, com
  timeouts, e aposentar o job antigo. Assim a tarefa 12 entra num lugar só.
- Uma falha no 12 não afeta os arquivos. Até o passo 1.6, o dashboard continua lendo os arquivos.
- **Pronto quando:** cinco rodadas diárias seguidas são carregadas sem intervenção.

### Passo 1.6: dashboard lê as tabelas

- **Leitura:** pelo `databricks-sql-connector`, o padrão dos exemplos do Databricks Apps, com o warehouse declarado
  como recurso no `app.yaml`.
- **Chave de troca:** a variável `FONTE_DADOS` (`arquivos` ou `tabelas`) no `app.yaml`. O padrão fica `arquivos` até a
  validação.
- **Primeiro alvo: as telas de histórico.** Séries de preço, vol ATM, r, q, P(knock-out) e VaR por data de mercado.
  É aí que as tabelas rendem mais. O detalhe de uma rodada (superfície 3D) pode continuar lendo o arquivo.
- **Cache:** `st.cache_data` com validade curta.
- **Testes:** AppTest no Streamlit 1.38, com uma fonte de dados falsa.
- **Pronto quando:** as mesmas telas mostram números idênticos com as duas fontes nas últimas rodadas.

### Passo 1.7 (opcional): espaço Genie sobre as tabelas

Um espaço AI/BI Genie com a descrição das tabelas e das colunas. Permite perguntas em linguagem natural sobre o
histórico, como "como evoluiu a P(knock-out) do contrato de referência do S&P 500 em novembro?".

**Entregas da fase 1:** `tabelas.py` com testes, notebook 12, carga do passado, jobs com a carga diária, dashboard
com `FONTE_DADOS` e a seção 7 da arquitetura atualizada.

---

## 4. Fase 2: VaR

**Hoje:** o notebook 03 calcula, para a posição comprada no contrato de referência,

$$\Delta V \approx \Delta\,(S_1 - S_0) + \tfrac{1}{2}\,\Gamma\,(S_1 - S_0)^2,$$

com $S_1$ lognormal, $h = 1/252$ e drift zero. As fraquezas declaradas: a aproximação de Taylor perde precisão perto
da barreira, só o spot é chocado e a distribuição não é histórica.

### Passo 2a: reprecificação completa (só spot)

A troca é usar $\Delta V = V(S_1) - V(S_0)$, sem Taylor. Os cenários e a configuração continuam os de hoje: 100.000,
99% e 1 dia.

**Função nova em `risk.py`:** `full_revaluation_var_es(value_fn, spot, volatility, config)`, em que `value_fn`
recebe um vetor de spots e devolve um vetor de valores.

**Dois jeitos de calcular os valores:**

1. **Fórmula fechada de Reiner-Rubinstein** (`analytical.price_down_and_out_call_analytical`).
   - Precisa ser vetorizada: hoje aceita um escalar e usa `if spot <= barrier`.
   - Cobre o contrato de referência dos dois mercados.
   - Supõe monitoramento contínuo, como o Monte Carlo com ponte browniana do motor. Um teste confere os dois.
2. **Grade e interpolação**, para os produtos sem fórmula (Desk, Quanto com vol local).
   - Reprecificar por Monte Carlo, com os mesmos números aleatórios, numa grade de spot (por exemplo, 41 pontos em
     ±5 desvios do choque diário).
   - Interpolar com PCHIP, que é monótona e não oscila.
   - Tratar à parte a região além da barreira.

**O que o delta-gamma não captura: a barreira tocada dentro do horizonte.**
- Se $S_1$ fica além da barreira, a opção morre e a perda é $V_0$, menos o rebate, se houver.
- Mesmo com $S_1$ do lado vivo, o caminho pode ter tocado a barreira durante o dia. Pela ponte browniana, para uma
  barreira de baixa,

  $$P(\text{toque} \mid S_0, S_1) = \exp\!\left(-\frac{2\,\ln(S_0/H)\,\ln(S_1/H)}{\sigma^2 h}\right).$$

  Cada cenário sorteia o toque com essa probabilidade, a mesma lógica do motor.

**Notebook 03:**
- Widget `metodo_var` com `delta_gamma` (padrão), `reprecificacao` ou `ambos`.
- O RISK ganha os campos `var_reprecificacao`, `es_reprecificacao` e `metodo_var`, sem remover nenhum dos atuais.
  A mudança é aditiva e não quebra o dashboard nem a carga do 12.
- Um parâmetro decide se o P&L inclui a passagem de um dia (carrego). O padrão fica sem carrego, como hoje, para a
  comparação ser justa.

**Testes:**
1. **Vanilla com resposta exata.** O valor da call é crescente no spot, então o VaR de 99% é
   $V_0 - V(S_{1\%})$, com $S_{1\%} = S_0\,e^{-\sigma^2 h/2 + \sigma\sqrt{h}\,z_{0{,}01}}$. A simulação precisa bater
   com esse número dentro do erro do quantil. O mesmo vale para a call down-and-out sem o toque intradiário.
2. **Longe da barreira:** reprecificação e delta-gamma ficam próximas.
3. **Cenários além da barreira:** perdem exatamente $V_0$ menos o rebate.
4. **Fórmula vetorizada igual à escalar**, e fórmula igual ao Monte Carlo com ponte browniana dentro de 3 erros-padrão
   numa grade de casos.
5. **ES ≥ VaR** e reprodutibilidade com a mesma seed.

**Pronto quando:** os dois métodos rodaram sobre o histórico (pelas tabelas ou reprocessando os RISK), as diferenças
foram apresentadas e o usuário aprovou a troca do padrão. A mudança de metodologia é então declarada na seção 6.2
do WP.

### Passo 2b: choques conjuntos

| Fator | Choque | Calibração inicial |
| :--- | :--- | :--- |
| Spot | log-retorno | IBOV: série histórica do índice (fonte longa a definir). SPX: `spx_fechamentos.parquet` do notebook 09 |
| Vol | deslocamento paralelo da vol, em pontos; a inclinação fica para depois | vol ATM das tabelas (histórico curto); até haver histórico, um beta da vol contra o retorno do spot |
| Juros | deslocamento paralelo, em bp | r BRL pela curva DI (B3/ANBIMA); r USD pela série do notebook 08 |
| Câmbio (Quanto) | log-retorno do USDBRL | PTAX (`ptax.parquet` do notebook 09) |

- **Covariância:** EWMA diária com λ = 0,94 (RiskMetrics). As correlações saem da mesma janela.
- **Cenários:** choques conjuntos normais, reprecificados em cada cenário.
  - Com a fórmula fechada, os 100.000 cenários são avaliados direto, vetorizados.
  - Com Monte Carlo, usa-se uma grade em spot × vol × juros com interpolação.
- **Relatório:** VaR só de spot, só de vol, só de juros e conjunto, mais o efeito da diversificação.
- **Vol local:** o preço com vol local do 09 exige chocar a superfície implícita e recalibrar o Dupire, o que é mais
  pesado. Fica para uma etapa própria, se for preciso.
- **Testes:**
  - com vol e juros sem choque, o resultado é igual ao do 2a;
  - fatores com correlação zero se combinam como esperado;
  - as matrizes de covariância são simétricas e positivas definidas.

### Passo 2c: simulação histórica e backtesting

**Simulação histórica.** Aplicar à posição de hoje as variações diárias dos fatores nos últimos N pregões: N = 250
quando houver, cerca de 80 hoje (desde 03/08/2026). A versão filtrada reescala cada variação pela razão entre a vol
EWMA de hoje e a vol EWMA daquele dia, o que aproveita melhor janelas curtas.

**Backtesting.**
- P&L hipotético diário do contrato de referência: o mesmo contrato, reprecificado no mercado de D com prazo menor
  em um dia, menos o seu valor no mercado de D−1.
- Esse P&L é comparado com o VaR calculado em D−1.
- Testes estatísticos:
  - Kupiec: a proporção de exceções;
  - Christoffersen: a independência das exceções;
  - semáforo de Basileia (250 observações a 99%): verde de 0 a 4 exceções, amarelo de 5 a 9, vermelho de 10 em
    diante.

**Tabelas novas:** `pnl_daily` e `var_backtest`.

**Histórico disponível:**
- No ritmo de uma rodada por pregão, 250 observações desde 03/08/2026 só existem por volta do fim de julho de 2027.
- Para encurtar isso, verificar até quando a B3 e a ThetaData servem arquivos antigos. A ThetaData serviu desde
  03/08/2026 no notebook 08.
- Antes de chegar a 250, o backtest aparece com a ressalva de que o teste de Kupiec tem pouco poder com poucas
  observações.

**Dashboard:** na aba de risco, a série do VaR contra o P&L realizado, com as exceções marcadas.

### Passo 2d: VaR de carteira

- **Tabela `positions`:** id, mercado, produto, parâmetros em JSON, lado, quantidade, data de entrada e origem (Desk
  ou COE).
- **Agregação:** cada produto tem a sua `value_fn`. O P&L da carteira em cada cenário é a soma dos P&L das posições;
  dele saem o VaR e o ES da carteira.
- **Contribuições (Euler):** a contribuição de cada posição ao ES é a média do seu P&L nos cenários da cauda da
  carteira.
- **Uso no COE:** esta é a base do risco do COE. A perna de título usa o fator de juros do 2b.

---

## 5. Fase 3: exercício americano (sob demanda)

**Gatilho:** a decisão de incluir um produto com exercício americano. Por exemplo, opções de ações da B3 (em que as
calls costumam ser americanas) ou ETFs americanos (SPY, QQQ).

### Passo 3.1: motor `pricing_american.py`

- **Árvore binomial CRR** como referência simples.
- **EDP como método principal:** Crank-Nicolson em log-spot, com exercício antecipado por projeção ou PSOR. Trata na
  mesma grade:
  - a barreira, como condição de contorno;
  - o rebate;
  - dividendos discretos, como salto no spot na data ex;
  - mais tarde, a vol local, com σ(S, t) na grade.
- As gregas saem da grade: delta e gamma direto e theta pelo passo de tempo.
- **Testes:**
  - call americana sem dividendos igual à europeia;
  - put americana maior ou igual à europeia e ao valor intrínseco;
  - árvore e EDP convergem para o mesmo valor;
  - EDP europeia com barreira igual a Reiner-Rubinstein;
  - valores publicados na tabela 1 de Longstaff e Schwartz (2001).

### Passo 3.2: Longstaff-Schwartz no Monte Carlo

Para produtos dependentes da trajetória com exercício antecipado. A validação é contra a EDP nos casos que os dois
cobrem.

### Passo 3.3: camada de mercado

- **Vol implícita:** inverter o preço americano (EDP ou árvore) em σ.
- **r e q:** a paridade put-call vale só como desigualdade para opções americanas. Por isso:
  - r vem de fonte externa (DI ou SOFR);
  - os dividendos vêm de projeção discreta (proventos anunciados) ou de um q calibrado com o próprio modelo
    americano.
- **Limpeza:** preço maior ou igual ao valor intrínseco, além dos filtros de liquidez que já existem.
- **Adaptador novo:** por exemplo `markets/b3_acoes.py`, com `MarketSpec(exercise="american")`.

### Passo 3.4: Desk e dashboard

- Produtos americanos marcados como tal.
- Cache do preço, porque a EDP é mais lenta que a fórmula fechada.
- Medir o tempo por clique no app, que hoje fica abaixo de 1 segundo.

---

## 6. Marcos

| Marco | Conteúdo | Depende de | Pronto quando |
| :--- | :--- | :--- | :--- |
| M1 | Esquema e `tabelas.py` | — | testes verdes |
| M2 | Notebook 12 e carga do passado | M1 e as decisões do 1.0 | contagens e números batem com os arquivos |
| M3 | Carga diária nos jobs (e job único do IBOV, se aprovado) | M2 | cinco rodadas seguidas carregadas sem intervenção |
| M4 | Dashboard lendo as tabelas | M3 | telas idênticas com as duas fontes |
| M5 | VaR com reprecificação | — | testes verdes e comparação histórica aprovada pelo usuário |
| M6 | Choques conjuntos | M5; M2 para calibrar | decomposição por fator no RISK |
| M7 | VaR histórico e backtesting | M3 e M6 | backtest no dashboard |
| M8 | VaR de carteira | M7 | carteira do Desk com VaR e contribuições |
| M9 | Exercício americano | decisão de produto | testes de referência verdes |

Cada marco segue o mesmo caminho até a produção: testes, merge no `main`, Pull no Git folder de produção, jobs e app
ajustados por prompt para a Genie, e conferência da primeira rodada real.

---

## 7. Decisões pendentes

1. Catálogo, schema e SQL warehouse (custo).
2. Unificar os dois jobs do IBOV junto com a fase 1?
3. Quando trocar o padrão do VaR para a reprecificação.
4. Janela e versão do VaR histórico: simulação histórica pura ou filtrada, e o valor de N.
5. Fonte longa da série histórica do Ibovespa.
6. Primeiro produto americano, e se ele entra antes ou depois do COE.

---

## 8. Documentação a atualizar ao longo do caminho

| Documento | O que muda |
| :--- | :--- |
| `WORKING_PAPER.md` | 6.2 (VaR); 7.3 (arquivos e tabelas); 10, itens 4, 6 e 8; 11 |
| `README.md` | notebook 12, jobs, tabelas e documentos |
| `ARQUITETURA_MULTIATIVOS.md` | seção 7 com o esquema final e o clustering no lugar da partição |
| `INVENTARIO_MODELOS.md` | produtos americanos e métodos (EDP, árvore, Longstaff-Schwartz) |
| `dashboard/README.md` | fonte de dados (`FONTE_DADOS`) e warehouse |
