# Working Paper: Quantitative Pricing & Risk Dashboard para Opções com Barreira no Ibovespa

**Autor**: Luiz Henrique Felipe Rocha
**Data**: Novembro de 2026
**Stack**: Databricks (Serverless, Unity Catalog, Databricks Apps) · Python · Streamlit
**Repositório**: `LuizRochaSP/LuizRocha_Databricks_OptionsPrice` (branch `main`, histórico linear)

---

## Abstract

Este projeto entrega um pipeline end-to-end de precificação e risco de opções com barreira sobre o Ibovespa, construído integralmente no Databricks. O motor quantitativo cobre quatro famílias de estruturas (A: Knock-Out/Knock-In simples com rebate; B: Double Barriers com convenções de ordem KI/KO parametrizáveis e rebate; C: estruturas combinadas Down, Up e Knock-In; D: binários de toque simples e duplos e Range Accrual; além das barreiras dependentes da trajetória Parisian, Step e Soft), precificados por Monte Carlo com correção de discretização de barreira via Brownian Bridge e redução de variância por variável de controle (Black-Scholes vanilla). A implementação é validada contra a solução fechada de Reiner-Rubinstein com erro relativo de 0,0115% e verifica a paridade Knock-In/Knock-Out com erro de 0,0000%. Os dados de mercado são coletados diariamente dos arquivos públicos da B3 (SPRD, PR, IN, IR), com esteira automatizada de precificação, validação e risco (VaR delta-gamma, 100.000 cenários, 99%/1 dia) rodando em produção sobre dados D-1 desde 03/08/2026. O resultado é entregue ao gestor através de um Structuring Desk interativo com 30 estratégias simuláveis em tempo real, implantado como aplicação institucional (dark mode) no Databricks Apps.

## 1. Introdução

Opções com barreira (Knock-Out e Knock-In) ocupam posição central no arsenal de mesas de tesouraria e gestores de carteira por dois motivos diretos:

1. **Redução de prêmio**: a condição de barreira embute um evento extremo que barateia a opção em relação ao vanilla equivalente — uma Down-and-Out Call a 70% do spot custa uma fração da call comum, viabilizando proteção e alavancagem com custo de carry menor.
2. **Estruturação flexível**: a mesma condição permite montar collars, spreads, forwards e binários de toque com perfis de risco sob medida para o cliente.

O desafio histórico é duplo: a precificação exige tratamento numérico rigoroso (o evento de barreira é sensível à discretização do caminho do ativo) e a operação diária exige uma esteira de dados confiável. Este projeto ataca os dois lados — motor matemático validado analiticamente e pipeline de dados de produção — em uma única plataforma.

## 2. Arquitetura de Dados e Tecnologia

O fluxo operacional é diário (execução ~08:02, dados D-1) e compõe-se de três rotinas encadeadas:

- **Ingestão**: coleta dos arquivos públicos da B3 (SPRD, PR, IN e IR, em ZIP/XML parseados com `lxml`) para pastas `data/YYYY-MM-DD/`, com backfill histórico completo desde 03/08/2026.
- **Precificação**: construção da superfície de volatilidade implícita e cálculo dos preços via Monte Carlo (200.000 caminhos, 252 passos, seed fixa), persistindo `RESULTS_superficieVol_*.csv` em Unity Catalog.
- **Validação e Risco**: artefatos `VALIDATION_*.json` (decisão auditável, APROVADO / APROVADO COM RESSALVAS / REPROVADO) e `RISK_*.json` (gregas, VaR e Expected Shortfall), encadeados por hash SHA-256 do arquivo de resultados — cada relatório de risco referencia exatamente o conjunto de dados que o originou.

A infraestrutura é totalmente serverless (AWS): orquestração no Databricks Jobs, armazenamento em Unity Catalog (Delta Tables) e interface implantada no Databricks Apps com tema institucional dark mode, sem sobrecarga de administração de clusters. Todo o código vive em repositório Git com branch única `main`, vendoring do pacote de pricing para o dashboard, e gates de sintaxe antes de cada commit.

## 3. Metodologia Matemática

O ativo-objeto (Ibovespa) é modelado por Movimento Browniano Geométrico sob a medida neutra ao risco, com taxa doméstica e dividend yield extraídos das curvas construídas a partir dos próprios arquivos da B3:

$$\frac{dS_t}{S_t} = (r - q)\,dt + \sigma\,dW_t$$

A precificação de cada estrutura segue quatro princípios:

1. **Monte Carlo com 200.000 caminhos e 252 passos discretos** por ano de maturidade, com seed fixa para reprodutibilidade auditável.
2. **Correção de discretização via Brownian Bridge**: o toque de barreira entre dois passos de simulação não é observável diretamente; a ponte browniana fornece a probabilidade condicional de travessia intra-passo, eliminando o viés de monitoramento discreto.
3. **Redução de variância por variável de controle**: cada precificador usa o payoff vanilla (com solução exata de Black-Scholes) como control variate, reduzindo o erro padrão em ordens de magnitude sem aumentar o custo de simulação.
4. **Intervalos de confiança de 95% reportados junto ao preço** em toda a cadeia (motor, validação, dashboard).

Para os binários de toque (Família D), o mesmo arcabouço estende-se à probabilidade de primeira passagem, com validação contra a fórmula analítica de primeira passagem do processo de Wiener.

## 4. Validação e Resultados (O Diferencial)

A validação quantitativa é o núcleo do projeto e foi conduzida em três frentes independentes:

- **Fórmula fechada**: implementação da solução de Reiner-Rubinstein para Down-and-Out Call, cobrindo ambos os regimes de strike (K ≥ H e K < H). O erro relativo do Monte Carlo contra a solução exata caiu para **0,0115%** após a combinação Brownian Bridge + variável de controle.
- **Paridade Knock-In/Knock-Out**: a identidade Vanilla = Knock-Out + Knock-In foi verificada em todas as direções (call/put) com erro de **0,0000%** — a soma dos preços simulados das duas condições de barreira reproduz o vanilla com precisão de ponto flutuante.
- **Validação econômica de estruturas combinadas**: cada estrutura da Família C reproduz exatamente a combinação de suas pernas (erro relativo 10⁻¹²), e o Knock-Out Forward converge para a paridade put-call analítica (S₀·e^(−qT) − K·e^(−rT)) quando a barreira é levada ao infinito. Notavelmente, o smoke revelou que o KO Forward custa **mais** que o forward vanilla: no knock-out as duas pernas morrem e o payoff vira zero em vez de S_T − K < 0 — a barreira elimina o downside do crash e agrega valor (prêmio da proteção implícita).

**Produção**: a esteira roda diariamente desde 03/08/2026. No ciclo mais recente (data de mercado 16/11/2026), o contrato de referência (Down-and-Out Call, strike ATM, barreira 70% do spot) foi precificado em R$ 24.969,61 com gregas completas — delta 0,8848; vega 195,03; theta −50,10; rho 593,87 — e VaR delta-gamma de R$ 3.709,76 a 99% / 1 dia (Expected Shortfall R$ 4.099,88), sobre 100.000 cenários de choque lognormal.

## 5. O Structuring Desk

A interface do gestor é um simulador interativo de estruturas, disponível 24/7 via Databricks Apps:

- **30 estratégias simuláveis em tempo real**: as 8 condições simples de barreira (Down/Up × In/Out × Call/Put) com rebate configurável, as barreiras dependentes da trajetória (Parisian/ParAsian em dias úteis, Step com mudança de nível e Soft com janela de monitoramento, todas com Call/Put, Down/Up e Out/In), a Família B completa (Double Knock-Out/Knock-In e dupla barreira KI/KO com convenção de ordem parametrizável: KI_first, KO_after_KI, KO_first ou sequential; Call ou Put), Knock-Out e Knock-In Straddle, as estruturas combinadas da Família C (Strangle, Risk Reversal, Collar, Call Spread, Put Spread, Knock-Out e Knock-In Forward, com variantes Up para Collar, Spreads e Strangle) e a Família D (One-Touch/No-Touch Up e Down, Double One-Touch/No-Touch e Range Accrual com taxa de accrual % a.a.) com payout/notional parametrizável em reais e benchmark analítico exibido ao lado do Monte Carlo.
- **Diagrama de payoff desenhado ao vivo** por estratégia, com a região de knock-out explícita e a barreira marcada no gráfico.
- **Travas de UX de tesouraria**: o desk bloqueia ou alerta sobre configurações economicamente degeneradas (barreira já violada pelo spot, prêmio praticamente zero, crédito líquido em estruturas vendidas), com mensagens institucionais que orientam o ajuste de parâmetros.
- **Métricas de risco junto ao preço**: cada cálculo reporta preço, intervalo de confiança, P(barreira caída) e prêmio da condição de barreira (diferença contra o vanilla), dando ao estruturador a leitura do custo econômico da barreira.

O gestor monta a estrutura, vê o payoff e o preço com IC em segundos — sem depender de planilha ou de área de risco para um first cut.

## 6. Limitações e Próximos Passos

O projeto documenta explicitamente suas fronteiras atuais:

- **Parisian contínua**: a Parisian é implementada com fechamentos diários (padrão de term sheet); a versão de tempo contínuo de excursão (Chesney-Jeanblanc-Yor) é extensão possível. Step e Soft já usam monitoramento contínuo via Brownian Bridge.
- **Calibração da superfície**: a superfície de volatilidade é interpolação dos strikes/vencimentos líquidos da B3; o upgrade natural é calibração paramétrica SVI/SABR com arbitragem-free constraints.
- **VaR model-based**: o VaR atual é delta-gamma com choques lognormais de spot (100.000 cenários, 99%/1 dia) e não incorpora choques conjuntos de volatilidade e curva de juros; as três limitações estão declaradas no próprio JSON de risco. A aproximação também perde precisão perto da barreira (o payoff ali é altamente não-linear).
- **Convenção de pagamento**: binários e rebate são pagos no vencimento; a convenção de pagamento no toque é extensão direta pendente.
- **Testes unitários formais** (KIs, combinadas, binários) para além dos gates e smokes econômicos que rodam hoje.

## 7. Conclusão

O projeto entrega uma ferramenta de nível institucional que integra três disciplinas em um único fluxo: rigor matemático (motor validado contra solução fechada com erro de 0,0115% e paridade exata), engenharia de software (pipeline diário auditável, com hash chaining dos artefatos, Git disciplinado e gates de deploy) e UX de tesouraria (desk interativo com 30 estratégias, travas de degenerescência e visualização de payoff em tempo real). Todo o stack é serverless e sustentável por uma pessoa, com custo operacional marginal. Para a mesa, o resultado prático é imediato: capacidade de estruturar, precificar e medir risco de opções com barreira no Ibovespa em segundos, com embasamento quantitativo auditável em cada número apresentado.
