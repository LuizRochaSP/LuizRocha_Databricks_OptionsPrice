# Databricks notebook source
# MAGIC %md
# MAGIC # Caderno de equações — Precificação de uma opção com barreira no Ibovespa
# MAGIC 
# MAGIC ## Do entendimento do produto à redução de variância
# MAGIC 
# MAGIC Este capítulo reúne, em uma única sequência, as equações utilizadas para precificar uma **call europeia down-and-out sem rebate** sobre o Ibovespa.
# MAGIC 
# MAGIC 1. Entender o produto
# MAGIC 2. Definir os dados de mercado
# MAGIC 3. Calcular a call vanilla por Black–Scholes
# MAGIC 4. Definir a dinâmica neutra ao risco
# MAGIC 5. Simular as trajetórias por Monte Carlo
# MAGIC 6. Corrigir a observação da barreira com Brownian Bridge
# MAGIC 7. Reduzir a variância usando a vanilla como controle
# MAGIC 
# MAGIC > **Objetivo:** calcular o valor presente esperado do payoff futuro sob uma medida neutra ao risco. Este é um modelo matemático de precificação, e não uma regressão para prever o Ibovespa.

# COMMAND ----------

# MAGIC %md
# MAGIC # Notação
# MAGIC 
# MAGIC | Símbolo | Significado |
# MAGIC |---|---|
# MAGIC | $S_0$ | nível atual do Ibovespa |
# MAGIC | $S_t$ | nível do Ibovespa no instante $t$ |
# MAGIC | $S_T$ | nível no vencimento |
# MAGIC | $K$ | strike ou preço de exercício |
# MAGIC | $H$ | barreira inferior |
# MAGIC | $T$ | prazo até o vencimento, em anos |
# MAGIC | $r$ | taxa de juros continuamente composta |
# MAGIC | $q$ | dividend yield ou carry contínuo |
# MAGIC | $\sigma$ | volatilidade anualizada |
# MAGIC | $N$ | número de trajetórias |
# MAGIC | $M$ | passos no tempo por trajetória |
# MAGIC | $\Delta t=T/M$ | tamanho de cada passo |
# MAGIC | $\Phi$ e $\phi$ | distribuição acumulada e densidade da normal padrão |
# MAGIC | $\mathbf{1}_A$ | indicador do evento $A$ |
# MAGIC | $\mathbb{E}^{\mathbb Q}$ | esperança sob a medida neutra ao risco |

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Entender o produto
# MAGIC 
# MAGIC ## 1.1 Call vanilla
# MAGIC 
# MAGIC O payoff de uma call europeia vanilla é:
# MAGIC 
# MAGIC $
# MAGIC X_{\mathrm{vanilla}}=(S_T-K)^+=\max(S_T-K,0)
# MAGIC $
# MAGIC 
# MAGIC Ela paga somente quando $S_T>K$.
# MAGIC 
# MAGIC ## 1.2 Call down-and-out sem rebate
# MAGIC 
# MAGIC Defina o menor nível alcançado pela trajetória contínua:
# MAGIC 
# MAGIC $
# MAGIC m_T=\min_{0\le t\le T}S_t
# MAGIC $
# MAGIC 
# MAGIC O payoff é:
# MAGIC 
# MAGIC $
# MAGIC X_{\mathrm{barreira}}
# MAGIC =
# MAGIC (S_T-K)^+\mathbf{1}_{\{m_T>H\}}
# MAGIC $
# MAGIC 
# MAGIC ou:
# MAGIC 
# MAGIC $
# MAGIC X_{\mathrm{barreira}}
# MAGIC =
# MAGIC \begin{cases}
# MAGIC (S_T-K)^+, & \text{se }S_t>H\text{ para todo }t\in[0,T],\\
# MAGIC 0, & \text{se }S_t\le H\text{ em algum instante.}
# MAGIC \end{cases}
# MAGIC $
# MAGIC 
# MAGIC A opção deixa de existir quando toca ou cruza $H$. Como é **sem rebate**, não existe pagamento compensatório no knock-out.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1.3 Relação entre vanilla e barreira
# MAGIC 
# MAGIC Para cada trajetória:
# MAGIC 
# MAGIC $
# MAGIC 0\le X_{\mathrm{barreira}}\le X_{\mathrm{vanilla}}
# MAGIC $
# MAGIC 
# MAGIC Consequentemente:
# MAGIC 
# MAGIC $
# MAGIC 0\le V_{\mathrm{barreira}}\le C_{\mathrm{BS}}
# MAGIC $
# MAGIC 
# MAGIC A vanilla e a barreira usam o mesmo valor terminal $S_T$. A diferença é que a barreira multiplica o payoff por um indicador de sobrevivência. Essa relação será essencial no passo 7.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Definir os dados de mercado
# MAGIC 
# MAGIC O vetor de entradas é:
# MAGIC 
# MAGIC $
# MAGIC \Theta=(S_0,K,H,T,r,q,\sigma)
# MAGIC $
# MAGIC 
# MAGIC Exemplo do projeto:
# MAGIC 
# MAGIC $
# MAGIC S_0=150.000,\quad K=155.000,\quad H=120.000,\quad T=0{,}5
# MAGIC $
# MAGIC 
# MAGIC $
# MAGIC r=12\%\text{ a.a.},\quad q=0\%\text{ a.a.},\quad \sigma=22\%\text{ a.a.}
# MAGIC $
# MAGIC 
# MAGIC As taxas $r$ e $q$ precisam usar a mesma convenção. Neste capítulo elas são continuamente compostas.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.1 Desconto, taxa e carry
# MAGIC 
# MAGIC O fator de desconto é:
# MAGIC 
# MAGIC $
# MAGIC D(0,T)=e^{-rT}
# MAGIC $
# MAGIC 
# MAGIC Se a entrada for uma taxa efetiva anual:
# MAGIC 
# MAGIC $
# MAGIC r=\ln(1+r_{\mathrm{efetiva}})
# MAGIC $
# MAGIC 
# MAGIC O carry líquido do índice é:
# MAGIC 
# MAGIC $
# MAGIC b=r-q
# MAGIC $
# MAGIC 
# MAGIC Sob hipóteses simplificadoras, o preço futuro satisfaz:
# MAGIC 
# MAGIC $
# MAGIC F_{0,T}=S_0e^{(r-q)T}
# MAGIC $
# MAGIC 
# MAGIC Logo, o dividend yield implícito é:
# MAGIC 
# MAGIC $
# MAGIC q_{\mathrm{impl}}(T)
# MAGIC =
# MAGIC r(T)-\frac{1}{T}\ln\left(\frac{F_{0,T}}{S_0}\right)
# MAGIC $
# MAGIC 
# MAGIC ou, usando fatores de desconto:
# MAGIC 
# MAGIC $
# MAGIC F_{0,T}=S_0\frac{e^{-qT}}{e^{-rT}}
# MAGIC $
# MAGIC 
# MAGIC Na prática, uma curva $q(T)$ por vencimento é mais realista do que um único valor constante.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.2 Volatilidade
# MAGIC 
# MAGIC Com retornos logarítmicos diários:
# MAGIC 
# MAGIC $
# MAGIC R_t=\ln\left(\frac{S_t}{S_{t-1}}\right)
# MAGIC $
# MAGIC 
# MAGIC uma estimativa histórica simples é:
# MAGIC 
# MAGIC $
# MAGIC \widehat{\sigma}_{\mathrm{hist}}
# MAGIC =
# MAGIC \operatorname{desvio\ padrão}(R_t)\sqrt{252}
# MAGIC $
# MAGIC 
# MAGIC Para precificação, normalmente usamos volatilidade implícita compatível com prazo, strike e mercado. Uma $\sigma$ constante ignora smile, skew e estrutura a termo.

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Calcular a vanilla por Black–Scholes
# MAGIC 
# MAGIC A call vanilla com dividend yield contínuo é:
# MAGIC 
# MAGIC $
# MAGIC C_{\mathrm{BS}}
# MAGIC =
# MAGIC S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2)
# MAGIC $
# MAGIC 
# MAGIC com:
# MAGIC 
# MAGIC $
# MAGIC d_1=
# MAGIC \frac{\ln(S_0/K)+(r-q+\tfrac12\sigma^2)T}
# MAGIC {\sigma\sqrt T}
# MAGIC $
# MAGIC 
# MAGIC $
# MAGIC d_2=d_1-\sigma\sqrt T
# MAGIC $
# MAGIC 
# MAGIC Os termos podem ser lidos como:
# MAGIC 
# MAGIC $
# MAGIC \underbrace{S_0e^{-qT}\Phi(d_1)}_{\text{parcela associada ao ativo}}
# MAGIC -
# MAGIC \underbrace{Ke^{-rT}\Phi(d_2)}_{\text{strike descontado}}
# MAGIC $

# COMMAND ----------

# MAGIC %md
# MAGIC # 3.1 Distribuição terminal do modelo
# MAGIC 
# MAGIC Sob a dinâmica lognormal neutra ao risco:
# MAGIC 
# MAGIC $
# MAGIC \ln S_T
# MAGIC \sim
# MAGIC \mathcal N\left(
# MAGIC \ln S_0+(r-q-\tfrac12\sigma^2)T,\,
# MAGIC \sigma^2T
# MAGIC \right)
# MAGIC $
# MAGIC 
# MAGIC Assim:
# MAGIC 
# MAGIC $
# MAGIC S_T
# MAGIC =
# MAGIC S_0\exp\left[
# MAGIC (r-q-\tfrac12\sigma^2)T+\sigma\sqrt T\,Z
# MAGIC \right],
# MAGIC \qquad Z\sim\mathcal N(0,1)
# MAGIC $
# MAGIC 
# MAGIC A fórmula Black–Scholes fornecerá uma referência exata, dentro das hipóteses do modelo, para a vanilla.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Definir a dinâmica neutra ao risco
# MAGIC 
# MAGIC Sob uma medida real $\mathbb P$, um modelo simplificado seria:
# MAGIC 
# MAGIC $
# MAGIC \frac{dS_t}{S_t}
# MAGIC =
# MAGIC (\mu-q)dt+\sigma dW_t^{\mathbb P}
# MAGIC $
# MAGIC 
# MAGIC Para precificação por não arbitragem, utilizamos a medida neutra ao risco $\mathbb Q$:
# MAGIC 
# MAGIC $
# MAGIC \frac{dS_t}{S_t}
# MAGIC =
# MAGIC (r-q)dt+\sigma dW_t^{\mathbb Q}
# MAGIC $
# MAGIC 
# MAGIC O retorno esperado $\mu$ é substituído pelo carry $r-q$. Isso não é uma previsão de crescimento do índice; é a dinâmica usada para valorar o payoff de forma consistente com preços de mercado.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.1 Solução do movimento geométrico browniano
# MAGIC 
# MAGIC Aplicando o lema de Itô:
# MAGIC 
# MAGIC $
# MAGIC d\ln S_t
# MAGIC =
# MAGIC \left(r-q-\frac12\sigma^2\right)dt
# MAGIC +
# MAGIC \sigma dW_t^{\mathbb Q}
# MAGIC $
# MAGIC 
# MAGIC Em um intervalo $\Delta t$:
# MAGIC 
# MAGIC $
# MAGIC S_{t+\Delta t}
# MAGIC =
# MAGIC S_t\exp\left[
# MAGIC \left(r-q-\frac12\sigma^2\right)\Delta t
# MAGIC +
# MAGIC \sigma\sqrt{\Delta t}\,Z
# MAGIC \right]
# MAGIC $
# MAGIC 
# MAGIC com:
# MAGIC 
# MAGIC $
# MAGIC Z\sim\mathcal N(0,1)
# MAGIC $
# MAGIC 
# MAGIC Essa é a solução exata do GBM nos pontos da grade.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.2 Princípio de precificação
# MAGIC 
# MAGIC Para um payoff $X_T$ pago no vencimento:
# MAGIC 
# MAGIC $
# MAGIC V_0=e^{-rT}\mathbb E^{\mathbb Q}[X_T]
# MAGIC $
# MAGIC 
# MAGIC Para a vanilla:
# MAGIC 
# MAGIC $
# MAGIC C_0=e^{-rT}\mathbb E^{\mathbb Q}[(S_T-K)^+]
# MAGIC $
# MAGIC 
# MAGIC Para a down-and-out call:
# MAGIC 
# MAGIC $
# MAGIC V_0^{\mathrm{DOC}}
# MAGIC =
# MAGIC e^{-rT}\mathbb E^{\mathbb Q}
# MAGIC \left[
# MAGIC (S_T-K)^+\mathbf{1}_{\{m_T>H\}}
# MAGIC \right]
# MAGIC $
# MAGIC 
# MAGIC É essa esperança que o Monte Carlo aproximará.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Simular trajetórias por Monte Carlo
# MAGIC 
# MAGIC Dividimos $[0,T]$ em $M$ passos:
# MAGIC 
# MAGIC $
# MAGIC t_j=j\Delta t,\qquad
# MAGIC \Delta t=\frac{T}{M},\qquad j=0,\ldots,M
# MAGIC $
# MAGIC 
# MAGIC Para a trajetória $i=1,\ldots,N$:
# MAGIC 
# MAGIC $
# MAGIC S_{i,j+1}
# MAGIC =
# MAGIC S_{i,j}\exp\left[
# MAGIC \left(r-q-\frac12\sigma^2\right)\Delta t
# MAGIC +
# MAGIC \sigma\sqrt{\Delta t}\,Z_{i,j}
# MAGIC \right]
# MAGIC $
# MAGIC 
# MAGIC $
# MAGIC Z_{i,j}\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),
# MAGIC \qquad S_{i,0}=S_0
# MAGIC $
# MAGIC 
# MAGIC Uma semente fixa torna o experimento reproduzível.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.1 Payoffs simulados
# MAGIC 
# MAGIC O indicador discreto de sobrevivência seria:
# MAGIC 
# MAGIC $
# MAGIC I_i^{\mathrm{disc}}
# MAGIC =
# MAGIC \mathbf{1}_{\{\min_{j=0,\ldots,M}S_{i,j}>H\}}
# MAGIC $
# MAGIC 
# MAGIC Payoff descontado da barreira:
# MAGIC 
# MAGIC $
# MAGIC X_i=e^{-rT}(S_{i,M}-K)^+I_i
# MAGIC $
# MAGIC 
# MAGIC Payoff descontado da vanilla, usando a mesma trajetória:
# MAGIC 
# MAGIC $
# MAGIC Y_i=e^{-rT}(S_{i,M}-K)^+
# MAGIC $
# MAGIC 
# MAGIC Estimadores brutos:
# MAGIC 
# MAGIC $
# MAGIC \overline X=\frac1N\sum_{i=1}^N X_i
# MAGIC $
# MAGIC 
# MAGIC $
# MAGIC \overline Y=\frac1N\sum_{i=1}^N Y_i
# MAGIC $
# MAGIC 
# MAGIC Pela lei dos grandes números:
# MAGIC 
# MAGIC $
# MAGIC \overline X\xrightarrow[N\to\infty]{}V_0^{\mathrm{DOC}}
# MAGIC $

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.2 Erro padrão e intervalo de confiança
# MAGIC 
# MAGIC Variância amostral:
# MAGIC 
# MAGIC $
# MAGIC s_X^2
# MAGIC =
# MAGIC \frac{1}{N-1}\sum_{i=1}^N(X_i-\overline X)^2
# MAGIC $
# MAGIC 
# MAGIC Erro padrão da média:
# MAGIC 
# MAGIC $
# MAGIC SE(\overline X)=\frac{s_X}{\sqrt N}
# MAGIC $
# MAGIC 
# MAGIC Intervalo aproximado de 95%:
# MAGIC 
# MAGIC $
# MAGIC IC_{95\%}
# MAGIC =
# MAGIC \left[
# MAGIC \overline X-1{,}96SE,\,
# MAGIC \overline X+1{,}96SE
# MAGIC \right]
# MAGIC $
# MAGIC 
# MAGIC Como:
# MAGIC 
# MAGIC $
# MAGIC SE\propto\frac1{\sqrt N}
# MAGIC $
# MAGIC 
# MAGIC reduzir o erro pela metade apenas aumentando a força bruta exige cerca de quatro vezes mais trajetórias.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.3 Probabilidade de knock-out
# MAGIC 
# MAGIC O estimador da probabilidade neutra ao risco de knock-out é:
# MAGIC 
# MAGIC $
# MAGIC \widehat p_{\mathrm{KO}}
# MAGIC =
# MAGIC \frac1N\sum_{i=1}^N
# MAGIC \mathbf{1}_{\{\text{trajetória }i\text{ atingiu }H\}}
# MAGIC $
# MAGIC 
# MAGIC A probabilidade de sobrevivência é:
# MAGIC 
# MAGIC $
# MAGIC \widehat p_{\mathrm{surv}}=1-\widehat p_{\mathrm{KO}}
# MAGIC $
# MAGIC 
# MAGIC Essas probabilidades estão sob $\mathbb Q$. Elas não são automaticamente previsões de risco real sob $\mathbb P$.

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Brownian Bridge
# MAGIC 
# MAGIC Mesmo quando dois pontos consecutivos estão acima de $H$, a trajetória contínua pode ter cruzado a barreira entre eles.
# MAGIC 
# MAGIC Defina:
# MAGIC 
# MAGIC $
# MAGIC x_j=\ln S_{i,j},\qquad x_{j+1}=\ln S_{i,j+1},\qquad h=\ln H
# MAGIC $
# MAGIC 
# MAGIC Se $S_{i,j}>H$ e $S_{i,j+1}>H$, a probabilidade condicional de cruzamento é:
# MAGIC 
# MAGIC $
# MAGIC p_{i,j}^{\mathrm{hit}}
# MAGIC =
# MAGIC \exp\left[
# MAGIC -\frac{2(x_j-h)(x_{j+1}-h)}
# MAGIC {\sigma^2\Delta t}
# MAGIC \right]
# MAGIC $
# MAGIC 
# MAGIC Forma equivalente:
# MAGIC 
# MAGIC $
# MAGIC p_{i,j}^{\mathrm{hit}}
# MAGIC =
# MAGIC \exp\left[
# MAGIC -\frac{
# MAGIC 2\ln(S_{i,j}/H)\ln(S_{i,j+1}/H)
# MAGIC }{
# MAGIC \sigma^2\Delta t
# MAGIC }
# MAGIC \right]
# MAGIC $
# MAGIC 
# MAGIC Se algum extremo for menor ou igual a $H$:
# MAGIC 
# MAGIC $
# MAGIC p_{i,j}^{\mathrm{hit}}=1
# MAGIC $

# COMMAND ----------

# MAGIC %md
# MAGIC # 6.1 Aplicação da Brownian Bridge
# MAGIC 
# MAGIC ## Método A — sorteio
# MAGIC 
# MAGIC Geramos:
# MAGIC 
# MAGIC $
# MAGIC U_{i,j}\sim\operatorname{Uniforme}(0,1)
# MAGIC $
# MAGIC 
# MAGIC e registramos cruzamento quando:
# MAGIC 
# MAGIC $
# MAGIC U_{i,j}<p_{i,j}^{\mathrm{hit}}
# MAGIC $
# MAGIC 
# MAGIC ## Método B — peso de sobrevivência
# MAGIC 
# MAGIC Em cada intervalo:
# MAGIC 
# MAGIC $
# MAGIC p_{i,j}^{\mathrm{surv}}=1-p_{i,j}^{\mathrm{hit}}
# MAGIC $
# MAGIC 
# MAGIC Para a trajetória completa:
# MAGIC 
# MAGIC $
# MAGIC P_i^{\mathrm{surv}}
# MAGIC =
# MAGIC \prod_{j=0}^{M-1}(1-p_{i,j}^{\mathrm{hit}})
# MAGIC $
# MAGIC 
# MAGIC Payoff suavizado:
# MAGIC 
# MAGIC $
# MAGIC X_i^{\mathrm{BB}}
# MAGIC =
# MAGIC e^{-rT}(S_{i,M}-K)^+P_i^{\mathrm{surv}}
# MAGIC $
# MAGIC 
# MAGIC O método por peso costuma introduzir menos ruído do que novos sorteios uniformes.

# COMMAND ----------

# MAGIC %md
# MAGIC # 6.2 Alcance da correção
# MAGIC 
# MAGIC A Brownian Bridge corrige cruzamentos não observados entre os pontos da grade, dentro das hipóteses do GBM.
# MAGIC 
# MAGIC Ela não corrige:
# MAGIC 
# MAGIC - volatilidade ou carry mal calibrados;
# MAGIC - saltos e gaps;
# MAGIC - volatilidade estocástica;
# MAGIC - liquidez e custos de transação;
# MAGIC - diferenças contratuais;
# MAGIC - risco geral de modelo.
# MAGIC 
# MAGIC Ela trata um viés numérico específico, e não todos os erros de precificação.

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Redução de variância com a vanilla
# MAGIC 
# MAGIC Usamos:
# MAGIC 
# MAGIC $
# MAGIC X_i=\text{payoff descontado da barreira}
# MAGIC $
# MAGIC 
# MAGIC $
# MAGIC Y_i=\text{payoff descontado da vanilla na mesma trajetória}
# MAGIC $
# MAGIC 
# MAGIC Como Black–Scholes fornece:
# MAGIC 
# MAGIC $
# MAGIC \mathbb E^{\mathbb Q}[Y_i]=C_{\mathrm{BS}}
# MAGIC $
# MAGIC 
# MAGIC definimos a observação corrigida:
# MAGIC 
# MAGIC $
# MAGIC X_i^{\mathrm{CV}}
# MAGIC =
# MAGIC X_i-\beta(Y_i-C_{\mathrm{BS}})
# MAGIC $
# MAGIC 
# MAGIC e o estimador final:
# MAGIC 
# MAGIC $
# MAGIC \widehat V_{\mathrm{CV}}
# MAGIC =
# MAGIC \overline X-\widehat\beta(\overline Y-C_{\mathrm{BS}})
# MAGIC $
# MAGIC 
# MAGIC O símbolo $\widehat{\phantom V}$ indica uma estimativa calculada com amostra finita.

# COMMAND ----------

# MAGIC %md
# MAGIC # 7.1 Coeficiente ótimo
# MAGIC 
# MAGIC A variância da observação corrigida é:
# MAGIC 
# MAGIC $
# MAGIC \operatorname{Var}(X-\beta Y)
# MAGIC =
# MAGIC \operatorname{Var}(X)
# MAGIC +\beta^2\operatorname{Var}(Y)
# MAGIC -2\beta\operatorname{Cov}(X,Y)
# MAGIC $
# MAGIC 
# MAGIC Derivando em relação a $\beta$ e igualando a zero:
# MAGIC 
# MAGIC $
# MAGIC 2\beta\operatorname{Var}(Y)
# MAGIC -2\operatorname{Cov}(X,Y)=0
# MAGIC $
# MAGIC 
# MAGIC Logo:
# MAGIC 
# MAGIC $
# MAGIC \beta^*
# MAGIC =
# MAGIC \frac{\operatorname{Cov}(X,Y)}
# MAGIC {\operatorname{Var}(Y)}
# MAGIC $
# MAGIC 
# MAGIC Na amostra:
# MAGIC 
# MAGIC $
# MAGIC \widehat\beta
# MAGIC =
# MAGIC \frac{
# MAGIC \sum_{i=1}^N(X_i-\overline X)(Y_i-\overline Y)
# MAGIC }{
# MAGIC \sum_{i=1}^N(Y_i-\overline Y)^2
# MAGIC }
# MAGIC $

# COMMAND ----------

# MAGIC %md
# MAGIC # 7.2 Intuição da correção
# MAGIC 
# MAGIC Se:
# MAGIC 
# MAGIC $
# MAGIC \overline Y>C_{\mathrm{BS}}
# MAGIC $
# MAGIC 
# MAGIC a vanilla simulada ficou acima do valor exato. Com correlação positiva e $\widehat\beta>0$, retiramos parte desse excesso da barreira.
# MAGIC 
# MAGIC Se:
# MAGIC 
# MAGIC $
# MAGIC \overline Y<C_{\mathrm{BS}}
# MAGIC $
# MAGIC 
# MAGIC a diferença é negativa e a correção aumenta a estimativa da barreira.
# MAGIC 
# MAGIC O ruído observável da vanilla é:
# MAGIC 
# MAGIC $
# MAGIC \overline Y-C_{\mathrm{BS}}
# MAGIC $
# MAGIC 
# MAGIC Como vanilla e barreira usam os mesmos caminhos, parte desse ruído também está em $\overline X$. A variável de controle procura removê-lo.

# COMMAND ----------

# MAGIC %md
# MAGIC # 7.3 Ganho teórico
# MAGIC 
# MAGIC Defina:
# MAGIC 
# MAGIC $
# MAGIC \rho_{XY}
# MAGIC =
# MAGIC \frac{\operatorname{Cov}(X,Y)}
# MAGIC {\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}}
# MAGIC $
# MAGIC 
# MAGIC Com o coeficiente ótimo:
# MAGIC 
# MAGIC $
# MAGIC \operatorname{Var}(X^{\mathrm{CV}})
# MAGIC =
# MAGIC \operatorname{Var}(X)(1-\rho_{XY}^2)
# MAGIC $
# MAGIC 
# MAGIC Quanto mais próximo $|\rho_{XY}|$ estiver de 1, maior tende a ser a redução de variância.
# MAGIC 
# MAGIC O erro padrão corrigido é:
# MAGIC 
# MAGIC $
# MAGIC s_{\mathrm{CV}}^2
# MAGIC =
# MAGIC \frac1{N-1}\sum_{i=1}^N
# MAGIC (X_i^{\mathrm{CV}}-\overline{X^{\mathrm{CV}}})^2
# MAGIC $
# MAGIC 
# MAGIC $
# MAGIC SE_{\mathrm{CV}}=\frac{s_{\mathrm{CV}}}{\sqrt N}
# MAGIC $
# MAGIC 
# MAGIC $
# MAGIC IC_{95\%}^{\mathrm{CV}}
# MAGIC =
# MAGIC \widehat V_{\mathrm{CV}}\pm1{,}96SE_{\mathrm{CV}}
# MAGIC $

# COMMAND ----------

# MAGIC %md
# MAGIC # Encadeamento completo
# MAGIC 
# MAGIC ## 1 — Produto
# MAGIC 
# MAGIC $
# MAGIC X_{\mathrm{barreira}}
# MAGIC =
# MAGIC (S_T-K)^+\mathbf{1}_{\{\min_{0\le t\le T}S_t>H\}}
# MAGIC $
# MAGIC 
# MAGIC ## 2 — Mercado
# MAGIC 
# MAGIC $
# MAGIC \Theta=(S_0,K,H,T,r,q,\sigma)
# MAGIC $
# MAGIC 
# MAGIC ## 3 — Vanilla analítica
# MAGIC 
# MAGIC $
# MAGIC C_{\mathrm{BS}}
# MAGIC =
# MAGIC S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2)
# MAGIC $
# MAGIC 
# MAGIC ## 4 — Dinâmica neutra ao risco
# MAGIC 
# MAGIC $
# MAGIC \frac{dS_t}{S_t}=(r-q)dt+\sigma dW_t^{\mathbb Q}
# MAGIC $
# MAGIC 
# MAGIC ## 5 — Monte Carlo
# MAGIC 
# MAGIC $
# MAGIC S_{t+\Delta t}
# MAGIC =
# MAGIC S_t\exp[(r-q-\tfrac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}Z]
# MAGIC $
# MAGIC 
# MAGIC ## 6 — Brownian Bridge
# MAGIC 
# MAGIC $
# MAGIC p_{\mathrm{hit}}
# MAGIC =
# MAGIC \exp\left[
# MAGIC -\frac{2\ln(S_t/H)\ln(S_{t+\Delta t}/H)}
# MAGIC {\sigma^2\Delta t}
# MAGIC \right]
# MAGIC $
# MAGIC 
# MAGIC ## 7 — Variável de controle
# MAGIC 
# MAGIC $
# MAGIC \widehat V_{\mathrm{CV}}
# MAGIC =
# MAGIC \overline X-\widehat\beta(\overline Y-C_{\mathrm{BS}})
# MAGIC $

# COMMAND ----------

# MAGIC %md
# MAGIC # Exemplo do projeto
# MAGIC 
# MAGIC Entradas:
# MAGIC 
# MAGIC $
# MAGIC S_0=150.000,\ K=155.000,\ H=120.000,\ T=0{,}5,\ r=0{,}12,\ q=0,\ \sigma=0{,}22
# MAGIC $
# MAGIC 
# MAGIC Configuração ilustrativa:
# MAGIC 
# MAGIC $
# MAGIC N=50.000,\qquad M=63
# MAGIC $
# MAGIC 
# MAGIC Resultado já observado no projeto, sujeito ao método e à semente:
# MAGIC 
# MAGIC | Medida | Resultado aproximado |
# MAGIC |---|---:|
# MAGIC | Preço da barreira | 11.322,54 pontos |
# MAGIC | Preço vanilla | 11.327,88 pontos |
# MAGIC | Erro padrão | 1,01 ponto |
# MAGIC | Intervalo de 95% | [11.320,55; 11.324,52] |
# MAGIC | Probabilidade neutra ao risco de knock-out | 9,22% |
# MAGIC 
# MAGIC A proximidade dos preços ocorreu porque a barreira estava relativamente distante do spot no cenário utilizado. Não é uma regra geral.

# COMMAND ----------

# MAGIC %md
# MAGIC # Verificações de consistência
# MAGIC 
# MAGIC 1. A opção down-and-out deve começar viva:
# MAGIC $
# MAGIC H<S_0
# MAGIC $
# MAGIC 
# MAGIC 2. Limites do preço:
# MAGIC $
# MAGIC 0\le V_{\mathrm{barreira}}\le C_{\mathrm{BS}}
# MAGIC $
# MAGIC 
# MAGIC 3. Barreira muito baixa:
# MAGIC $
# MAGIC H\downarrow0\Longrightarrow V_{\mathrm{barreira}}\to C_{\mathrm{BS}}
# MAGIC $
# MAGIC 
# MAGIC 4. Barreira se aproxima do spot:
# MAGIC $
# MAGIC H\uparrow S_0\Longrightarrow V_{\mathrm{barreira}}\to0
# MAGIC $
# MAGIC para monitoramento contínuo.
# MAGIC 
# MAGIC 5. Convergência estatística:
# MAGIC $
# MAGIC SE\propto1/\sqrt N
# MAGIC $
# MAGIC 
# MAGIC 6. Ao aumentar $M$, o preço deve estabilizar. A Brownian Bridge deve reduzir a sensibilidade ao número de passos.
# MAGIC 
# MAGIC 7. Mesmos parâmetros e mesma semente devem reproduzir o resultado.

# COMMAND ----------

# MAGIC %md
# MAGIC # Limitações
# MAGIC 
# MAGIC ## Do produto
# MAGIC 
# MAGIC - O contrato é extinto ao atingir a barreira, mesmo que o índice depois se recupere.
# MAGIC - Sem rebate, não há compensação pelo knock-out.
# MAGIC - Liquidez, spread e regras exatas de monitoramento importam.
# MAGIC - O preço e o hedge podem mudar rapidamente perto da barreira.
# MAGIC 
# MAGIC ## Do modelo
# MAGIC 
# MAGIC O GBM–Black–Scholes pressupõe:
# MAGIC 
# MAGIC $
# MAGIC \sigma=\text{constante},\qquad r,q=\text{determinísticos}
# MAGIC $
# MAGIC 
# MAGIC e trajetórias contínuas, sem saltos. Na vida real existem smile e skew, volatilidade variável, gaps, curvas de juros e dividendos, custos, hedge discreto e risco de liquidez.
# MAGIC 
# MAGIC > A variável de controle reduz o **erro estatístico do Monte Carlo**. Ela não elimina erro de modelo, calibração ou dados.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusão
# MAGIC 
# MAGIC A identidade central é:
# MAGIC 
# MAGIC $
# MAGIC \boxed{
# MAGIC V_0=e^{-rT}\mathbb E^{\mathbb Q}[X_T]
# MAGIC }
# MAGIC $
# MAGIC 
# MAGIC Para a opção com barreira:
# MAGIC 
# MAGIC $
# MAGIC V_0^{\mathrm{DOC}}
# MAGIC =
# MAGIC e^{-rT}\mathbb E^{\mathbb Q}
# MAGIC \left[
# MAGIC (S_T-K)^+\mathbf{1}_{\{\min S_t>H\}}
# MAGIC \right]
# MAGIC $
# MAGIC 
# MAGIC O Monte Carlo aproxima essa esperança; a Brownian Bridge trata cruzamentos entre os pontos da grade; e a vanilla remove parte do ruído comum:
# MAGIC 
# MAGIC $
# MAGIC \boxed{
# MAGIC \widehat V_{\mathrm{CV}}
# MAGIC =
# MAGIC \overline X-\widehat\beta(\overline Y-C_{\mathrm{BS}})
# MAGIC }
# MAGIC $
# MAGIC 
# MAGIC Os sete passos constituem um processo único: **definir o contrato, calibrar o mercado, obter uma referência analítica, modelar sob $\mathbb Q$, simular, tratar a barreira e aumentar a eficiência estatística**.
