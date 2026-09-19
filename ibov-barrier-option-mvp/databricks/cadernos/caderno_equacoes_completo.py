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
# MAGIC | S₀ | nível atual do Ibovespa |
# MAGIC | Sₜ | nível do Ibovespa no instante t |
# MAGIC | Sₜ (vencimento) | nível no vencimento |
# MAGIC | K | strike ou preço de exercício |
# MAGIC | H | barreira inferior |
# MAGIC | T | prazo até o vencimento, em anos |
# MAGIC | r | taxa de juros continuamente composta |
# MAGIC | q | dividend yield ou carry contínuo |
# MAGIC | σ | volatilidade anualizada |
# MAGIC | N | número de trajetórias |
# MAGIC | M | passos no tempo por trajetória |
# MAGIC | Δt = T/M | tamanho de cada passo |
# MAGIC | Φ e φ | distribuição acumulada e densidade da normal padrão |
# MAGIC | indicador 1₍A₎ | indicador do evento A |
# MAGIC | E sob Q | esperança sob a medida neutra ao risco |

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Entender o produto
# MAGIC
# MAGIC ## 1.1 Call vanilla
# MAGIC
# MAGIC O payoff de uma call europeia vanilla é:
# MAGIC
# MAGIC $$X_{\mathrm{vanilla}}=(S_T-K)^+=\max(S_T-K,0)$$
# MAGIC
# MAGIC Ela paga somente quando Sₜ > K.
# MAGIC
# MAGIC ## 1.2 Call down-and-out sem rebate
# MAGIC
# MAGIC Defina o menor nível alcançado pela trajetória contínua:
# MAGIC
# MAGIC $$m_T=\min_{0\le t\le T}S_t$$
# MAGIC
# MAGIC O payoff é:
# MAGIC
# MAGIC $$X_{\mathrm{barreira}}=(S_T-K)^+\mathbf{1}_{\{m_T>H\}}$$
# MAGIC
# MAGIC ou:
# MAGIC
# MAGIC $$X_{\mathrm{barreira}}=\begin{cases} (S_T-K)^+, & \text{se }S_t>H\text{ para todo }t\in[0,T],\\ 0, & \text{se }S_t\le H\text{ em algum instante.} \end{cases}$$
# MAGIC
# MAGIC A opção deixa de existir quando toca ou cruza H. Como é **sem rebate**, não existe pagamento compensatório no knock-out.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1.3 Relação entre vanilla e barreira
# MAGIC
# MAGIC Para cada trajetória:
# MAGIC
# MAGIC $$0\le X_{\mathrm{barreira}}\le X_{\mathrm{vanilla}}$$
# MAGIC
# MAGIC Consequentemente:
# MAGIC
# MAGIC $$0\le V_{\mathrm{barreira}}\le C_{\mathrm{BS}}$$
# MAGIC
# MAGIC A vanilla e a barreira usam o mesmo valor terminal Sₜ (vencimento). A diferença é que a barreira multiplica o payoff por um indicador de sobrevivência. Essa relação será essencial no passo 7.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Definir os dados de mercado
# MAGIC
# MAGIC O vetor de entradas é:
# MAGIC
# MAGIC $$\Theta=(S_0,K,H,T,r,q,\sigma)$$
# MAGIC
# MAGIC Exemplo do projeto:
# MAGIC
# MAGIC $$S_0=150.000,\quad K=155.000,\quad H=120.000,\quad T=0{,}5$$
# MAGIC
# MAGIC $$r=12\%\text{ a.a.},\quad q=0\%\text{ a.a.},\quad \sigma=22\%\text{ a.a.}$$
# MAGIC
# MAGIC As taxas r e q precisam usar a mesma convenção. Neste capítulo elas são continuamente compostas.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.1 Desconto, taxa e carry
# MAGIC
# MAGIC O fator de desconto é:
# MAGIC
# MAGIC $$D(0,T)=e^{-rT}$$
# MAGIC
# MAGIC Se a entrada for uma taxa efetiva anual:
# MAGIC
# MAGIC $$r=\ln(1+r_{\mathrm{efetiva}})$$
# MAGIC
# MAGIC O carry líquido do índice é:
# MAGIC
# MAGIC $$b=r-q$$
# MAGIC
# MAGIC Sob hipóteses simplificadoras, o preço futuro satisfaz:
# MAGIC
# MAGIC $$F_{0,T}=S_0e^{(r-q)T}$$
# MAGIC
# MAGIC Logo, o dividend yield implícito é:
# MAGIC
# MAGIC $$q_{\mathrm{impl}}(T)=r(T)-\frac{1}{T}\ln\left(\frac{F_{0,T}}{S_0}\right)$$
# MAGIC
# MAGIC ou, usando fatores de desconto:
# MAGIC
# MAGIC $$F_{0,T}=S_0\frac{e^{-qT}}{e^{-rT}}$$
# MAGIC
# MAGIC Na prática, uma curva q(T) por vencimento é mais realista do que um único valor constante.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.2 Volatilidade
# MAGIC
# MAGIC Com retornos logarítmicos diários:
# MAGIC
# MAGIC $$R_t=\ln\left(\frac{S_t}{S_{t-1}}\right)$$
# MAGIC
# MAGIC uma estimativa histórica simples é:
# MAGIC
# MAGIC $$\widehat{\sigma}_{\mathrm{hist}}=\mathrm{DP}(R_t)\sqrt{252}$$
# MAGIC
# MAGIC Para precificação, normalmente usamos volatilidade implícita compatível com prazo, strike e mercado. Uma σ constante ignora smile, skew e estrutura a termo.

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Calcular a vanilla por Black–Scholes
# MAGIC
# MAGIC A call vanilla com dividend yield contínuo é:
# MAGIC
# MAGIC $$C_{\mathrm{BS}}=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2)$$
# MAGIC
# MAGIC com:
# MAGIC
# MAGIC $$d_1=\frac{\ln(S_0/K)+(r-q+\tfrac12\sigma^2)T} {\sigma\sqrt T}$$
# MAGIC
# MAGIC $$d_2=d_1-\sigma\sqrt T$$
# MAGIC
# MAGIC Os termos podem ser lidos como:
# MAGIC
# MAGIC $$\underbrace{S_0e^{-qT}\Phi(d_1)}_{\text{parcela associada ao ativo}} - \underbrace{Ke^{-rT}\Phi(d_2)}_{\text{strike descontado}}$$

# COMMAND ----------

# MAGIC %md
# MAGIC # 3.1 Distribuição terminal do modelo
# MAGIC
# MAGIC Sob a dinâmica lognormal neutra ao risco:
# MAGIC
# MAGIC $$\ln S_T \sim \mathcal N\left( \ln S_0+(r-q-\tfrac12\sigma^2)T,\, \sigma^2T \right)$$
# MAGIC
# MAGIC Assim:
# MAGIC
# MAGIC $$S_T=S_0\exp\left[ (r-q-\tfrac12\sigma^2)T+\sigma\sqrt T\,Z \right], \qquad Z\sim\mathcal N(0,1)$$
# MAGIC
# MAGIC A fórmula Black–Scholes fornecerá uma referência exata, dentro das hipóteses do modelo, para a vanilla.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Definir a dinâmica neutra ao risco
# MAGIC
# MAGIC Sob uma medida real P, um modelo simplificado seria:
# MAGIC
# MAGIC $$\frac{dS_t}{S_t}=(\mu-q)dt+\sigma dW_t^{\mathbb P}$$
# MAGIC
# MAGIC Para precificação por não arbitragem, utilizamos a medida neutra ao risco Q:
# MAGIC
# MAGIC $$\frac{dS_t}{S_t}=(r-q)dt+\sigma dW_t^{\mathbb Q}$$
# MAGIC
# MAGIC O retorno esperado μ é substituído pelo carry r − q. Isso não é uma previsão de crescimento do índice; é a dinâmica usada para valorar o payoff de forma consistente com preços de mercado.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.1 Solução do movimento geométrico browniano
# MAGIC
# MAGIC Aplicando o lema de Itô:
# MAGIC
# MAGIC $$d\ln S_t=\left(r-q-\frac12\sigma^2\right)dt + \sigma dW_t^{\mathbb Q}$$
# MAGIC
# MAGIC Em um intervalo Δt:
# MAGIC
# MAGIC $$S_{t+\Delta t}=S_t\exp\left[ \left(r-q-\frac12\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z \right]$$
# MAGIC
# MAGIC com:
# MAGIC
# MAGIC $$Z\sim\mathcal N(0,1)$$
# MAGIC
# MAGIC Essa é a solução exata do GBM nos pontos da grade.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.2 Princípio de precificação
# MAGIC
# MAGIC Para um payoff Xₜ pago no vencimento:
# MAGIC
# MAGIC $$V_0=e^{-rT}\mathbb E^{\mathbb Q}[X_T]$$
# MAGIC
# MAGIC Para a vanilla:
# MAGIC
# MAGIC $$C_0=e^{-rT}\mathbb E^{\mathbb Q}[(S_T-K)^+]$$
# MAGIC
# MAGIC Para a down-and-out call:
# MAGIC
# MAGIC $$V_0^{\mathrm{DOC}}=e^{-rT}\mathbb E^{\mathbb Q} \left[ (S_T-K)^+\mathbf{1}_{\{m_T>H\}} \right]$$
# MAGIC
# MAGIC É essa esperança que o Monte Carlo aproximará.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Simular trajetórias por Monte Carlo
# MAGIC
# MAGIC Dividimos [0,T] em M passos:
# MAGIC
# MAGIC $$t_j=j\Delta t,\qquad \Delta t=\frac{T}{M},\qquad j=0,\ldots,M$$
# MAGIC
# MAGIC Para a trajetória i = 1,…,N:
# MAGIC
# MAGIC $$S_{i,j+1}=S_{i,j}\exp\left[ \left(r-q-\frac12\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_{i,j} \right]$$
# MAGIC
# MAGIC $$Z_{i,j}\overset{\mathrm{iid}}{\sim}\mathcal N(0,1), \qquad S_{i,0}=S_0$$
# MAGIC
# MAGIC Uma semente fixa torna o experimento reproduzível.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.1 Payoffs simulados
# MAGIC
# MAGIC O indicador discreto de sobrevivência seria:
# MAGIC
# MAGIC $$I_i^{\mathrm{disc}}=\mathbf{1}_{\{\min_{j=0,\ldots,M}S_{i,j}>H\}}$$
# MAGIC
# MAGIC Payoff descontado da barreira:
# MAGIC
# MAGIC $$X_i=e^{-rT}(S_{i,M}-K)^+I_i$$
# MAGIC
# MAGIC Payoff descontado da vanilla, usando a mesma trajetória:
# MAGIC
# MAGIC $$Y_i=e^{-rT}(S_{i,M}-K)^+$$
# MAGIC
# MAGIC Estimadores brutos:
# MAGIC
# MAGIC $$\overline X=\frac1N\sum_{i=1}^N X_i$$
# MAGIC
# MAGIC $$\overline Y=\frac1N\sum_{i=1}^N Y_i$$
# MAGIC
# MAGIC Pela lei dos grandes números:
# MAGIC
# MAGIC $$\overline X\xrightarrow[N\to\infty]{}V_0^{\mathrm{DOC}}$$

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.2 Erro padrão e intervalo de confiança
# MAGIC
# MAGIC Variância amostral:
# MAGIC
# MAGIC $$s_X^2=\frac{1}{N-1}\sum_{i=1}^N(X_i-\overline X)^2$$
# MAGIC
# MAGIC Erro padrão da média:
# MAGIC
# MAGIC $$SE(\overline X)=\frac{s_X}{\sqrt N}$$
# MAGIC
# MAGIC Intervalo aproximado de 95%:
# MAGIC
# MAGIC $$IC_{95\%}=\left[ \overline X-1{,}96SE,\, \overline X+1{,}96SE \right]$$
# MAGIC
# MAGIC Como:
# MAGIC
# MAGIC $$SE\propto\frac1{\sqrt N}$$
# MAGIC
# MAGIC reduzir o erro pela metade apenas aumentando a força bruta exige cerca de quatro vezes mais trajetórias.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.3 Probabilidade de knock-out
# MAGIC
# MAGIC O estimador da probabilidade neutra ao risco de knock-out é:
# MAGIC
# MAGIC $$\widehat p_{\mathrm{KO}}=\frac1N\sum_{i=1}^N \mathbf{1}_{\{\text{trajetória }i\text{ atingiu }H\}}$$
# MAGIC
# MAGIC A probabilidade de sobrevivência é:
# MAGIC
# MAGIC $$\widehat p_{\mathrm{surv}}=1-\widehat p_{\mathrm{KO}}$$
# MAGIC
# MAGIC Essas probabilidades estão sob Q. Elas não são automaticamente previsões de risco real sob P.

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Brownian Bridge
# MAGIC
# MAGIC Mesmo quando dois pontos consecutivos estão acima de H, a trajetória contínua pode ter cruzado a barreira entre eles.
# MAGIC
# MAGIC Defina:
# MAGIC
# MAGIC $$x_j=\ln S_{i,j},\qquad x_{j+1}=\ln S_{i,j+1},\qquad h=\ln H$$
# MAGIC
# MAGIC Se Sᵢⱼ > H e Sᵢ,ⱼ₊₁ > H, a probabilidade condicional de cruzamento é:
# MAGIC
# MAGIC $$p_{i,j}^{\mathrm{hit}}=\exp\left[ -\frac{2(x_j-h)(x_{j+1}-h)} {\sigma^2\Delta t} \right]$$
# MAGIC
# MAGIC Forma equivalente:
# MAGIC
# MAGIC $$p_{i,j}^{\mathrm{hit}}=\exp\left[ -\frac{ 2\ln(S_{i,j}/H)\ln(S_{i,j+1}/H) }{ \sigma^2\Delta t } \right]$$
# MAGIC
# MAGIC Se algum extremo for menor ou igual a H:
# MAGIC
# MAGIC $$p_{i,j}^{\mathrm{hit}}=1$$

# COMMAND ----------

# MAGIC %md
# MAGIC # 6.1 Aplicação da Brownian Bridge
# MAGIC
# MAGIC ## Método A — sorteio
# MAGIC
# MAGIC Geramos:
# MAGIC
# MAGIC $$U_{i,j}\sim\operatorname{Uniforme}(0,1)$$
# MAGIC
# MAGIC e registramos cruzamento quando:
# MAGIC
# MAGIC $$U_{i,j}<p_{i,j}^{\mathrm{hit}}$$
# MAGIC
# MAGIC ## Método B — peso de sobrevivência
# MAGIC
# MAGIC Em cada intervalo:
# MAGIC
# MAGIC $$p_{i,j}^{\mathrm{surv}}=1-p_{i,j}^{\mathrm{hit}}$$
# MAGIC
# MAGIC Para a trajetória completa:
# MAGIC
# MAGIC $$P_i^{\mathrm{surv}}=\prod_{j=0}^{M-1}(1-p_{i,j}^{\mathrm{hit}})$$
# MAGIC
# MAGIC Payoff suavizado:
# MAGIC
# MAGIC $$X_i^{\mathrm{BB}}=e^{-rT}(S_{i,M}-K)^+P_i^{\mathrm{surv}}$$
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
# MAGIC $$X_i=\text{payoff descontado da barreira}$$
# MAGIC
# MAGIC $$Y_i=\text{payoff descontado da vanilla na mesma trajetória}$$
# MAGIC
# MAGIC Como Black–Scholes fornece:
# MAGIC
# MAGIC $$\mathbb E^{\mathbb Q}[Y_i]=C_{\mathrm{BS}}$$
# MAGIC
# MAGIC definimos a observação corrigida:
# MAGIC
# MAGIC $$X_i^{\mathrm{CV}}=X_i-\beta(Y_i-C_{\mathrm{BS}})$$
# MAGIC
# MAGIC e o estimador final:
# MAGIC
# MAGIC $$\widehat V_{\mathrm{CV}}=\overline X-\widehat\beta(\overline Y-C_{\mathrm{BS}})$$
# MAGIC
# MAGIC O símbolo acento circunflexo indica uma estimativa calculada com amostra finita.

# COMMAND ----------

# MAGIC %md
# MAGIC # 7.1 Coeficiente ótimo
# MAGIC
# MAGIC A variância da observação corrigida é:
# MAGIC
# MAGIC $$\operatorname{Var}(X-\beta Y)=\operatorname{Var}(X) +\beta^2\operatorname{Var}(Y) -2\beta\operatorname{Cov}(X,Y)$$
# MAGIC
# MAGIC Derivando em relação a β e igualando a zero:
# MAGIC
# MAGIC $$2\beta\operatorname{Var}(Y) -2\operatorname{Cov}(X,Y)=0$$
# MAGIC
# MAGIC Logo:
# MAGIC
# MAGIC $$\beta^*=\frac{\operatorname{Cov}(X,Y)} {\operatorname{Var}(Y)}$$
# MAGIC
# MAGIC Na amostra:
# MAGIC
# MAGIC $$\widehat\beta=\frac{ \sum_{i=1}^N(X_i-\overline X)(Y_i-\overline Y) }{ \sum_{i=1}^N(Y_i-\overline Y)^2 }$$

# COMMAND ----------

# MAGIC %md
# MAGIC # 7.2 Intuição da correção
# MAGIC
# MAGIC Se:
# MAGIC
# MAGIC $$\overline Y>C_{\mathrm{BS}}$$
# MAGIC
# MAGIC a vanilla simulada ficou acima do valor exato. Com correlação positiva e β estimado > 0, retiramos parte desse excesso da barreira.
# MAGIC
# MAGIC Se:
# MAGIC
# MAGIC $$\overline Y<C_{\mathrm{BS}}$$
# MAGIC
# MAGIC a diferença é negativa e a correção aumenta a estimativa da barreira.
# MAGIC
# MAGIC O ruído observável da vanilla é:
# MAGIC
# MAGIC $$\overline Y-C_{\mathrm{BS}}$$
# MAGIC
# MAGIC Como vanilla e barreira usam os mesmos caminhos, parte desse ruído também está em média de X. A variável de controle procura removê-lo.

# COMMAND ----------

# MAGIC %md
# MAGIC # 7.3 Ganho teórico
# MAGIC
# MAGIC Defina:
# MAGIC
# MAGIC $$\rho_{XY}=\frac{\operatorname{Cov}(X,Y)} {\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}}$$
# MAGIC
# MAGIC Com o coeficiente ótimo:
# MAGIC
# MAGIC $$\operatorname{Var}(X^{\mathrm{CV}})=\operatorname{Var}(X)(1-\rho_{XY}^2)$$
# MAGIC
# MAGIC Quanto mais próximo |ρXY| estiver de 1, maior tende a ser a redução de variância.
# MAGIC
# MAGIC O erro padrão corrigido é:
# MAGIC
# MAGIC $$s_{\mathrm{CV}}^2=\frac1{N-1}\sum_{i=1}^N (X_i^{\mathrm{CV}}-\overline{X^{\mathrm{CV}}})^2$$
# MAGIC
# MAGIC $$SE_{\mathrm{CV}}=\frac{s_{\mathrm{CV}}}{\sqrt N}$$
# MAGIC
# MAGIC $$IC_{95\%}^{\mathrm{CV}}=\widehat V_{\mathrm{CV}}\pm1{,}96SE_{\mathrm{CV}}$$

# COMMAND ----------

# MAGIC %md
# MAGIC # Encadeamento completo
# MAGIC
# MAGIC ## 1 — Produto
# MAGIC
# MAGIC $$X_{\mathrm{barreira}}=(S_T-K)^+\mathbf{1}_{\{\min_{0\le t\le T}S_t>H\}}$$
# MAGIC
# MAGIC ## 2 — Mercado
# MAGIC
# MAGIC $$\Theta=(S_0,K,H,T,r,q,\sigma)$$
# MAGIC
# MAGIC ## 3 — Vanilla analítica
# MAGIC
# MAGIC $$C_{\mathrm{BS}}=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2)$$
# MAGIC
# MAGIC ## 4 — Dinâmica neutra ao risco
# MAGIC
# MAGIC $$\frac{dS_t}{S_t}=(r-q)dt+\sigma dW_t^{\mathbb Q}$$
# MAGIC
# MAGIC ## 5 — Monte Carlo
# MAGIC
# MAGIC $$S_{t+\Delta t}=S_t\exp[(r-q-\tfrac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}Z]$$
# MAGIC
# MAGIC ## 6 — Brownian Bridge
# MAGIC
# MAGIC $$p_{\mathrm{hit}}=\exp\left[ -\frac{2\ln(S_t/H)\ln(S_{t+\Delta t}/H)} {\sigma^2\Delta t} \right]$$
# MAGIC
# MAGIC ## 7 — Variável de controle
# MAGIC
# MAGIC $$\widehat V_{\mathrm{CV}}=\overline X-\widehat\beta(\overline Y-C_{\mathrm{BS}})$$

# COMMAND ----------

# MAGIC %md
# MAGIC # Exemplo do projeto
# MAGIC
# MAGIC Entradas:
# MAGIC
# MAGIC $$S_0=150.000,\ K=155.000,\ H=120.000,\ T=0{,}5,\ r=0{,}12,\ q=0,\ \sigma=0{,}22$$
# MAGIC
# MAGIC Configuração ilustrativa:
# MAGIC
# MAGIC $$N=50.000,\qquad M=63$$
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
# MAGIC $$H<S_0$$
# MAGIC
# MAGIC 2. Limites do preço:
# MAGIC $$0\le V_{\mathrm{barreira}}\le C_{\mathrm{BS}}$$
# MAGIC
# MAGIC 3. Barreira muito baixa:
# MAGIC $$H\downarrow0\Longrightarrow V_{\mathrm{barreira}}\to C_{\mathrm{BS}}$$
# MAGIC
# MAGIC 4. Barreira se aproxima do spot:
# MAGIC $$H\uparrow S_0\Longrightarrow V_{\mathrm{barreira}}\to0$$
# MAGIC para monitoramento contínuo.
# MAGIC
# MAGIC 5. Convergência estatística:
# MAGIC $$SE\propto1/\sqrt N$$
# MAGIC
# MAGIC 6. Ao aumentar M, o preço deve estabilizar. A Brownian Bridge deve reduzir a sensibilidade ao número de passos.
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
# MAGIC $$\sigma=\text{constante},\qquad r,q=\text{determinísticos}$$
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
# MAGIC $$\boxed{ V_0=e^{-rT}\mathbb E^{\mathbb Q}[X_T] }$$
# MAGIC
# MAGIC Para a opção com barreira:
# MAGIC
# MAGIC $$V_0^{\mathrm{DOC}}=e^{-rT}\mathbb E^{\mathbb Q} \left[ (S_T-K)^+\mathbf{1}_{\{\min S_t>H\}} \right]$$
# MAGIC
# MAGIC O Monte Carlo aproxima essa esperança; a Brownian Bridge trata cruzamentos entre os pontos da grade; e a vanilla remove parte do ruído comum:
# MAGIC
# MAGIC $$\boxed{ \widehat V_{\mathrm{CV}}=\overline X-\widehat\beta(\overline Y-C_{\mathrm{BS}}) }$$
# MAGIC
# MAGIC Os sete passos constituem um processo único: **definir o contrato, calibrar o mercado, obter uma referência analítica, modelar sob Q, simular, tratar a barreira e aumentar a eficiência estatística**.