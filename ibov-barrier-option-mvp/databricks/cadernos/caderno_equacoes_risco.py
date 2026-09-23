# Databricks notebook source
# MAGIC %md
# MAGIC # Caderno de equações — métricas de risco da opção com barreira
# MAGIC
# MAGIC Este caderno formaliza as convenções do notebook operacional
# MAGIC `03_analise_risco`. Ele é material explicativo e não participa do fluxo diário.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Gregas
# MAGIC
# MAGIC Para o valor da opção $V(S,\sigma,r,T)$:
# MAGIC
# MAGIC $$\Delta=\frac{\partial V}{\partial S},\qquad
# MAGIC \Gamma=\frac{\partial^2V}{\partial S^2}$$
# MAGIC
# MAGIC $$\text{Vega}_{1\%}=0{,}01\frac{\partial V}{\partial\sigma},\qquad
# MAGIC \text{Rho}_{1\%}=0{,}01\frac{\partial V}{\partial r}$$
# MAGIC
# MAGIC $$\Theta_{1d}=V(T-1/365)-V(T)$$
# MAGIC
# MAGIC As derivadas são estimadas por diferenças finitas centrais, com números
# MAGIC aleatórios comuns. Para opções com barreira, os sinais de Gamma e Vega não são
# MAGIC impostos: eles podem mudar perto da barreira.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. DV01 e convexidade de taxa
# MAGIC
# MAGIC Com $h=1$ bp $=10^{-4}$:
# MAGIC
# MAGIC $$DV01=\frac{V(r+h)-V(r-h)}{2}$$
# MAGIC
# MAGIC O DV01 é assinado e representa aproximadamente a variação do preço para uma
# MAGIC alta paralela de 1 bp na taxa. Para uma call, ele não segue necessariamente a
# MAGIC convenção positiva usada em títulos de renda fixa.
# MAGIC
# MAGIC A convexidade discreta de taxa por bp² é:
# MAGIC
# MAGIC $$C_r^{1bp}=V(r+h)-2V(r)+V(r-h)$$
# MAGIC
# MAGIC Ela mede a parcela de segunda ordem da sensibilidade à taxa. Não deve ser
# MAGIC confundida com Gamma, que é a convexidade em relação ao índice-objeto.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. VaR e Expected Shortfall
# MAGIC
# MAGIC A primeira implementação usa Monte Carlo delta-gamma de um dia. O choque do
# MAGIC índice segue:
# MAGIC
# MAGIC $$S_{t+h}=S_t\exp\left[(\mu-\tfrac12\sigma^2)h+
# MAGIC \sigma\sqrt{h}Z\right],\qquad Z\sim N(0,1)$$
# MAGIC
# MAGIC e o P&L aproximado é:
# MAGIC
# MAGIC $$\Delta V\approx\Delta\,\Delta S+
# MAGIC \frac12\Gamma(\Delta S)^2$$
# MAGIC
# MAGIC Para perdas $L=-\Delta V$ e confiança $\alpha$:
# MAGIC
# MAGIC $$VaR_\alpha=Q_\alpha(L)$$
# MAGIC
# MAGIC $$ES_\alpha=E[L\mid L\ge VaR_\alpha]$$
# MAGIC
# MAGIC Convenção inicial: horizonte de 1 dia útil, 252 dias por ano, confiança de
# MAGIC 99%, 100.000 cenários, seed fixa e drift anual igual a zero.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Limitações e governança
# MAGIC
# MAGIC - VaR e ES são model-based e usam apenas o risco do spot na aproximação
# MAGIC   delta-gamma; não são VaR histórico nem incorporam choques conjuntos de curva e vol.
# MAGIC - A aproximação pode perder precisão perto da barreira, onde o payoff é não suave.
# MAGIC - Os resultados devem carregar os `WARN` emitidos pelo notebook 02.
# MAGIC - O notebook 03 só executa quando existe `VALIDATION_*.json` com `FAIL=0`,
# MAGIC   referência ao mesmo `RESULTS` e SHA-256 coincidente.
# MAGIC - Cenários multivariados e stress testing serão tratados em uma implementação
# MAGIC   posterior e separada.
