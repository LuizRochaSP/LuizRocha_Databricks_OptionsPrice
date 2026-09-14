# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Curvas de mercado para a opção com barreira no Ibovespa
# MAGIC 
# MAGIC ## Curva de juros e dividend yield implícito
# MAGIC 
# MAGIC Este notebook transforma preços do Ibovespa à vista, taxas de juros e futuros de Ibovespa nos dois parâmetros exigidos pelo modelo:
# MAGIC 
# MAGIC - **r(T):** taxa zero livre de risco para o vencimento da opção;
# MAGIC - **q(T):** dividend yield ou carry implícito para o mesmo vencimento.
# MAGIC 
# MAGIC A relação central é:
# MAGIC 
# MAGIC $$F_{0,T}=S_0e^{(r(T)-q(T))T}$$
# MAGIC 
# MAGIC Isolando o dividend yield implícito:
# MAGIC 
# MAGIC $$q(T)=r(T)-\frac{1}{T}\ln\left(\frac{F_{0,T}}{S_0}\right)$$
# MAGIC 
# MAGIC > Os dados iniciais deste notebook são **ilustrativos**. Eles permitem validar toda a mecânica antes da conexão com fontes reais.

# COMMAND ----------

# MAGIC %md
# MAGIC # Fluxo do notebook
# MAGIC 
# MAGIC 1. Definir a data de avaliação e o spot;
# MAGIC 2. Informar vencimentos, taxas zero e futuros;
# MAGIC 3. Converter as taxas para capitalização contínua;
# MAGIC 4. Calcular fatores de desconto;
# MAGIC 5. extrair o dividend yield implícito;
# MAGIC 6. verificar a reconstrução dos futuros;
# MAGIC 7. interpolar r(T) e q(T) no prazo da opção;
# MAGIC 8. enviar os parâmetros ao modelo de barreira.
# MAGIC 
# MAGIC ## Convenções desta primeira versão
# MAGIC 
# MAGIC - contagem de tempo: ACT/365;
# MAGIC - taxa de entrada: efetiva anual;
# MAGIC - interpolação: linear nas taxas continuamente compostas;
# MAGIC - um único spot para todos os vencimentos;
# MAGIC - sem ajuste específico de feriados da B3 nesta versão.

# COMMAND ----------

# Comentário: imports usados na construção, interpolação e visualização das curvas.
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# COMMAND ----------

# Comentário: parâmetros gerais e dados de mercado ilustrativos; substitua apenas esta tabela quando tivermos dados reais.
valuation_date = date(2026, 9, 14)
spot_ibov = 150_000.0

market_quotes = pd.DataFrame(
    {
        "maturity_date": pd.to_datetime(
            [
                "2026-10-14",
                "2026-12-16",
                "2027-02-17",
                "2027-04-14",
                "2027-06-16",
            ]
        ),
        # Taxas zero efetivas anuais ilustrativas, em formato decimal.
        "zero_rate_effective": [0.1450, 0.1430, 0.1400, 0.1380, 0.1360],
        # Preços futuros ilustrativos do Ibovespa, em pontos.
        "ibov_future": [151_180.0, 153_580.0, 155_780.0, 157_710.0, 159_740.0],
    }
)

display(market_quotes)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Prazo e conversão da curva de juros
# MAGIC 
# MAGIC Para cada vencimento i, o prazo ACT/365 é:
# MAGIC 
# MAGIC $$T_i=\frac{\text{dias corridos entre avaliação e vencimento}}{365}$$
# MAGIC 
# MAGIC A taxa efetiva anual é convertida para taxa continuamente composta:
# MAGIC 
# MAGIC $$r_i=\ln(1+r_{i,\mathrm{efetiva}})$$
# MAGIC 
# MAGIC O fator de desconto é:
# MAGIC 
# MAGIC $$D(0,T_i)=e^{-r_iT_i}$$

# COMMAND ----------

# Comentário: calcula prazo ACT/365, taxa zero contínua e fator de desconto para cada vértice.
curve = market_quotes.copy()

valuation_ts = pd.Timestamp(valuation_date)
curve["days"] = (curve["maturity_date"] - valuation_ts).dt.days
curve["T"] = curve["days"] / 365.0

if (curve["days"] <= 0).any():
    raise ValueError("Todos os vencimentos devem ser posteriores à data de avaliação.")
if (curve["zero_rate_effective"] <= -1.0).any():
    raise ValueError("A taxa efetiva deve ser maior que -100%.")
if spot_ibov <= 0:
    raise ValueError("O spot do Ibovespa deve ser positivo.")

curve["r_continuous"] = np.log1p(curve["zero_rate_effective"])
curve["discount_factor"] = np.exp(-curve["r_continuous"] * curve["T"])

display(
    curve[
        [
            "maturity_date",
            "days",
            "T",
            "zero_rate_effective",
            "r_continuous",
            "discount_factor",
        ]
    ].style.format(
        {
            "T": "{:.6f}",
            "zero_rate_effective": "{:.4%}",
            "r_continuous": "{:.4%}",
            "discount_factor": "{:.6f}",
        }
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Extração do carry implícito
# MAGIC 
# MAGIC Do custo de carregamento:
# MAGIC 
# MAGIC $$F_{0,T}=S_0e^{(r-q)T}$$
# MAGIC 
# MAGIC obtemos:
# MAGIC 
# MAGIC $$q_i=r_i-\frac{\ln(F_{0,T_i}/S_0)}{T_i}$$
# MAGIC 
# MAGIC Também definimos o carry líquido:
# MAGIC 
# MAGIC $$b_i=r_i-q_i$$
# MAGIC 
# MAGIC O modelo de Black–Scholes e o movimento browniano geométrico utilizam exatamente esse carry no drift neutro ao risco.

# COMMAND ----------

# Comentário: extrai dividend yield implícito e carry líquido de cada futuro.
if (curve["ibov_future"] <= 0).any():
    raise ValueError("Todos os preços futuros devem ser positivos.")

curve["implied_q"] = (
    curve["r_continuous"]
    - np.log(curve["ibov_future"] / spot_ibov) / curve["T"]
)
curve["net_carry"] = curve["r_continuous"] - curve["implied_q"]

display(
    curve[
        [
            "maturity_date",
            "T",
            "ibov_future",
            "r_continuous",
            "implied_q",
            "net_carry",
        ]
    ].style.format(
        {
            "T": "{:.6f}",
            "ibov_future": "{:,.2f}",
            "r_continuous": "{:.4%}",
            "implied_q": "{:.4%}",
            "net_carry": "{:.4%}",
        }
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Verificação de consistência
# MAGIC 
# MAGIC Reconstruímos cada futuro usando as curvas calculadas:
# MAGIC 
# MAGIC $$\widehat F_{0,T_i}=S_0e^{(r_i-q_i)T_i}$$
# MAGIC 
# MAGIC O erro de reconstrução é:
# MAGIC 
# MAGIC $$\varepsilon_i=\widehat F_{0,T_i}-F_{0,T_i}$$
# MAGIC 
# MAGIC Como q foi extraído da própria equação do futuro, o erro deve ser praticamente zero, salvo arredondamento numérico.

# COMMAND ----------

# Comentário: reconstrói os futuros e testa se a extração do carry está numericamente correta.
curve["future_rebuilt"] = spot_ibov * np.exp(
    (curve["r_continuous"] - curve["implied_q"]) * curve["T"]
)
curve["rebuild_error_points"] = curve["future_rebuilt"] - curve["ibov_future"]
curve["rebuild_error_bps_of_future"] = (
    curve["rebuild_error_points"] / curve["ibov_future"] * 10_000
)

tolerance_points = 1e-6
if curve["rebuild_error_points"].abs().max() > tolerance_points:
    raise AssertionError("A reconstrução dos futuros excedeu a tolerância.")

display(
    curve[
        [
            "maturity_date",
            "ibov_future",
            "future_rebuilt",
            "rebuild_error_points",
        ]
    ].style.format(
        {
            "ibov_future": "{:,.4f}",
            "future_rebuilt": "{:,.4f}",
            "rebuild_error_points": "{:.10f}",
        }
    )
)

# COMMAND ----------

# Comentário: apresenta visualmente as curvas zero, dividend yield implícito e carry líquido.
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(curve["T"], 100 * curve["r_continuous"], marker="o", label="Taxa zero contínua r(T)")
ax.plot(curve["T"], 100 * curve["implied_q"], marker="o", label="Dividend yield q(T)")
ax.plot(curve["T"], 100 * curve["net_carry"], marker="o", label="Carry líquido r(T) − q(T)")

ax.set_title("Curvas de mercado — dados ilustrativos")
ax.set_xlabel("Prazo em anos")
ax.set_ylabel("Taxa anual contínua (%)")
ax.grid(alpha=0.3)
ax.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Interpolação para o prazo da opção
# MAGIC 
# MAGIC A opção pode vencer entre dois vértices disponíveis. Nesta primeira versão, usamos interpolação linear nas taxas contínuas.
# MAGIC 
# MAGIC Se Ta ≤ T ≤ Tb:
# MAGIC 
# MAGIC $$r(T)=r(T_a)+\frac{T-T_a}{T_b-T_a}\left[r(T_b)-r(T_a)\right]$$
# MAGIC 
# MAGIC A mesma expressão é aplicada a q(T).
# MAGIC 
# MAGIC > Não fazemos extrapolação silenciosa: o prazo da opção precisa estar dentro do intervalo coberto pela curva.

# COMMAND ----------

# Comentário: função de interpolação linear que impede extrapolação fora dos vértices observados.
def interpolate_curve(target_T: float, tenors, values, curve_name: str) -> float:
    tenors = np.asarray(tenors, dtype=float)
    values = np.asarray(values, dtype=float)

    if target_T < tenors.min() or target_T > tenors.max():
        raise ValueError(
            f"O prazo {target_T:.6f} está fora da curva {curve_name}: "
            f"[{tenors.min():.6f}, {tenors.max():.6f}]."
        )

    return float(np.interp(target_T, tenors, values))


# Prazo da opção usada no projeto: seis meses.
option_maturity = 0.5

rate_for_option = interpolate_curve(
    option_maturity,
    curve["T"],
    curve["r_continuous"],
    "de juros",
)
q_for_option = interpolate_curve(
    option_maturity,
    curve["T"],
    curve["implied_q"],
    "de dividend yield",
)
discount_for_option = np.exp(-rate_for_option * option_maturity)
forward_for_option = spot_ibov * np.exp(
    (rate_for_option - q_for_option) * option_maturity
)

parameters_for_pricing = pd.DataFrame(
    {
        "parameter": [
            "Spot",
            "Prazo",
            "Taxa contínua r(T)",
            "Dividend yield q(T)",
            "Carry líquido r(T)-q(T)",
            "Fator de desconto",
            "Forward interpolado",
        ],
        "value": [
            spot_ibov,
            option_maturity,
            rate_for_option,
            q_for_option,
            rate_for_option - q_for_option,
            discount_for_option,
            forward_for_option,
        ],
    }
)

display(parameters_for_pricing)

# COMMAND ----------

# Comentário: resumo formatado dos parâmetros que serão enviados ao modelo de precificação.
print(f"Spot do Ibovespa:             {spot_ibov:,.2f} pontos")
print(f"Prazo da opção:               {option_maturity:.4f} ano")
print(f"Taxa contínua r(T):           {rate_for_option:.4%} a.a.")
print(f"Dividend yield q(T):          {q_for_option:.4%} a.a.")
print(f"Carry líquido r(T)-q(T):      {rate_for_option - q_for_option:.4%} a.a.")
print(f"Fator de desconto:            {discount_for_option:.6f}")
print(f"Forward teórico interpolado:  {forward_for_option:,.2f} pontos")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Integração com o modelo de barreira
# MAGIC 
# MAGIC O motor atual recebe parâmetros constantes até o vencimento. Portanto, resumimos cada curva pelo valor interpolado no prazo da opção:
# MAGIC 
# MAGIC $$r=r(T_{\mathrm{opção}})$$
# MAGIC 
# MAGIC $$q=q(T_{\mathrm{opção}})$$
# MAGIC 
# MAGIC A dinâmica usada na simulação passa a ser:
# MAGIC 
# MAGIC $$\frac{dS_t}{S_t}=(r(T)-q(T))dt+\sigma dW_t^{\mathbb Q}$$
# MAGIC 
# MAGIC Esta é uma melhora importante em relação à escolha arbitrária de q = 0, mas ainda é uma aproximação. Uma evolução futura permitirá taxas determinísticas variáveis ao longo de cada passo da trajetória.

# COMMAND ----------

# Comentário: localiza automaticamente o código-fonte do projeto e importa o motor de precificação.
import sys
from pathlib import Path

candidate_src_paths = [
    Path.cwd().parent / "src",  # execução a partir da pasta databricks
    Path.cwd() / "src",         # execução a partir da raiz do projeto
]

src_path = next((path for path in candidate_src_paths if path.exists()), None)
if src_path is None:
    raise FileNotFoundError(
        "A pasta src do projeto não foi encontrada. "
        "Confirme que este notebook está dentro de ibov-barrier-option-mvp/databricks."
    )

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ibov_barrier.pricing import (
    BarrierContract,
    MarketData,
    price_down_and_out_call,
)

market_from_curves = MarketData(
    spot=spot_ibov,
    rate=rate_for_option,
    dividend_yield=q_for_option,
    volatility=0.22,
)
contract = BarrierContract(
    strike=155_000.0,
    barrier=120_000.0,
    maturity=option_maturity,
)

result_from_curves = price_down_and_out_call(
    market_from_curves,
    contract,
    paths=50_000,
    steps=63,
    seed=42,
)

display(
    {
        "barrier_price": result_from_curves.price,
        "vanilla_price": result_from_curves.vanilla_price,
        "standard_error": result_from_curves.standard_error,
        "ci_95_low": result_from_curves.ci_low,
        "ci_95_high": result_from_curves.ci_high,
        "knock_out_probability": result_from_curves.knock_out_probability,
        "rate_from_curve": rate_for_option,
        "q_from_curve": q_for_option,
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Limitações e próximos aprimoramentos
# MAGIC 
# MAGIC Esta primeira versão já conecta corretamente a lógica de curvas ao modelo, mas ainda utiliza dados ilustrativos.
# MAGIC 
# MAGIC Próximas etapas:
# MAGIC 
# MAGIC 1. importar a curva DI/zero de uma fonte real;
# MAGIC 2. importar spot e futuros de Ibovespa por vencimento;
# MAGIC 3. aplicar calendário de dias úteis e convenções da B3;
# MAGIC 4. tratar preços ausentes, liquidez e contratos pouco negociados;
# MAGIC 5. comparar interpolação em taxas com interpolação em fatores de desconto;
# MAGIC 6. armazenar a fotografia diária das curvas;
# MAGIC 7. permitir r(t) e q(t) variáveis ao longo da simulação;
# MAGIC 8. criar controles de qualidade e alertas de arbitragem.
# MAGIC 
# MAGIC ## Controle essencial
# MAGIC 
# MAGIC O preço futuro reconstruído deve permanecer compatível com os dados de entrada:
# MAGIC 
# MAGIC $$F_{0,T}=S_0e^{(r(T)-q(T))T}$$
# MAGIC 
# MAGIC A curva não deve ser considerada “de mercado” enquanto spot, taxas e futuros continuarem ilustrativos.
