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
# MAGIC > Este notebook lê diretamente os arquivos públicos da B3 armazenados em `data/AAAA-MM-DD`. O arquivo **IR** fornece o IBOV à vista e o arquivo **SPRD** fornece os ajustes de DI1 e IND.

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

# Comentário: importa apenas bibliotecas padrão e pacotes já disponíveis no Databricks.
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Leitura dos arquivos reais da B3
# MAGIC
# MAGIC Para cada data de mercado, o notebook utiliza:
# MAGIC
# MAGIC | Arquivo | Conteúdo usado |
# MAGIC |---|---|
# MAGIC | IR | fechamento do índice IBOV |
# MAGIC | SPRD | taxa de ajuste dos futuros DI1 e preço de ajuste dos futuros IND |
# MAGIC | IN | cadastro completo dos instrumentos; fica arquivado, mas não precisa ser lido nesta etapa |
# MAGIC
# MAGIC O fluxo é:
# MAGIC
# MAGIC 1. escolher a data;
# MAGIC 2. abrir os ZIPs sem extrair manualmente;
# MAGIC 3. retirar o spot do IBOV;
# MAGIC 4. montar a curva DI1;
# MAGIC 5. associar a taxa DI ao vencimento de cada futuro IND;
# MAGIC 6. calcular o dividend yield implícito.

# COMMAND ----------

# Comentário: escolha aqui a fotografia de mercado; use None para selecionar automaticamente a última pasta completa.
MARKET_DATE = None

# Prazo da opção usado adiante no notebook, em anos ACT/365.
option_maturity = 0.5

# COMMAND ----------

# Comentário: localiza a raiz do projeto independentemente de o notebook rodar na raiz ou na pasta databricks.
def find_project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Não encontrei a raiz do projeto. Confirme que existem as pastas data e src."
    )


project_root = find_project_root()
data_root = project_root / "data"


def complete_market_folders(root: Path) -> list[Path]:
    return sorted(
        folder
        for folder in root.iterdir()
        if folder.is_dir()
        and list(folder.glob("IR*.zip"))
        and list(folder.glob("SPRD*.zip"))
    )


available_folders = complete_market_folders(data_root)
if not available_folders:
    raise FileNotFoundError(
        f"Nenhuma pasta completa foi encontrada em {data_root}. "
        "São necessários pelo menos um IR*.zip e um SPRD*.zip."
    )

market_folder = (
    data_root / MARKET_DATE
    if MARKET_DATE is not None
    else available_folders[-1]
)

if market_folder not in available_folders:
    available_dates = [folder.name for folder in available_folders]
    raise FileNotFoundError(
        f"A pasta {market_folder.name} não contém IR e SPRD. "
        f"Datas completas disponíveis: {available_dates}"
    )

valuation_date = date.fromisoformat(market_folder.name)
ir_zip = sorted(market_folder.glob("IR*.zip"))[-1]
sprd_zip = sorted(market_folder.glob("SPRD*.zip"))[-1]

display(
    pd.DataFrame(
        {
            "item": ["Data de avaliação", "Pasta", "Arquivo IR", "Arquivo SPRD"],
            "valor": [
                valuation_date.isoformat(),
                str(market_folder),
                ir_zip.name,
                sprd_zip.name,
            ],
        }
    )
)

# COMMAND ----------

# Comentário: remove os namespaces do XML e transforma as folhas de cada registro em um dicionário simples.
def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def flatten_element(element) -> dict:
    row = {}
    for child in element.iter():
        value = (child.text or "").strip()
        if value:
            row.setdefault(local_name(child.tag), value)
    return row


def read_latest_xml(zip_path: Path):
    with ZipFile(zip_path) as archive:
        xml_members = [
            info for info in archive.infolist()
            if info.filename.lower().endswith(".xml")
        ]
        if not xml_members:
            raise ValueError(f"Nenhum XML encontrado em {zip_path.name}.")

        # Alguns ZIPs trazem mais de uma publicação. Usamos a mais recente.
        latest_member = max(xml_members, key=lambda info: info.date_time)
        xml_bytes = archive.read(latest_member)
        return ET.fromstring(xml_bytes), latest_member.filename


def rows_from_tag(root, record_tag: str) -> list[dict]:
    return [
        flatten_element(element)
        for element in root.iter()
        if local_name(element.tag) == record_tag
    ]


ir_root, ir_xml_name = read_latest_xml(ir_zip)
sprd_root, sprd_xml_name = read_latest_xml(sprd_zip)

print(f"XML usado para o índice:     {ir_xml_name}")
print(f"XML usado para derivativos:  {sprd_xml_name}")

# COMMAND ----------

# Comentário: extrai o fechamento oficial do IBOV do relatório de índices BVBG.087.01.
index_rows = pd.DataFrame(rows_from_tag(ir_root, "IndxInf"))

ibov_rows = index_rows.loc[index_rows["TckrSymb"].eq("IBOV")].copy()
if len(ibov_rows) != 1:
    raise ValueError(
        f"Esperava exatamente um registro IBOV no IR; encontrei {len(ibov_rows)}."
    )

spot_field = "ClsgPric" if "ClsgPric" in ibov_rows.columns else "IndxVal"
spot_ibov = float(ibov_rows.iloc[0][spot_field])

display(
    ibov_rows[
        [
            column
            for column in [
                "TckrSymb",
                "OpngPric",
                "MinPric",
                "MaxPric",
                "ClsgPric",
                "IndxVal",
                "OscnVal",
            ]
            if column in ibov_rows.columns
        ]
    ]
)

print(f"Spot do Ibovespa em {valuation_date:%d/%m/%Y}: {spot_ibov:,.2f} pontos")

# COMMAND ----------

# Comentário: extrai os registros DI1 e IND do relatório simplificado de derivativos BVBG.187.01.
price_report = pd.DataFrame(rows_from_tag(sprd_root, "PricRpt"))

di_quotes = price_report.loc[
    price_report["TckrSymb"].str.startswith("DI1", na=False)
].copy()
ind_quotes = price_report.loc[
    price_report["TckrSymb"].str.startswith("IND", na=False)
].copy()

# A taxa de ajuste do DI1 vem em percentual ao ano; o ajuste do IND vem em pontos.
di_quotes["zero_rate_effective"] = pd.to_numeric(
    di_quotes["AdjstdQtTax"], errors="coerce"
) / 100.0
ind_quotes["ibov_future"] = pd.to_numeric(
    ind_quotes["AdjstdQt"], errors="coerce"
)

di_quotes = (
    di_quotes.dropna(subset=["zero_rate_effective"])
    .drop_duplicates("TckrSymb", keep="last")
)
ind_quotes = (
    ind_quotes.dropna(subset=["ibov_future"])
    .drop_duplicates("TckrSymb", keep="last")
)

if di_quotes.empty:
    raise ValueError("Nenhuma taxa de ajuste DI1 foi encontrada no SPRD.")
if ind_quotes.empty:
    raise ValueError("Nenhum preço de ajuste IND foi encontrado no SPRD.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Datas de vencimento dos contratos
# MAGIC
# MAGIC O código da B3 contém uma letra para o mês e dois dígitos para o ano.
# MAGIC
# MAGIC - DI1: primeiro dia útil do mês de vencimento;
# MAGIC - IND: quarta-feira mais próxima do dia 15 do mês de vencimento.
# MAGIC
# MAGIC Nesta versão, “dia útil” considera segunda a sexta-feira. O uso do calendário
# MAGIC oficial da B3 será o próximo refinamento.

# COMMAND ----------

# Comentário: converte códigos como DI1F27 e INDZ26 em datas de vencimento.
MONTH_CODES = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


def first_weekday(year: int, month: int) -> date:
    maturity = date(year, month, 1)
    while maturity.weekday() >= 5:
        maturity += timedelta(days=1)
    return maturity


def nearest_wednesday_to_15(year: int, month: int) -> date:
    candidates = [
        date(year, month, day)
        for day in range(12, 19)
        if date(year, month, day).weekday() == 2
    ]
    return min(candidates, key=lambda maturity: abs(maturity.day - 15))


def contract_maturity(ticker: str) -> date:
    month_code = ticker[-3]
    year = 2000 + int(ticker[-2:])
    month = MONTH_CODES[month_code]

    if ticker.startswith("DI1"):
        return first_weekday(year, month)
    if ticker.startswith("IND"):
        return nearest_wednesday_to_15(year, month)
    raise ValueError(f"Contrato não reconhecido: {ticker}")


di_quotes["maturity_date"] = pd.to_datetime(
    di_quotes["TckrSymb"].map(contract_maturity)
)
ind_quotes["maturity_date"] = pd.to_datetime(
    ind_quotes["TckrSymb"].map(contract_maturity)
)

valuation_ts = pd.Timestamp(valuation_date)
di_quotes = di_quotes.loc[di_quotes["maturity_date"] > valuation_ts].sort_values(
    "maturity_date"
)
ind_quotes = ind_quotes.loc[ind_quotes["maturity_date"] > valuation_ts].sort_values(
    "maturity_date"
)

display(
    di_quotes[
        ["TckrSymb", "maturity_date", "zero_rate_effective", "AdjstdQtTax"]
    ].style.format({"zero_rate_effective": "{:.4%}"})
)
display(
    ind_quotes[
        ["TckrSymb", "maturity_date", "ibov_future"]
    ].style.format({"ibov_future": "{:,.2f}"})
)

# COMMAND ----------

# Comentário: interpola a curva DI nos vencimentos dos futuros IND e monta a tabela usada pelo restante do notebook.
di_days = (di_quotes["maturity_date"] - valuation_ts).dt.days.to_numpy()
di_rates = di_quotes["zero_rate_effective"].to_numpy(dtype=float)

ind_quotes["days"] = (ind_quotes["maturity_date"] - valuation_ts).dt.days
inside_di_curve = ind_quotes["days"].between(di_days.min(), di_days.max())
excluded_ind = ind_quotes.loc[~inside_di_curve].copy()
ind_quotes = ind_quotes.loc[inside_di_curve].copy()

if ind_quotes.empty:
    raise ValueError(
        "Nenhum vencimento IND ficou dentro do intervalo coberto pela curva DI1."
    )

ind_quotes["zero_rate_effective"] = np.interp(
    ind_quotes["days"].to_numpy(),
    di_days,
    di_rates,
)

market_quotes = ind_quotes[
    [
        "TckrSymb",
        "maturity_date",
        "zero_rate_effective",
        "ibov_future",
    ]
].rename(columns={"TckrSymb": "future_ticker"})

display(
    market_quotes.style.format(
        {
            "zero_rate_effective": "{:.4%}",
            "ibov_future": "{:,.2f}",
        }
    )
)

if not excluded_ind.empty:
    print(
        "Contratos IND fora do intervalo da curva DI e não utilizados:",
        excluded_ind["TckrSymb"].tolist(),
    )

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

ax.set_title(f"Curvas de mercado B3 — {valuation_date:%d/%m/%Y}")
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
# MAGIC Esta versão já lê spot, DI1 e IND diretamente dos arquivos reais da B3 armazenados no projeto.
# MAGIC
# MAGIC Próximas etapas:
# MAGIC
# MAGIC 1. substituir o calendário simplificado pelo calendário oficial da B3;
# MAGIC 2. validar vencimentos contra o cadastro IN;
# MAGIC 3. aplicar filtros de liquidez aos contratos futuros;
# MAGIC 4. comparar interpolação em taxas com interpolação em fatores de desconto;
# MAGIC 5. permitir r(t) e q(t) variáveis ao longo da simulação;
# MAGIC 6. criar controles de qualidade e alertas de arbitragem.
# MAGIC
# MAGIC ## Controle essencial
# MAGIC
# MAGIC O preço futuro reconstruído deve permanecer compatível com os dados de entrada:
# MAGIC
# MAGIC $$F_{0,T}=S_0e^{(r(T)-q(T))T}$$
# MAGIC
# MAGIC A curva agora é construída com dados de mercado, mas ainda depende de convenções simplificadas de calendário e de interpolação.