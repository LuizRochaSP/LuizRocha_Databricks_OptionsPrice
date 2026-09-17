# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "lxml",
# ]
# ///
# MAGIC %md
# MAGIC # Superfície de volatilidade implícita das opções de Ibovespa
# MAGIC
# MAGIC Este notebook executa os seis passos seguintes do projeto:
# MAGIC
# MAGIC 1. ler eficientemente o cadastro `IN`;
# MAGIC 2. selecionar as opções europeias sobre o IBOV;
# MAGIC 3. cruzar cadastro e preços do `PR`;
# MAGIC 4. aplicar filtros de qualidade e liquidez;
# MAGIC 5. calcular a volatilidade implícita por Black–Scholes;
# MAGIC 6. construir smiles por vencimento e uma primeira superfície.
# MAGIC
# MAGIC A volatilidade implícita é o valor de $\sigma$ que resolve:
# MAGIC
# MAGIC $$V_{BS}(S_0,K,T,r,q,\sigma)=V_{mercado}$$
# MAGIC
# MAGIC | Símbolo | Descrição |
# MAGIC |---|---|
# MAGIC | $S_0$ | nível do Ibovespa na data de avaliação |
# MAGIC | $K$ | strike da opção |
# MAGIC | $T$ | prazo até o vencimento, em anos |
# MAGIC | $r$ | taxa contínua livre de risco para o prazo $T$ |
# MAGIC | $q$ | dividend yield implícito para o prazo $T$ |
# MAGIC | $\sigma$ | volatilidade implícita procurada |
# MAGIC | $V_{mercado}$ | prêmio observado no PriceReport |

# COMMAND ----------

# Comentário: verifica se o lxml está disponível; instala e reinicia somente quando necessário.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("lxml") is None:
    print("lxml não encontrado. Iniciando a instalação...")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "lxml>=6.0"]
    )

    print("lxml instalado. Reiniciando o Python...")
    dbutils.library.restartPython()

else:
    import lxml
    from lxml import etree

    print("Python:", sys.version.split()[0])
    print("lxml instalado:", lxml.__version__)
    print("Importação do etree: OK")

# COMMAND ----------

# Comentário: importa as bibliotecas usadas para XML, curvas, inversão de Black–Scholes e gráficos.
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from scipy.interpolate import griddata
from scipy.optimize import brentq
from scipy.stats import norm

# COMMAND ----------

# Comentário: escolha a fotografia de mercado e os filtros mínimos da amostra.
MARKET_DATE = None
MIN_TRADES = 1
MIN_OPEN_INTEREST = 1
MIN_IV = 0.01
MAX_IV = 2.00

# Parâmetros da opção com barreira usada no projeto.
TARGET_STRIKE = 155_000.0
TARGET_MATURITY = 0.5

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Localização dos arquivos
# MAGIC
# MAGIC Para a mesma data, usamos quatro fontes:
# MAGIC
# MAGIC | Arquivo | Informação |
# MAGIC |---|---|
# MAGIC | `IR` | fechamento do IBOV |
# MAGIC | `SPRD` | DI1 e futuros IND |
# MAGIC | `IN` | strike, vencimento, tipo e estilo da opção |
# MAGIC | `PR` | preços e dados de negociação |

# COMMAND ----------

# Comentário: encontra a raiz do projeto e valida os quatro arquivos necessários.
def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Não encontrei a raiz contendo as pastas data e src.")


# Comentário: localiza a pasta solicitada ou seleciona automaticamente
# a pasta mais recente que contenha os quatro arquivos necessários.

project_root = find_project_root()
data_root = project_root / "data"

required_patterns = (
    "IN*.zip",
    "IR*.zip",
    "SPRD*.zip",
    "PR*.zip",
)

if MARKET_DATE is None:
    candidate_folders = sorted(
        [
            folder
            for folder in data_root.iterdir()
            if folder.is_dir()
        ],
        key=lambda folder: folder.name,
        reverse=True,
    )

    market_folder = next(
        (
            folder
            for folder in candidate_folders
            if all(any(folder.glob(pattern)) for pattern in required_patterns)
        ),
        None,
    )

    if market_folder is None:
        raise FileNotFoundError(
            "Nenhuma pasta contém simultaneamente os arquivos IN, IR, SPRD e PR."
        )

    MARKET_DATE = market_folder.name

else:
    market_folder = data_root / MARKET_DATE

    if not market_folder.exists():
        raise FileNotFoundError(
            f"A pasta de mercado não foi encontrada: {market_folder}"
        )

print("Data de mercado selecionada:", MARKET_DATE)
print("Pasta selecionada:", market_folder)


def one_file(pattern: str) -> Path:
    files = sorted(market_folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Arquivo {pattern} não encontrado em {market_folder}.")
    return files[-1]


in_zip = one_file("IN*.zip")
ir_zip = one_file("IR*.zip")
sprd_zip = one_file("SPRD*.zip")
pr_zip = one_file("PR*.zip")
valuation_date = date.fromisoformat(MARKET_DATE)
valuation_ts = pd.Timestamp(valuation_date)

display(pd.DataFrame({
    "fonte": ["IN", "IR", "SPRD", "PR"],
    "arquivo": [in_zip.name, ir_zip.name, sprd_zip.name, pr_zip.name],
}))

# COMMAND ----------

# Comentário: funções genéricas para remover namespaces e ler a publicação XML mais recente de cada ZIP.
def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def flatten_element(element) -> dict:
    row = {}
    for child in element.iter():
        value = (child.text or "").strip()
        if value:
            row.setdefault(local_name(child.tag), value)
    return row


def latest_xml_member(archive: ZipFile):
    members = [x for x in archive.infolist() if x.filename.lower().endswith(".xml")]
    if not members:
        raise ValueError("O ZIP não contém XML.")
    return max(members, key=lambda x: x.date_time)


def read_small_xml(zip_path: Path):
    with ZipFile(zip_path) as archive:
        member = latest_xml_member(archive)
        return etree.fromstring(archive.read(member), parser=etree.XMLParser(huge_tree=True)), member.filename

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Leitura eficiente do cadastro IN
# MAGIC
# MAGIC O `IN` é muito grande. Por isso, o XML não é carregado inteiro na memória.
# MAGIC O leitor percorre um instrumento por vez e guarda somente registros cujo ativo seja `IBOV`.

# COMMAND ----------

# Comentário: lê o IN em streaming e conserva somente opções europeias sobre o Ibovespa.
def read_ibov_option_catalog(zip_path: Path) -> tuple[pd.DataFrame, str]:
    rows = []
    with ZipFile(zip_path) as archive:
        member = latest_xml_member(archive)
        with archive.open(member) as xml_file:
            context = etree.iterparse(
                xml_file,
                events=("end",),
                tag="{*}Instrm",
                huge_tree=True,
            )
            for _, element in context:
                row = flatten_element(element)
                if (
                    row.get("Asst") == "IBOV"
                    and row.get("OptnTp") in {"CALL", "PUTT"}
                    and row.get("ExrcPric")
                    and row.get("XprtnDt")
                ):
                    rows.append(row)

                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]

    return pd.DataFrame(rows), member.filename


catalog, in_member = read_ibov_option_catalog(in_zip)

catalog = catalog.rename(columns={
    "TckrSymb": "ticker",
    "ExrcPric": "strike",
    "XprtnDt": "maturity_date",
    "OptnTp": "option_type",
    "OptnStyle": "option_style",
})
catalog["strike"] = pd.to_numeric(catalog["strike"], errors="coerce")
catalog["maturity_date"] = pd.to_datetime(catalog["maturity_date"], errors="coerce")
catalog["option_type"] = catalog["option_type"].replace({"PUTT": "PUT"})
catalog = catalog.dropna(subset=["ticker", "strike", "maturity_date"])
catalog = catalog.drop_duplicates("ticker", keep="last")

print(f"XML cadastral utilizado: {in_member}")
print(f"Opções IBOV no cadastro: {len(catalog):,}")
display(catalog[["ticker", "option_type", "option_style", "strike", "maturity_date"]].head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Leitura do PriceReport e cruzamento
# MAGIC
# MAGIC Os quatro XMLs existentes no `PR` são publicações sucessivas. Usamos a publicação
# MAGIC mais recente e procuramos todos os registros cujo ticker começa com `IBOV`.

# COMMAND ----------

# Comentário: lê em streaming a última publicação do PriceReport e extrai preços das opções IBOV.
def read_ibov_prices(zip_path: Path) -> tuple[pd.DataFrame, str]:
    rows = []
    with ZipFile(zip_path) as archive:
        member = latest_xml_member(archive)
        with archive.open(member) as xml_file:
            context = etree.iterparse(
                xml_file,
                events=("end",),
                tag="{*}PricRpt",
                huge_tree=True,
            )
            for _, element in context:
                row = flatten_element(element)
                if row.get("TckrSymb", "").startswith("IBOV"):
                    rows.append(row)
                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]
    return pd.DataFrame(rows), member.filename


prices, pr_member = read_ibov_prices(pr_zip)
prices = prices.rename(columns={
    "TckrSymb": "ticker",
    "FrstPric": "first_price",
    "MinPric": "min_price",
    "MaxPric": "max_price",
    "TradAvrgPric": "average_price",
    "LastPric": "last_price",
    "BestBidPric": "best_bid",
    "BestAskPric": "best_ask",
    "RglrTxsQty": "trades",
    "OpnIntrst": "open_interest",
})

numeric_price_columns = [
    "first_price", "min_price", "max_price", "average_price", "last_price",
    "best_bid", "best_ask", "trades", "open_interest",
]
for column in numeric_price_columns:
    if column not in prices:
        prices[column] = np.nan
    prices[column] = pd.to_numeric(prices[column], errors="coerce")

prices = prices.drop_duplicates("ticker", keep="last")
options = catalog.merge(prices, on="ticker", how="inner", validate="one_to_one")

print(f"XML de preços utilizado: {pr_member}")
print(f"Tickers IBOV no PR: {len(prices):,}")
print(f"Cruzamentos IN × PR: {len(options):,}")

# COMMAND ----------

# Comentário: escolhe mid quando bid e ask são válidos; caso contrário usa o último preço negociado.
valid_spread = (
    options["best_bid"].notna()
    & options["best_ask"].notna()
    & (options["best_bid"] > 0)
    & (options["best_ask"] >= options["best_bid"])
)
options["market_price"] = np.where(
    valid_spread,
    (options["best_bid"] + options["best_ask"]) / 2.0,
    options["last_price"],
)
options["price_source"] = np.where(valid_spread, "mid", "last")
options["T"] = (options["maturity_date"] - valuation_ts).dt.days / 365.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Reconstrução das curvas $r(T)$ e $q(T)$
# MAGIC
# MAGIC O futuro do Ibovespa obedece a:
# MAGIC
# MAGIC $$F_{0,T}=S_0e^{(r(T)-q(T))T}$$
# MAGIC
# MAGIC Isolando o dividend yield:
# MAGIC
# MAGIC $$q(T)=r(T)-\frac{1}{T}\ln\left(\frac{F_{0,T}}{S_0}\right)$$
# MAGIC
# MAGIC O `IR` fornece $S_0$, o DI1 fornece $r(T)$ e o futuro IND fornece $F_{0,T}$.

# COMMAND ----------

# Comentário: extrai o fechamento do IBOV e os ajustes dos contratos DI1 e IND.
ir_root, ir_member = read_small_xml(ir_zip)
sprd_root, sprd_member = read_small_xml(sprd_zip)

index_rows = [
    flatten_element(x) for x in ir_root.iter()
    if local_name(x.tag) == "IndxInf"
]
index_df = pd.DataFrame(index_rows)
ibov_row = index_df.loc[index_df["TckrSymb"].eq("IBOV")]
if len(ibov_row) != 1:
    raise ValueError(f"Esperava um IBOV no IR; encontrei {len(ibov_row)}.")
spot = float(ibov_row.iloc[0]["ClsgPric"])

price_rows = [
    flatten_element(x) for x in sprd_root.iter()
    if local_name(x.tag) == "PricRpt"
]
derivatives = pd.DataFrame(price_rows)

MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


def first_weekday(year: int, month: int) -> date:
    result = date(year, month, 1)
    while result.weekday() >= 5:
        result += timedelta(days=1)
    return result


def nearest_wednesday(year: int, month: int) -> date:
    candidates = [
        date(year, month, day)
        for day in range(12, 19)
        if date(year, month, day).weekday() == 2
    ]
    return min(candidates, key=lambda x: abs(x.day - 15))


def contract_maturity(ticker: str) -> date:
    month = MONTH_CODES[ticker[-3]]
    year = 2000 + int(ticker[-2:])
    return first_weekday(year, month) if ticker.startswith("DI1") else nearest_wednesday(year, month)


di = derivatives.loc[derivatives["TckrSymb"].str.startswith("DI1", na=False)].copy()
ind = derivatives.loc[derivatives["TckrSymb"].str.startswith("IND", na=False)].copy()
di["rate_effective"] = pd.to_numeric(di["AdjstdQtTax"], errors="coerce") / 100.0
ind["future"] = pd.to_numeric(ind["AdjstdQt"], errors="coerce")
di = di.dropna(subset=["rate_effective"]).drop_duplicates("TckrSymb")
ind = ind.dropna(subset=["future"]).drop_duplicates("TckrSymb")
di["maturity"] = pd.to_datetime(di["TckrSymb"].map(contract_maturity))
ind["maturity"] = pd.to_datetime(ind["TckrSymb"].map(contract_maturity))
di["T"] = (di["maturity"] - valuation_ts).dt.days / 365.0
ind["T"] = (ind["maturity"] - valuation_ts).dt.days / 365.0
di = di.loc[di["T"] > 0].sort_values("T")
ind = ind.loc[ind["T"].between(di["T"].min(), di["T"].max())].sort_values("T")

ind["r_effective"] = np.interp(ind["T"], di["T"], di["rate_effective"])
ind["r"] = np.log1p(ind["r_effective"])
ind["q"] = ind["r"] - np.log(ind["future"] / spot) / ind["T"]

curve = ind[["T", "r", "q", "future"]].drop_duplicates("T").sort_values("T")


def interpolate_market_curve(target_tenors) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target_tenors, dtype=float)
    if target.min() < curve["T"].min() or target.max() > curve["T"].max():
        raise ValueError("Há vencimentos de opções fora do intervalo coberto pelos futuros IND.")
    return (
        np.interp(target, curve["T"], curve["r"]),
        np.interp(target, curve["T"], curve["q"]),
    )

print(f"Spot IBOV: {spot:,.2f}")
display(curve.style.format({"r": "{:.4%}", "q": "{:.4%}", "future": "{:,.2f}"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Filtros e volatilidade implícita
# MAGIC
# MAGIC Para uma call europeia com dividend yield contínuo:
# MAGIC
# MAGIC $$C=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2)$$
# MAGIC
# MAGIC Para uma put europeia:
# MAGIC
# MAGIC $$P=Ke^{-rT}\Phi(-d_2)-S_0e^{-qT}\Phi(-d_1)$$
# MAGIC
# MAGIC com:
# MAGIC
# MAGIC $$d_1=\frac{\ln(S_0/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt{T}},\qquad d_2=d_1-\sigma\sqrt{T}$$

# COMMAND ----------

# Comentário: aplica filtros de prazo, estilo, negociação, posição em aberto e cobertura das curvas.
filter_summary = []


def register_filter(name: str, frame: pd.DataFrame) -> None:
    filter_summary.append({"etapa": name, "quantidade": len(frame)})


filtered = options.copy()
register_filter("Cruzamento IN × PR", filtered)

filtered = filtered.loc[filtered["option_style"].eq("EURO")]
register_filter("Somente opções europeias", filtered)

filtered = filtered.loc[filtered["T"] > 0]
register_filter("Vencimento futuro", filtered)

filtered = filtered.loc[filtered["market_price"].gt(0)]
register_filter("Preço positivo", filtered)

filtered = filtered.loc[filtered["trades"].fillna(0).ge(MIN_TRADES)]
register_filter(f"Ao menos {MIN_TRADES} negócio(s)", filtered)

filtered = filtered.loc[filtered["open_interest"].fillna(0).ge(MIN_OPEN_INTEREST)]
register_filter("Posição em aberto positiva", filtered)

filtered = filtered.loc[
    filtered["T"].between(curve["T"].min(), curve["T"].max())
].copy()
register_filter("Prazo coberto pelas curvas", filtered)

filtered["r"], filtered["q"] = interpolate_market_curve(filtered["T"])
filtered["forward"] = spot * np.exp((filtered["r"] - filtered["q"]) * filtered["T"])
filtered["log_moneyness"] = np.log(filtered["strike"] / filtered["forward"])

display(pd.DataFrame(filter_summary))

# COMMAND ----------

# Comentário: funções Black–Scholes, limites de arbitragem e inversão numérica da volatilidade.
def black_scholes_price(
    option_type: str,
    spot_value: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    sqrt_t = np.sqrt(maturity)
    d1 = (
        np.log(spot_value / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot_value * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)

    if option_type == "CALL":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def arbitrage_bounds(row) -> tuple[float, float]:
    discounted_spot = spot * np.exp(-row["q"] * row["T"])
    discounted_strike = row["strike"] * np.exp(-row["r"] * row["T"])
    if row["option_type"] == "CALL":
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


def implied_volatility(row) -> float:
    lower, upper = arbitrage_bounds(row)
    if not (lower < row["market_price"] < upper):
        return np.nan

    def objective(volatility: float) -> float:
        return black_scholes_price(
            row["option_type"],
            spot,
            row["strike"],
            row["T"],
            row["r"],
            row["q"],
            volatility,
        ) - row["market_price"]

    try:
        return float(brentq(objective, 1e-4, 5.0, xtol=1e-10, maxiter=200))
    except ValueError:
        return np.nan


filtered["implied_vol"] = filtered.apply(implied_volatility, axis=1)
iv_data = filtered.dropna(subset=["implied_vol"]).copy()
register_filter("Preço dentro dos limites de arbitragem", iv_data)

iv_data = iv_data.loc[iv_data["implied_vol"].between(MIN_IV, MAX_IV)].copy()
register_filter(f"Volatilidade entre {MIN_IV:.0%} e {MAX_IV:.0%}", iv_data)

display(pd.DataFrame(filter_summary).drop_duplicates("etapa", keep="last"))
display(
    iv_data[
        [
            "ticker", "option_type", "maturity_date", "strike", "market_price",
            "price_source", "trades", "open_interest", "r", "q", "implied_vol",
        ]
    ].sort_values(["maturity_date", "strike"])
    .style.format({
        "strike": "{:,.0f}", "market_price": "{:,.2f}", "r": "{:.4%}",
        "q": "{:.4%}", "implied_vol": "{:.2%}",
    })
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Smiles e superfície
# MAGIC
# MAGIC Utilizamos como eixo horizontal o log-moneyness forward:
# MAGIC
# MAGIC $$k=\ln\left(\frac{K}{F_{0,T}}\right)$$
# MAGIC
# MAGIC | Valor de $k$ | Interpretação |
# MAGIC |---|---|
# MAGIC | $k<0$ | strike abaixo do forward |
# MAGIC | $k=0$ | opção aproximadamente at-the-money forward |
# MAGIC | $k>0$ | strike acima do forward |

# COMMAND ----------

# Comentário: desenha um smile separado para cada vencimento com observações suficientes.
fig, ax = plt.subplots(figsize=(11, 6))

plotted = 0
for maturity_date, group in iv_data.groupby("maturity_date"):
    if len(group) < 3:
        continue
    group = group.sort_values("log_moneyness")
    ax.plot(
        group["log_moneyness"],
        100 * group["implied_vol"],
        marker="o",
        linewidth=1,
        label=maturity_date.strftime("%d/%m/%Y"),
    )
    plotted += 1

ax.axvline(0.0, color="black", linewidth=1, alpha=0.5)
ax.set_title(f"Smiles de volatilidade implícita — {valuation_date:%d/%m/%Y}")
ax.set_xlabel("Log-moneyness forward: ln(K/F)")
ax.set_ylabel("Volatilidade implícita (% a.a.)")
ax.grid(alpha=0.25)
if plotted:
    ax.legend(ncol=2, fontsize=8)
plt.show()

# COMMAND ----------

# Comentário: interpola as observações em uma grade bidimensional e apresenta a primeira superfície.
surface_data = iv_data.drop_duplicates(["T", "log_moneyness"]).copy()

if len(surface_data) < 6 or surface_data["T"].nunique() < 2:
    raise ValueError("Há poucos pontos ou vencimentos para construir uma superfície bidimensional.")

t_grid = np.linspace(surface_data["T"].min(), surface_data["T"].max(), 60)
k_low, k_high = surface_data["log_moneyness"].quantile([0.05, 0.95])
k_grid = np.linspace(k_low, k_high, 80)
K_GRID, T_GRID = np.meshgrid(k_grid, t_grid)

points = surface_data[["log_moneyness", "T"]].to_numpy()
values = surface_data["implied_vol"].to_numpy()
IV_GRID = griddata(points, values, (K_GRID, T_GRID), method="linear")

fig, ax = plt.subplots(figsize=(11, 6))
contour = ax.contourf(K_GRID, T_GRID, 100 * IV_GRID, levels=20, cmap="viridis")
ax.scatter(
    surface_data["log_moneyness"],
    surface_data["T"],
    c=100 * surface_data["implied_vol"],
    cmap="viridis",
    edgecolor="white",
    s=30,
)
fig.colorbar(contour, ax=ax, label="Volatilidade implícita (% a.a.)")
ax.set_title("Primeira superfície de volatilidade implícita do IBOV")
ax.set_xlabel("Log-moneyness forward: ln(K/F)")
ax.set_ylabel("Prazo T em anos")
plt.show()

# COMMAND ----------

# Comentário: estima a volatilidade para o strike e prazo da opção com barreira.
target_r, target_q = interpolate_market_curve([TARGET_MATURITY])
target_forward = spot * np.exp((target_r[0] - target_q[0]) * TARGET_MATURITY)
target_k = np.log(TARGET_STRIKE / target_forward)

target_iv = griddata(
    points,
    values,
    np.array([[target_k, TARGET_MATURITY]]),
    method="linear",
)[0]

if np.isnan(target_iv):
    target_iv = griddata(
        points,
        values,
        np.array([[target_k, TARGET_MATURITY]]),
        method="nearest",
    )[0]
    interpolation_method = "nearest — alvo fora do casco convexo"
else:
    interpolation_method = "linear"

surface_result = pd.DataFrame({
    "Parâmetro": [
        "Spot",
        "Strike alvo",
        "Prazo alvo",
        "Forward alvo",
        "Log-moneyness alvo",
        "Taxa r(T)",
        "Dividend yield q(T)",
        "Volatilidade implícita",
        "Método de interpolação",
    ],
    "Valor": [
        f"{spot:,.2f}",
        f"{TARGET_STRIKE:,.2f}",
        f"{TARGET_MATURITY:.4f}",
        f"{target_forward:,.2f}",
        f"{target_k:.6f}",
        f"{100 * target_r[0]:.4f}% a.a.",
        f"{100 * target_q[0]:.4f}% a.a.",
        f"{100 * target_iv:.4f}% a.a.",
        interpolation_method,
    ],
})

display(surface_result)

print(f"Volatilidade fixa anterior: {22.0:.2f}% a.a.")
print(
    f"Volatilidade obtida da superfície: "
    f"{100 * target_iv:.2f}% a.a."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Limitações desta primeira superfície
# MAGIC
# MAGIC 1. O preço pode ser o último negócio e não uma cotação executável.
# MAGIC 2. Há poucas opções com bid e ask simultaneamente disponíveis.
# MAGIC 3. Um negócio isolado pode estar defasado em relação ao fechamento do índice.
# MAGIC 4. A interpolação linear não garante ausência de arbitragem estática.
# MAGIC 5. O calendário ainda considera aproximações na construção das curvas DI1 e IND.
# MAGIC 6. Uma opção com barreira é sensível ao skew de volatilidade; utilizar somente a
# MAGIC    volatilidade no strike da opção é uma primeira aproximação, não o modelo final.
# MAGIC 7. Uma evolução profissional pode usar SVI/SABR, pesos por liquidez e calibração
# MAGIC    conjunta de calls e puts por preço forward.