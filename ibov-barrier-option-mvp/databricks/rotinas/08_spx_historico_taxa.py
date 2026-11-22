# Databricks notebook source
# MAGIC %md
# MAGIC # SPX: histórico diário, r acima da SOFR e método antigo x novo
# MAGIC
# MAGIC Roda a fotografia D-1 do notebook 07 em **todos os pregões** de um intervalo (padrão: 03/08/2026 até D-1,
# MAGIC o último pregão com dados) e monta um painel por dia e por vencimento.
# MAGIC
# MAGIC Cada dia é processado duas vezes sobre as mesmas cotações, para comparar:
# MAGIC
# MAGIC - **antigo** (`rate_method="atm"`): taxa de cada vencimento pela regressão de C − P em K com os 10 pares
# MAGIC   mais próximos do ATM; prazos < 3 meses usam a taxa do vencimento confiável mais curto;
# MAGIC - **novo** (`rate_method="smooth"`, padrão do pipeline): taxa de cada vencimento com T >= 3m pela regressão
# MAGIC   ampla (pares em ±10% do ATM), reta r(T) robusta ajustada a essas taxas e aplicada a todos os prazos,
# MAGIC   descarte de vencimentos com resíduo incoerente e alerta de dia com baixa cobertura.
# MAGIC
# MAGIC O **box spread** (menor x maior strike da faixa de ±10%, executado nas pontas) continua como referência
# MAGIC independente: comprar no ask = emprestar (taxa mais baixa); vender no bid = tomar (taxa mais alta).
# MAGIC
# MAGIC A cadeia bruta de cada dia fica salva em disco (parquet). Rodar de novo não chama a API para os dias já
# MAGIC baixados.
# MAGIC
# MAGIC **Uso:** ajuste os widgets, rode tudo e copie a saída da última célula (relatório em texto).

# COMMAND ----------

# MAGIC %pip install --upgrade "grpcio>=1.80.0" thetadata python-dotenv "exchange-calendars==4.13.2" "lxml>=6.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import importlib

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
if not notebook_path.startswith("/Workspace"):
    notebook_path = "/Workspace" + notebook_path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
SRC = os.path.join(project_dir, "src")
if not os.path.isdir(os.path.join(SRC, "ibov_barrier", "markets")):
    raise RuntimeError(f"{SRC} não tem ibov_barrier/markets: o Git folder precisa estar na branch feature/sp500.")

for name in [m for m in sys.modules if m == "ibov_barrier" or m.startswith("ibov_barrier.")]:
    del sys.modules[name]
while SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
importlib.invalidate_caches()

import ibov_barrier
print("Código de:", ibov_barrier.__file__)

# COMMAND ----------

import json
import time as _time
import traceback
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ibov_barrier.markets.base import CALL, PUT, CleaningReport
from ibov_barrier.markets.parity import implied_forwards
from ibov_barrier.markets.vol import implied_volatility
from ibov_barrier.markets.cleaning import clean_quotes
from ibov_barrier.markets.quality import low_coverage_flags
from ibov_barrier.markets.spx import SPX_SPEC, SPXAdapter, ThetaDataSource

MIN_T_RATE = 0.25        # a partir daqui cada vencimento mede a própria taxa
WIDE_BAND = 0.10         # faixa |ln(K/F)| do box
PAUSE_S = 0.5            # pausa entre chamadas à API

# Pasta fora do Git folder (os parquets não devem ir para o repositório).
user_home = "/".join(notebook_path.split("/")[:4])          # /Workspace/Users/<usuário>
dbutils.widgets.text("inicio", "2026-08-03", "Início (AAAA-MM-DD)")
dbutils.widgets.text("fim", "", "Fim (vazio = D-1)")
dbutils.widgets.text("pasta", f"{user_home}/spx_dados", "Pasta dos dados")

source = ThetaDataSource()
probe = SPXAdapter(lambda r, d: None)
start = date.fromisoformat(dbutils.widgets.get("inicio").strip())
end_txt = dbutils.widgets.get("fim").strip()
end = date.fromisoformat(end_txt) if end_txt else probe.market_date()
DATA_DIR = Path(dbutils.widgets.get("pasta").strip())
RAW_DIR = DATA_DIR / "brutos"
RAW_DIR.mkdir(parents=True, exist_ok=True)

sessions = [s.date() for s in xcals.get_calendar("XNYS").sessions_in_range(start.isoformat(), end.isoformat())]
print(f"Pregões NYSE de {start} a {end}: {len(sessions)} | pasta: {DATA_DIR} | chave: {source.key_source}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download (só os dias que ainda não estão em disco)

# COMMAND ----------

def call_api(fn, *args, tries=3):
    for k in range(tries):
        try:
            out = fn(*args)
            _time.sleep(PAUSE_S)
            return out
        except Exception as exc:
            if "NoDataFound" in type(exc).__name__ or k == tries - 1:
                raise
            _time.sleep(2 * (k + 1))


def download_day(d):
    """Grava SPX.parquet, SPXW.parquet e meta.json. O meta.json é gravado por último: marca o dia completo."""
    folder = RAW_DIR / d.isoformat()
    meta_path = folder / "meta.json"
    if meta_path.exists():
        return "em disco"
    folder.mkdir(parents=True, exist_ok=True)
    rows = {}
    for root in SPX_SPEC.roots:
        raw = call_api(source.option_chain, root, d)
        raw.to_parquet(folder / f"{root}.parquet", index=False)
        rows[root] = len(raw)
    meta = {"data": d.isoformat(), "spot": call_api(source.index_close, d), "linhas": rows}
    try:
        meta["sofr"] = call_api(source.sofr, d)
    except Exception as exc:
        meta["sofr"] = None
        meta["sofr_erro"] = f"{type(exc).__name__}: {exc}"[:200]
    meta_path.write_text(json.dumps(meta))
    return "baixado"


download_status = {}
for d in sessions:
    try:
        download_status[d] = download_day(d)
    except Exception as exc:
        download_status[d] = f"falhou: {type(exc).__name__}: {str(exc)[:150]}"
    print(d, download_status[d])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Painel por vencimento e resumo por dia (antigo x novo, mesmas cotações)

# COMMAND ----------

def load_meta(d):
    return json.loads((RAW_DIR / d.isoformat() / "meta.json").read_text())


def adapter_for(d, meta, method):
    folder = RAW_DIR / d.isoformat()
    return SPXAdapter(lambda root, _d: pd.read_parquet(folder / f"{root}.parquet"),
                      spot_source=lambda _d: meta["spot"], rate_method=method)


def pair_quotes(group):
    """Uma linha por strike com bid/ask de call e put (cotações já limpas)."""
    q = group.loc[(group["bid"] > 0) & (group["ask"] >= group["bid"])]
    side = lambda t: q.loc[q["option_type"].eq(t)].groupby("strike")[["bid", "ask"]].mean()
    return side(CALL).join(side(PUT), lsuffix="_c", rsuffix="_p", how="inner").reset_index()


def box_rates(group, T, forward):
    """Box entre o menor e o maior strike em ±WIDE_BAND: taxa de empréstimo (ask), mid e tomada (bid)."""
    p = pair_quotes(group)
    p = p.loc[np.abs(np.log(p["strike"] / forward)) <= WIDE_BAND]
    out = {"pares amplos": len(p)}
    if len(p) >= 2:
        lo, hi = p.iloc[0], p.iloc[-1]
        width = hi["strike"] - lo["strike"]
        cost_ask = lo["ask_c"] - hi["bid_c"] + hi["ask_p"] - lo["bid_p"]    # comprar o box = emprestar
        value_bid = lo["bid_c"] - hi["ask_c"] + hi["bid_p"] - lo["ask_p"]   # vender o box = tomar
        mid = (cost_ask + value_bid) / 2
        rate = lambda v: -np.log(v / width) / T if 0 < v < width * 1.5 else np.nan
        out.update({"box largura": width, "r box empresta": rate(cost_ask),
                    "r box mid": rate(mid), "r box toma": rate(value_bid)})
    return out


def forwards_frame(snap, tag):
    f = pd.DataFrame([{"vencimento": p.expiration, "T": p.T, f"forward {tag}": p.forward,
                       f"r {tag}": p.rate, f"resíduo {tag}": p.residual_std,
                       "r do vencimento": p.fit_rate, "no ajuste": p.in_fit} for p in snap.forwards])
    f[f"q {tag}"] = f[f"r {tag}"] - np.log(f[f"forward {tag}"] / snap.spot) / f["T"]
    o = snap.options
    atm = o.assign(dist=o["log_moneyness"].abs()).sort_values("dist").groupby("expiration")["implied_vol"].first()
    f[f"vol ATM {tag}"] = f["vencimento"].map(atm)
    if tag == "antigo":
        f = f.drop(columns=["r do vencimento", "no ajuste"])
    return f


def near(frame, t, col):
    sub = frame.dropna(subset=[col])
    return float(sub.iloc[(sub["T"] - t).abs().argmin()][col]) if len(sub) else np.nan


def old_forwards_frame(quotes, new, adp):
    """Método antigo só no que o relatório usa: forwards (r, q, resíduo) e vol ATM.

    A vol ATM antiga é recalculada no MESMO contrato escolhido pelo método novo, com forward/r/q antigos.
    Assim o Δ vol mede só o efeito do forward e da taxa, e não se calcula a vol implícita de toda a
    superfície uma segunda vez (era cerca de metade do tempo do notebook).
    """
    warnings = []
    pts = implied_forwards(quotes, discount_curve=adp.rate_curve, rate_method="atm", report_warnings=warnings)
    f = pd.DataFrame([{"vencimento": p.expiration, "T": p.T, "forward antigo": p.forward, "r antigo": p.rate,
                       "resíduo antigo": p.residual_std} for p in pts])
    f["q antigo"] = f["r antigo"] - np.log(f["forward antigo"] / new.spot) / f["T"]
    o = new.options
    atm = o.assign(dist=o["log_moneyness"].abs()).sort_values("dist").groupby(["expiration", "T"]).head(1)
    atm = atm.merge(f, left_on=["expiration", "T"], right_on=["vencimento", "T"], how="inner")
    vols = {e: implied_volatility(t, price, new.spot, k, T, r, q) for e, t, price, k, T, r, q in zip(
        atm["vencimento"], atm["option_type"], atm["market_price"], atm["strike"], atm["T"],
        atm["r antigo"], atm["q antigo"])}
    f["vol ATM antigo"] = f["vencimento"].map(vols)
    return f, warnings


def process_day(d):
    meta = load_meta(d)
    adp = adapter_for(d, meta, "smooth")
    new = adp.load_snapshot(d)
    quotes = clean_quotes(adp.load_chain(d), adp.cleaning, CleaningReport())
    old_f, old_warnings = old_forwards_frame(quotes, new, adp)
    f = forwards_frame(new, "novo").merge(old_f, on=["vencimento", "T"], how="outer")
    boxes = []
    for _, row in f.iterrows():
        fwd = row["forward novo"] if pd.notna(row["forward novo"]) else row["forward antigo"]
        g = quotes.loc[quotes["expiration"].eq(row["vencimento"]) & np.isclose(quotes["T"], row["T"])]
        boxes.append(box_rates(g, row["T"], fwd))
    panel = pd.concat([f.reset_index(drop=True), pd.DataFrame(boxes)], axis=1)
    panel.insert(0, "data", d)
    panel["sofr"] = meta.get("sofr")
    panel["Δ vol ATM (pts)"] = (panel["vol ATM novo"] - panel["vol ATM antigo"]) * 100

    def count(warnings, word):
        return sum(word in w for w in warnings)

    new_atm = panel.dropna(subset=["vol ATM novo"])
    day = {"data": d, "spot": new.spot, "sofr": meta.get("sofr"),
           "pontos": len(new.options),
           "vencimentos": len(new.forwards), "vencimentos antigo": len(old_f),
           "avisos": len(new.report.warnings), "avisos antigo": len(old_warnings),
           "fora do ajuste": count(new.report.warnings, "fora da curva"),
           "descartados": count(new.report.warnings, "descartado"),
           "descartados antigo": count(old_warnings, "descartado"),
           "vol ATM 1m": near(new_atm, 1 / 12, "vol ATM novo"), "vol ATM 12m": near(new_atm, 1.0, "vol ATM novo"),
           "r 1a antigo": near(panel, 1.0, "r antigo"), "r 1a novo": near(panel, 1.0, "r novo"),
           "q 1a antigo": near(panel, 1.0, "q antigo"), "q 1a novo": near(panel, 1.0, "q novo"),
           "avisos novo (texto)": " | ".join(new.report.warnings)[:400]}
    return panel, day


panels, days, process_status = [], [], {}
_t0 = _time.time()
for k, d in enumerate(sessions, 1):
    if k % 10 == 0 or k == len(sessions):
        print(f"processando {k}/{len(sessions)} ({d}) | {(_time.time() - _t0) / 60:.1f} min", flush=True)
    if not (RAW_DIR / d.isoformat() / "meta.json").exists():
        process_status[d] = "sem dados em disco"
        continue
    try:
        panel_d, day = process_day(d)
        panels.append(panel_d)
        days.append(day)
        process_status[d] = "ok"
    except Exception as exc:
        process_status[d] = f"falhou: {type(exc).__name__}: {str(exc)[:150]}"
        print(d, process_status[d])

panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
daily = pd.DataFrame(days).sort_values("data").reset_index(drop=True) if days else pd.DataFrame()
RATES = {"antigo": "r antigo", "novo": "r novo", "vencimento": "r do vencimento",
         "box empresta": "r box empresta", "box mid": "r box mid", "box toma": "r box toma"}
if len(panel):
    for k, c in RATES.items():
        panel[f"{k} - SOFR (bps)"] = (panel[c] - panel["sofr"]) * 1e4
    daily = daily.join(low_coverage_flags(daily["pontos"]))
    panel.to_parquet(DATA_DIR / "painel_vencimentos.parquet", index=False)
    daily.to_parquet(DATA_DIR / "resumo_diario.parquet", index=False)
print(f"Dias processados: {len(daily)} de {len(sessions)} | linhas no painel: {len(panel)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gráfico: r − SOFR por dia, antigo x novo (vencimentos com T >= 3 meses)

# COMMAND ----------

BG, GRID, MUTED = "#0B0B0B", "#2A2A2A", "#8C8C8C"
WARM = ["#FFF3B0", "#FFC233", "#FF8A1F", "#F0432F", "#D6246E"]
AXIS = dict(gridcolor=GRID, zerolinecolor=MUTED, linecolor=MUTED, showline=True,
            tickfont=dict(color="white"), title_font=dict(color="white"))
SP = {k: f"{k} - SOFR (bps)" for k in RATES}

long_ = panel.loc[panel["T"] >= MIN_T_RATE] if len(panel) else panel
if len(long_):
    med = long_.groupby("data")[list(SP.values())].median()
    disp = long_.groupby("data")[[SP["antigo"], SP["novo"]]].std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=med.index, y=med[SP["box toma"]], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=med.index, y=med[SP["box empresta"]], mode="lines", fill="tonexty",
                             fillcolor="rgba(255,138,31,0.15)", line=dict(width=0), name="box: empresta-toma"))
    for i, (k, sym, dash) in enumerate([("antigo", "circle", "dot"), ("novo", "square", "solid")]):
        fig.add_trace(go.Scatter(
            x=med.index, y=med[SP[k]], mode="lines+markers", name=f"{k}: mediana",
            line=dict(color=WARM[1 + 2 * i], dash=dash, width=2), marker=dict(symbol=sym, size=6, color=WARM[1 + 2 * i]),
            customdata=disp[SP[k]], hovertemplate="%{x}<br>%{y:+.0f} bps<br>desvio entre vencimentos %{customdata:.0f} bps"))
    low = daily.loc[daily["baixa cobertura"].fillna(False), "data"]
    for x in low:
        fig.add_vline(x=x, line_color=WARM[3], line_width=1, opacity=0.35)
    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="white"), height=480,
                      title=dict(text="<b>SPX: r − SOFR (bps), antigo x novo · linhas vermelhas = baixa cobertura</b>",
                                 x=0.5),
                      legend=dict(bgcolor="#141414", bordercolor="#3A3A3A", borderwidth=1),
                      margin=dict(t=70, l=60, r=20, b=50))
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(title_text="bps", range=[-150, 350], **AXIS)
    fig.show(config={"displaylogo": False, "scrollZoom": True,
                     "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "spx_r_antigo_x_novo"}})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Relatório único para enviar (copie a saída inteira)

# COMMAND ----------

def _df(frame, spec=None):
    spec = {c: f for c, f in (spec or {}).items() if c in frame.columns}
    fmt = {c: (lambda f: (lambda v: "-" if pd.isna(v) else f.format(v)))(f) for c, f in spec.items()}
    with pd.option_context("display.max_rows", 1000, "display.max_columns", None, "display.width", 400,
                           "display.max_colwidth", 400):
        print(frame.to_string(index=False, formatters=fmt))


def _block(title, fn):
    print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)
    try:
        fn()
    except Exception as exc:
        print(f"[não disponível] {type(exc).__name__}: {exc}")
        print("".join(traceback.format_exc().splitlines(keepends=True)[-4:]))


BPS = "{:+.0f}"


def _r_status():
    st = pd.DataFrame({"data": sessions, "download": [download_status.get(d) for d in sessions],
                       "processamento": [process_status.get(d) for d in sessions]})
    print(f"pregões: {len(sessions)} | processados: {(st['processamento'] == 'ok').sum()}")
    bad = st.loc[st["processamento"] != "ok"]
    if len(bad):
        print("\ndias com problema:")
        _df(bad)


def _r_daily():
    m = long_.groupby("data")[[SP["antigo"], SP["novo"], SP["box empresta"], SP["box toma"]]].median()
    m.columns = ["med antigo", "med novo", "med box empresta", "med box toma"]
    s = long_.groupby("data")[[SP["antigo"], SP["novo"]]].std()
    s.columns = ["desvio antigo", "desvio novo"]
    d = daily.merge(m.join(s).reset_index(), on="data", how="left")
    cols = ["data", "spot", "pontos", "razão", "baixa cobertura", "vencimentos antigo", "vencimentos",
            "fora do ajuste", "descartados", "sofr", "r 1a antigo", "r 1a novo", "q 1a antigo", "q 1a novo",
            "med antigo", "med novo", "desvio antigo", "desvio novo", "med box empresta", "med box toma"]
    _df(d[cols], {"spot": "{:,.2f}", "razão": "{:.0%}", "sofr": "{:.3%}", "r 1a antigo": "{:.3%}",
                  "r 1a novo": "{:.3%}", "q 1a antigo": "{:.3%}", "q 1a novo": "{:.3%}",
                  **{c: BPS for c in cols if c.startswith(("med ", "desvio "))}})


def _r_buckets():
    b = long_.assign(faixa=pd.cut(long_["T"], [0.25, 0.5, 1.0, 1.5, 2.1],
                                  labels=["3-6m", "6-12m", "12-18m", "18-24m"], right=False))
    rows = []
    for faixa, g in b.groupby("faixa", observed=True):
        row = {"faixa": faixa, "obs": len(g)}
        for k, c in SP.items():
            row[f"{k} p25/med/p75"] = f"{g[c].quantile(.25):+.0f} / {g[c].median():+.0f} / {g[c].quantile(.75):+.0f}"
        row["% SOFR < box empresta"] = (g["sofr"] < g["r box empresta"]).mean()
        rows.append(row)
    _df(pd.DataFrame(rows), {"% SOFR < box empresta": "{:.0%}"})


def _r_before_after():
    def line(name, a, b, fmt="{:.1f}"):
        print(f"{name:<62} antigo {fmt.format(a):>10} | novo {fmt.format(b):>10}")

    s = long_.groupby("data")[[SP["antigo"], SP["novo"]]].std()
    line("desvio entre vencimentos de r - SOFR (T>=3m), mediana dos dias, bps",
         s[SP["antigo"]].median(), s[SP["novo"]].median())
    line("idem, pior dia, bps", s[SP["antigo"]].max(), s[SP["novo"]].max())
    line("|r - SOFR| por vencimento, percentil 99, bps",
         long_[SP["antigo"]].abs().quantile(.99), long_[SP["novo"]].abs().quantile(.99))
    line("variação diária do r 1a, desvio-padrão, bps",
         daily["r 1a antigo"].diff().std() * 1e4, daily["r 1a novo"].diff().std() * 1e4)
    line("variação diária do q 1a, desvio-padrão, bps",
         daily["q 1a antigo"].diff().std() * 1e4, daily["q 1a novo"].diff().std() * 1e4)
    line("dias com q 1a fora de [0%, 3%]", (~daily["q 1a antigo"].between(0, .03)).sum(),
         (~daily["q 1a novo"].between(0, .03)).sum(), "{:.0f}")
    line("vencimentos por dia, mediana", daily["vencimentos antigo"].median(), daily["vencimentos"].median(), "{:.0f}")
    line("vencimentos descartados por resíduo (total)", daily["descartados antigo"].sum(),
         daily["descartados"].sum(), "{:.0f}")
    print(f"{'taxas fora da curva r(T), ignoradas no ajuste (total)':<62} novo {daily['fora do ajuste'].sum():>10.0f}")
    dv = panel["Δ vol ATM (pts)"].abs()
    print(f"\nimpacto na vol ATM (novo - antigo), pontos de vol: mediana {dv.median():.3f} | p95 {dv.quantile(.95):.3f} "
          f"| máx {dv.max():.3f}")
    worst = panel.loc[dv.nlargest(5).index, ["data", "vencimento", "T", "vol ATM antigo", "vol ATM novo",
                                             "Δ vol ATM (pts)", SP["antigo"], SP["novo"]]]
    _df(worst, {"T": "{:.3f}", "vol ATM antigo": "{:.2%}", "vol ATM novo": "{:.2%}", "Δ vol ATM (pts)": "{:+.2f}",
                SP["antigo"]: BPS, SP["novo"]: BPS})
    low = daily.loc[daily["baixa cobertura"].fillna(False)]
    print(f"\ndias com baixa cobertura (< 70% da mediana dos 20 pregões anteriores não marcados): {len(low)}")
    if len(low):
        _df(low[["data", "pontos", "referência", "razão"]], {"referência": "{:,.0f}", "razão": "{:.0%}"})
    warn_days = daily.loc[daily["avisos"] > 0, ["data", "avisos novo (texto)"]]
    print(f"\navisos do método novo ({len(warn_days)} dias):")
    if len(warn_days):
        _df(warn_days)


def _r_box_cost():
    w = (long_["r box toma"] - long_["r box empresta"]) * 1e4
    print(f"largura da faixa box (toma - empresta), bps: p25 {w.quantile(.25):.0f} | mediana {w.median():.0f} "
          f"| p75 {w.quantile(.75):.0f}")


def _r_last_day():
    last = panel.loc[panel["data"].eq(panel["data"].max())].sort_values("T")
    cols = ["vencimento", "T", "no ajuste", "resíduo antigo", "resíduo novo", SP["antigo"], SP["novo"],
            SP["vencimento"], SP["box empresta"], SP["box toma"], "Δ vol ATM (pts)"]
    _df(last[cols], {"T": "{:.3f}", "resíduo antigo": "{:.3f}", "resíduo novo": "{:.3f}",
                     "Δ vol ATM (pts)": "{:+.2f}", **{c: BPS for c in SP.values()}})


print(f"RELATÓRIO SPX HISTÓRICO (antigo x novo) | {start} a {end} | gerado em "
      f"{pd.Timestamp.now(tz='America/Sao_Paulo'):%Y-%m-%d %H:%M} (Brasília)")
print("antigo = 10 pares ATM por vencimento | novo = curva r(T) suavizada sobre a regressão ampla (±10%) | "
      "vencimento = regressão ampla de cada vencimento | box = menor x maior strike em ±10%, nas pontas")
for _t, _f in [("1. Status por dia", _r_status),
               ("2. Resumo diário (med/desvio = entre vencimentos com T >= 3m, r - SOFR em bps)", _r_daily),
               ("3. r - SOFR por faixa de prazo, todos os dias (bps)", _r_buckets),
               ("4. Antes x depois", _r_before_after),
               ("5. Custo do box", _r_box_cost),
               ("6. Último dia, por vencimento", _r_last_day)]:
    _block(_t, _f)
print("\nFIM DO RELATÓRIO")
