# Databricks notebook source
# MAGIC %md
# MAGIC # SPX: histórico diário e investigação de r acima da SOFR
# MAGIC
# MAGIC Roda a fotografia D-1 do notebook 07 em **todos os pregões** de um intervalo (padrão: 01/10/2026 até D-1)
# MAGIC e monta um painel por dia e por vencimento. O objetivo é ter uma base sólida para entender por que o r
# MAGIC implícito pela paridade fica ~100 bps acima da SOFR.
# MAGIC
# MAGIC Para cada vencimento, a taxa é medida de três jeitos sobre as mesmas cotações:
# MAGIC
# MAGIC - **atual**: regressão de C − P em K com os 10 pares mais próximos do ATM (o que o pipeline usa);
# MAGIC - **ampla**: a mesma regressão com todos os pares em |ln(K/F)| <= 10% (inclinação mais bem medida);
# MAGIC - **box**: box spread entre o menor e o maior strike dessa faixa, executado nas pontas.
# MAGIC   Comprar o box no ask = emprestar dinheiro (taxa mais baixa); vender no bid = tomar (taxa mais alta).
# MAGIC   Se a SOFR ficar **abaixo** da taxa de empréstimo do box, o excesso não é ruído de mid: o mercado de
# MAGIC   opções realmente embute uma taxa acima da SOFR.
# MAGIC
# MAGIC A cadeia bruta de cada dia fica salva em disco (parquet). Rodar de novo não chama a API para os dias já
# MAGIC baixados, e análises futuras podem ser refeitas sem a API.
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
from datetime import date, timedelta
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ibov_barrier.markets.base import CALL, PUT, CleaningReport
from ibov_barrier.markets.cleaning import clean_quotes
from ibov_barrier.markets.spx import SPX_SPEC, SPXAdapter, ThetaDataSource

MIN_T_RATE = 0.25        # a partir daqui a taxa vem da regressão (mesmo padrão do pipeline)
WIDE_BAND = 0.10         # faixa |ln(K/F)| da regressão ampla e do box
PAUSE_S = 0.5            # pausa entre chamadas à API

# Pasta fora do Git folder (os parquets não devem ir para o repositório).
user_home = "/".join(notebook_path.split("/")[:4])          # /Workspace/Users/<usuário>
dbutils.widgets.text("inicio", "2026-10-01", "Início (AAAA-MM-DD)")
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
# MAGIC ## 2. Painel por vencimento e resumo por dia

# COMMAND ----------

def load_meta(d):
    return json.loads((RAW_DIR / d.isoformat() / "meta.json").read_text())


def adapter_for(d, meta):
    folder = RAW_DIR / d.isoformat()
    return SPXAdapter(lambda root, _d: pd.read_parquet(folder / f"{root}.parquet"),
                      spot_source=lambda _d: meta["spot"])


def pair_quotes(group):
    """Uma linha por strike com bid/ask de call e put (cotações já limpas)."""
    q = group.loc[(group["bid"] > 0) & (group["ask"] >= group["bid"])]
    side = lambda t: q.loc[q["option_type"].eq(t)].groupby("strike")[["bid", "ask"]].mean()
    return side(CALL).join(side(PUT), lsuffix="_c", rsuffix="_p", how="inner").reset_index()


def rate_variants(group, T, forward):
    """Taxa pela regressão ampla e limites do box spread (empréstimo no ask, tomada no bid)."""
    p = pair_quotes(group)
    p = p.loc[np.abs(np.log(p["strike"] / forward)) <= WIDE_BAND]
    out = {"pares amplos": len(p)}
    if len(p) >= 3:
        y = (p["bid_c"] + p["ask_c"]) / 2 - (p["bid_p"] + p["ask_p"]) / 2
        slope, _ = np.polyfit(p["strike"], y, 1)
        out["r ampla"] = -np.log(-slope) / T if -slope > 0 else np.nan
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


def process_day(d):
    meta = load_meta(d)
    adp = adapter_for(d, meta)
    snap = adp.load_snapshot(d)
    quotes = clean_quotes(adp.load_chain(d), adp.cleaning, CleaningReport())
    sofr = meta.get("sofr")
    rows = []
    for p in snap.forwards:
        g = quotes.loc[quotes["expiration"].eq(p.expiration) & np.isclose(quotes["T"], p.T)]
        row = {"data": d, "vencimento": p.expiration, "T": p.T, "forward": p.forward,
               "q": p.rate - np.log(p.forward / snap.spot) / p.T, "r atual": p.rate,
               "resíduo": p.residual_std, "origem": "regressão" if p.T >= MIN_T_RATE else "emprestado"}
        row.update(rate_variants(g, p.T, p.forward))
        rows.append(row)
    panel = pd.DataFrame(rows)
    panel["sofr"] = sofr
    opts = snap.options
    atm = (opts.assign(dist=opts["log_moneyness"].abs()).sort_values("dist")
           .groupby("expiration").first()[["T", "implied_vol"]].sort_values("T"))
    near = lambda f, t, c: float(f.iloc[(f["T"] - t).abs().argmin()][c])
    day = {"data": d, "spot": snap.spot, "vencimentos": len(snap.forwards), "pontos": len(opts),
           "avisos": len(snap.report.warnings), "sofr": sofr,
           "vol ATM 1m": near(atm, 1 / 12, "implied_vol"), "vol ATM 12m": near(atm, 1.0, "implied_vol"),
           "q 1a": near(panel, 1.0, "q"), "r 1a": near(panel, 1.0, "r atual")}
    return panel, day


panels, days, process_status = [], [], {}
for d in sessions:
    if not (RAW_DIR / d.isoformat() / "meta.json").exists():
        process_status[d] = "sem dados em disco"
        continue
    try:
        panel, day = process_day(d)
        panels.append(panel)
        days.append(day)
        process_status[d] = "ok"
    except Exception as exc:
        process_status[d] = f"falhou: {type(exc).__name__}: {str(exc)[:150]}"
        print(d, process_status[d])

panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
daily = pd.DataFrame(days)
METHODS = ["r atual", "r ampla", "r box empresta", "r box mid", "r box toma"]
if len(panel):
    for m in METHODS:
        panel[f"{m} - SOFR (bps)"] = (panel[m] - panel["sofr"]) * 1e4
    panel.to_parquet(DATA_DIR / "painel_vencimentos.parquet", index=False)
    daily.to_parquet(DATA_DIR / "resumo_diario.parquet", index=False)
print(f"Dias processados: {len(daily)} de {len(sessions)} | linhas no painel: {len(panel)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gráfico: r − SOFR ao longo dos dias (mediana dos vencimentos com T >= 3 meses)

# COMMAND ----------

BG, GRID, MUTED = "#0B0B0B", "#2A2A2A", "#8C8C8C"
WARM = ["#FFF3B0", "#FFC233", "#FF8A1F", "#F0432F", "#D6246E"]
AXIS = dict(gridcolor=GRID, zerolinecolor=MUTED, linecolor=MUTED, showline=True,
            tickfont=dict(color="white"), title_font=dict(color="white"))

long_ = panel.loc[panel["T"] >= MIN_T_RATE] if len(panel) else panel
if len(long_):
    med = long_.groupby("data")[[f"{m} - SOFR (bps)" for m in METHODS]].median()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=med.index, y=med["r box toma - SOFR (bps)"], mode="lines",
                             line=dict(color=MUTED, width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=med.index, y=med["r box empresta - SOFR (bps)"], mode="lines",
                             fill="tonexty", fillcolor="rgba(255,138,31,0.18)", line=dict(color=MUTED, width=0),
                             name="box: faixa empresta-toma"))
    for i, (m, sym, dash) in enumerate([("r atual", "circle", "solid"), ("r ampla", "square", "dash"),
                                        ("r box mid", "diamond", "dot")]):
        fig.add_trace(go.Scatter(x=med.index, y=med[f"{m} - SOFR (bps)"], mode="lines+markers", name=m,
                                 line=dict(color=WARM[i + 1], dash=dash, width=2),
                                 marker=dict(symbol=sym, size=7, color=WARM[i + 1])))
    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="white"), height=460,
                      title=dict(text="<b>SPX: r implícito − SOFR (bps), mediana T >= 3m</b>", x=0.5),
                      legend=dict(bgcolor="#141414", bordercolor="#3A3A3A", borderwidth=1),
                      margin=dict(t=70, l=60, r=20, b=50))
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(title_text="bps", **AXIS)
    fig.show(config={"displaylogo": False, "scrollZoom": True,
                     "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "spx_r_menos_sofr"}})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Relatório único para enviar (copie a saída inteira)

# COMMAND ----------

def _df(frame, spec=None):
    spec = {c: f for c, f in (spec or {}).items() if c in frame.columns}
    fmt = {c: (lambda f: (lambda v: "-" if pd.isna(v) else f.format(v)))(f) for c, f in spec.items()}
    with pd.option_context("display.max_rows", 1000, "display.max_columns", None, "display.width", 300):
        print(frame.to_string(index=False, formatters=fmt))


def _block(title, fn):
    print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)
    try:
        fn()
    except Exception as exc:
        print(f"[não disponível] {type(exc).__name__}: {exc}")
        print("".join(traceback.format_exc().splitlines(keepends=True)[-4:]))


BPS = "{:+.0f}"
SPREADS = [f"{m} - SOFR (bps)" for m in METHODS]


def _r_status():
    st = pd.DataFrame({"data": sessions, "download": [download_status.get(d) for d in sessions],
                       "processamento": [process_status.get(d) for d in sessions]})
    print(f"pregões: {len(sessions)} | processados: {(st['processamento'] == 'ok').sum()}")
    bad = st.loc[st["processamento"] != "ok"]
    if len(bad):
        print("\ndias com problema:")
        _df(bad)


def _r_daily():
    d = daily.copy()
    m = long_.groupby("data")[SPREADS].median().reset_index()
    share = long_.assign(acima=long_["sofr"] < long_["r box empresta"]).groupby("data")["acima"].mean()
    d = d.merge(m, on="data", how="left").merge(share.rename("% venc SOFR < box empresta").reset_index(),
                                                 on="data", how="left")
    _df(d, {"spot": "{:,.2f}", "sofr": "{:.3%}", "vol ATM 1m": "{:.2%}", "vol ATM 12m": "{:.2%}",
            "q 1a": "{:.3%}", "r 1a": "{:.3%}", "% venc SOFR < box empresta": "{:.0%}",
            **{c: BPS for c in SPREADS}})


def _r_buckets():
    b = long_.assign(faixa=pd.cut(long_["T"], [0.25, 0.5, 1.0, 1.5, 2.1],
                                  labels=["3-6m", "6-12m", "12-18m", "18-24m"], right=False))
    rows = []
    for faixa, g in b.groupby("faixa", observed=True):
        row = {"faixa": faixa, "obs": len(g)}
        for c in SPREADS:
            row[f"{c.split(' - ')[0]} p25/med/p75"] = (
                f"{g[c].quantile(.25):+.0f} / {g[c].median():+.0f} / {g[c].quantile(.75):+.0f}")
        row["% SOFR < box empresta"] = (g["sofr"] < g["r box empresta"]).mean()
        rows.append(row)
    _df(pd.DataFrame(rows), {"% SOFR < box empresta": "{:.0%}"})


def _r_box_cost():
    """Quanto o spread bid/ask do box vale em bps: mostra se a faixa empresta-toma é larga ou estreita."""
    w = (long_["r box toma"] - long_["r box empresta"]) * 1e4
    print(f"largura da faixa box (toma - empresta), bps: p25 {w.quantile(.25):.0f} | mediana {w.median():.0f} "
          f"| p75 {w.quantile(.75):.0f}")
    print(f"largura do box (pontos de índice), mediana: {long_['box largura'].median():,.0f} | "
          f"pares na faixa ampla, mediana: {long_['pares amplos'].median():.0f}")


def _r_last_day():
    last = panel.loc[panel["data"].eq(panel["data"].max())]
    _df(last[["vencimento", "T", "origem", "resíduo", "pares amplos", "sofr"] + SPREADS],
        {"T": "{:.3f}", "resíduo": "{:.3f}", "sofr": "{:.3%}", "pares amplos": "{:.0f}",
         **{c: BPS for c in SPREADS}})


print(f"RELATÓRIO SPX HISTÓRICO | {start} a {end} | gerado em "
      f"{pd.Timestamp.now(tz='America/Sao_Paulo'):%Y-%m-%d %H:%M} (Brasília)")
print(f"métodos: atual = 10 pares ATM | ampla = pares em |ln(K/F)| <= {WIDE_BAND:.0%} | "
      f"box = menor x maior strike dessa faixa, nas pontas")
for _t, _f in [("1. Status por dia", _r_status),
               ("2. Resumo diário (spreads = mediana dos vencimentos com T >= 3m)", _r_daily),
               ("3. r - SOFR por faixa de prazo, todos os dias (bps)", _r_buckets),
               ("4. Custo do box", _r_box_cost),
               ("5. Último dia, por vencimento", _r_last_day)]:
    _block(_t, _f)
print("\nFIM DO RELATÓRIO")
