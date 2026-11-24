"""Gráficos Plotly no padrão visual do projeto: fundo preto, texto branco, rampa quente,
marcador e traço distintos por série, barra com zoom e download PNG.

Usado pelos notebooks (``html_figura`` + ``displayHTML`` dentro de ``div.nodarkmode``, para o
modo escuro do Databricks não inverter as cores) e pelo dashboard (``st.plotly_chart(theme=None)``).
"""
from __future__ import annotations

from typing import Iterable

BG, GRID, MUTED = "#0B0B0B", "#2A2A2A", "#8C8C8C"
QUENTES = ["#FFF3B0", "#FFC233", "#FF8A1F", "#F0432F", "#D6246E"]
SIMBOLOS = ["circle", "square", "triangle-up", "diamond", "triangle-down"]
TRACOS = ["solid", "dash", "dashdot", "dot", "longdashdot"]
EIXO = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, showline=True, tickfont=dict(color="white"), title_font=dict(color="white"))


def config_barra(nome_arquivo: str) -> dict:
    return {
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "filename": nome_arquivo, "scale": 2},
    }


def estilo(fig, titulo: str, altura: int = 440, eixo_x: str = "", eixo_y: str = ""):
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="white", size=12),
        colorway=QUENTES,
        title=dict(text=f"<b>{titulo}</b>", x=0.5, xanchor="center", font=dict(color="white", size=16)),
        legend=dict(bgcolor="#141414", bordercolor="#3A3A3A", borderwidth=1, font=dict(color="white")),
        hoverlabel=dict(bgcolor="#141414", bordercolor="#3A3A3A", font=dict(color="white")),
        height=altura,
        margin=dict(t=80, l=60, r=20, b=50),
        dragmode="zoom",
        separators=",.",  # vírgula decimal e ponto no milhar (padrão brasileiro)
    )
    fig.update_xaxes(title_text=eixo_x, **EIXO)
    fig.update_yaxes(title_text=eixo_y, **EIXO)
    fig.update_annotations(font=dict(color="white"))
    return fig


def linha(i: int, nome: str, x, y, modo: str = "lines+markers", hover: str | None = None):
    import plotly.graph_objects as go

    cor = QUENTES[i % len(QUENTES)] if i >= 0 else MUTED
    return go.Scatter(
        x=list(x),
        y=list(y),
        name=nome,
        mode=modo,
        line=dict(color=cor, width=2, dash=TRACOS[i % len(TRACOS)] if i >= 0 else "dot"),
        marker=dict(color=cor, size=7, symbol=SIMBOLOS[i % len(SIMBOLOS)] if i >= 0 else "x"),
        hovertemplate=hover,
    )


def barras(nome: str, x, y, i: int = 1):
    import plotly.graph_objects as go

    cores = [QUENTES[3] if v < 0 else QUENTES[1] for v in y]
    return go.Bar(x=list(x), y=list(y), name=nome, marker=dict(color=cores, line=dict(color=BG, width=1)))


def figura(series: Iterable, titulo: str, eixo_x: str = "", eixo_y: str = "", altura: int = 440):
    import plotly.graph_objects as go

    fig = go.Figure()
    for s in series:
        fig.add_trace(s)
    return estilo(fig, titulo, altura, eixo_x, eixo_y)


def html_figura(fig, nome_arquivo: str) -> str:
    """HTML para ``displayHTML`` no Databricks, protegido do modo escuro."""
    corpo = fig.to_html(full_html=False, include_plotlyjs="cdn", config=config_barra(nome_arquivo))
    return f'<div class="nodarkmode" style="background:{BG}">{corpo}</div>'
