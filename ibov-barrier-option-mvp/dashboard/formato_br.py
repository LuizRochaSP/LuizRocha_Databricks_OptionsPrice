"""Números do dashboard no padrão brasileiro: ponto no milhar, vírgula no decimal.

O app formata números em dezenas de lugares com f-strings ("{:,.2f}", "{:.2%}"). Em vez de trocar
cada uma, ``instalar`` envolve as funções do Streamlit que exibem texto, tabela e gráfico e troca os
separadores na saída:

- texto (metric, caption, markdown sem HTML, info/success/warning/error): ``br_texto``;
- st.dataframe com DataFrame: Styler com decimal "," e milhar "."; com Styler, segue o formato
  que a chamada já definiu (``sem_vazios`` e os ``style.format`` usam decimal/thousands);
- st.plotly_chart: ``separators=",."`` (eixos e hover) e ``br_texto`` em anotações, títulos,
  nomes das séries e textos das barras.
"""
from __future__ import annotations

import re

import pandas as pd

_SEP = re.compile(r"(?<=\d)[,.](?=\d)")


def br_texto(texto):
    """"166,376.43" -> "166.376,43"; "21.15%" -> "21,15%". Só troca vírgula/ponto entre dígitos."""
    if not isinstance(texto, str):
        return texto
    return _SEP.sub(lambda m: "." if m.group(0) == "," else ",", texto)


def br(valor, casas: int = 2) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    return br_texto(f"{valor:,.{casas}f}")


def estilo_br(df: pd.DataFrame, casas: int = 4):
    """Styler padrão para DataFrame cru: floats com milhar e ``casas`` decimais, no padrão brasileiro."""
    fmt = {c: f"{{:,.{casas}f}}" for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    return df.style.format(fmt, decimal=",", thousands=".", na_rep="—")


def figura_br(fig):
    fig.update_layout(separators=",.")
    if fig.layout.title and fig.layout.title.text:
        fig.layout.title.text = br_texto(fig.layout.title.text)
    for a in fig.layout.annotations or ():
        a.text = br_texto(a.text)
    for tr in fig.data:
        if isinstance(getattr(tr, "name", None), str):
            tr.name = br_texto(tr.name)
        txt = getattr(tr, "text", None)
        if isinstance(txt, str):
            tr.text = br_texto(txt)
        elif txt is not None and len(txt) and all(isinstance(t, str) for t in txt):
            tr.text = [br_texto(t) for t in txt]
    return fig


def instalar(st) -> None:
    """Envolve as funções de exibição do Streamlit (uma vez por processo)."""
    from streamlit.delta_generator import DeltaGenerator as DG

    if getattr(DG, "_formato_br", False):
        return

    def texto(nome):
        orig = getattr(DG, nome)

        def f(self, body, *a, **k):
            if not k.get("unsafe_allow_html"):  # CSS/HTML fica como está ("0.85em")
                body = br_texto(body)
            return orig(self, body, *a, **k)
        return f

    orig_metric, orig_df, orig_plot = DG.metric, DG.dataframe, DG.plotly_chart

    def metric(self, label, value, delta=None, *a, **k):
        return orig_metric(self, label, br_texto(value), br_texto(delta), *a, **k)

    def dataframe(self, data=None, *a, **k):
        if isinstance(data, pd.DataFrame):
            data = estilo_br(data)
        return orig_df(self, data, *a, **k)

    def plotly_chart(self, fig, *a, **k):
        return orig_plot(self, figura_br(fig), *a, **k)

    novos = {"metric": metric, "dataframe": dataframe, "plotly_chart": plotly_chart}
    for nome in ("caption", "markdown", "info", "success", "warning", "error"):
        novos[nome] = texto(nome)
    for nome, f in novos.items():
        setattr(DG, nome, f)
        # st.metric etc. são métodos já ligados ao container principal
        setattr(st, nome, f.__get__(st._main, DG))
    DG._formato_br = True
