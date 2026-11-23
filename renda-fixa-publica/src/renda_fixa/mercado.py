"""Monta o "mercado do dia" a partir do arquivo ANBIMA: VNAs, curvas pré e real, NSS e tabelas.

Também compara a curva pré dos títulos com a curva de ajustes de DI1 da B3 (``comparar_di1``).

Uso típico:
    df = anbima.baixar_mercado_secundario(date(2026, 11, 19), cache_dir="rf_dados")
    m = montar_mercado(df)
    tabela_titulos(m)   # validação título a título + analíticos
    tabela_curvas(m)    # pré, real, inflação implícita por prazo
    comparar_di1(m, b3.baixar_ajustes_b3(m.ref))   # DI1 da B3 x pré dos títulos
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .curvas.bootstrap import bootstrap_titulos, inflacao_implicita
from .calendario import dias_uteis
from .curvas.curva import CurvaPlana, CurvaZero
from .curvas.nss import CurvaNSS, ajustar_nss_precos
from .dados.anbima import VNAImplicito, vna_implicito
from .derivativos.futuros import DI1, curva_di1
from .titulos import criar_titulo

SUPORTADOS = ("LTN", "NTN-F", "NTN-B", "NTN-C", "LFT")
VERTICES_PADRAO = (21, 63, 126, 252, 378, 504, 756, 1008, 1260, 1512, 2016, 2520, 3024, 3780, 5040, 6300, 7560)


@dataclass
class MercadoDia:
    ref: date
    df: pd.DataFrame
    vna_lft: VNAImplicito | None
    vna_ntnb: VNAImplicito | None
    curva_pre: CurvaZero
    curva_real: CurvaZero | None
    nss_pre: CurvaNSS | None
    nss_real: CurvaNSS | None
    vna_ntnc: VNAImplicito | None = None

    def vna(self, titulo: str) -> float | None:
        if titulo == "LFT" and self.vna_lft:
            return self.vna_lft.vna
        if titulo.startswith("NTN-B") and self.vna_ntnb:
            return self.vna_ntnb.vna
        if titulo == "NTN-C" and self.vna_ntnc:
            return self.vna_ntnc.vna
        return None

    @property
    def curva_igpm(self) -> CurvaPlana | None:
        """Só há uma NTN-C em circulação: a "curva" de IGP-M é plana na taxa dela (usada no risco)."""
        sub = self.df[(self.df["titulo"] == "NTN-C") & self.df["taxa_indicativa"].notna()]
        if sub.empty:
            return None
        return CurvaPlana(float(sub["taxa_indicativa"].iloc[-1]), nome="IGPM")

    @property
    def curvas(self) -> dict:
        out = {"PRE": self.curva_pre}
        if self.curva_real is not None:
            out["IPCA"] = self.curva_real
        igpm = self.curva_igpm
        if igpm is not None:
            out["IGPM"] = igpm
        return out


def _itens(df: pd.DataFrame, nomes: tuple[str, ...]):
    sub = df[df["titulo"].isin(nomes) & df["pu"].notna()]
    return [criar_titulo(r.titulo, r.vencimento) for r in sub.itertuples()], sub["pu"].tolist()


def montar_mercado(df: pd.DataFrame, ajustar_nss: bool = True) -> MercadoDia:
    df = df[df["titulo"].isin(SUPORTADOS)].copy()
    ref = df["data_referencia"].iloc[0]
    vna_lft = vna_implicito(df, "LFT") if (df["titulo"] == "LFT").any() else None
    vna_ntnb = vna_implicito(df, "NTN-B") if (df["titulo"] == "NTN-B").any() else None
    vna_ntnc = vna_implicito(df, "NTN-C") if (df["titulo"] == "NTN-C").any() else None

    tit_pre, pu_pre = _itens(df, ("LTN", "NTN-F"))
    curva_pre = bootstrap_titulos(tit_pre, pu_pre, ref, nome="PRE")
    curva_real = nss_pre = nss_real = None
    tit_real, pu_real = _itens(df, ("NTN-B",))
    if tit_real:
        vnas = [vna_ntnb.vna] * len(tit_real)
        curva_real = bootstrap_titulos(tit_real, pu_real, ref, vnas, nome="IPCA")
    if ajustar_nss:
        nss_pre = ajustar_nss_precos(tit_pre, pu_pre, ref, nome="PRE NSS")
        if tit_real:
            nss_real = ajustar_nss_precos(tit_real, pu_real, ref, [vna_ntnb.vna] * len(tit_real), nome="IPCA NSS")
    return MercadoDia(ref, df, vna_lft, vna_ntnb, curva_pre, curva_real, nss_pre, nss_real, vna_ntnc)


def tabela_titulos(m: MercadoDia) -> pd.DataFrame:
    """Cada título: PU do motor vs ANBIMA, taxa recuperada e analíticos."""
    linhas = []
    for r in m.df.itertuples():
        t = criar_titulo(r.titulo, r.vencimento)
        vna = m.vna(r.titulo)
        pu_motor = t.pu(r.taxa_indicativa, m.ref, vna)
        a = t.analiticos(r.taxa_indicativa, m.ref, vna)
        linha = {
            "titulo": r.titulo,
            "vencimento": r.vencimento,
            "du": a.du,
            "taxa_anbima": r.taxa_indicativa,
            "pu_anbima": r.pu,
            "pu_motor": pu_motor,
            "erro_pu": round(pu_motor - r.pu, 6),
            "taxa_motor": t.taxa(r.pu, m.ref, vna),
            "duration_mac": a.duration_macaulay,
            "duration_mod": a.duration_modificada,
            "dv01": a.dv01,
            "convexidade": a.convexidade,
        }
        nss = m.nss_pre if t.indexador == "PRE" else m.nss_real if t.indexador == "IPCA" else None
        if nss is not None:
            pc = t.pu_curva(nss, m.ref, vna)
            linha["pu_nss"] = pc
            linha["residuo_nss_bps"] = (pc - r.pu) / a.dv01 if a.dv01 else np.nan
        linhas.append(linha)
    return pd.DataFrame(linhas)


def curvas_por_taxas(df_dia: pd.DataFrame) -> tuple[CurvaZero | None, CurvaZero | None]:
    """Curvas pré e real de um dia só com as taxas (sem PU nem VNA).

    O bootstrap é feito em cotação (VNA = 100), então serve para qualquer fonte com
    ``titulo``, ``vencimento`` e ``taxa_indicativa`` (ANBIMA ou Tesouro Direto).
    """
    ref = df_dia["data_referencia"].iloc[0]
    curvas = []
    for nomes, rotulo in ((("LTN", "NTN-F"), "PRE"), (("NTN-B", "NTN-B Principal"), "IPCA")):
        sub = df_dia[df_dia["titulo"].isin(nomes) & df_dia["taxa_indicativa"].notna()]
        tit, pus = [], []
        for r in sub.itertuples():
            t = criar_titulo(r.titulo, r.vencimento)
            if not t.fluxos(ref):
                continue
            vna = 100.0 if t.usa_vna else None
            tit.append(t)
            pus.append(t.pu_bruto(float(r.taxa_indicativa), ref, vna))
        if not tit:
            curvas.append(None)
            continue
        vnas = [100.0 if t.usa_vna else None for t in tit]
        curvas.append(bootstrap_titulos(tit, pus, ref, vnas, nome=rotulo))
    return curvas[0], curvas[1]


def historico_curvas(dados: pd.DataFrame | list[pd.DataFrame], vertices=VERTICES_PADRAO) -> dict[str, pd.DataFrame]:
    """Taxas zero (bootstrap, sem NSS) por dia nos vértices fixos: {"PRE": df, "IPCA": df}.

    ``dados``: um DataFrame com vários dias (coluna ``data_referencia``) ou uma lista de
    DataFrames diários, da ANBIMA ou do Tesouro Direto. Cada saída tem uma linha por data e
    uma coluna por vértice (DU). Serve para VaR histórico (variações em bps) e PCA da curva.
    """
    if isinstance(dados, list):
        dados = pd.concat(dados, ignore_index=True)
    pre, real = {}, {}
    for ref, df_dia in dados.groupby("data_referencia"):
        try:
            c_pre, c_real = curvas_por_taxas(df_dia)
        except Exception:
            continue
        if c_pre is not None:
            pre[ref] = c_pre.taxas(vertices)
        if c_real is not None:
            real[ref] = c_real.taxas(vertices)
    cols = [int(v) for v in vertices]
    return {
        "PRE": pd.DataFrame.from_dict(pre, orient="index", columns=cols).sort_index(),
        "IPCA": pd.DataFrame.from_dict(real, orient="index", columns=cols).sort_index(),
    }


def variacoes_diarias_bps(historico: pd.DataFrame) -> pd.DataFrame:
    """Variações (bps) só entre dias úteis consecutivos; lacunas na série não viram choque de 1 dia."""
    from .calendario import dias_uteis

    h = historico.sort_index()
    datas = list(h.index)
    manter = [dias_uteis(a, b) == 1 for a, b in zip(datas[:-1], datas[1:])]
    d = h.diff().iloc[1:] * 100.0
    return d[manter]


def tabela_curvas(m: MercadoDia, vertices=VERTICES_PADRAO) -> pd.DataFrame:
    dus = np.array(vertices, dtype=float)
    out = pd.DataFrame({"du": dus.astype(int), "anos": dus / 252.0})
    out["pre"] = m.curva_pre.taxas(dus)
    if m.nss_pre is not None:
        out["pre_nss"] = m.nss_pre.taxas(dus)
    if m.curva_real is not None:
        out["real"] = m.curva_real.taxas(dus)
        out["inflacao_implicita"] = inflacao_implicita(m.curva_pre, m.curva_real, dus)
        if m.nss_real is not None:
            out["real_nss"] = m.nss_real.taxas(dus)
    # termos de 1 ano
    out["termo_pre_1a"] = [m.curva_pre.taxa_termo(d, d + 252) for d in dus]
    return out


def comparar_di1(m: MercadoDia, ajustes: pd.DataFrame) -> pd.DataFrame:
    """Curva de ajustes de DI1 da B3 contra a curva pré dos títulos, contrato a contrato.

    ``ajustes``: saída de ``dados.b3.baixar_ajustes_b3`` do mesmo dia de ``m``.
    Colunas: taxa de ajuste do DI1, taxa zero dos títulos (bootstrap e NSS) no mesmo prazo,
    spread DI1 - títulos (bps), PU de ajuste da B3 x PU do motor e a taxa a termo de cada curva
    entre o contrato anterior e o atual.
    """
    di = ajustes[ajustes["ativo"] == "DI1"].sort_values("vencimento")
    datas = set(di["data_referencia"])
    if datas != {m.ref}:
        raise ValueError(f"Ajustes da B3 de {sorted(datas)} e mercado de {m.ref}: datas diferentes.")
    linhas, du_ant, di_ant, pre_ant = [], None, None, None
    for r in di.itertuples():
        du = dias_uteis(m.ref, r.vencimento)
        if du <= 0:
            continue
        tx = float(r.taxa_ajuste)
        pre = m.curva_pre.taxa(du)
        linha = {
            "contrato": r.contrato,
            "vencimento": r.vencimento,
            "du": du,
            "taxa_di1": tx,
            "taxa_pre_titulos": pre,
            "spread_bps": (tx - pre) * 100.0,
            "pu_ajuste_b3": float(r.pu_ajuste),
            "pu_motor": DI1(r.vencimento).pu(tx, m.ref),
            "contratos": r.contratos,
        }
        if m.nss_pre is not None:
            linha["taxa_pre_nss"] = m.nss_pre.taxa(du)
            linha["spread_nss_bps"] = (tx - linha["taxa_pre_nss"]) * 100.0
        if du_ant is not None:
            linha["termo_di1"] = _termo(di_ant, du_ant, tx, du)
            linha["termo_titulos"] = _termo(pre_ant, du_ant, pre, du)
        linhas.append(linha)
        du_ant, di_ant, pre_ant = du, tx, pre
    out = pd.DataFrame(linhas)
    out["erro_pu"] = (out["pu_motor"] - out["pu_ajuste_b3"]).round(2)
    # além do último título prefixado a curva dos títulos é extrapolada (taxa constante)
    out["extrapolado"] = out["du"] > max(m.curva_pre.dus)
    return out


def _termo(t1: float, du1: int, t2: float, du2: int) -> float:
    f1 = (1 + t1 / 100.0) ** (du1 / 252.0)
    f2 = (1 + t2 / 100.0) ** (du2 / 252.0)
    return ((f2 / f1) ** (252.0 / (du2 - du1)) - 1.0) * 100.0


def curva_di1_b3(ajustes: pd.DataFrame, ref: date) -> CurvaZero:
    """Curva zero flat-forward com as taxas de ajuste de DI1 da B3 do dia."""
    di = ajustes[(ajustes["ativo"] == "DI1") & (ajustes["data_referencia"] == ref)]
    if di.empty:
        raise ValueError(f"Sem ajustes de DI1 em {ref}.")
    return curva_di1(list(zip(di["vencimento"], di["taxa_ajuste"].astype(float))), ref)
