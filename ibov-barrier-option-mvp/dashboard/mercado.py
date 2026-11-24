"""Mercado do dia para o Structuring Desk: spot, r(T), q(T) e vol no strike de cada ativo.

As regras são as dos notebooks que gravam os arquivos, para o Desk reproduzir o preço do pipeline:

- **Vol no strike:** a do notebook 01 — interpolação linear da vol implícita no plano ln(K/F) x T
  (triangulação de Delaunay, igual ao ``griddata(..., method="linear")``) e vizinho mais próximo quando
  o ponto cai fora do casco convexo dos dados.
- **IBOV (notebook 01):** r(T) contínua da curva DI1 em flat-forward (r·DU linear entre vértices,
  extrapolada pelo segmento da ponta, como o ``FlatForwardInterpolator``) e q(T) linear em DU.
- **SPX (notebook 07):** r(T) e q(T) da paridade put-call, lineares em T entre os vencimentos. Com o q
  de cada vencimento o forward do Desk bate com o forward da paridade, que é o mesmo usado no
  ln(K/F) da superfície.

Prazos em anos: DU/252 no IBOV e ACT/365 no SPX, como nos notebooks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

DU_ANO = 252
PRAZO_PADRAO = 0.5  # prazo padrão do Desk (6 meses): referência dos valores iniciais


def _linear_com_ponta(x: float, xs, ys) -> float:
    """Interpolação linear; fora da grade, segue a reta do primeiro ou do último segmento."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) == 1:
        return float(ys[0])
    i = 0 if x <= xs[0] else len(xs) - 2 if x >= xs[-1] else int(np.searchsorted(xs, x) - 1)
    return float(ys[i] + (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) * (x - xs[i]))


@dataclass
class MercadoDia:
    """Spot, curvas e superfície de um ativo numa data. ``r``, ``q`` e ``vol`` recebem T em anos."""

    ativo: str
    data: str
    spot: float
    fonte: str
    r_tenores: np.ndarray       # eixo das curvas r/q (DU no IBOV, anos no SPX)
    r_valores: np.ndarray
    q_tenores: np.ndarray
    q_valores: np.ndarray
    pontos: np.ndarray          # (ln(K/F), T) da superfície
    vols: np.ndarray
    flat_forward: bool          # IBOV: r em flat-forward sobre DU
    _lin: object = field(default=None, init=False, repr=False)
    _viz: object = field(default=None, init=False, repr=False)

    def _eixo(self, T: float) -> float:
        return T * DU_ANO if self.flat_forward else T

    def r(self, T: float) -> float:
        x = self._eixo(T)
        if self.flat_forward:
            return _linear_com_ponta(x, self.r_tenores, self.r_valores * self.r_tenores) / x
        return float(np.interp(x, self.r_tenores, self.r_valores))

    def q(self, T: float) -> float:
        return float(np.interp(self._eixo(T), self.q_tenores, self.q_valores))

    def forward(self, T: float) -> float:
        return self.spot * math.exp((self.r(T) - self.q(T)) * T)

    def vol(self, strike: float, T: float) -> tuple[float, str]:
        """Vol implícita no strike e prazo, e a regra usada ("linear" ou "vizinho mais próximo")."""
        if self._lin is None:
            self._lin = LinearNDInterpolator(self.pontos, self.vols)
            self._viz = NearestNDInterpolator(self.pontos, self.vols)
        k = math.log(strike / self.forward(T))
        v = float(np.asarray(self._lin(k, T)).ravel()[0])
        if np.isfinite(v):
            return v, "linear"
        return float(np.asarray(self._viz(k, T)).ravel()[0]), "vizinho mais próximo"

    def vol_atm(self, T: float) -> float:
        return self.vol(self.forward(T), T)[0]

    def cobertura(self) -> str:
        """Faixa de prazos da superfície, para avisar quando o Desk extrapola."""
        t = self.pontos[:, 1]
        return f"{t.min():.2f} a {t.max():.2f} ano"


def _pontos(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sd = frame.dropna(subset=["log_moneyness", "T", "implied_vol"]).drop_duplicates(["T", "log_moneyness"])
    return sd[["log_moneyness", "T"]].to_numpy(dtype=float), sd["implied_vol"].to_numpy(dtype=float)


def mercado_ibov(metadata: dict, curve: pd.DataFrame, superficie: pd.DataFrame) -> MercadoDia | None:
    """Do arquivo do notebook 01: ``metadata``, ``curve`` (tenor_bd, r_cont, q_cont) e ``surface_data``."""
    if not metadata or curve is None or superficie is None or len(curve) < 2:
        return None
    pts, vols = _pontos(superficie)
    if len(pts) < 3:
        return None
    c = curve.dropna(subset=["tenor_bd", "r_cont", "q_cont"]).sort_values("tenor_bd")
    t = c["tenor_bd"].to_numpy(dtype=float)
    return MercadoDia(ativo="IBOV", data=str(metadata.get("market_date", "—")), spot=float(metadata["spot"]),
                      fonte="notebook 01 (curva DI1 e superfície do IBOV)",
                      r_tenores=t, r_valores=c["r_cont"].to_numpy(dtype=float),
                      q_tenores=t, q_valores=c["q_cont"].to_numpy(dtype=float),
                      pontos=pts, vols=vols, flat_forward=True)


def mercado_spx(bundle: dict | None) -> MercadoDia | None:
    """Do arquivo do notebook 07: ``metadata``, ``forwards`` (T, r, q) e ``options`` (pontos da superfície)."""
    if not bundle or "forwards" not in bundle or "options" not in bundle:
        return None
    pts, vols = _pontos(bundle["options"])
    f = bundle["forwards"].dropna(subset=["T", "r", "q"]).sort_values("T")
    if len(pts) < 3 or f.empty:
        return None
    t = f["T"].to_numpy(dtype=float)
    meta = bundle.get("metadata", {})
    return MercadoDia(ativo="SPX", data=str(meta.get("market_date", "—")), spot=float(meta["spot"]),
                      fonte="notebook 07 (paridade put-call e superfície do SPX)",
                      r_tenores=t, r_valores=f["r"].to_numpy(dtype=float),
                      q_tenores=t, q_valores=f["q"].to_numpy(dtype=float),
                      pontos=pts, vols=vols, flat_forward=False)


def padroes(m: MercadoDia | None, reserva: dict, T: float = PRAZO_PADRAO) -> dict:
    """Valores iniciais do Desk: spot, r, q e vol ATM no prazo padrão; sem arquivo, os de reserva."""
    if m is None:
        return dict(reserva, origem=None)
    return {"spot": m.spot, "rate": m.r(T), "q": m.q(T), "vol": m.vol_atm(T), "origem": m.data}


def referencia_quanto(bundle: dict | None) -> dict | None:
    """Câmbio, juro em BRL, vol do câmbio e correlação da última execução do notebook 09."""
    meta = (bundle or {}).get("metadata") or {}
    if not all(meta.get(k) is not None for k in ("X0", "r_brl", "vol_fx", "rho")):
        return None
    return {"X0": float(meta["X0"]), "r_brl": float(meta["r_brl"]), "vol_fx": float(meta["vol_fx"]),
            "rho": float(meta["rho"]), "data": str(meta.get("market_date", "—")),
            "janela_rho": meta.get("janela_rho")}


# ---------------------------------------------------------------- resumo do SPX (aba Resumo e histórico)

def _mais_proximo(frame: pd.DataFrame, t: float, col: str) -> float:
    return float(frame.iloc[(frame["T"] - t).abs().argmin()][col])


def resumo_spx(bundle: dict) -> dict:
    """Números do dia (a mesma regra do resumo do notebook 07, mas em número e não em texto)."""
    opts, fwd = bundle["options"], bundle["forwards"]
    atm = (opts.assign(dist=opts["log_moneyness"].abs()).sort_values("dist", kind="stable")
           .groupby("expiration", as_index=False).first().sort_values("T"))
    meta = bundle.get("metadata", {})
    return {"data": str(meta.get("market_date", "—")), "spot": float(meta.get("spot", np.nan)),
            "vol ATM 1m": _mais_proximo(atm, 1 / 12, "implied_vol"),
            "vol ATM 12m": _mais_proximo(atm, 1.0, "implied_vol"),
            "r 1a": _mais_proximo(fwd, 1.0, "r"), "q 1a": _mais_proximo(fwd, 1.0, "q"),
            "SOFR": meta.get("sofr"), "vencimentos": meta.get("vencimentos"), "pontos": meta.get("pontos")}
