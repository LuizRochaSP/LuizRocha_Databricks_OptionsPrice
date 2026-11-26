"""Mercado de uma rodada aprovada do pipeline (etapa A5 do plano de COE).

**Rodada aprovada.** `ultima_rodada_aprovada(pasta)` lê a evidência `VALIDATION_<prefixo>_*.json` mais recente e
aplica as exigências do notebook 03:
- schema 1, FAIL = 0, aprovada para risco, decisão APROVADO ou APROVADO COM RESSALVAS;
- o RESULTS da validação existe e tem o SHA-256 registrado (um RESULTS alterado depois da validação é recusado);
- RESULTS no schema 2 do notebook 01, com a mesma data de mercado da validação.

Uma evidência não aprovada é pulada (e listada em `ignoradas`). Uma falha de integridade (SHA-256, arquivo ausente,
datas divergentes) para tudo: o estruturador precisa saber, e a rodada anterior não entra no lugar sem aviso.

**Estado de mercado do IBOV** (`estado_ibov`), com as regras do notebook 01:
- **Curva DI pré.** Os vértices de `curve` (`tenor_bd`, `r_cont`) voltam à taxa efetiva (e^r − 1) e viram a
  mesma `DiscountCurve` flat-forward do 01. Nos vértices e entre eles, r(T) é a do 01; depois do último vértice,
  segue o forward do último segmento.
- **Dividendos.** q(T) linear em DU sobre os vértices de `curve` (`q_cont`), a regra do Desk. O 01 interpola q
  nos vencimentos do IND, que o RESULTS não guarda; por isso o q reconstruído pode diferir um pouco do q do 01
  entre os vértices. `conferir_referencia` mostra a diferença no contrato de referência do 01.
- **Vol.** `SuperficieVol`: linear no plano (ln(K/F), T) dos pontos de `surface_data` (a triangulação do
  `griddata(..., method="linear")`), com o vizinho mais próximo fora do casco convexo. F usa o spot, o r e o q da
  rodada e fica fixo: a superfície é "sticky strike", e o choque de spot das gregas não a move.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from ibov_barrier.produtos.mercado import DU_ANO, CurvaDI, EstadoMercado, MercadoAtivo

DECISOES_APROVADAS = ("APROVADO", "APROVADO COM RESSALVAS")
SCHEMA_VALIDACAO = 1
SCHEMA_RESULTS_IBOV = 2


# ------------------------------------------------------------------------------------------ rodada

@dataclass(frozen=True)
class Rodada:
    """Uma rodada aprovada: a evidência de validação, o RESULTS conferido e os objetos dele."""

    validacao_path: Path
    results_path: Path
    sha256: str
    validacao: dict
    objetos: dict
    ignoradas: tuple[str, ...] = ()

    @property
    def metadata(self) -> dict:
        return self.objetos["metadata"]

    @property
    def data_mercado(self) -> date:
        return date.fromisoformat(str(self.metadata["market_date"])[:10])

    @property
    def decisao(self) -> str:
        return str(self.validacao["decision"])

    @property
    def warn_count(self) -> int:
        return int(self.validacao.get("warn_count", 0))

    @property
    def ressalvas(self) -> list:
        return list(self.validacao.get("warnings", []))

    def proveniencia(self) -> dict:
        """Para o relatório e o JSON do notebook 12."""
        return {"validation_file": self.validacao_path.name, "results_file": self.results_path.name,
                "results_sha256": self.sha256, "market_date": self.data_mercado.isoformat(),
                "validation_decision": self.decisao, "validation_warn_count": self.warn_count}


def _carimbo(nome: str, padrao: re.Pattern) -> datetime | None:
    m = padrao.match(nome)
    return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S") if m else None


def _motivo_reprovacao(validacao: dict) -> str | None:
    if validacao.get("schema_version") != SCHEMA_VALIDACAO:
        return f"schema_version {validacao.get('schema_version')} (esperado {SCHEMA_VALIDACAO})"
    if validacao.get("fail_count") != 0 or not validacao.get("approved_for_risk"):
        return f"não aprovada para risco (FAIL={validacao.get('fail_count')})"
    if validacao.get("decision") not in DECISOES_APROVADAS:
        return f"decisão {validacao.get('decision')!r}"
    return None


def ultima_rodada_aprovada(pasta: Path | str, prefixo: str = "superficieVol", *,
                           arquivo_validacao: str | None = None) -> Rodada:
    """A rodada aprovada mais recente em `pasta` (ou a de `arquivo_validacao`, para backfill)."""
    from ibov_barrier.results_bundle import read_bundle

    pasta = Path(pasta)
    padrao = re.compile(rf"^VALIDATION_{re.escape(prefixo)}_(\d{{8}})_(\d{{6}})\.json$")
    if arquivo_validacao:
        candidatos = [pasta / arquivo_validacao]
        if not candidatos[0].is_file():
            raise RuntimeError(f"Validação não encontrada: {candidatos[0]}")
    else:
        datados = [(c, p) for p in pasta.glob(f"VALIDATION_{prefixo}_*.json") if (c := _carimbo(p.name, padrao))]
        candidatos = [p for _, p in sorted(datados, reverse=True)]
        if not candidatos:
            raise RuntimeError(f"Nenhuma validação VALIDATION_{prefixo}_*.json em {pasta}. "
                               "Rode o pipeline (01 e 02) antes.")

    ignoradas = []
    for caminho in candidatos:
        validacao = json.loads(caminho.read_text(encoding="utf-8"))
        motivo = _motivo_reprovacao(validacao)
        if motivo:
            if arquivo_validacao:
                raise RuntimeError(f"{caminho.name} não está aprovada: {motivo}")
            ignoradas.append(f"{caminho.name}: {motivo}")
            continue
        results = pasta / str(validacao.get("results_file", ""))
        if not results.is_file():
            raise RuntimeError(f"RESULTS aprovado não encontrado: {results}")
        sha = hashlib.sha256(results.read_bytes()).hexdigest()
        if sha != validacao.get("results_sha256"):
            raise RuntimeError(f"{results.name} foi alterado depois da validação (SHA-256 divergente).")
        objetos = read_bundle(results)
        meta = objetos.get("metadata", {})
        if prefixo == "superficieVol" and meta.get("schema_version") != SCHEMA_RESULTS_IBOV:
            raise RuntimeError(f"{results.name}: RESULTS do IBOV com schema_version {meta.get('schema_version')} "
                               f"(esperado {SCHEMA_RESULTS_IBOV}).")
        if validacao.get("market_date") != meta.get("market_date"):
            raise RuntimeError("Data de mercado divergente entre RESULTS e VALIDATION.")
        return Rodada(caminho, results, sha, validacao, objetos, tuple(ignoradas))
    raise RuntimeError("Nenhuma validação aprovada: " + "; ".join(ignoradas))


# -------------------------------------------------------------------------------------- superfície

@dataclass(frozen=True, eq=False)
class SuperficieVol:
    """Vol implícita do notebook 01 como função `vol(strike, prazo)` para o `MercadoAtivo`.

    `pontos`: (ln(K/F), T) de cada observação; `vols`: a vol implícita. `curva` e `dividendos` dão o r e o q do
    forward F(T) = spot · e^((r − q) T), fixos na rodada.
    """

    spot: float
    curva: object
    dividendos: object
    pontos: np.ndarray
    vols: np.ndarray
    _linear: object = field(default=None, init=False, repr=False)
    _vizinho: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        pontos = np.asarray(self.pontos, dtype=float)
        vols = np.asarray(self.vols, dtype=float)
        if pontos.ndim != 2 or pontos.shape[1] != 2 or len(pontos) != len(vols):
            raise ValueError("pontos deve ser (n, 2) com (ln(K/F), T), e vols ter n valores")
        if len(pontos) < 3 or len(np.unique(pontos[:, 1])) < 2:
            raise ValueError("a superfície precisa de pelo menos 3 pontos em 2 vencimentos")
        if not (np.all(np.isfinite(pontos)) and np.all(np.isfinite(vols)) and np.all(vols > 0)):
            raise ValueError("pontos e vols devem ser finitos, e as vols positivas")
        object.__setattr__(self, "pontos", pontos)
        object.__setattr__(self, "vols", vols)
        object.__setattr__(self, "_linear", LinearNDInterpolator(pontos, vols))
        object.__setattr__(self, "_vizinho", NearestNDInterpolator(pontos, vols))

    def forward(self, prazo: float) -> float:
        r = float(self.curva.taxa_em(prazo))
        q = float(self.dividendos(prazo))
        return self.spot * math.exp((r - q) * prazo)

    def com_regra(self, strike: float, prazo: float) -> tuple[float, str]:
        """(vol, regra): "linear" dentro do casco convexo dos pontos, "vizinho mais próximo" fora."""
        k = math.log(float(strike) / self.forward(prazo))
        v = float(np.asarray(self._linear(k, prazo)).ravel()[0])
        if np.isfinite(v):
            return v, "linear"
        return float(np.asarray(self._vizinho(k, prazo)).ravel()[0]), "vizinho mais próximo"

    def __call__(self, strike: float, prazo: float) -> float:
        return self.com_regra(strike, prazo)[0]

    @property
    def prazo_maximo(self) -> float:
        return float(self.pontos[:, 1].max())

    @property
    def prazo_minimo(self) -> float:
        return float(self.pontos[:, 1].min())


# --------------------------------------------------------------------------------------------- estado

def _q_linear_em_du(tenores_du: np.ndarray, q: np.ndarray):
    def dividendos(prazo: float) -> float:
        return float(np.interp(float(prazo) * DU_ANO, tenores_du, q))
    return dividendos


def estado_ibov(rodada: Rodada | dict, *, spread_emissor_bp: float = 0.0, ativo: str = "IBOV") -> EstadoMercado:
    """`EstadoMercado` do IBOV a partir de uma `Rodada` (ou dos objetos de um RESULTS do notebook 01)."""
    from ibov_barrier.di_curve.curve import DiscountCurve

    objetos = rodada.objetos if isinstance(rodada, Rodada) else rodada
    for nome in ("metadata", "curve", "surface_data"):
        if nome not in objetos:
            raise ValueError(f"RESULTS sem o objeto {nome!r}")
    meta = objetos["metadata"]
    curva_df = objetos["curve"].dropna(subset=["tenor_bd", "r_cont", "q_cont"]).sort_values("tenor_bd")
    if len(curva_df) < 2:
        raise ValueError("a curva do RESULTS precisa de pelo menos 2 vértices")
    tenores = curva_df["tenor_bd"].to_numpy(dtype=float)
    curva = CurvaDI(DiscountCurve.from_di1(tenores, np.expm1(curva_df["r_cont"].to_numpy(dtype=float))))
    dividendos = _q_linear_em_du(tenores, curva_df["q_cont"].to_numpy(dtype=float))

    sup = objetos["surface_data"].dropna(subset=["log_moneyness", "T", "implied_vol"])
    sup = sup.drop_duplicates(["T", "log_moneyness"])
    spot = float(meta["spot"])
    superficie = SuperficieVol(spot=spot, curva=curva, dividendos=dividendos,
                               pontos=sup[["log_moneyness", "T"]].to_numpy(dtype=float),
                               vols=sup["implied_vol"].to_numpy(dtype=float))
    data = date.fromisoformat(str(meta["market_date"])[:10]) if meta.get("market_date") else None
    return EstadoMercado(ativos={ativo: MercadoAtivo(spot=spot, vol=superficie, dividendos=dividendos)},
                         curvas={"BRL": curva}, spread_emissor_bp=spread_emissor_bp, data=data)


def conferir_referencia(estado: EstadoMercado, metadata: dict, ativo: str = "IBOV") -> list[dict]:
    """r, q e vol do estado no contrato de referência do notebook 01 (target_strike, target_maturity), lado a
    lado com os números que o 01 gravou. A diferença do q vem da interpolação (ver o docstring do módulo)."""
    prazo = float(metadata["target_maturity"])
    strike = float(metadata["target_strike"])
    mercado = estado.ativo(ativo)
    r = float(estado.curva("BRL").taxa_em(prazo))
    q = mercado.dividendos_em(prazo)
    vol = float(mercado.vol_em(strike, prazo))
    linhas = []
    for grandeza, rodada, reconstruido in (("r(T)", metadata.get("target_r"), r), ("q(T)", metadata.get("target_q"), q),
                                           ("vol no strike", metadata.get("target_iv"), vol)):
        rodada = None if rodada is None else float(rodada)
        linhas.append({"grandeza": grandeza, "notebook_01": rodada, "reconstruido": reconstruido,
                       "diferenca": None if rodada is None else reconstruido - rodada})
    return linhas
