"""Completude das execuções (RESULTS -> VALIDATION -> RISK) e disparo do job que roda a cadeia toda.

Cada execução do notebook 01 grava ``RESULTS_superficieVol_<ts>.csv``. O 02 grava
``VALIDATION_superficieVol_<ts>.json`` só quando aprova (FAIL=0) e o 03 grava
``RISK_superficieVol_<ts>.json`` a partir dessa validação. Uma execução só está completa com os três.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_NOME = re.compile(r"^(RESULTS|VALIDATION|RISK)_superficieVol_(\d{8}_\d{6})\.(csv|json)$")
JOB_PADRAO = "opcoes-pipeline-ibov"


def status_execucoes(nomes) -> pd.DataFrame:
    """Uma linha por RESULTS (mais recente primeiro): tem validação? tem risco? o que falta?"""
    achados: dict[str, set] = {}
    for n in nomes:
        m = _NOME.match(str(n).split("/")[-1])
        if m:
            achados.setdefault(m.group(2), set()).add(m.group(1))
    linhas = []
    for ts, tipos in achados.items():
        if "RESULTS" not in tipos:
            continue
        val, risk = "VALIDATION" in tipos, "RISK" in tipos
        linhas.append({"ts": ts, "validacao": val, "risco": risk, "completo": val and risk,
                       "falta": None if val and risk else ("02" if not val else "03")})
    cols = ["ts", "validacao", "risco", "completo", "falta"]
    return pd.DataFrame(linhas, columns=cols).sort_values("ts", ascending=False, ignore_index=True)


def rotulo(ts: str, st_df: pd.DataFrame) -> str:
    linha = st_df[st_df["ts"] == ts]
    if linha.empty or linha["completo"].iloc[0]:
        return f"{ts} · completo"
    return f"{ts} · sem risco" if linha["falta"].iloc[0] == "03" else f"{ts} · não validado"


def indice_padrao(st_df: pd.DataFrame) -> int:
    """A execução completa mais recente; sem nenhuma completa, a mais recente."""
    ok = st_df.index[st_df["completo"]]
    return int(ok[0]) if len(ok) else 0


def motivo(ts: str, st_df: pd.DataFrame) -> str | None:
    """Texto do que falta na execução ``ts`` (None se está completa)."""
    linha = st_df[st_df["ts"] == ts]
    if linha.empty or linha["completo"].iloc[0]:
        return None
    if linha["falta"].iloc[0] == "02":
        return (f"A execução {ts} não tem validação: o notebook 02 não rodou para ela ou reprovou "
                "(com FAIL o 02 não grava a evidência). Sem validação o 03 não calcula gregas nem VaR.")
    return (f"A execução {ts} foi validada pelo 02, mas o notebook 03 não rodou para ela: "
            "faltam gregas, DV01 e VaR.")


# ------------------------------------------------------------------ job
_ESTADOS = {"SUCCESS": "ok", "FAILED": "falhou", "TIMEDOUT": "estourou o tempo", "INTERNAL_ERROR": "erro interno",
            "CANCELED": "cancelada", "UPSTREAM_FAILED": "não rodou", "UPSTREAM_CANCELED": "não rodou",
            "EXCLUDED": "não rodou", "RUNNING": "rodando", "PENDING": "aguardando", "BLOCKED": "aguardando",
            "QUEUED": "na fila", "WAITING_FOR_RETRY": "aguardando nova tentativa", "TERMINATING": "terminando"}
_FALHAS = {"FAILED", "TIMEDOUT", "INTERNAL_ERROR"}


def estado_pt(estado: str | None, resultado: str | None) -> str:
    chave = resultado or estado
    return _ESTADOS.get(chave, chave.lower() if chave else "aguardando")


def _unicas(tarefas):
    """Uma linha por etapa (a lista do Databricks repete a etapa a cada nova tentativa: fica a última)."""
    vistas = {}
    for k, e, r in tarefas:
        vistas[k] = (k, e, r)
    return [vistas[k] for k in sorted(vistas)]


def andamento_texto(tarefas) -> str:
    return " | ".join(f"{k}: {estado_pt(e, r)}" for k, e, r in _unicas(tarefas))


def resumo_falha(tarefas) -> str:
    """"falhou em 00_coleta; 01_pricer, 02_validacao, 03_risco não rodaram"."""
    unicas = _unicas(tarefas)
    falhou = [k for k, _, r in unicas if r in _FALHAS]
    parou = [k for k, _, r in unicas if r and r not in _FALHAS and r != "SUCCESS"]
    partes = []
    if falhou:
        partes.append("falhou em " + ", ".join(falhou))
    if parou:
        partes.append(", ".join(parou) + (" não rodou" if len(parou) == 1 else " não rodaram"))
    return "; ".join(partes)


def _valor(x) -> str | None:
    return None if x is None else str(getattr(x, "value", x))


@dataclass
class Progresso:
    run_id: int
    ativo: bool
    resultado: str | None  # SUCCESS, FAILED, ... (None enquanto roda)
    inicio_ms: int | None
    tarefas: list  # [(task_key, estado, resultado)]
    url: str | None = None


class Pipeline:
    """Job do Databricks com as tarefas 00 -> 01 -> 02 -> 03 (achado pelo nome ou pelo id)."""

    def __init__(self, w, nome: str = JOB_PADRAO, job_id: int | str | None = None):
        self.w, self.nome = w, nome
        self._job_id = int(job_id) if job_id else None

    def job_id(self) -> int | None:
        if self._job_id is None:
            job = next(iter(self.w.jobs.list(name=self.nome)), None)
            self._job_id = job.job_id if job else None
        return self._job_id

    def _progresso(self, run) -> Progresso:
        estado = run.state
        ciclo = _valor(getattr(estado, "life_cycle_state", None))
        ativo = ciclo in {"QUEUED", "PENDING", "RUNNING", "TERMINATING", "BLOCKED", "WAITING_FOR_RETRY"}
        tarefas = [(t.task_key, _valor(t.state.life_cycle_state) if t.state else None,
                    _valor(t.state.result_state) if t.state else None) for t in (run.tasks or [])]
        return Progresso(run.run_id, ativo, None if ativo else _valor(getattr(estado, "result_state", None)),
                         run.start_time, sorted(tarefas), getattr(run, "run_page_url", None))

    def ultima(self) -> Progresso | None:
        """A rodada mais recente do job (ativa ou não)."""
        jid = self.job_id()
        run = next(iter(self.w.jobs.list_runs(job_id=jid, limit=1, expand_tasks=True)), None) if jid else None
        return self._progresso(run) if run else None

    def acompanhar(self, run_id: int) -> Progresso:
        return self._progresso(self.w.jobs.get_run(run_id))

    def disparar(self) -> int:
        """Roda o job, a não ser que já haja uma rodada ativa (aí devolve a ativa)."""
        atual = self.ultima()
        if atual and atual.ativo:
            return atual.run_id
        espera = self.w.jobs.run_now(job_id=self.job_id())
        return getattr(espera, "run_id", None) or espera.response.run_id
