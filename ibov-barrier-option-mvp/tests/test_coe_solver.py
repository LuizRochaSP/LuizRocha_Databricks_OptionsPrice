"""Testes do solver de margem alvo (etapa A4 do plano).

- Ida e volta: parâmetro -> margem (decomposição do A3) -> parâmetro, em cada parâmetro livre das variantes do A3,
  com vol plana e com smile, curva DI inclinada e spread do emissor.
- Participação e proteção: a solução direta é igual à conta fechada com o valor do call spread e o fator de
  desconto do emissor.
- Monotonicidade: a margem é monotônica em cada parâmetro livre, no sentido esperado; uma estrutura não monotônica
  (straddle no strike) é recusada, e um parâmetro declarado linear que não é linear é pego.
- Fora do alcance: proteção de 100% com juro baixo e prazo curto é inviável; alvo abaixo da margem sem cap é folga.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, replace
from math import log

import numpy as np

from ibov_barrier.coe import COE, ParametroLivre, Participacao, TermsheetBase, Titulo, resolver
from ibov_barrier.produtos import CallSpread, CurvaDI, CurvaPlana, MercadoAtivo, Vanilla
from tests.test_coe_estrutura import S0, VARIANTES, _estado, _ts
from tests.test_coe_titulo import INCLINADA

CUSTOS = dict(custo_distribuicao=0.03, custo_operacional=0.005, reservas=0.005)

# Sentido esperado da margem quando o parâmetro sobe: +1 sobe, −1 cai.
SENTIDO = {
    ("alta", "participacao"): -1, ("alta", "protecao"): -1, ("alta", "cap"): -1, ("alta", "strike"): +1,
    ("baixa", "participacao"): -1, ("baixa", "protecao"): -1, ("baixa", "cap"): -1, ("baixa", "strike"): -1,
}


def _smile(strike, prazo):
    """Skew com vol plana fora de ±60% em log-moneyness, como a extrapolação do pipeline. O smile do A3 explode
    perto do strike zero (arbitragem), e o solver recusa a margem não monotônica que ele cria."""
    x = min(max(log(strike / S0), -0.6), 0.6)
    return 0.22 - 0.10 * x + 0.15 * x * x


def _mercados():
    plano = _estado(spot=133_000.0, spread_bp=100.0)
    com_smile = replace(_estado(curva=CurvaDI(INCLINADA), spread_bp=110.0),
                        ativos={"IBOV": MercadoAtivo(spot=133_000.0, vol=_smile, dividendos=0.04)})
    return {"vol plana": plano, "smile e curva inclinada": com_smile}


def _margem(ts, e, **custos) -> float:
    return COE(ts).decompor(e, **custos).margem / ts.nominal


@dataclass(frozen=True, kw_only=True)
class _Straddle(TermsheetBase):
    """Estrutura de teste: título + straddle no strike. O valor do straddle tem mínimo perto do forward."""

    strike: float = 1.0
    strike_linear: bool = False

    def opcoes(self):
        k = self.strike * self.nivel_inicial
        quantidade = self.nominal / self.nivel_inicial
        return quantidade * (Vanilla(self.ativo, k, self.vencimento, "call")
                             + Vanilla(self.ativo, k, self.vencimento, "put"))

    def payoff_opcoes(self, finais):
        return self.nominal * np.abs(np.asarray(finais[self.ativo]) / self.nivel_inicial - self.strike)

    def parametros_livres(self):
        return {"strike": ParametroLivre("strike", "strike", 0.6, 1.6, linear=self.strike_linear)}


class TestIdaEVolta(unittest.TestCase):
    def test_todos_os_parametros_livres_nas_variantes(self):
        for mercado, e in _mercados().items():
            for nome, campos in VARIANTES.items():
                ts = _ts(vencimento=1.5, **campos)
                for parametro, livre in ts.parametros_livres().items():
                    original = getattr(ts, parametro)
                    if original is None:
                        continue
                    with self.subTest(mercado=mercado, variante=nome, parametro=parametro):
                        alvo = _margem(ts, e, **CUSTOS)
                        sol = resolver(ts, e, parametro, margem_alvo=alvo, **CUSTOS)
                        self.assertEqual(sol.situacao, "resolvido")
                        self.assertTrue(sol.viavel)
                        self.assertEqual(sol.metodo, "direto" if livre.linear else "raiz")
                        self.assertAlmostEqual(sol.valor, original, delta=1e-8)
                        self.assertAlmostEqual(sol.margem, alvo, delta=1e-10)

    def test_resto_do_termsheet_fica_igual(self):
        ts = _ts(protecao=0.97, strike=1.02, cap=0.25)
        sol = resolver(ts, _estado(spread_bp=100.0), "cap", margem_alvo=0.05, **CUSTOS)
        self.assertEqual(sol.situacao, "resolvido")
        self.assertEqual(sol.termsheet, replace(ts, cap=sol.valor))
        self.assertNotEqual(sol.valor, ts.cap)

    def test_aceita_o_coe(self):
        ts, e = _ts(), _estado(spread_bp=100.0)
        a = resolver(ts, e, "participacao", margem_alvo=0.02, **CUSTOS)
        b = resolver(COE(ts), e, "participacao", margem_alvo=0.02, **CUSTOS)
        self.assertEqual(a.valor, b.valor)

    def test_o_nominal_nao_muda_a_solucao(self):
        e = _estado(spread_bp=100.0)
        for parametro in ("participacao", "cap", "strike"):
            with self.subTest(parametro):
                a = resolver(_ts(nominal=1_000.0), e, parametro, margem_alvo=0.015, **CUSTOS)
                b = resolver(_ts(nominal=5_000_000.0), e, parametro, margem_alvo=0.015, **CUSTOS)
                self.assertAlmostEqual(a.valor, b.valor, delta=1e-10)
                self.assertAlmostEqual(b.decomposicao.margem, 0.015 * 5_000_000.0, delta=1e-4)


class TestSolucaoDireta(unittest.TestCase):
    def setUp(self):
        self.e = _estado(curva=CurvaDI(INCLINADA), spread_bp=120.0)
        self.n = 1_000_000.0
        self.disponivel = (1.0 - 0.04) * self.n

    def test_participacao_pela_conta_fechada(self):
        ts = _ts(nominal=self.n, participacao=1.0)
        alvo = 0.02
        titulo = float(Titulo(vencimento=1.0, principal=self.n).valor(self.e))
        call_spread = float(CallSpread("IBOV", S0, 1.2 * S0, 1.0).valor(self.e))
        esperado = (self.disponivel - titulo - alvo * self.n) / (self.n / S0 * call_spread)
        sol = resolver(ts, self.e, "participacao", margem_alvo=alvo, **CUSTOS)
        self.assertEqual(sol.metodo, "direto")
        self.assertAlmostEqual(sol.valor, esperado, delta=1e-12)

    def test_protecao_pela_conta_fechada(self):
        ts = _ts(nominal=self.n, participacao=1.2, cap=0.3)
        alvo = 0.01
        fator = float(Titulo(vencimento=1.0, principal=1.0).valor(self.e))
        opcoes = float(COE(ts).opcoes().valor(self.e))
        esperado = (self.disponivel - opcoes - alvo * self.n) / (self.n * fator)
        sol = resolver(ts, self.e, "protecao", margem_alvo=alvo, **CUSTOS)
        self.assertEqual(sol.metodo, "direto")
        self.assertAlmostEqual(sol.valor, esperado, delta=1e-12)
        self.assertLess(sol.valor, 1.0)

    def test_alvo_na_ponta_do_alcance(self):
        ts = _ts(nominal=self.n)
        teto = _margem(replace(ts, participacao=0.0), self.e, **CUSTOS)
        sol = resolver(ts, self.e, "participacao", margem_alvo=teto, **CUSTOS)
        self.assertEqual(sol.situacao, "resolvido")
        self.assertAlmostEqual(sol.valor, 0.0, delta=1e-12)
        self.assertAlmostEqual(sol.margem_maxima, teto, delta=1e-14)


class TestMonotonicidade(unittest.TestCase):
    def test_margem_monotonica_no_sentido_esperado(self):
        e = _mercados()["smile e curva inclinada"]
        for nome, campos in VARIANTES.items():
            ts = _ts(vencimento=1.5, **campos)
            for parametro, livre in ts.parametros_livres().items():
                with self.subTest(variante=nome, parametro=parametro):
                    xs = np.linspace(livre.minimo, livre.maximo, 41)
                    margens = np.array([_margem(replace(ts, **{parametro: x}), e, **CUSTOS) for x in xs])
                    passos = np.diff(margens) * SENTIDO[(ts.direcao, parametro)]
                    self.assertTrue(np.all(passos >= -1e-12), f"{parametro}: {passos.min()}")
                    self.assertGreater(margens.max() - margens.min(), 1e-3)

    def test_estrutura_nao_monotonica_e_recusada(self):
        ts = _Straddle(ativo="IBOV", nivel_inicial=S0, vencimento=1.0)
        with self.assertRaisesRegex(ValueError, "monotônica"):
            resolver(ts, _estado(), "strike", margem_alvo=-0.05)

    def test_smile_com_arbitragem_e_recusado(self):
        """Vol que explode perto do strike zero: abaixo de ~60% do spot, a put de strike mais baixo passa a valer
        mais, e a margem deixa de cair com o cap (a partir de cap ~40%). Restrito a 35%, o solver resolve."""
        def explosivo(strike, prazo):
            return 0.20 + 0.15 * (S0 / strike - 1.0) ** 2 + 0.05 * max(0.0, S0 / strike - 1.0)

        e = replace(_estado(), ativos={"IBOV": MercadoAtivo(spot=S0, vol=explosivo, dividendos=0.04)})
        ts = _ts(direcao="baixa", cap=0.25, vencimento=1.5)
        with self.assertRaisesRegex(ValueError, "monotônica"):
            resolver(ts, e, "cap", margem_alvo=0.0, **CUSTOS)
        alvo = _margem(ts, e, **CUSTOS)
        sol = resolver(ts, e, "cap", margem_alvo=alvo, intervalo=(0.05, 0.35), **CUSTOS)
        self.assertAlmostEqual(sol.valor, 0.25, delta=1e-8)

    def test_intervalo_restrito_resolve(self):
        e = _estado()
        ts = _Straddle(ativo="IBOV", nivel_inicial=S0, vencimento=1.0, strike=1.4)
        alvo = _margem(ts, e)
        sol = resolver(replace(ts, strike=1.0), e, "strike", margem_alvo=alvo, intervalo=(1.2, 1.6))
        self.assertEqual(sol.situacao, "resolvido")
        self.assertAlmostEqual(sol.valor, 1.4, delta=1e-8)
        self.assertEqual(sol.intervalo, (1.2, 1.6))

    def test_parametro_declarado_linear_que_nao_e(self):
        e = _estado()
        ts = _Straddle(ativo="IBOV", nivel_inicial=S0, vencimento=1.0, strike_linear=True)
        alvo = _margem(replace(ts, strike=0.8), e)
        with self.assertRaisesRegex(RuntimeError, "linear"):
            resolver(ts, e, "strike", margem_alvo=alvo, intervalo=(0.6, 1.0))


class TestForaDoAlcance(unittest.TestCase):
    def test_protecao_total_com_juro_baixo_e_prazo_curto_e_inviavel(self):
        e = _estado(curva=CurvaPlana(log(1.045)), spread_bp=50.0)
        ts = _ts(vencimento=0.5, participacao=1.0)
        fator = 1.05 ** -0.5
        for parametro in ("participacao", "cap", "strike"):
            with self.subTest(parametro):
                sol = resolver(ts, e, parametro, margem_alvo=0.01, **CUSTOS)
                self.assertEqual(sol.situacao, "inviavel")
                self.assertFalse(sol.viavel)
                self.assertIsNone(sol.valor)
                self.assertAlmostEqual(sol.margem_maxima, 1.0 - fator - 0.04, delta=1e-5)
                self.assertAlmostEqual(sol.margem, sol.margem_maxima, delta=1e-12)
                self.assertIn("inviável", sol.mensagem)
                self.assertIn("o título custa 97,59% do nominal", sol.mensagem)
        self.assertEqual(resolver(ts, e, "participacao", margem_alvo=0.01, **CUSTOS).termsheet.participacao, 0.0)
        # A saída é reduzir a proteção.
        sol = resolver(ts, e, "protecao", margem_alvo=0.01, **CUSTOS)
        self.assertEqual(sol.situacao, "resolvido")
        self.assertLess(sol.valor, 0.95)

    def test_folga_quando_nem_sem_cap_a_margem_cai_ao_alvo(self):
        e = _estado(spread_bp=100.0)
        ts = _ts(vencimento=3.0, participacao=0.5)
        sem_cap = _margem(replace(ts, cap=None), e, **CUSTOS)
        sol = resolver(ts, e, "cap", margem_alvo=0.01, **CUSTOS)
        self.assertGreater(sem_cap, 0.01)
        self.assertEqual(sol.situacao, "folga")
        self.assertIsNone(sol.valor)
        self.assertAlmostEqual(sol.margem_minima, sem_cap, delta=1e-6)
        self.assertEqual(sol.termsheet.cap, 10.0)
        self.assertIn("na prática sem cap", sol.mensagem)

    def test_folga_na_baixa(self):
        sol = resolver(_ts(direcao="baixa", participacao=1.0), _estado(spread_bp=100.0), "cap",
                       margem_alvo=0.02, **CUSTOS)
        self.assertEqual(sol.situacao, "folga")
        self.assertIn("piso no zero", sol.mensagem)


class TestParametrosLivres(unittest.TestCase):
    def test_intervalos_sao_termsheets_validos(self):
        for nome, campos in VARIANTES.items():
            ts = _ts(**campos)
            for parametro, livre in ts.parametros_livres().items():
                with self.subTest(variante=nome, parametro=parametro):
                    for x in (livre.minimo, livre.maximo):
                        replace(ts, **{parametro: x})

    def test_intervalos_seguem_strike_e_cap(self):
        livres = _ts(strike=1.05, cap=0.3).parametros_livres()
        self.assertAlmostEqual(livres["cap"].minimo, 0.05 + 1e-6)
        self.assertAlmostEqual(livres["strike"].maximo, 1.3 - 1e-6)
        baixa = _ts(direcao="baixa", strike=0.95, cap=0.25).parametros_livres()
        self.assertAlmostEqual(baixa["cap"].minimo, 0.05 + 1e-6)
        self.assertAlmostEqual(baixa["strike"].minimo, 0.75 + 1e-6)
        self.assertTrue(livres["participacao"].linear and livres["protecao"].linear)
        self.assertFalse(livres["cap"].linear or livres["strike"].linear)

    def test_base_nao_declara_parametros(self):
        self.assertEqual(TermsheetBase.parametros_livres(_ts()), {})


class TestValidacao(unittest.TestCase):
    def setUp(self):
        self.ts, self.e = _ts(), _estado(spread_bp=100.0)

    def test_entradas_recusadas(self):
        casos = [
            dict(parametro="barreira", margem_alvo=0.02),
            dict(parametro="participacao", margem_alvo=2.0),
            dict(parametro="participacao", margem_alvo=float("nan")),
            dict(parametro="cap", margem_alvo=0.02, pontos=2),
            dict(parametro="cap", margem_alvo=0.02, intervalo=(0.5, 0.1)),
            dict(parametro="participacao", margem_alvo=0.02, custo_distribuicao=3.0),
        ]
        for caso in casos:
            with self.subTest(**caso), self.assertRaises(ValueError):
                resolver(self.ts, self.e, **caso)

    def test_mensagem_lista_os_livres(self):
        with self.assertRaisesRegex(ValueError, "participacao, protecao, cap, strike"):
            resolver(self.ts, self.e, "barreira", margem_alvo=0.02)

    def test_estado_vetorial_e_recusado(self):
        with self.assertRaises(ValueError):
            resolver(self.ts, self.e.com_spot("IBOV", np.array([S0, 1.1 * S0])), "cap", margem_alvo=0.02)

    def test_tipo_errado(self):
        with self.assertRaises(TypeError):
            resolver("COE", self.e, "cap", margem_alvo=0.02)


class TestSolucao(unittest.TestCase):
    def test_mensagem_e_dict(self):
        e = _estado(spread_bp=100.0)
        sol = resolver(_ts(participacao=1.0), e, "participacao", margem_alvo=0.02, **CUSTOS)
        self.assertTrue(sol.mensagem.startswith("Participação de "))
        self.assertTrue(sol.mensagem.endswith(" dá a margem de 2%."))
        self.assertTrue(sol.decomposicao.viavel)
        self.assertEqual(sol.coe.termsheet, sol.termsheet)
        dados = sol.como_dict()
        for chave in ("parametro", "situacao", "valor", "margem_alvo", "margem", "margem_minima", "margem_maxima",
                      "metodo", "termsheet", "mensagem", "decomposicao"):
            self.assertIn(chave, dados)
        self.assertAlmostEqual(dados["decomposicao"]["margem_pct"], 0.02, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
