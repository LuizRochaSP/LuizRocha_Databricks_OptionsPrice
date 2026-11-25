"""Testes da perna de título do COE (etapa A2 do plano).

- Com spread zero, o título é uma LTN na mesma curva.
- O valor cai quando o spread sobe, com a derivada analítica (CS01).
- A soma dos DV01 por vértice bate com o DV01 paralelo.
- 100% do CDI equivale a spread zero; ida e volta entre % do CDI e spread em bp.
"""
from __future__ import annotations

import importlib.util
import unittest
from datetime import date
from math import exp, log

import numpy as np

from ibov_barrier.coe import (
    Titulo,
    fator_desconto_emissor,
    fator_desconto_percentual_cdi,
    percentual_cdi_de_spread,
    spread_de_percentual_cdi,
)
from ibov_barrier.produtos import (
    CallSpread,
    CurvaDI,
    CurvaPlana,
    EstadoMercado,
    MercadoAtivo,
    Vanilla,
    dv01_por_vertice,
    gregas,
)

HAS_DI_CURVE = all(importlib.util.find_spec(m) is not None for m in ("exchange_calendars", "lxml"))


class _CurvaVertices:
    """Curva DI mínima com a interface da `DiscountCurve`: taxas efetivas por vértice (DU), ln(DF) linear em T
    entre vértices (flat forward), taxa zero constante antes do primeiro vértice e forward da ponta depois do
    último. Não importa `di_curve` (que exige exchange_calendars)."""

    def __init__(self, vertices_du, taxas):
        self.tenors_bd = np.asarray(vertices_du, dtype=float)
        self.rates = np.asarray(taxas, dtype=float)
        self._t = np.concatenate(([0.0], self.tenors_bd / 252.0))
        self._ld = np.concatenate(([0.0], -self.tenors_bd / 252.0 * np.log1p(self.rates)))

    def log_df(self, du):
        t = float(du) / 252.0
        if t <= self._t[-1]:
            return float(np.interp(t, self._t, self._ld))
        inclinacao = (self._ld[-1] - self._ld[-2]) / (self._t[-1] - self._t[-2])
        return float(self._ld[-1] + inclinacao * (t - self._t[-1]))

    def rate(self, du):
        return float(np.expm1(-self.log_df(du) / (du / 252.0)))

    def bump_parallel(self, bp):
        return _CurvaVertices(self.tenors_bd, self.rates + bp * 1e-4)

    def bump_key(self, du, bp):
        idx = int(np.argmin(np.abs(self.tenors_bd - du)))
        taxas = self.rates.copy()
        taxas[idx] += bp * 1e-4
        return _CurvaVertices(self.tenors_bd, taxas)


def _estado(curva_di, spread_bp=0.0) -> EstadoMercado:
    return EstadoMercado(ativos={"IBOV": MercadoAtivo(spot=130_000.0, vol=0.22, dividendos=0.03)},
                         curvas={"BRL": CurvaDI(curva_di)}, spread_emissor_bp=spread_bp)


PLANA_14 = _CurvaVertices([21, 252, 504, 756], [0.14, 0.14, 0.14, 0.14])
INCLINADA = _CurvaVertices([21, 126, 252, 504, 756], [0.149, 0.146, 0.141, 0.138, 0.137])


class TestDescontoDoEmissor(unittest.TestCase):
    def test_spread_zero_e_a_curva_pre(self):
        curva = CurvaDI(INCLINADA)
        for prazo in (0.3, 1.0, 1.75, 3.0):
            self.assertAlmostEqual(fator_desconto_emissor(curva, prazo, 0.0), curva.fator_desconto(prazo), places=14)

    def test_spread_soma_na_taxa_efetiva(self):
        curva = CurvaDI(PLANA_14)
        self.assertAlmostEqual(fator_desconto_emissor(curva, 2.0, 150.0), 1.155 ** -2.0, places=14)

    def test_prazo_vencido_vale_um(self):
        self.assertEqual(fator_desconto_emissor(CurvaDI(PLANA_14), 0.0, 200.0), 1.0)
        np.testing.assert_array_equal(fator_desconto_emissor(CurvaDI(PLANA_14), 0.0, np.array([1.0, 2.0])), [1, 1])

    def test_spread_vetorial(self):
        spreads = np.array([0.0, 100.0, 250.0])
        vetor = fator_desconto_emissor(CurvaDI(PLANA_14), 1.0, spreads)
        np.testing.assert_allclose(vetor, (1.14 + spreads * 1e-4) ** -1.0, rtol=1e-14)


class TestPercentualDoCDI(unittest.TestCase):
    def test_cem_por_cento_do_cdi_e_spread_zero(self):
        for prazo in (0.5, 1.0, 2.5):
            self.assertAlmostEqual(spread_de_percentual_cdi(CurvaDI(INCLINADA), 100.0, prazo), 0.0, places=8)

    def test_valor_conhecido_em_curva_plana(self):
        cdi_diario = 1.14 ** (1 / 252) - 1
        fator = (1 + 1.10 * cdi_diario) ** 252
        esperado = (fator - 1 - 0.14) * 1e4
        self.assertAlmostEqual(spread_de_percentual_cdi(CurvaDI(PLANA_14), 110.0, 1.0), esperado, places=8)
        # 252 produtos diários: o arredondamento acumula perto de 1e-14.
        self.assertAlmostEqual(fator_desconto_percentual_cdi(CurvaDI(PLANA_14), 1.0, 110.0), 1 / fator, places=12)

    def test_ida_e_volta(self):
        curva = CurvaDI(INCLINADA)
        for percentual in (95.0, 100.0, 107.5, 120.0):
            spread = spread_de_percentual_cdi(curva, percentual, 1.5)
            self.assertAlmostEqual(percentual_cdi_de_spread(curva, spread, 1.5), percentual, places=8)

    def test_monotonico_no_percentual(self):
        curva = CurvaDI(INCLINADA)
        spreads = [spread_de_percentual_cdi(curva, p, 2.0) for p in (90.0, 100.0, 110.0, 130.0)]
        self.assertEqual(spreads, sorted(spreads))
        self.assertLess(spreads[0], 0.0)
        self.assertGreater(spreads[2], 0.0)

    def test_o_mesmo_percentual_vale_spreads_diferentes_por_prazo(self):
        curva = CurvaDI(INCLINADA)
        self.assertNotAlmostEqual(spread_de_percentual_cdi(curva, 110.0, 0.5),
                                  spread_de_percentual_cdi(curva, 110.0, 3.0), places=2)

    def test_fora_da_faixa(self):
        with self.assertRaises(ValueError):
            percentual_cdi_de_spread(CurvaDI(PLANA_14), 1e6, 1.0)
        with self.assertRaises(ValueError):
            fator_desconto_percentual_cdi(CurvaDI(PLANA_14), 1.0, -5.0)
        with self.assertRaises(ValueError):
            spread_de_percentual_cdi(CurvaDI(PLANA_14), 110.0, 0.001)


class TestTitulo(unittest.TestCase):
    def test_spread_zero_e_ltn_na_mesma_curva(self):
        du = 378
        titulo = Titulo(vencimento=du / 252, principal=1_000.0)
        taxa = INCLINADA.rate(du)
        self.assertAlmostEqual(titulo.valor(_estado(INCLINADA)), 1_000.0 / (1 + taxa) ** (du / 252), places=9)

    def test_valor_cai_com_o_spread(self):
        titulo = Titulo(vencimento=2.0, principal=1_000.0)
        valores = [titulo.valor(_estado(PLANA_14, s)) for s in (0.0, 50.0, 150.0, 400.0)]
        self.assertEqual(valores, sorted(valores, reverse=True))
        self.assertAlmostEqual(valores[2], 1_000.0 / 1.155**2, places=9)

    def test_cupons(self):
        titulo = Titulo(vencimento=2.0, principal=1_000.0, cupons=((1.0, 60.0), (0.5, 30.0), (2.0, 60.0)))
        e = _estado(INCLINADA, 80.0)
        curva = e.curva("BRL")
        esperado = sum(v * fator_desconto_emissor(curva, t, 80.0) for t, v in ((0.5, 30.0), (1.0, 60.0), (2.0, 1060.0)))
        self.assertAlmostEqual(titulo.valor(e), esperado, places=9)
        self.assertEqual(titulo.cupons, ((0.5, 30.0), (1.0, 60.0), (2.0, 60.0)))

    def test_tempo_decorrido_tira_os_fluxos_pagos(self):
        titulo = Titulo(vencimento=2.0, principal=1_000.0, cupons=((0.5, 30.0), (1.0, 60.0)))
        e = _estado(PLANA_14, 100.0).avancar(0.6)
        esperado = 60.0 * 1.15**-0.4 + 1_000.0 * 1.15**-1.4
        self.assertAlmostEqual(titulo.valor(e), esperado, places=9)
        self.assertAlmostEqual(titulo.valor(_estado(PLANA_14).com_tempo(2.0)), 1_000.0, places=12)
        self.assertEqual(titulo.valor(_estado(PLANA_14).com_tempo(2.5)), 0.0)

    def test_vetorial_igual_ao_escalar(self):
        titulo = Titulo(vencimento=1.5, principal=100.0, cupons=((0.75, 5.0),))
        e = _estado(INCLINADA).com_spread(np.array([0.0, 120.0, 300.0]))
        np.testing.assert_allclose(titulo.valor(e), [titulo.valor(c) for c in e.cenarios()], rtol=1e-14)

    def test_curva_plana_continua(self):
        e = EstadoMercado(ativos={}, curvas={"BRL": CurvaPlana(0.13)})
        self.assertAlmostEqual(Titulo(vencimento=1.5, principal=1.0).valor(e), exp(-0.13 * 1.5), places=14)

    def test_taxa_do_emissor(self):
        titulo = Titulo(vencimento=1.0, principal=1.0)
        self.assertAlmostEqual(titulo.taxa_emissor(_estado(PLANA_14, 175.0)), 0.1575, places=12)

    def test_metadados_e_payoff(self):
        titulo = Titulo(vencimento=2.0, principal=1_000.0, cupons=((2.0, 50.0),))
        self.assertEqual(titulo.subjacentes(), ())
        self.assertEqual(titulo.observacoes(), ())
        self.assertEqual([(f.tempo, f.valor, f.descricao) for f in titulo.fluxos()],
                         [(2.0, 50.0, "cupom fixo"), (2.0, 1_000.0, "principal")])
        np.testing.assert_array_equal(titulo.payoff({"IBOV": np.array([1.0, 2.0, 3.0])}), [1_050.0] * 3)
        with self.assertRaises(TypeError):
            Titulo(vencimento=2.0, cupons=((1.0, 5.0),)).payoff({"IBOV": np.array([1.0])})

    def test_validacao(self):
        with self.assertRaises(ValueError):
            Titulo(vencimento=0.0)
        with self.assertRaises(ValueError):
            Titulo(vencimento=1.0, principal=-1.0)
        with self.assertRaises(ValueError):
            Titulo(vencimento=1.0, cupons=((1.5, 1.0),))


class TestRiscoDoTitulo(unittest.TestCase):
    def test_cs01_analitico(self):
        titulo = Titulo(vencimento=2.0, principal=1_000.0)
        g = gregas(titulo, _estado(PLANA_14, 100.0))
        esperado = -2.0 * 1_000.0 * 1.15**-3.0 * 1e-4
        self.assertAlmostEqual(g.cs01 / esperado, 1.0, delta=1e-7)
        self.assertEqual(g.delta, {})
        self.assertIn("cs01", g.como_dict())

    def test_dv01_igual_ao_cs01_em_curva_plana(self):
        # Spread aditivo na taxa efetiva: deslocar a curva plana ou o spread em 1 bp é a mesma coisa.
        titulo = Titulo(vencimento=1.5, principal=1_000.0)
        g = gregas(titulo, _estado(PLANA_14, 60.0))
        self.assertAlmostEqual(g.dv01["BRL"], g.cs01, places=10)

    def test_soma_dos_dv01_por_vertice_e_o_paralelo(self):
        titulo = Titulo(vencimento=2.0, principal=1_000.0, cupons=((0.5, 30.0), (1.0, 30.0), (1.5, 30.0)))
        e = _estado(INCLINADA, 90.0)
        por_vertice = dv01_por_vertice(titulo, e)
        self.assertEqual(set(por_vertice), {21.0, 126.0, 252.0, 504.0, 756.0})
        self.assertAlmostEqual(sum(por_vertice.values()) / gregas(titulo, e).dv01["BRL"], 1.0, delta=1e-6)
        self.assertAlmostEqual(por_vertice[756.0], 0.0, places=12)  # nenhum fluxo depois de 504 DU

    def test_theta(self):
        titulo = Titulo(vencimento=1.0, principal=1_000.0)
        g = gregas(titulo, _estado(PLANA_14, 50.0))
        self.assertAlmostEqual(g.theta, 1_000.0 * (1.145 ** -(1 - 1 / 252) - 1.145**-1), places=9)

    def test_dv01_por_vertice_exige_curva_com_vertices(self):
        e = EstadoMercado(ativos={}, curvas={"BRL": CurvaPlana(0.13)})
        with self.assertRaises(TypeError):
            dv01_por_vertice(Titulo(vencimento=1.0), e)


class TestTituloMaisOpcoes(unittest.TestCase):
    """O COE mínimo do primeiro caso, ainda sem termsheet: título + participação num call spread do IBOV."""

    def test_carteira_soma_e_gregas_separadas(self):
        e = _estado(INCLINADA, 120.0)
        titulo = Titulo(vencimento=1.0, principal=1_000.0)
        cs = CallSpread("IBOV", 130_000.0, 156_000.0, 1.0)
        participacao = 0.8 * 1_000.0 / 130_000.0
        coe = titulo + participacao * cs
        self.assertTrue(coe.depende_do_spread)
        self.assertAlmostEqual(coe.valor(e), titulo.valor(e) + participacao * cs.valor(e), places=9)
        g, g_titulo, g_cs = gregas(coe, e), gregas(titulo, e), gregas(cs, e)
        self.assertAlmostEqual(g.cs01, g_titulo.cs01, places=12)     # as opções não dependem do spread
        self.assertAlmostEqual(g.delta["IBOV"], participacao * g_cs.delta["IBOV"], places=12)
        self.assertAlmostEqual(g.dv01["BRL"], g_titulo.dv01["BRL"] + participacao * g_cs.dv01["BRL"], places=9)
        finais = {"IBOV": np.array([100_000.0, 143_000.0, 200_000.0])}
        np.testing.assert_allclose(coe.payoff(finais), [1_000.0, 1_080.0, 1_160.0])

    def test_opcao_sozinha_nao_tem_cs01(self):
        g = gregas(Vanilla("IBOV", 130_000.0, 1.0), _estado(INCLINADA, 120.0))
        self.assertIsNone(g.cs01)
        self.assertNotIn("cs01", g.como_dict())


@unittest.skipUnless(HAS_DI_CURVE, "exchange_calendars/lxml ausentes")
class TestComCurvaReal(unittest.TestCase):
    def setUp(self):
        from ibov_barrier.di_curve import DiscountCurve

        self.curva = DiscountCurve.from_di1([21, 126, 252, 504, 756], [0.149, 0.146, 0.141, 0.138, 0.137])

    def test_spread_zero_e_ltn_price(self):
        from ibov_barrier.di_curve.conventions import business_days
        from ibov_barrier.fixed_income.ltn import ltn_price

        referencia, vencimento = date(2026, 11, 25), date(2028, 1, 1)
        du = business_days(referencia, vencimento)
        e = EstadoMercado(ativos={}, curvas={"BRL": CurvaDI(self.curva)})
        esperado = ltn_price(self.curva.rate(du) * 100.0, referencia, vencimento)
        self.assertAlmostEqual(Titulo(vencimento=du / 252, principal=1_000.0).valor(e), esperado, places=8)

    def test_soma_dos_dv01_por_vertice_e_o_paralelo(self):
        e = EstadoMercado(ativos={}, curvas={"BRL": CurvaDI(self.curva)}, spread_emissor_bp=90.0)
        titulo = Titulo(vencimento=2.0, principal=1_000.0, cupons=((0.5, 30.0), (1.0, 30.0), (1.5, 30.0)))
        soma = sum(dv01_por_vertice(titulo, e).values())
        self.assertAlmostEqual(soma / gregas(titulo, e).dv01["BRL"], 1.0, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
