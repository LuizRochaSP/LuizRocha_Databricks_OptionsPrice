"""Testes da interface de produtos (etapa A1 do plano de COE).

O critério central é "mesmo número": cada adaptador reproduz o pricer original. Depois vêm a avaliação vetorial
(igual à escalar, cenário a cenário), a composição em carteira e as gregas por diferenças finitas contra as
fórmulas analíticas.
"""
from __future__ import annotations

import importlib.util
import unittest
from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm

from ibov_barrier.analytical import _bs_digital, price_down_and_out_call_analytical
from ibov_barrier.pricing import BarrierContract, MarketData, black_scholes_call, price_down_and_out_call
from ibov_barrier.pricing_barriers import _black_scholes_put_by_parity
from ibov_barrier.pricing_extensions import price_no_touch, price_one_touch, price_single_barrier
from ibov_barrier.pricing_quanto import QuantoMarket, price_quanto_vanilla, quanto_market_data
from ibov_barrier.produtos import (
    BarreiraSimples,
    CallSpread,
    Carteira,
    CurvaDI,
    CurvaPlana,
    Digital,
    EstadoMercado,
    MercadoAtivo,
    MercadoCambio,
    NoTouch,
    OneTouch,
    Produto,
    Quanto,
    Vanilla,
    gregas,
)

HAS_DI_CURVE = all(importlib.util.find_spec(m) is not None for m in ("exchange_calendars", "lxml"))

MC = dict(caminhos=4_000, passos=20, semente=7)
MC_PRICER = dict(paths=4_000, steps=20, seed=7)

CASOS_VANILLA = [
    # spot, strike, prazo, r, q, vol
    (130_000.0, 130_000.0, 1.0, 0.13, 0.03, 0.22),
    (130_000.0, 156_000.0, 0.5, 0.14, 0.00, 0.18),
    (130_000.0, 100_000.0, 2.0, 0.12, 0.04, 0.30),
    (7_000.0, 6_500.0, 0.25, 0.045, 0.012, 0.16),
    (100.0, 140.0, 3.0, 0.10, 0.02, 0.45),
]


def _md(s, r, q, v):
    return MarketData(spot=s, rate=r, dividend_yield=q, volatility=v)


def _estado_quanto(spot=7_000.0, vol=0.17, q=0.013, r_usd=0.042, r_brl=0.138, fx=5.35, vol_fx=0.16, rho=-0.3):
    return EstadoMercado(
        ativos={"SPX": MercadoAtivo(spot=spot, vol=vol, dividendos=q, moeda="USD")},
        curvas={"USD": CurvaPlana(r_usd), "BRL": CurvaPlana(r_brl)},
        cambios={"USDBRL": MercadoCambio(spot=fx, vol=vol_fx)},
        correlacoes={("SPX", "USDBRL"): rho},
    )


def _quanto_market(spot=7_000.0, vol=0.17, q=0.013, r_usd=0.042, r_brl=0.138, fx=5.35, vol_fx=0.16, rho=-0.3):
    return QuantoMarket(spot=spot, rate_usd=r_usd, dividend_yield=q, vol_asset=vol, fx_spot=fx, rate_brl=r_brl,
                        vol_fx=vol_fx, correlation=rho)


class _CurvaFalsa:
    """Curva com a mesma interface da `DiscountCurve` (taxa efetiva plana, DU/252), sem exchange_calendars."""

    def __init__(self, taxa_efetiva: float):
        self.taxa = taxa_efetiva

    def rate(self, du):
        return self.taxa

    def log_df(self, du):
        return -du / 252.0 * log(1.0 + self.taxa)

    def bump_parallel(self, bp):
        return _CurvaFalsa(self.taxa + bp * 1e-4)

    def bump_key(self, du, bp):
        return _CurvaFalsa(self.taxa + bp * 1e-4)


# ------------------------------------------------------------------------------------------ estado

class TestEstadoMercado(unittest.TestCase):
    def test_estado_simples_escalar(self):
        e = EstadoMercado.simples(spot=100.0, taxa=0.1, vol=0.2)
        self.assertEqual(e.n_cenarios, 1)
        self.assertFalse(e.vetorial)
        self.assertEqual(e.ativo("IBOV").spot, 100.0)

    def test_vetor_de_cenarios(self):
        e = EstadoMercado.simples(spot=np.array([90.0, 100.0, 110.0]), taxa=0.1, vol=0.2)
        self.assertEqual(e.n_cenarios, 3)
        self.assertTrue(e.vetorial)
        c = e.cenario(2)
        self.assertFalse(c.vetorial)
        self.assertEqual(c.ativo("IBOV").spot, 110.0)
        self.assertEqual([x.ativo("IBOV").spot for x in e.cenarios()], [90.0, 100.0, 110.0])
        with self.assertRaises(IndexError):
            e.cenario(3)

    def test_vetor_de_tamanho_um_vale_para_todos(self):
        e = EstadoMercado.simples(spot=np.array([90.0, 100.0]), taxa=np.array([0.1]), vol=0.2)
        self.assertEqual(e.n_cenarios, 2)
        self.assertEqual(e.cenario(1).curva("BRL").taxa, 0.1)

    def test_tamanhos_diferentes_rejeitados(self):
        with self.assertRaises(ValueError):
            EstadoMercado.simples(spot=np.array([90.0, 100.0, 110.0]), taxa=np.array([0.1, 0.2]), vol=0.2)

    def test_choques_nao_alteram_o_original(self):
        e = EstadoMercado.simples(spot=100.0, taxa=0.1, vol=0.2)
        e2 = e.com_spot("IBOV", 120.0).com_choque_vol("IBOV", 0.05).avancar(0.1).com_spread(150.0)
        self.assertEqual(e.ativo("IBOV").spot, 100.0)
        self.assertEqual(e.ativo("IBOV").choque_vol, 0.0)
        self.assertEqual(e.tempo, 0.0)
        self.assertEqual(e2.ativo("IBOV").spot, 120.0)
        self.assertAlmostEqual(e2.ativo("IBOV").vol_em(100.0, 1.0), 0.25)
        self.assertEqual(e2.tempo, 0.1)
        self.assertEqual(e2.spread_emissor_bp, 150.0)

    def test_vol_como_funcao_do_strike(self):
        sorriso = lambda k, t: 0.20 - 0.10 * (k / 100.0 - 1.0)  # noqa: E731
        m = MercadoAtivo(spot=100.0, vol=sorriso, choque_vol=0.01)
        self.assertAlmostEqual(m.vol_em(90.0, 1.0), 0.22)
        self.assertAlmostEqual(m.vol_em(110.0, 1.0), 0.20)

    def test_vol_chocada_negativa_rejeitada(self):
        m = MercadoAtivo(spot=100.0, vol=0.05, choque_vol=-0.10)
        with self.assertRaises(ValueError):
            m.vol_em(100.0, 1.0)

    def test_spot_invalido_rejeitado(self):
        with self.assertRaises(ValueError):
            MercadoAtivo(spot=-1.0, vol=0.2)
        with self.assertRaises(ValueError):
            MercadoAtivo(spot=np.array([100.0, np.nan]), vol=0.2)

    def test_correlacao_simetrica(self):
        e = _estado_quanto(rho=-0.3)
        self.assertEqual(e.correlacao("SPX", "USDBRL"), -0.3)
        self.assertEqual(e.correlacao("USDBRL", "SPX"), -0.3)
        with self.assertRaises(KeyError):
            e.correlacao("IBOV", "USDBRL")
        with self.assertRaises(ValueError):
            EstadoMercado(ativos={}, curvas={}, correlacoes={("A", "B"): 1.5})

    def test_faltas_com_mensagem(self):
        e = EstadoMercado.simples(spot=100.0, taxa=0.1, vol=0.2)
        with self.assertRaisesRegex(KeyError, "SPX"):
            e.ativo("SPX")
        with self.assertRaisesRegex(KeyError, "USD"):
            e.curva("USD")


class TestCurvas(unittest.TestCase):
    def test_curva_plana(self):
        c = CurvaPlana(0.12)
        self.assertEqual(c.taxa_em(2.0), 0.12)
        self.assertAlmostEqual(c.fator_desconto(2.0), exp(-0.24))
        self.assertAlmostEqual(c.deslocar(10).taxa, 0.121)
        cv = CurvaPlana(np.array([0.1, 0.2]))
        np.testing.assert_allclose(cv.fator_desconto(1.0), np.exp([-0.1, -0.2]))

    def test_curva_di_converte_taxa_efetiva_em_continua(self):
        c = CurvaDI(_CurvaFalsa(0.14))
        self.assertAlmostEqual(c.taxa_em(1.5), log(1.14), places=14)
        self.assertAlmostEqual(c.fator_desconto(1.5), 1.14 ** -1.5, places=14)
        self.assertEqual(c.fator_desconto(0.0), 1.0)
        self.assertAlmostEqual(c.deslocar(100).taxa_em(1.0), log(1.15), places=14)
        self.assertAlmostEqual(c.deslocar_vertice(252, -100).taxa_em(1.0), log(1.13), places=14)

    def test_opcao_com_curva_di_igual_a_curva_plana_equivalente(self):
        estado_di = EstadoMercado(ativos={"IBOV": MercadoAtivo(spot=130_000.0, vol=0.2, dividendos=0.03)},
                                  curvas={"BRL": CurvaDI(_CurvaFalsa(0.14))})
        estado_plano = EstadoMercado.simples(spot=130_000.0, taxa=log(1.14), vol=0.2, dividendos=0.03)
        v = Vanilla("IBOV", 135_000.0, 1.0)
        self.assertAlmostEqual(v.valor(estado_di), v.valor(estado_plano), places=8)

    @unittest.skipUnless(HAS_DI_CURVE, "exchange_calendars/lxml ausentes")
    def test_curva_di_real_segue_a_regra_do_notebook_01(self):
        from ibov_barrier.di_curve import DiscountCurve

        dc = DiscountCurve.from_di1([21, 126, 252, 504], [0.149, 0.145, 0.140, 0.138])
        c = CurvaDI(dc)
        for prazo in (0.1, 0.5, 1.0, 1.7, 2.0):
            self.assertAlmostEqual(c.taxa_em(prazo), float(np.log1p(dc.rate(prazo * 252))), places=14)
            self.assertAlmostEqual(c.fator_desconto(prazo), dc.df(prazo * 252), places=14)


# ----------------------------------------------------------------------------------------- europeus

class TestVanilla(unittest.TestCase):
    def test_call_mesmo_numero(self):
        for s, k, t, r, q, v in CASOS_VANILLA:
            e = EstadoMercado.simples(spot=s, taxa=r, vol=v, dividendos=q)
            esperado = black_scholes_call(_md(s, r, q, v), BarrierContract(k, k, t))
            self.assertAlmostEqual(Vanilla("IBOV", k, t).valor(e), esperado, delta=1e-10 * s)

    def test_put_mesmo_numero(self):
        for s, k, t, r, q, v in CASOS_VANILLA:
            e = EstadoMercado.simples(spot=s, taxa=r, vol=v, dividendos=q)
            md, contrato = _md(s, r, q, v), BarrierContract(k, k, t)
            esperado = _black_scholes_put_by_parity(md, contrato, black_scholes_call(md, contrato))
            self.assertAlmostEqual(Vanilla("IBOV", k, t, "put").valor(e), esperado, delta=1e-9 * s)

    def test_vol_zero_mesmo_numero(self):
        e = EstadoMercado.simples(spot=100.0, taxa=0.1, vol=0.0, dividendos=0.02)
        esperado = black_scholes_call(_md(100.0, 0.1, 0.02, 0.0), BarrierContract(95.0, 95.0, 1.0))
        self.assertAlmostEqual(Vanilla("IBOV", 95.0, 1.0).valor(e), esperado, places=12)

    def test_vetorial_igual_ao_escalar(self):
        spots = np.array([110_000.0, 130_000.0, 150_000.0])
        choques = np.array([-0.02, 0.0, 0.05])
        e = EstadoMercado.simples(spot=130_000.0, taxa=0.13, vol=0.22, dividendos=0.03)
        ev = e.com_spot("IBOV", spots).com_choque_vol("IBOV", choques)
        for produto in (Vanilla("IBOV", 135_000.0, 1.0), Vanilla("IBOV", 120_000.0, 0.5, "put")):
            vetor = produto.valor(ev)
            self.assertEqual(vetor.shape, (3,))
            escalar = [produto.valor(c) for c in ev.cenarios()]
            np.testing.assert_allclose(vetor, escalar, rtol=0, atol=1e-9)

    def test_paridade_put_call(self):
        s, k, t, r, q, v = CASOS_VANILLA[2]
        e = EstadoMercado.simples(spot=s, taxa=r, vol=v, dividendos=q)
        c, p = Vanilla("IBOV", k, t).valor(e), Vanilla("IBOV", k, t, "put").valor(e)
        self.assertAlmostEqual(c - p, s * exp(-q * t) - k * exp(-r * t), delta=1e-8 * s)

    def test_tempo_decorrido_reduz_o_prazo(self):
        e = EstadoMercado.simples(spot=100.0, taxa=0.1, vol=0.2).avancar(0.25)
        esperado = black_scholes_call(_md(100.0, 0.1, 0.0, 0.2), BarrierContract(105.0, 105.0, 0.75))
        self.assertAlmostEqual(Vanilla("IBOV", 105.0, 1.0).valor(e), esperado, places=10)

    def test_vencida_vale_o_intrinseco(self):
        e = EstadoMercado.simples(spot=np.array([90.0, 110.0]), taxa=0.1, vol=0.2).com_tempo(1.0)
        np.testing.assert_allclose(Vanilla("IBOV", 100.0, 1.0).valor(e), [0.0, 10.0])
        np.testing.assert_allclose(Vanilla("IBOV", 100.0, 1.0, "put").valor(e), [10.0, 0.0])

    def test_payoff_e_metadados(self):
        v = Vanilla("IBOV", 100.0, 1.0, "put")
        np.testing.assert_allclose(v.payoff({"IBOV": np.array([80.0, 120.0])}), [20.0, 0.0])
        self.assertEqual(v.subjacentes(), ("IBOV",))
        self.assertEqual(v.observacoes()[0].tipo, "vencimento")
        self.assertIsNone(v.fluxos()[0].valor)

    def test_validacao(self):
        with self.assertRaises(ValueError):
            Vanilla("IBOV", 100.0, 1.0, "forward")
        with self.assertRaises(ValueError):
            Vanilla("IBOV", -1.0, 1.0)
        with self.assertRaises(ValueError):
            Vanilla("IBOV", 100.0, 0.0)


class TestCallSpread(unittest.TestCase):
    def setUp(self):
        self.e = EstadoMercado.simples(spot=130_000.0, taxa=0.13, vol=0.22, dividendos=0.03)

    def test_mesmo_numero_que_duas_calls(self):
        md = _md(130_000.0, 0.13, 0.03, 0.22)
        esperado = (black_scholes_call(md, BarrierContract(130_000.0, 1.0, 1.0))
                    - black_scholes_call(md, BarrierContract(156_000.0, 1.0, 1.0)))
        cs = CallSpread("IBOV", 130_000.0, 156_000.0, 1.0)
        self.assertAlmostEqual(cs.valor(self.e), esperado, delta=1e-9 * 130_000)

    def test_limites_e_payoff(self):
        cs = CallSpread("IBOV", 130_000.0, 156_000.0, 1.0)
        valor = cs.valor(self.e)
        self.assertGreater(valor, 0.0)
        self.assertLess(valor, 26_000.0 * exp(-0.13))
        finais = {"IBOV": np.array([100_000.0, 140_000.0, 200_000.0])}
        np.testing.assert_allclose(cs.payoff(finais), [0.0, 10_000.0, 26_000.0])

    def test_vol_no_strike_de_cada_perna(self):
        sorriso = lambda k, t: 0.25 - 0.20 * (k / 130_000.0 - 1.0)  # noqa: E731
        e = EstadoMercado(ativos={"IBOV": MercadoAtivo(spot=130_000.0, vol=sorriso, dividendos=0.03)},
                          curvas={"BRL": CurvaPlana(0.13)})
        md_baixo, md_alto = _md(130_000.0, 0.13, 0.03, sorriso(130_000.0, 1)), _md(130_000.0, 0.13, 0.03,
                                                                                   sorriso(156_000.0, 1))
        esperado = (black_scholes_call(md_baixo, BarrierContract(130_000.0, 1.0, 1.0))
                    - black_scholes_call(md_alto, BarrierContract(156_000.0, 1.0, 1.0)))
        self.assertAlmostEqual(CallSpread("IBOV", 130_000.0, 156_000.0, 1.0).valor(e), esperado, delta=1e-4)

    def test_strikes_invertidos_rejeitados(self):
        with self.assertRaises(ValueError):
            CallSpread("IBOV", 150.0, 100.0, 1.0)


class TestDigital(unittest.TestCase):
    def test_call_mesmo_numero(self):
        for s, k, t, r, q, v in CASOS_VANILLA:
            e = EstadoMercado.simples(spot=s, taxa=r, vol=v, dividendos=q)
            esperado = 1_000.0 * _bs_digital(s, k, t, r, q, v)
            self.assertAlmostEqual(Digital("IBOV", k, t, pagamento=1_000.0).valor(e), esperado, places=9)

    def test_call_mais_put_paga_com_certeza(self):
        e = EstadoMercado.simples(spot=100.0, taxa=0.1, vol=0.25, dividendos=0.02)
        soma = Digital("IBOV", 105.0, 2.0).valor(e) + Digital("IBOV", 105.0, 2.0, "put").valor(e)
        self.assertAlmostEqual(soma, exp(-0.2), places=12)

    def test_limite_do_call_spread_estreito(self):
        e = EstadoMercado.simples(spot=100.0, taxa=0.1, vol=0.25, dividendos=0.02)
        h = 1e-3
        spread = (Vanilla("IBOV", 105.0 - h, 1.0).valor(e) - Vanilla("IBOV", 105.0 + h, 1.0).valor(e)) / (2 * h)
        self.assertAlmostEqual(Digital("IBOV", 105.0, 1.0).valor(e), spread, places=6)

    def test_vetorial_igual_ao_escalar(self):
        e = EstadoMercado.simples(spot=np.array([90.0, 100.0, 110.0]), taxa=0.1, vol=0.25)
        d = Digital("IBOV", 100.0, 1.0, "put", pagamento=10.0)
        np.testing.assert_allclose(d.valor(e), [d.valor(c) for c in e.cenarios()], atol=1e-12)

    def test_payoff(self):
        d = Digital("IBOV", 100.0, 1.0, pagamento=7.0)
        np.testing.assert_allclose(d.payoff({"IBOV": np.array([99.0, 101.0])}), [0.0, 7.0])


# ------------------------------------------------------------------------------- dependentes de trajetória

class TestBarreiraSimples(unittest.TestCase):
    def setUp(self):
        self.s, self.r, self.q, self.v = 130_000.0, 0.13, 0.03, 0.22
        self.e = EstadoMercado.simples(spot=self.s, taxa=self.r, vol=self.v, dividendos=self.q)

    def test_oito_tipos_com_e_sem_rebate_mesmo_numero(self):
        md = _md(self.s, self.r, self.q, self.v)
        for tipo in ("call", "put"):
            for direcao, barreira in (("down", 110_000.0), ("up", 155_000.0)):
                for knock in ("out", "in"):
                    for rebate in (0.0, 2_000.0):
                        with self.subTest(tipo=tipo, direcao=direcao, knock=knock, rebate=rebate):
                            produto = BarreiraSimples(ativo="IBOV", strike=132_000.0, barreira=barreira,
                                                      vencimento=0.5, tipo=tipo, direcao=direcao, knock=knock,
                                                      rebate=rebate, **MC)
                            esperado = price_single_barrier(
                                md, BarrierContract(132_000.0, barreira, 0.5), option_type=tipo,
                                direction=direcao, knock=knock, rebate=rebate, **MC_PRICER)
                            self.assertEqual(produto.valor(self.e), esperado.price)
                            self.assertEqual(produto.resultado(self.e), esperado)

    def test_doc_sem_rebate_igual_ao_pricer_legado(self):
        produto = BarreiraSimples(ativo="IBOV", strike=130_000.0, barreira=115_000.0, vencimento=1.0, **MC)
        legado = price_down_and_out_call(_md(self.s, self.r, self.q, self.v),
                                         BarrierContract(130_000.0, 115_000.0, 1.0), **MC_PRICER)
        self.assertAlmostEqual(produto.valor(self.e), legado.price, delta=1e-8 * self.s)

    def test_vetorial_igual_ao_escalar(self):
        produto = BarreiraSimples(ativo="IBOV", strike=130_000.0, barreira=115_000.0, vencimento=1.0, **MC)
        ev = self.e.com_spot("IBOV", np.array([120_000.0, 130_000.0, 140_000.0]))
        np.testing.assert_array_equal(produto.valor(ev), [produto.valor(c) for c in ev.cenarios()])

    def test_tempo_decorrido(self):
        produto = BarreiraSimples(ativo="IBOV", strike=130_000.0, barreira=115_000.0, vencimento=1.0, **MC)
        esperado = price_single_barrier(_md(self.s, self.r, self.q, self.v),
                                        BarrierContract(130_000.0, 115_000.0, 0.6), **MC_PRICER).price
        self.assertEqual(produto.valor(self.e.avancar(0.4)), esperado)

    def test_vencida_levanta_erro(self):
        produto = BarreiraSimples(ativo="IBOV", strike=130_000.0, barreira=115_000.0, vencimento=1.0, **MC)
        with self.assertRaisesRegex(ValueError, "vencido"):
            produto.valor(self.e.com_tempo(1.0))

    def test_observacoes_e_validacao(self):
        produto = BarreiraSimples(ativo="IBOV", strike=130_000.0, barreira=115_000.0, vencimento=1.0, **MC)
        tipos = [o.tipo for o in produto.observacoes()]
        self.assertEqual(tipos, ["barreira_continua", "vencimento"])
        self.assertFalse(hasattr(produto, "payoff"))
        with self.assertRaises(ValueError):
            BarreiraSimples(ativo="IBOV", strike=1.0, barreira=1.0, vencimento=1.0, knock="talvez")
        with self.assertRaises(ValueError):
            BarreiraSimples(ativo="IBOV", strike=1.0, barreira=1.0, vencimento=1.0, rebate=-1.0)


class TestToques(unittest.TestCase):
    def setUp(self):
        self.md = _md(130_000.0, 0.13, 0.03, 0.22)
        self.e = EstadoMercado.simples(spot=130_000.0, taxa=0.13, vol=0.22, dividendos=0.03)

    def test_one_touch_e_no_touch_mesmo_numero(self):
        for direcao, barreira in (("down", 115_000.0), ("up", 150_000.0)):
            contrato = BarrierContract(barreira, barreira, 0.5)
            ot = OneTouch(ativo="IBOV", barreira=barreira, vencimento=0.5, direcao=direcao, pagamento=100.0, **MC)
            nt = NoTouch(ativo="IBOV", barreira=barreira, vencimento=0.5, direcao=direcao, pagamento=100.0, **MC)
            self.assertEqual(ot.resultado(self.e), price_one_touch(self.md, contrato, direction=direcao,
                                                                   payout=100.0, **MC_PRICER))
            self.assertEqual(nt.resultado(self.e), price_no_touch(self.md, contrato, direction=direcao,
                                                                  payout=100.0, **MC_PRICER))
            self.assertAlmostEqual(ot.valor(self.e) + nt.valor(self.e), 100.0 * exp(-0.13 * 0.5), places=9)

    def test_vol_lida_na_barreira(self):
        sorriso = lambda k, t: 0.20 + 0.3 * (1.0 - k / 130_000.0)  # noqa: E731
        e = EstadoMercado(ativos={"IBOV": MercadoAtivo(spot=130_000.0, vol=sorriso, dividendos=0.03)},
                          curvas={"BRL": CurvaPlana(0.13)})
        ot = OneTouch(ativo="IBOV", barreira=115_000.0, vencimento=0.5, direcao="down", **MC)
        md = _md(130_000.0, 0.13, 0.03, sorriso(115_000.0, 0.5))
        esperado = price_one_touch(md, BarrierContract(115_000.0, 115_000.0, 0.5), direction="down", payout=1.0,
                                   **MC_PRICER)
        self.assertEqual(ot.valor(e), esperado.price)


# ------------------------------------------------------------------------------------------- quanto

class TestQuanto(unittest.TestCase):
    def setUp(self):
        self.e = _estado_quanto()
        self.qm = _quanto_market()

    def test_vanilla_quanto_mesmo_numero(self):
        for tipo in ("call", "put"):
            for strike in (6_500.0, 7_000.0, 7_800.0):
                produto = Vanilla("SPX", strike, 1.5, tipo, quanto=Quanto("USDBRL", 5.40))
                esperado = price_quanto_vanilla(self.qm, strike, 1.5, option_type=tipo, fx_fixed=5.40)
                self.assertAlmostEqual(produto.valor(self.e), esperado, delta=1e-9 * esperado)

    def test_barreira_quanto_igual_ao_desk(self):
        # O Desk precifica a barreira Quanto com `quanto_market_data` e multiplica pelo câmbio fixo.
        produto = BarreiraSimples(ativo="SPX", strike=7_000.0, barreira=6_000.0, vencimento=1.0, rebate=50.0,
                                  quanto=Quanto("USDBRL", 5.40), **MC)
        esperado = price_single_barrier(quanto_market_data(self.qm), BarrierContract(7_000.0, 6_000.0, 1.0),
                                        rebate=50.0, **MC_PRICER)
        self.assertEqual(produto.valor(self.e), esperado.price * 5.40)
        self.assertEqual(produto.resultado(self.e).standard_error, esperado.standard_error * 5.40)

    def test_digital_quanto_contra_call_spread_quanto_estreito(self):
        h = 0.01
        spread = (price_quanto_vanilla(self.qm, 7_200.0 - h, 1.0, fx_fixed=1.0)
                  - price_quanto_vanilla(self.qm, 7_200.0 + h, 1.0, fx_fixed=1.0)) / (2 * h)
        digital = Digital("SPX", 7_200.0, 1.0, pagamento=1_000.0, quanto=Quanto("USDBRL"))
        self.assertAlmostEqual(digital.valor(self.e), 1_000.0 * spread, places=4)

    def test_correlacao_muda_o_preco(self):
        produto = Vanilla("SPX", 7_000.0, 1.0, quanto=Quanto("USDBRL", 5.40))
        negativa = produto.valor(_estado_quanto(rho=-0.5))
        positiva = produto.valor(_estado_quanto(rho=0.5))
        self.assertGreater(negativa, positiva)  # rho < 0 aumenta o drift do SPX em BRL

    def test_quanto_sem_cambio_fixo_em_payoff_em_pontos(self):
        with self.assertRaisesRegex(ValueError, "cambio_fixo"):
            Vanilla("SPX", 7_000.0, 1.0, quanto=Quanto("USDBRL"))

    def test_par_incompativel_com_a_moeda_do_ativo(self):
        e = EstadoMercado(ativos={"IBOV": MercadoAtivo(spot=130_000.0, vol=0.2)},
                          curvas={"BRL": CurvaPlana(0.13), "USD": CurvaPlana(0.04)},
                          cambios={"USDBRL": MercadoCambio(spot=5.4, vol=0.15)},
                          correlacoes={("IBOV", "USDBRL"): -0.5})
        with self.assertRaisesRegex(ValueError, "moeda base"):
            Vanilla("IBOV", 130_000.0, 1.0, quanto=Quanto("USDBRL", 5.4)).valor(e)


# ------------------------------------------------------------------------------------------ carteira

class TestCarteira(unittest.TestCase):
    def setUp(self):
        self.e = EstadoMercado.simples(spot=130_000.0, taxa=0.13, vol=0.22, dividendos=0.03)

    def test_adaptadores_sao_produtos(self):
        produtos = [Vanilla("IBOV", 1.0, 1.0), CallSpread("IBOV", 1.0, 2.0, 1.0), Digital("IBOV", 1.0, 1.0),
                    BarreiraSimples(ativo="IBOV", strike=1.0, barreira=1.0, vencimento=1.0),
                    OneTouch(ativo="IBOV", barreira=1.0, vencimento=1.0, direcao="up"),
                    NoTouch(ativo="IBOV", barreira=1.0, vencimento=1.0, direcao="up")]
        for p in produtos:
            self.assertIsInstance(p, Produto)
        self.assertIsInstance(produtos[0] + produtos[1], Produto)

    def test_aritmetica(self):
        c1, c2 = Vanilla("IBOV", 130_000.0, 1.0), Vanilla("IBOV", 156_000.0, 1.0)
        cs = CallSpread("IBOV", 130_000.0, 156_000.0, 1.0)
        self.assertAlmostEqual((c1 - c2).valor(self.e), cs.valor(self.e), places=8)
        self.assertAlmostEqual((0.8 * cs).valor(self.e), 0.8 * cs.valor(self.e), places=8)
        self.assertAlmostEqual((np.float64(0.8) * cs).valor(self.e), 0.8 * cs.valor(self.e), places=8)
        self.assertAlmostEqual((-cs).valor(self.e), -cs.valor(self.e), places=8)
        self.assertAlmostEqual(sum([c1, c2]).valor(self.e), c1.valor(self.e) + c2.valor(self.e), places=8)
        carteira = (c1 + c2) + (c1 - c2)
        self.assertIsInstance(carteira, Carteira)
        self.assertEqual(len(carteira.pernas), 4)
        self.assertAlmostEqual(carteira.valor(self.e), 2 * c1.valor(self.e), places=8)

    def test_vetorial_mistura_formula_e_monte_carlo(self):
        carteira = Vanilla("IBOV", 130_000.0, 1.0) + 2.0 * BarreiraSimples(
            ativo="IBOV", strike=130_000.0, barreira=115_000.0, vencimento=1.0, **MC)
        ev = self.e.com_spot("IBOV", np.array([125_000.0, 135_000.0]))
        np.testing.assert_allclose(carteira.valor(ev), [carteira.valor(c) for c in ev.cenarios()], atol=1e-8)

    def test_metadados_e_payoff(self):
        carteira = Vanilla("IBOV", 130_000.0, 2.0) - Digital("IBOV", 120_000.0, 1.0, pagamento=500.0) + \
            Vanilla("SPX", 7_000.0, 1.0)
        self.assertEqual(carteira.subjacentes(), ("IBOV", "SPX"))
        self.assertEqual([f.tempo for f in carteira.fluxos()], [1.0, 1.0, 2.0])
        self.assertEqual(carteira.vencimento, 2.0)
        finais = {"IBOV": np.array([110_000.0, 140_000.0]), "SPX": np.array([7_100.0, 6_000.0])}
        np.testing.assert_allclose(carteira.payoff(finais), [0.0 - 0.0 + 100.0, 10_000.0 - 500.0 + 0.0])
        with self.assertRaises(TypeError):
            (carteira + OneTouch(ativo="IBOV", barreira=1.0, vencimento=1.0, direcao="up")).payoff(finais)

    def test_objeto_que_nao_e_produto(self):
        with self.assertRaises(TypeError):
            Vanilla("IBOV", 1.0, 1.0) + "call"


# -------------------------------------------------------------------------------------------- gregas

class TestGregas(unittest.TestCase):
    def setUp(self):
        self.s, self.k, self.t, self.r, self.q, self.v = 130_000.0, 135_000.0, 1.0, 0.13, 0.03, 0.22
        self.e = EstadoMercado.simples(spot=self.s, taxa=self.r, vol=self.v, dividendos=self.q)

    def test_call_contra_formulas_analiticas(self):
        s, k, t, r, q, v = self.s, self.k, self.t, self.r, self.q, self.v
        d1 = (log(s / k) + (r - q + 0.5 * v**2) * t) / (v * sqrt(t))
        d2 = d1 - v * sqrt(t)
        g = gregas(Vanilla("IBOV", k, t), self.e)
        self.assertAlmostEqual(g.delta["IBOV"], exp(-q * t) * norm.cdf(d1), delta=1e-4)
        self.assertAlmostEqual(g.gamma["IBOV"] / (exp(-q * t) * norm.pdf(d1) / (s * v * sqrt(t))), 1.0, delta=1e-3)
        self.assertAlmostEqual(g.vega["IBOV"] / (s * exp(-q * t) * norm.pdf(d1) * sqrt(t) * 0.01), 1.0, delta=1e-3)
        self.assertAlmostEqual(g.dv01["BRL"] / (k * t * exp(-r * t) * norm.cdf(d2) * 1e-4), 1.0, delta=1e-3)
        esperado_theta = (black_scholes_call(_md(s, r, q, v), BarrierContract(k, k, t - 1 / 252))
                          - black_scholes_call(_md(s, r, q, v), BarrierContract(k, k, t)))
        self.assertAlmostEqual(g.theta, esperado_theta, places=8)

    def test_carteira_e_linear(self):
        c1, c2 = Vanilla("IBOV", 130_000.0, 1.0), Vanilla("IBOV", 150_000.0, 0.5, "put")
        g1, g2, g = gregas(c1, self.e), gregas(c2, self.e), gregas(2.0 * c1 - c2, self.e)
        for nome in ("delta", "gamma", "vega", "dv01"):
            for chave, valor in getattr(g, nome).items():
                combinado = 2.0 * getattr(g1, nome)[chave] - getattr(g2, nome)[chave]
                self.assertAlmostEqual(valor, combinado, delta=1e-6 * max(1.0, abs(combinado)))
        self.assertAlmostEqual(g.theta, 2.0 * g1.theta - g2.theta, places=6)

    def test_delta_monte_carlo_contra_reiner_rubinstein(self):
        # Números aleatórios comuns: o delta do Monte Carlo fica perto do analítico mesmo com poucos caminhos.
        s, k, h, t, r, q, v = self.s, 130_000.0, 115_000.0, 1.0, self.r, self.q, self.v
        produto = BarreiraSimples(ativo="IBOV", strike=k, barreira=h, vencimento=t, caminhos=40_000, passos=100)
        g = gregas(produto, self.e)
        passo = 0.01 * s
        analitico = (price_down_and_out_call_analytical(s + passo, k, h, t, r, q, v)
                     - price_down_and_out_call_analytical(s - passo, k, h, t, r, q, v)) / (2 * passo)
        self.assertAlmostEqual(g.delta["IBOV"], analitico, delta=0.02 * analitico)

    def test_quanto_tem_dv01_nas_duas_curvas(self):
        g = gregas(Vanilla("SPX", 7_000.0, 1.0, quanto=Quanto("USDBRL", 5.40)), _estado_quanto())
        self.assertEqual(set(g.dv01), {"USD", "BRL"})
        self.assertGreater(g.dv01["USD"], 0.0)   # r em USD sobe o drift do SPX
        self.assertLess(g.dv01["BRL"], 0.0)      # r em BRL só desconta (o drift não depende dele)
        self.assertIn("SPX", g.delta)

    def test_estado_vetorial_rejeitado(self):
        ev = self.e.com_spot("IBOV", np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            gregas(Vanilla("IBOV", 1.0, 1.0), ev)

    def test_como_dict(self):
        d = gregas(Vanilla("IBOV", self.k, self.t), self.e).como_dict()
        self.assertEqual(set(d), {"valor", "delta[IBOV]", "gamma[IBOV]", "vega[IBOV]", "dv01[BRL]", "theta"})


if __name__ == "__main__":
    unittest.main()
