"""Testes do COE como composição (etapa A3 do plano).

- O termsheet valida datas, níveis e percentuais (percentuais em fração: 100% = 1.0).
- O payoff escrito direto do termsheet é igual ao payoff das pernas (adaptadores).
- A soma das pernas é o valor do COE.
- Monte Carlo independente: o payoff simulado no vencimento, descontado, bate com o valor. A simulação só usa o
  payoff do termsheet e os dados de mercado (spot, DF, dividendos, vol), não usa os adaptadores nem Black-Scholes.
- A decomposição fecha: preço de emissão = título + opções + custos + reservas + margem.
"""
from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace
from datetime import date
from math import exp, log, sqrt

import numpy as np

from ibov_barrier.coe import COE, Participacao, TermsheetBase, Titulo
from ibov_barrier.produtos import (
    CallSpread,
    Carteira,
    CurvaDI,
    CurvaPlana,
    EstadoMercado,
    MercadoAtivo,
    gregas,
)
from tests.test_coe_titulo import INCLINADA, PLANA_14

HAS_CALENDARIO = importlib.util.find_spec("exchange_calendars") is not None

S0 = 130_000.0


def _ts(**campos) -> Participacao:
    base = dict(ativo="IBOV", nivel_inicial=S0, vencimento=1.0, nominal=1_000.0, participacao=0.8, cap=0.2)
    base.update(campos)
    return Participacao(**base)


def _estado(spot=S0, vol=0.22, dividendos=0.04, curva=None, spread_bp=0.0) -> EstadoMercado:
    curva = CurvaPlana(log(1.14)) if curva is None else curva
    return EstadoMercado(ativos={"IBOV": MercadoAtivo(spot=spot, vol=vol, dividendos=dividendos)},
                         curvas={"BRL": curva}, spread_emissor_bp=spread_bp)


VARIANTES = {
    "alta com cap": dict(),
    "alta sem cap": dict(cap=None),
    "alta com strike 105%": dict(strike=1.05, cap=0.3),
    "baixa com cap": dict(direcao="baixa", cap=0.25),
    "baixa sem cap": dict(direcao="baixa", cap=None, participacao=0.6),
    "proteção 95% e participação 130%": dict(protecao=0.95, participacao=1.3, cap=0.15),
}


class TestTermsheet(unittest.TestCase):
    def test_propriedades(self):
        ts = _ts()
        self.assertAlmostEqual(ts.quantidade, 1_000.0 * 0.8 / S0)
        self.assertEqual(ts.nivel_strike, S0)
        self.assertAlmostEqual(ts.nivel_cap, 1.2 * S0)
        self.assertAlmostEqual(ts.pagamento_maximo, 1.16)
        self.assertEqual(ts.dias_uteis, 252)
        self.assertIsNone(_ts(cap=None).pagamento_maximo)
        baixa = _ts(direcao="baixa", cap=0.25)
        self.assertAlmostEqual(baixa.nivel_cap, 0.75 * S0)
        self.assertAlmostEqual(baixa.pagamento_maximo, 1.0 + 0.8 * 0.25)
        self.assertAlmostEqual(_ts(direcao="baixa", cap=None).pagamento_maximo, 1.8)

    def test_descricao(self):
        self.assertEqual(_ts(vencimento=1.5).descricao(),
                         "COE IBOV: proteção 100%, participação 80% na alta, cap 20%, 378 DU")
        self.assertEqual(_ts(strike=1.05, cap=None, protecao=0.975).descricao(),
                         "COE IBOV: proteção 97,5%, participação 80% na alta, strike 105%, 252 DU")

    def test_percentual_digitado_como_numero_inteiro_e_recusado(self):
        for campos in (dict(protecao=100.0), dict(participacao=80.0), dict(cap=20.0), dict(strike=100.0)):
            with self.subTest(**campos), self.assertRaisesRegex(ValueError, "100% = 1.0"):
                _ts(**campos)

    def test_validacao(self):
        invalidos = [
            dict(ativo=""), dict(nivel_inicial=0.0), dict(vencimento=0.0), dict(nominal=-1.0),
            dict(protecao=-0.1), dict(participacao=-0.5), dict(cap=0.0), dict(direcao="lateral"),
            dict(strike=1.25, cap=0.2),                  # cap abaixo do strike
            dict(direcao="baixa", cap=1.0),              # queda de 100% não é piso
            dict(direcao="baixa", strike=0.7, cap=0.3),  # piso acima do strike
            dict(data_emissao=date(2026, 11, 25)),       # só uma data
            dict(data_emissao=date(2026, 11, 25), data_vencimento=date(2026, 11, 25)),
        ]
        for campos in invalidos:
            with self.subTest(**campos), self.assertRaises(ValueError):
                _ts(**campos)

    def test_participacao_zero_e_so_o_titulo(self):
        coe = COE(_ts(participacao=0.0))
        e = _estado(spread_bp=80.0)
        self.assertAlmostEqual(coe.valor(e), coe.titulo().valor(e), places=10)

    def test_termsheet_imutavel_e_substituivel(self):
        ts = _ts()
        with self.assertRaises(Exception):
            ts.participacao = 0.9
        self.assertEqual(replace(ts, participacao=0.9).participacao, 0.9)

    def test_base_sem_estrutura(self):
        coe = COE(TermsheetBase(ativo="IBOV", nivel_inicial=S0, vencimento=1.0))
        with self.assertRaises(NotImplementedError):
            coe.valor(_estado())
        with self.assertRaises(TypeError):
            COE(object())

    @unittest.skipUnless(HAS_CALENDARIO, "exchange_calendars ausente")
    def test_por_datas_conta_dias_uteis_da_b3(self):
        from ibov_barrier.di_curve.conventions import business_days

        emissao, vencimento = date(2026, 11, 25), date(2028, 11, 27)
        ts = Participacao.por_datas(data_emissao=emissao, data_vencimento=vencimento, ativo="IBOV",
                                    nivel_inicial=S0, participacao=0.9, cap=0.3)
        du = business_days(emissao, vencimento)
        self.assertEqual(ts.dias_uteis, du)
        self.assertAlmostEqual(ts.vencimento, du / 252)
        self.assertEqual((ts.data_emissao, ts.data_vencimento), (emissao, vencimento))
        self.assertTrue(495 <= du <= 510)
        with self.assertRaises(ValueError):
            Participacao.por_datas(data_emissao=vencimento, data_vencimento=emissao, ativo="IBOV", nivel_inicial=S0)


class TestPayoff(unittest.TestCase):
    def test_pontos_conhecidos_alta_com_cap(self):
        coe = COE(_ts())
        finais = {"IBOV": np.array([100_000.0, 130_000.0, 143_000.0, 156_000.0, 200_000.0])}
        np.testing.assert_allclose(coe.payoff(finais), [1_000.0, 1_000.0, 1_080.0, 1_160.0, 1_160.0])

    def test_pontos_conhecidos_baixa_e_protecao_parcial(self):
        coe = COE(_ts(direcao="baixa", cap=0.25, protecao=0.95))
        finais = {"IBOV": np.array([200_000.0, 130_000.0, 117_000.0, 97_500.0, 50_000.0])}
        np.testing.assert_allclose(coe.payoff(finais), [950.0, 950.0, 1_030.0, 1_150.0, 1_150.0])

    def test_sem_cap_e_com_strike(self):
        finais = {"IBOV": np.array([130_000.0, 136_500.0, 260_000.0])}
        np.testing.assert_allclose(COE(_ts(cap=None)).payoff(finais), [1_000.0, 1_040.0, 1_800.0])
        np.testing.assert_allclose(COE(_ts(cap=None, strike=1.05)).payoff(finais), [1_000.0, 1_000.0, 1_760.0])

    def test_payoff_do_termsheet_igual_ao_das_pernas(self):
        finais = {"IBOV": np.linspace(40_000.0, 260_000.0, 441)}
        for nome, campos in VARIANTES.items():
            with self.subTest(nome):
                coe = COE(_ts(**campos))
                np.testing.assert_allclose(coe.payoff(finais), coe.pernas().payoff(finais), rtol=1e-13, atol=1e-9)


class TestComposicao(unittest.TestCase):
    @staticmethod
    def _smile(strike, prazo):
        return 0.20 + 0.15 * (S0 / strike - 1.0) ** 2 + 0.05 * max(0.0, S0 / strike - 1.0)

    def test_soma_das_pernas_e_o_valor(self):
        e = _estado(curva=CurvaDI(INCLINADA), spread_bp=110.0)
        e = replace(e, ativos={"IBOV": MercadoAtivo(spot=133_000.0, vol=self._smile, dividendos=0.04)})
        for nome, campos in VARIANTES.items():
            with self.subTest(nome):
                coe = COE(_ts(vencimento=1.5, **campos))
                pernas = float(coe.titulo().valor(e)) + float(coe.opcoes().valor(e))
                self.assertAlmostEqual(coe.valor(e), pernas, places=8)

    def test_pernas_no_mesmo_numero_dos_produtos_soltos(self):
        e = _estado(curva=CurvaDI(INCLINADA), spread_bp=110.0)
        ts = _ts(nominal=5_000_000.0, protecao=0.97)
        coe = COE(ts)
        titulo = Titulo(vencimento=1.0, principal=5_000_000.0 * 0.97)
        cs = CallSpread("IBOV", S0, 1.2 * S0, 1.0)
        self.assertAlmostEqual(coe.titulo().valor(e), titulo.valor(e), places=6)
        self.assertAlmostEqual(coe.opcoes().valor(e), ts.quantidade * cs.valor(e), places=6)

    def test_vetorial_igual_ao_escalar(self):
        coe = COE(_ts(cap=0.25))
        e = _estado(spot=np.array([110_000.0, 130_000.0, 150_000.0]), spread_bp=np.array([50.0, 100.0, 150.0]))
        np.testing.assert_allclose(coe.valor(e), [coe.valor(c) for c in e.cenarios()], rtol=1e-13)

    def test_linear_no_nominal(self):
        e = _estado(spread_bp=90.0)
        self.assertAlmostEqual(COE(_ts(nominal=2_500.0)).valor(e) / COE(_ts()).valor(e), 2.5, places=12)

    def test_gregas_por_perna(self):
        e = _estado(curva=CurvaDI(INCLINADA), spread_bp=100.0)
        coe = COE(_ts())
        g, g_titulo, g_opcoes = gregas(coe, e), gregas(coe.titulo(), e), gregas(coe.opcoes(), e)
        self.assertAlmostEqual(g.cs01, g_titulo.cs01, places=10)
        self.assertIsNone(g_opcoes.cs01)
        self.assertAlmostEqual(g.delta["IBOV"], g_opcoes.delta["IBOV"], places=12)
        self.assertAlmostEqual(g.vega["IBOV"], g_opcoes.vega["IBOV"], places=10)
        self.assertAlmostEqual(g.dv01["BRL"], g_titulo.dv01["BRL"] + g_opcoes.dv01["BRL"], places=9)

    def test_livro_de_coes(self):
        e = _estado(spread_bp=100.0)
        a, b = COE(_ts()), COE(_ts(direcao="baixa", cap=0.3, nominal=3_000.0))
        livro = a + 2.0 * b
        self.assertIsInstance(livro, Carteira)
        self.assertTrue(livro.depende_do_spread)
        self.assertAlmostEqual(livro.valor(e), a.valor(e) + 2.0 * b.valor(e), places=9)
        self.assertIsNotNone(gregas(livro, e).cs01)

    def test_metadados(self):
        coe = COE(_ts(protecao=0.95))
        self.assertEqual(coe.subjacentes(), ("IBOV",))
        self.assertEqual(coe.vencimento, 1.0)
        principal, rendimento = coe.fluxos()
        self.assertEqual((principal.tempo, principal.valor, principal.moeda), (1.0, 950.0, "BRL"))
        self.assertIsNone(rendimento.valor)
        self.assertIn("cap 20%", rendimento.descricao)
        (obs,) = coe.observacoes()
        self.assertEqual((obs.tipo, obs.nivel, obs.ativo), ("vencimento", S0, "IBOV"))

    def test_vencido_vale_o_payoff(self):
        coe = COE(_ts())
        e = _estado(spot=150_000.0, spread_bp=100.0).com_tempo(1.0)
        self.assertAlmostEqual(coe.valor(e), float(coe.payoff({"IBOV": np.array([150_000.0])})[0]), places=9)


def _monte_carlo(coe: COE, estado: EstadoMercado, n: int = 1_000_000, semente: int = 2026) -> tuple[float, float]:
    """Valor por simulação do S_T log-normal na medida neutra ao risco, com variáveis antitéticas.

    Só usa dados de mercado e o payoff do termsheet. O principal protegido é desconto na curva do emissor (DI pré +
    spread, em taxa efetiva); o resto do payoff, na curva livre de risco. Devolve (valor, erro-padrão).
    """
    ts = coe.termsheet
    mercado = estado.ativo(ts.ativo)
    t = ts.vencimento
    df_livre = float(estado.curva(ts.moeda).fator_desconto(t))
    r = -log(df_livre) / t
    df_emissor = ((df_livre ** (-1.0 / t)) + float(estado.spread_emissor_bp) * 1e-4) ** (-t)
    q, sigma = mercado.dividendos_em(t), float(mercado.vol)
    z = np.random.default_rng(semente).standard_normal(n)
    deriva = (r - q - 0.5 * sigma**2) * t
    protegido = ts.nominal * ts.protecao
    pares = 0.5 * sum(coe.payoff({ts.ativo: float(mercado.spot) * np.exp(deriva + sinal * sigma * sqrt(t) * z)})
                      - protegido for sinal in (1.0, -1.0))
    valor = protegido * df_emissor + df_livre * pares.mean()
    return valor, df_livre * pares.std(ddof=1) / sqrt(n)


class TestMonteCarloIndependente(unittest.TestCase):
    def _confere(self, coe, e):
        mc, erro = _monte_carlo(coe, e)
        valor = float(coe.valor(e))
        self.assertLess(abs(mc - valor), 4.0 * erro + 1e-9, f"MC {mc:.4f} ± {erro:.4f} contra {valor:.4f}")
        self.assertLess(erro / coe.termsheet.nominal, 2e-4)

    def test_spread_zero_payoff_descontado_na_curva_plana(self):
        e = _estado(spot=133_000.0)
        for nome, campos in VARIANTES.items():
            with self.subTest(nome):
                self._confere(COE(_ts(**campos)), e)

    def test_curva_di_e_spread_do_emissor(self):
        e = _estado(curva=CurvaDI(INCLINADA), spread_bp=125.0, vol=0.25, dividendos=0.035)
        for nome, campos in VARIANTES.items():
            with self.subTest(nome):
                self._confere(COE(_ts(vencimento=2.0, **campos)), e)

    def test_spread_so_move_o_titulo(self):
        coe = COE(_ts(protecao=0.98))
        e0, e1 = _estado(curva=CurvaDI(PLANA_14)), _estado(curva=CurvaDI(PLANA_14), spread_bp=150.0)
        esperado = 1_000.0 * 0.98 * (1.155**-1.0 - 1.14**-1.0)
        self.assertAlmostEqual(coe.valor(e1) - coe.valor(e0), esperado, places=9)


class TestDecomposicao(unittest.TestCase):
    def setUp(self):
        self.e = _estado(spread_bp=100.0)
        self.coe = COE(_ts(nominal=1_000_000.0, participacao=1.0))
        self.d = self.coe.decompor(self.e, custo_distribuicao=0.03, custo_operacional=0.005, reservas=0.005)

    def test_cascata_fecha(self):
        d = self.d
        self.assertAlmostEqual(d.titulo + d.opcoes + d.custos + d.reservas + d.margem, d.preco_emissao, places=6)
        p = d.percentuais()
        self.assertAlmostEqual(sum(v for k, v in p.items() if k != "preco_emissao"), p["preco_emissao"], places=12)
        self.assertEqual([r for r, _, _ in d.linhas()],
                         ["Preço de emissão", "Título (proteção)", "Opções", "Custos de distribuição",
                          "Custos operacionais", "Reservas", "Margem"])
        for _, reais, fracao in d.linhas():
            self.assertAlmostEqual(fracao, reais / 1_000_000.0, places=14)

    def test_numeros_conhecidos(self):
        d = self.d
        self.assertAlmostEqual(d.titulo, 1_000_000.0 / 1.15, places=4)
        self.assertAlmostEqual(d.taxa_emissor, 0.15, places=12)
        self.assertEqual(d.spread_emissor_bp, 100.0)
        self.assertAlmostEqual(d.opcoes, float(self.coe.opcoes().valor(self.e)), places=6)
        self.assertAlmostEqual(d.valor_justo, float(self.coe.valor(self.e)), places=6)
        self.assertAlmostEqual(d.custos, 35_000.0)
        self.assertAlmostEqual(d.reservas, 5_000.0)

    def test_margem_sobe_com_o_spread_e_cai_com_a_participacao(self):
        margens = [self.coe.decompor(self.e.com_spread(s)).margem for s in (0.0, 100.0, 200.0)]
        self.assertEqual(margens, sorted(margens))
        cs = CallSpread("IBOV", S0, 1.2 * S0, 1.0).valor(self.e)
        m1 = COE(_ts(nominal=1_000_000.0, participacao=0.9)).decompor(self.e).margem
        m2 = COE(_ts(nominal=1_000_000.0, participacao=1.0)).decompor(self.e).margem
        self.assertAlmostEqual(m1 - m2, 0.1 * 1_000_000.0 / S0 * cs, places=6)

    def test_viabilidade(self):
        self.assertTrue(self.d.viavel)
        sem_cap = COE(_ts(nominal=1_000_000.0, participacao=1.0, cap=None)).decompor(
            self.e, custo_distribuicao=0.03, custo_operacional=0.005, reservas=0.005)
        self.assertFalse(sem_cap.viavel)
        self.assertLess(sem_cap.margem, 0.0)

    def test_percentuais_nao_dependem_do_nominal(self):
        outro = COE(_ts(nominal=1_000.0, participacao=1.0)).decompor(
            self.e, custo_distribuicao=0.03, custo_operacional=0.005, reservas=0.005)
        for chave, valor in self.d.percentuais().items():
            self.assertAlmostEqual(outro.percentuais()[chave], valor, places=12)

    def test_como_dict(self):
        dados = self.d.como_dict()
        for chave in ("titulo", "titulo_pct", "opcoes_pct", "margem", "margem_pct", "valor_justo", "taxa_emissor"):
            self.assertIn(chave, dados)
        self.assertAlmostEqual(dados["margem_pct"], self.d.margem / 1_000_000.0, places=14)

    def test_validacao(self):
        for campos in (dict(custo_distribuicao=3.0), dict(reservas=-0.01), dict(preco_emissao=0.0),
                       dict(custo_operacional=float("nan"))):
            with self.subTest(**campos), self.assertRaises(ValueError):
                self.coe.decompor(self.e, **campos)
        with self.assertRaises(ValueError):
            self.coe.decompor(self.e.com_spread(np.array([50.0, 100.0])))


if __name__ == "__main__":
    unittest.main()
