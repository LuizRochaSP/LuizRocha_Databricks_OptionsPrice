import unittest
from datetime import date

from renda_fixa.arredondamento import arredondar, truncar
from renda_fixa.calendario import (
    dia_util_anterior,
    dias_uteis,
    eh_dia_util,
    feriados_nacionais,
    pascoa,
    proximo_dia_util,
    soma_dias_uteis,
    soma_meses,
)


class TestCalendario(unittest.TestCase):
    def test_pascoa(self):
        self.assertEqual(pascoa(2024), date(2024, 3, 31))
        self.assertEqual(pascoa(2025), date(2025, 4, 20))
        self.assertEqual(pascoa(2026), date(2026, 4, 5))
        self.assertEqual(pascoa(2027), date(2027, 3, 28))

    def test_feriados_2026(self):
        f = set(feriados_nacionais(2026))
        for d in [
            date(2026, 1, 1),
            date(2026, 2, 16),
            date(2026, 2, 17),
            date(2026, 4, 3),
            date(2026, 4, 21),
            date(2026, 5, 1),
            date(2026, 6, 4),
            date(2026, 9, 7),
            date(2026, 10, 12),
            date(2026, 11, 2),
            date(2026, 11, 15),
            date(2026, 11, 20),
            date(2026, 12, 25),
        ]:
            self.assertIn(d, f)
        self.assertNotIn(date(2023, 11, 20), set(feriados_nacionais(2023)))

    def test_dias_uteis(self):
        self.assertEqual(dias_uteis(date(2026, 11, 19), date(2026, 11, 23)), 1)  # 20/11 feriado
        self.assertEqual(dias_uteis(date(2026, 11, 18), date(2026, 11, 19)), 1)
        self.assertEqual(dias_uteis(date(2026, 11, 19), date(2026, 11, 19)), 0)
        self.assertEqual(dias_uteis(date(2026, 11, 23), date(2026, 11, 19)), -1)
        # vencimento em feriado conta como no próximo dia útil
        self.assertEqual(dias_uteis(date(2026, 11, 19), date(2027, 1, 1)), dias_uteis(date(2026, 11, 19), date(2027, 1, 4)))

    def test_navegacao(self):
        self.assertFalse(eh_dia_util(date(2026, 11, 20)))
        self.assertTrue(eh_dia_util(date(2026, 11, 19)))
        self.assertEqual(proximo_dia_util(date(2026, 11, 20)), date(2026, 11, 23))
        self.assertEqual(dia_util_anterior(date(2026, 11, 23)), date(2026, 11, 19))
        self.assertEqual(soma_dias_uteis(date(2026, 11, 19), 1), date(2026, 11, 23))
        self.assertEqual(soma_dias_uteis(date(2026, 11, 23), -1), date(2026, 11, 19))
        self.assertEqual(soma_meses(date(2026, 8, 31), 6), date(2027, 2, 28))
        self.assertEqual(soma_meses(date(2027, 5, 15), -6), date(2026, 11, 15))

    def test_fora_do_calendario(self):
        with self.assertRaises(ValueError):
            dias_uteis(date(1980, 1, 1), date(2026, 1, 1))


class TestArredondamento(unittest.TestCase):
    def test_truncar(self):
        self.assertEqual(truncar(955.0380919999, 6), 955.038091)
        self.assertEqual(truncar(955.038091, 6), 955.038091)
        self.assertEqual(truncar(99.99999, 4), 99.9999)
        self.assertEqual(truncar(-1.23456, 2), -1.23)

    def test_arredondar(self):
        self.assertEqual(arredondar(1.23455, 4), 1.2346)
        self.assertEqual(arredondar(2.5, 0), 3.0)
        self.assertEqual(arredondar(1.0000000149, 8), 1.00000001)


if __name__ == "__main__":
    unittest.main()
