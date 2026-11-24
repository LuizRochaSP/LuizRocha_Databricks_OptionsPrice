"""ZIPs da B3 truncados não podem passar por "já baixado" (falha do job em 24/11/2026: PR261123.zip)."""
import io
import tempfile
import unittest
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ibov_barrier.b3_arquivos import gravar_atomico, pasta_completa, zip_integro, zips_invalidos

D = date(2026, 11, 23)
PREFIXOS = ("IN", "IR", "PR", "SPRD")


def zip_bytes(nome="x.xml", conteudo=b"<a/>" * 5000) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as z:
        z.writestr(nome, conteudo)
    return buf.getvalue()


class B3ArquivosTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self.tmp.name) / D.isoformat()
        self.pasta.mkdir()
        for p in PREFIXOS:
            (self.pasta / f"{p}261123.zip").write_bytes(zip_bytes())

    def tearDown(self):
        self.tmp.cleanup()

    def test_pasta_integra(self):
        self.assertTrue(pasta_completa(self.pasta, PREFIXOS, D))
        self.assertEqual(zips_invalidos(self.pasta, PREFIXOS, D), [])

    def test_zip_truncado_nao_conta_como_baixado(self):
        pr = self.pasta / "PR261123.zip"
        dados = pr.read_bytes()
        pr.write_bytes(dados[: len(dados) // 2])  # começa com PK, sem diretório central no fim
        self.assertTrue(pr.read_bytes().startswith(b"PK"))
        self.assertFalse(zip_integro(pr))
        self.assertFalse(pasta_completa(self.pasta, PREFIXOS, D))
        self.assertEqual(zips_invalidos(self.pasta, PREFIXOS, D), ["PR261123.zip"])

    def test_crc_errado_e_arquivo_faltando(self):
        dados = bytearray(zip_bytes(conteudo=b"abc" * 100))
        i = dados.find(b"PK\x03\x04") + 30 + len("x.xml")  # primeiro byte comprimido
        dados[i] ^= 0xFF
        (self.pasta / "IR261123.zip").write_bytes(bytes(dados))
        (self.pasta / "SPRD261123.zip").unlink()
        self.assertEqual(zips_invalidos(self.pasta, PREFIXOS, D), ["IR261123.zip", "SPRD261123.zip"])

    def test_bytes_e_pasta_inexistente(self):
        self.assertTrue(zip_integro(zip_bytes()))
        self.assertFalse(zip_integro(b"PK\x03\x04 lixo"))
        self.assertFalse(zip_integro(b""))
        self.assertFalse(pasta_completa(self.pasta.parent / "2026-11-24", PREFIXOS, D))

    def test_gravacao_atomica(self):
        destino = self.pasta / "PR261123.zip"
        novo = zip_bytes(conteudo=b"<b/>" * 10)
        gravar_atomico(destino, novo)
        self.assertEqual(destino.read_bytes(), novo)
        self.assertFalse(list(self.pasta.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
