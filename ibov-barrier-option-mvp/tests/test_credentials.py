"""Resolução da chave ThetaData: ordem das fontes, limpeza e nunca vazar o valor."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from ibov_barrier.markets import credentials as cred

FAKE = "td1_prod_00000000000000000000000000000000"


class CredentialTests(unittest.TestCase):
    def env(self, **values):
        clean = {k: v for k, v in os.environ.items()
                 if k not in (cred.ENV_VAR, cred.FILE_ENV_VAR, "DATABRICKS_RUNTIME_VERSION")}
        clean.update(values)
        return mock.patch.dict(os.environ, clean, clear=True)

    def test_argument_wins(self):
        with self.env(THETADATA_API_KEY="td1_prod_env_key_xxxxxxxxxxxx"):
            self.assertEqual(cred.resolve_thetadata_key(FAKE).source, "argumento")

    def test_env_before_file(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "k.txt"
            f.write_text("td1_prod_file_key_xxxxxxxxxxx", encoding="utf-8")
            with self.env(THETADATA_API_KEY=FAKE, THETADATA_KEY_FILE=str(f)):
                r = cred.resolve_thetadata_key()
        self.assertEqual((r.value, r.source), (FAKE, "variável de ambiente THETADATA_API_KEY"))

    def test_file_trimmed(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "k.txt"
            f.write_text(f'  "{FAKE}"\r\n', encoding="utf-8")
            with self.env(THETADATA_KEY_FILE=str(f)):
                r = cred.resolve_thetadata_key()
        self.assertEqual(r.value, FAKE)
        self.assertTrue(r.source.startswith("arquivo"))

    def test_databricks_secret_only_inside_databricks(self):
        with self.env(), mock.patch.object(cred, "DEFAULT_KEY_FILE", Path("nao_existe/tdk.txt")):
            with self.assertRaises(RuntimeError):
                cred.resolve_thetadata_key()
        with self.env(DATABRICKS_RUNTIME_VERSION="15.4"), \
                mock.patch.object(cred, "_databricks_secret", return_value=FAKE):
            self.assertIn("Databricks Secret", cred.resolve_thetadata_key().source)

    def test_internal_whitespace_rejected(self):
        with self.assertRaises(ValueError):
            cred.resolve_thetadata_key("td1_prod_abc def")

    def test_missing_key_message_and_no_leak(self):
        with self.env(), mock.patch.object(cred, "DEFAULT_KEY_FILE", Path("nao_existe/tdk.txt")):
            with self.assertRaisesRegex(RuntimeError, "set_thetadata_key"):
                cred.resolve_thetadata_key()
        r = cred.ResolvedKey(FAKE, "argumento")
        self.assertNotIn(FAKE, repr(r))
        self.assertNotIn(FAKE, cred.mask(FAKE))


if __name__ == "__main__":
    unittest.main()
