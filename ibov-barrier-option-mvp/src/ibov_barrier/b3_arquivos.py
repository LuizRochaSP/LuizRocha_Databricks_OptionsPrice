"""Integridade dos ZIPs diários da B3 em ``data/AAAA-MM-DD/`` (usado pelo notebook 00).

Um download interrompido deixava um ZIP truncado (começa com "PK", mas sem o diretório central no
fim). A pasta parecia completa, o 00 pulava a data como "já baixado" e a validação morria em
``BadZipFile`` em toda nova tentativa. Agora:

- ``pasta_completa`` só aceita a pasta se cada ZIP esperado abre e passa no CRC;
- ``zips_invalidos`` diz quais precisam ser baixados de novo;
- ``gravar_atomico`` grava em ``.tmp`` e renomeia, para uma queda no meio não deixar arquivo pela metade.
"""
from __future__ import annotations

import io
import os
import zlib
from datetime import date
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def zip_integro(origem) -> bool:
    """Caminho ou bytes de um ZIP que abre e passa no teste de CRC de todos os membros."""
    try:
        alvo = io.BytesIO(origem) if isinstance(origem, (bytes, bytearray)) else origem
        with ZipFile(alvo) as z:
            return z.testzip() is None and len(z.namelist()) > 0
    except (BadZipFile, OSError, EOFError, ValueError, zlib.error):  # zlib.error: dado comprimido danificado
        return False


def nome_zip(prefixo: str, d: date) -> str:
    return f"{prefixo}{d:%y%m%d}.zip"


def zips_invalidos(pasta: Path, prefixos, d: date) -> list[str]:
    """Nomes dos ZIPs esperados que faltam ou estão corrompidos na pasta da data."""
    return [nome_zip(p, d) for p in prefixos if not zip_integro(Path(pasta) / nome_zip(p, d))]


def pasta_completa(pasta: Path, prefixos, d: date) -> bool:
    return Path(pasta).is_dir() and not zips_invalidos(pasta, prefixos, d)


def gravar_atomico(destino: Path, dados: bytes) -> None:
    destino = Path(destino)
    tmp = destino.with_name(destino.name + ".tmp")
    tmp.write_bytes(dados)
    os.replace(tmp, destino)
