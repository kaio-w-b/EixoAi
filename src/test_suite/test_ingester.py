"""
tests/test_ingester.py
──────────────────────
Testes unitários para src/ingester.py.
Cobre: extração de texto, paginação, múltiplos PDFs e tratamento de erros.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

from test_suite.conftest import _title
from test_suite.runner import TestRunner
from test_suite.conftest import make_pdf

from ingester import (
    extract_text_from_pdf,
    extract_text_from_pdf_by_page,
    extract_text_from_multiple_pdfs,
)

def test_ingester() -> Tuple[int, int]:
    print(_title("MÓDULO: ingester.py"))


    runner = TestRunner()

    # ── Casos de erro ────────────────────────────────────────

    def t_file_not_found():
        """Arquivo inexistente deve levantar FileNotFoundError."""
        try:
            extract_text_from_pdf("/tmp/nao_existe_xyzabc.pdf")
            assert False, "Deveria lançar FileNotFoundError"
        except FileNotFoundError:
            pass

    def t_not_pdf():
        """Arquivo com extensão errada deve levantar ValueError."""
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.write(b"isso nao e pdf")
        tmp.close()
        try:
            extract_text_from_pdf(tmp.name)
            assert False, "Deveria lançar ValueError"
        except ValueError:
            pass
        finally:
            os.unlink(tmp.name)

    def t_invalid_dir():
        """Diretório inexistente deve levantar ValueError."""
        try:
            extract_text_from_multiple_pdfs("/tmp/dir_nao_existe_xyz")
            assert False, "Deveria lançar ValueError"
        except ValueError:
            pass

    # ── Casos de sucesso ─────────────────────────────────────

    def t_valid_pdf():
        """PDF válido deve retornar string (possivelmente vazia para PDF raw)."""
        pdf_path = make_pdf("Conteúdo de teste para extração.")
        try:
            text = extract_text_from_pdf(pdf_path)
            assert isinstance(text, str), "Resultado deve ser string"
        finally:
            os.unlink(pdf_path)

    def t_by_page():
        """extract_by_page deve retornar lista de dicts com 'page' e 'text'."""
        pdf_path = make_pdf("Página de teste.")
        try:
            pages = extract_text_from_pdf_by_page(pdf_path)
            assert isinstance(pages, list), "Deve retornar lista"
            if pages:
                assert "page" in pages[0], "Deve ter chave 'page'"
                assert "text" in pages[0], "Deve ter chave 'text'"
        finally:
            os.unlink(pdf_path)

    def t_empty_dir():
        """Diretório vazio deve retornar dict vazio."""
        with tempfile.TemporaryDirectory() as d:
            result = extract_text_from_multiple_pdfs(d)
            assert isinstance(result, dict), "Deve retornar dict"
            assert len(result) == 0, "Deve ser vazio"

    def t_multiple_pdfs():
        """Diretório com 3 PDFs deve retornar dict com 3 entradas."""
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                src = make_pdf(f"Documento número {i}")
                shutil.copy(src, Path(d) / f"doc_{i}.pdf")
                os.unlink(src)
            result = extract_text_from_multiple_pdfs(d)
            assert isinstance(result, dict), "Deve retornar dict"
            assert len(result) == 3, f"Deve ter 3 entradas (tem {len(result)})"

    # ── Registro ─────────────────────────────────────────────

    runner.run("FileNotFoundError em arquivo inexistente", t_file_not_found)
    runner.run("ValueError em arquivo não-PDF",           t_not_pdf)
    runner.run("ValueError em diretório inexistente",     t_invalid_dir)
    runner.run("Extração de PDF válido retorna string",   t_valid_pdf)
    runner.run("extract_by_page retorna lista de dicts",  t_by_page)
    runner.run("Diretório vazio retorna dict vazio",      t_empty_dir)
    runner.run("Processamento de múltiplos PDFs",         t_multiple_pdfs)

    return runner.summary()