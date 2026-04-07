"""
test_suite/conftest.py
──────────────────────
Utilitários compartilhados pelas suítes de teste:
  • Re-exporta helpers de output de utils.display
  • Carregamento do .env
  • Constante FALLBACK_TEXT (CF/88 resumido)
  • Fábrica de PDFs mínimos para testes
  • get_sample_pdf_path() — lê EVAL_PDF_PATH do ambiente
"""

import os
import logging
import tempfile
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s"
)

# ── Re-exportar helpers de display ───────────────────────
from utils.display import (
    GREEN, RED, YELLOW, BLUE, CYAN, BOLD, DIM, RESET,
    _ok, _fail, _warn, _info, _title,
)

# ── Carregamento do .env ──────────────────────────────────

def _load_env() -> None:
    """Carrega o .env da raiz do projeto (dois níveis acima de test_suite/)."""
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
    else:
        load_dotenv()

_load_env()

# ── PDF de amostra configurado pelo usuário ──────────────

def get_sample_pdf_path() -> str | None:
    """Retorna EVAL_PDF_PATH do .env, ou None se não definido/inexistente."""
    path = os.getenv("EVAL_PDF_PATH", "").strip()
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(_warn(f"EVAL_PDF_PATH definido mas arquivo não encontrado: {path}"))
        return None
    return str(p)

# ── Texto fixture (CF/88 resumido) ────────────────────────

FALLBACK_TEXT = """
Constituição da República Federativa do Brasil de 1988

Art. 1º A República Federativa do Brasil, formada pela união indissolúvel dos Estados
e Municípios e do Distrito Federal, constitui-se em Estado Democrático de Direito e
tem como fundamentos: I - a soberania; II - a cidadania; III - a dignidade da pessoa humana;
IV - os valores sociais do trabalho e da livre iniciativa; V - o pluralismo político.

Art. 5º Todos são iguais perante a lei, sem distinção de qualquer natureza, garantindo-se
aos brasileiros e aos estrangeiros residentes no País a inviolabilidade do direito à vida,
à liberdade, à igualdade, à segurança e à propriedade.

Art. 6º São direitos sociais a educação, a saúde, a alimentação, o trabalho, a moradia,
o transporte, o lazer, a segurança, a previdência social, a proteção à maternidade e à
infância, a assistência aos desamparados, na forma desta Constituição.
"""

# ── Fábrica de PDFs mínimos ───────────────────────────────

def make_pdf(text: str) -> str:
    """
    Cria um PDF temporário e retorna o caminho.
    Usa reportlab se disponível; caso contrário gera PDF raw mínimo.
    O chamador é responsável por deletar o arquivo após o uso.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        c = canvas.Canvas(tmp.name, pagesize=letter)
        y = 700
        for line in text.split("\n"):
            if y < 50:
                c.showPage()
                y = 700
            c.drawString(50, y, line[:90])
            y -= 15
        c.save()
        return tmp.name
    except ImportError:
        raw = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Test PDF Content) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000368 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(raw)
        tmp.close()
        return tmp.name


def make_fallback_pdf() -> str:
    """Cria PDF com o texto fallback da CF/88."""
    return make_pdf(FALLBACK_TEXT)