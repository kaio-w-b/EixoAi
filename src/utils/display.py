"""
utils/display.py
────────────────
Helpers de output para terminal: cores ANSI e funções de formatação.
Sem dependências internas — pode ser importado por qualquer módulo do projeto.
"""

# ── Cores ANSI ────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Formatadores ─────────────────────────────────────────
def _ok(msg: str)    -> str: return f"{GREEN}✔ {msg}{RESET}"
def _fail(msg: str)  -> str: return f"{RED}✗ {msg}{RESET}"
def _warn(msg: str)  -> str: return f"{YELLOW}⚠ {msg}{RESET}"
def _info(msg: str)  -> str: return f"{BLUE}ℹ {msg}{RESET}"
def _title(msg: str) -> str: return f"\n{BOLD}{CYAN}{'═'*60}\n  {msg}\n{'═'*60}{RESET}"