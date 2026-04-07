"""
test_suite/runner.py
────────────────────
TestResult  — dataclass com resultado de um caso de teste.
TestRunner  — executa funções de teste, captura erros e agrega resultados.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from utils.display import (
    GREEN, YELLOW, RED, BOLD, DIM, RESET,
    _ok, _fail, _info,
)


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    details: Dict = field(default_factory=dict)


class TestRunner:
    """
    Executa casos de teste individuais e agrega os resultados.

    Uso:
        runner = TestRunner()
        runner.run("Nome do teste", minha_funcao)
        passed, total = runner.summary()
    """

    def __init__(self):
        self.results: List[TestResult] = []

    def run(self, name: str, fn: Callable, *args, **kwargs) -> TestResult:
        start = time.perf_counter()
        try:
            fn(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            r = TestResult(name=name, passed=True, duration_ms=duration)
            print(_ok(f"{name}  ({duration:.1f}ms)"))
        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000
            r = TestResult(name=name, passed=False, duration_ms=duration, message=str(e))
            print(_fail(f"{name}  ({duration:.1f}ms)"))
            print(f"   {DIM}{e}{RESET}")
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            r = TestResult(name=name, passed=False, duration_ms=duration, message=str(e))
            print(_fail(f"{name}  ({duration:.1f}ms) — {type(e).__name__}: {e}"))
        self.results.append(r)
        return r

    def summary(self) -> Tuple[int, int]:
        passed = sum(1 for r in self.results if r.passed)
        total  = len(self.results)
        color  = GREEN if passed == total else (YELLOW if passed > 0 else RED)
        print(f"\n{color}{BOLD}Resultado: {passed}/{total} passaram{RESET}")
        if passed < total:
            print(_info("Falhas:"))
            for r in self.results:
                if not r.passed:
                    print(f"   • {r.name}: {r.message}")
        return passed, total