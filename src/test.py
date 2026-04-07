"""
test.py — Entrypoint principal da suite de testes EixoAI
=========================================================
Delega toda a lógica para os submódulos em tests/ e eval/.

Uso:
    python test.py                          # menu interativo
    python test.py --all                    # todos os testes unitários
    python test.py --module ingester        # módulo específico
    python test.py --eval doc.pdf           # avaliação RAG com PDF
    python test.py --eval doc.pdf --answers # idem + respostas da LLM
    python test.py --sample                 # avaliação com PDF do .env (CF/88)
    python test.py --sample --output r.json # idem + salva relatório
"""

import sys
import argparse
from pathlib import Path
import os

# Garantir que src/ está no path (necessário ao rodar como script direto)
sys.path.insert(0, str(Path(__file__).parent))

from test_suite.conftest import (
    _title, _info, _warn, _fail,
    GREEN, RED, BOLD, RESET,
    get_sample_pdf_path, make_fallback_pdf,
)
from test_suite.test_ingester    import test_ingester
from test_suite.test_retriever   import test_retriever
from test_suite.test_llm_chain   import test_llm_chain
from test_suite.test_integration import test_integration
from eval.qa_pairs          import CF88_QA_PAIRS
from eval.evaluator         import RAGEvaluator


# ─────────────────────────────────────────────────────────
# Agregador
# ─────────────────────────────────────────────────────────

def run_all_tests() -> bool:
    """Executa todas as suítes e retorna True se tudo passou."""
    totals = {"passed": 0, "total": 0}
    for fn in [test_ingester, test_retriever, test_llm_chain, test_integration]:
        p, t = fn()
        totals["passed"] += p
        totals["total"]  += t

    color = GREEN if totals["passed"] == totals["total"] else RED
    print(f"\n{color}{BOLD}{'═'*60}")
    print(f"  TOTAL GERAL: {totals['passed']}/{totals['total']} testes passaram")
    print(f"{'═'*60}{RESET}\n")
    return totals["passed"] == totals["total"]


# ─────────────────────────────────────────────────────────
# Menu interativo
# ─────────────────────────────────────────────────────────

def interactive_menu() -> None:
    while True:
        print(_title("EixoAI — Suite de Testes"))
        print("  1. Testar ingester.py")
        print("  2. Testar retriever.py")
        print("  3. Testar llm_chain.py")
        print("  4. Teste de integração")
        print("  5. Rodar TODOS os testes")
        print("  6. Avaliar RAG com PDF existente")
        print("  7. Avaliar RAG com PDF da CF/88 (via .env)")
        print("  0. Sair")
        print()

        choice = input("Escolha: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            test_ingester()
        elif choice == "2":
            test_retriever()
        elif choice == "3":
            test_llm_chain()
        elif choice == "4":
            test_integration()
        elif choice == "5":
            run_all_tests()
        elif choice == "6":
            path = input("Caminho do PDF: ").strip()
            if not path:
                print(_warn("Caminho vazio.")); continue
            if not Path(path).exists():
                print(_fail(f"Arquivo não encontrado: {path}")); continue

            n       = input("Número de queries (padrão 8): ").strip()
            n       = int(n) if n.isdigit() else 8
            llm     = input("Usar LLM para gerar Q&A? (s/N): ").strip().lower() == "s"
            answers = input("Gerar respostas da LLM? (s/N): ").strip().lower() == "s"

            evaluator = RAGEvaluator(top_k=5)
            report    = evaluator.evaluate(path, n_pairs=n, use_llm_qa=llm, evaluate_answer=answers)

            if input("\nSalvar relatório JSON? (s/N): ").strip().lower() == "s":
                out = input("Caminho (padrão: eval_report.json): ").strip() or "eval_report.json"
                evaluator.save_report(report, out)
        elif choice == "7":
            _run_cf88_eval(top_k=5, answers=False)
        else:
            print(_warn("Opção inválida."))

        input("\nPressione ENTER para continuar...")


def _run_cf88_eval(top_k: int, answers: bool, output: str | None = None) -> None:
    """Avalia o RAG com o PDF da CF/88 (ou texto fallback se não configurado)."""

    sample_path = get_sample_pdf_path()
    evaluator   = RAGEvaluator(top_k=top_k)

    if sample_path:
        print(_info(f"Usando PDF configurado: {sample_path}"))
        report = evaluator.evaluate(
            pdf_path=sample_path,
            qa_pairs=CF88_QA_PAIRS,
            evaluate_answer=answers,
        )
    else:
        print(_warn("EVAL_PDF_PATH não definido no .env — usando texto de amostra embutido."))
        print(_info("Adicione EVAL_PDF_PATH=C:\\caminho\\CF88.pdf no .env para usar o PDF real."))
        pdf_path = make_fallback_pdf()
        try:
            report = evaluator.evaluate(
                pdf_path=pdf_path,
                qa_pairs=CF88_QA_PAIRS[:3],
                evaluate_answer=False,
            )
        finally:
            os.unlink(pdf_path)

    if output:
        evaluator.save_report(report, output)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EixoAI — suite de testes e avaliação RAG")
    parser.add_argument("--all",     action="store_true",  help="Rodar todos os testes")
    parser.add_argument("--module",  choices=["ingester", "retriever", "llm_chain", "integration"],
                        help="Testar módulo específico")
    parser.add_argument("--eval",    metavar="PDF",        help="Avaliar RAG com PDF")
    parser.add_argument("--sample",  action="store_true",  help="Avaliar com PDF da CF/88 (via .env)")
    parser.add_argument("--pairs",   type=int, default=8,  help="Número de Q&A pairs (default: 8)")
    parser.add_argument("--topk",    type=int, default=5,  help="Top-K para busca (default: 5)")
    parser.add_argument("--no-llm",  action="store_true",  help="Heurística para Q&A (sem LLM)")
    parser.add_argument("--answers", action="store_true",  help="Gerar respostas da LLM na avaliação")
    parser.add_argument("--output",  metavar="JSON",       help="Salvar relatório em JSON")
    args = parser.parse_args()

    if args.all:
        sys.exit(0 if run_all_tests() else 1)
    elif args.module:
        fn = {
            "ingester":    test_ingester,
            "retriever":   test_retriever,
            "llm_chain":   test_llm_chain,
            "integration": test_integration,
        }[args.module]
        p, t = fn()
        sys.exit(0 if p == t else 1)
    elif args.eval:
        if not Path(args.eval).exists():
            print(_fail(f"Arquivo não encontrado: {args.eval}"))
            sys.exit(1)
        evaluator = RAGEvaluator(top_k=args.topk)
        report    = evaluator.evaluate(
            pdf_path=args.eval,
            n_pairs=args.pairs,
            use_llm_qa=not args.no_llm,
            evaluate_answer=args.answers,
        )
        if args.output:
            evaluator.save_report(report, args.output)
    elif args.sample:
        _run_cf88_eval(top_k=args.topk, answers=args.answers, output=args.output)
    else:
        interactive_menu()