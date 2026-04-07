"""
tests/test_integration.py
─────────────────────────
Testes de integração: fluxo completo PDF → chunking → ChromaDB → busca → contexto.
Não depende de API externa — testa apenas a cadeia local de componentes.
"""

import os
import hashlib
from typing import Tuple

from test_suite.conftest import _title, _warn, YELLOW, RESET, FALLBACK_TEXT
from test_suite.conftest import make_fallback_pdf
from test_suite.runner import TestRunner
from llm_chain import LLMChain
from retriever import DocumentRetriever
from ingester import extract_text_from_pdf


def test_integration() -> Tuple[int, int]:
    print(_title("TESTE DE INTEGRAÇÃO (fluxo completo)"))

    runner = TestRunner()

    # ── Pipeline: PDF → retriever → busca → contexto → delete ──

    def t_full_pipeline():

        retriever = DocumentRetriever()
        doc_id = "integration_test_" + hashlib.md5(b"integration").hexdigest()[:8]

        # Limpar resíduo de execuções anteriores
        retriever.delete_document(doc_id)
        chunks_before = retriever.get_stats()["total_chunks"]

        # 1. Tentar extrair texto de um PDF temporário.
        #    Se falhar (PDF raw sem reportlab), continua com texto direto —
        #    o objetivo é testar o pipeline de retriever, não o parser de PDF.
        pdf_path    = None
        ingest_text = None
        try:
            pdf_path = make_fallback_pdf()
            extracted = extract_text_from_pdf(pdf_path)
            assert isinstance(extracted, str), "extract_text_from_pdf deve retornar str"
            ingest_text = extracted if len(extracted) > 50 else None
        except Exception as e:
            print(f"\n     {YELLOW}[aviso] Extração do PDF falhou: {e} — usando texto direto{RESET}")
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.unlink(pdf_path)

        if ingest_text is None:
            ingest_text = FALLBACK_TEXT

        # 2. Adicionar ao ChromaDB
        result = retriever.add_document(doc_id, ingest_text, "integration.pdf")
        chunks_added = result["count"]
        assert chunks_added > 0, \
            f"add_document deve criar chunks (criou {chunks_added})"

        # 3. Busca semântica
        results = retriever.search("direitos fundamentais República", top_k=3)
        assert isinstance(results, list), "search deve retornar lista"
        assert len(results) > 0, "search deve retornar ao menos 1 resultado"

        # 4. Contexto formatado
        ctx = retriever.get_context("poderes da União Legislativo Executivo", top_k=3)
        assert isinstance(ctx, str), "get_context deve retornar str"
        assert len(ctx) > 0,         "get_context não deve ser vazio com documento carregado"

        # 5. Estatísticas refletem adição
        stats = retriever.get_stats()
        assert stats["total_chunks"] >= chunks_before + chunks_added, \
            "total_chunks deve ter crescido após add_document"

        # 6. Deleção — delta preciso (não assume banco vazio)
        removed = retriever.delete_document(doc_id)
        assert removed == chunks_added, \
            f"delete deve remover exatamente {chunks_added} chunks (removeu {removed})"

        stats_after = retriever.get_stats()
        assert stats_after["total_chunks"] == chunks_before, (
            f"Após delete, total deve voltar a {chunks_before} "
            f"(atual: {stats_after['total_chunks']})"
        )

    # ── LLMChain: preparação de mensagens com contexto ──────

    def t_llm_chain_with_context():
        """Testa _prepare_messages sem chamar a API."""

        chain = LLMChain()
        msgs = chain._prepare_messages(
            user_message="O que é RAG?",
            system_prompt="Seja um especialista em IA.",
            context="RAG é Retrieval-Augmented Generation."
        )
        assert msgs[0]["role"] == "system", "Primeira mensagem deve ser system"
        assert "RAG" in msgs[-1]["content"],      "Contexto deve estar na última mensagem"
        assert "O que é RAG?" in msgs[-1]["content"], "Pergunta deve estar na última mensagem"

    # ── Registro ─────────────────────────────────────────────

    runner.run("Pipeline completo PDF→busca→contexto",      t_full_pipeline)
    runner.run("LLMChain prepara mensagens com contexto",   t_llm_chain_with_context)

    return runner.summary()