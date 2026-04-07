"""
tests/test_retriever.py
───────────────────────
Testes unitários para src/retriever.py.
Cobre: inicialização, estratégias de chunking, normalização,
       add/search/context/stats/delete no ChromaDB.
"""

import hashlib
from typing import Tuple

from test_suite.conftest import _title
from test_suite.runner import TestRunner
from retriever import DocumentRetriever


# Fixture local: texto constitucional curto, independente de constantes globais.
_FIXTURE_TEXT = (
    "A República Federativa do Brasil constitui-se em Estado Democrático de Direito.\n\n"
    "São fundamentos da República: a soberania, a cidadania, a dignidade da pessoa humana,\n"
    "os valores sociais do trabalho e da livre iniciativa, e o pluralismo político.\n\n"
    "Todo o poder emana do povo, que o exerce por meio de representantes eleitos ou\n"
    "diretamente, nos termos desta Constituição.\n\n"
    "São Poderes da União, independentes e harmônicos entre si, o Legislativo, o Executivo\n"
    "e o Judiciário.\n\n"
    "Todos são iguais perante a lei, sem distinção de qualquer natureza, garantindo-se\n"
    "a inviolabilidade do direito à vida, à liberdade, à igualdade, à segurança e à propriedade."
)

_DOC_ID = "test_doc_" + hashlib.md5(b"retriever_suite").hexdigest()[:8]


def test_retriever() -> Tuple[int, int]:
    print(_title("MÓDULO: retriever.py"))


    runner = TestRunner()
    retriever: DocumentRetriever | None = None
    chunks_before_add = 0  # capturado em t_add_document, usado em t_delete_document

    # ── Inicialização ────────────────────────────────────────

    def t_init():
        nonlocal retriever
        retriever = DocumentRetriever()
        retriever.delete_document(_DOC_ID)   # limpar resíduo de execuções anteriores
        assert retriever is not None
        assert retriever.model is not None
        assert retriever.collection is not None

    # ── Estratégias de chunking ──────────────────────────────

    def t_chunk_fixed():
        r = DocumentRetriever()
        chunks = r._chunk_text_fixed(_FIXTURE_TEXT, chunk_size=200, overlap=20)
        assert isinstance(chunks, list), "Deve retornar lista"
        assert len(chunks) > 0, "Deve gerar ao menos 1 chunk"
        assert all("text"      in c for c in chunks), "Chunks devem ter 'text'"
        assert all("chunk_num" in c for c in chunks), "Chunks devem ter 'chunk_num'"

    def t_chunk_semantic():
        r = DocumentRetriever()
        chunks = r._chunk_text_semantic(_FIXTURE_TEXT, chunk_size=300)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(len(c["text"]) > 0 for c in chunks), "Chunks não podem ser vazios"

    def t_chunk_sentence():
        r = DocumentRetriever()
        chunks = r._chunk_text_sentence(_FIXTURE_TEXT, chunk_size=5, overlap=1)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    # ── Normalização ─────────────────────────────────────────

    def t_normalize():
        r = DocumentRetriever()
        result = r._normalize_text("  texto   com  espaços  ")
        assert "  " not in result, "Não deve ter espaços duplos"
        assert result == result.strip(), "Não deve ter espaços nas bordas"

    # ── ChromaDB: add / search / context / stats / delete ───

    def t_add_document():
        nonlocal retriever, chunks_before_add
        chunks_before_add = retriever.get_stats().get("total_chunks", 0)
        result = retriever.add_document(
            document_id=_DOC_ID,
            text=_FIXTURE_TEXT,
            source="test_document.pdf"
        )
        assert "count" in result, "Deve retornar 'count'"
        assert result["count"] > 0, "Deve gerar ao menos 1 chunk"
        assert "ids"   in result, "Deve retornar 'ids'"

    def t_search_returns_results():
        nonlocal retriever
        results = retriever.search("fundamentos da República", top_k=3)
        assert isinstance(results, list), "Deve retornar lista"
        assert len(results) > 0, "Deve retornar ao menos 1 resultado"
        assert all("text"      in r for r in results), "Resultados devem ter 'text'"
        assert all("relevance" in r for r in results), "Resultados devem ter 'relevance'"
        assert all("distance"  in r for r in results), "Resultados devem ter 'distance'"

    def t_search_relevance_order():
        nonlocal retriever
        results = retriever.search("poderes independentes harmônicos", top_k=5, rerank=True)
        if len(results) > 1:
            scores = [r["relevance"] for r in results]
            assert scores == sorted(scores, reverse=True), "Deve estar ordenado por relevância"

    def t_search_empty_query():
        nonlocal retriever
        results = retriever.search("", top_k=3)
        assert isinstance(results, list), "Query vazia deve retornar lista (sem exceção)"

    def t_get_context():
        nonlocal retriever
        context = retriever.get_context("direito à vida liberdade igualdade", top_k=3)
        assert isinstance(context, str), "Deve retornar string"
        assert len(context) > 0, "Contexto não deve estar vazio quando há documento"

    def t_get_stats():
        nonlocal retriever
        stats = retriever.get_stats()
        assert "total_chunks" in stats, "Stats deve ter 'total_chunks'"
        assert "model"        in stats, "Stats deve ter 'model'"
        assert stats["total_chunks"] > 0, "Deve ter chunks após add_document"

    def t_delete_document():
        nonlocal retriever, chunks_before_add
        count_before_delete = retriever.get_stats()["total_chunks"]
        assert count_before_delete > chunks_before_add, (
            "Pré-condição: banco deve ter mais chunks do que antes do add "
            f"(antes={chunks_before_add}, atual={count_before_delete})"
        )

        removed = retriever.delete_document(_DOC_ID)
        assert removed > 0, f"delete_document deve remover chunks (removeu {removed})"

        count_after = retriever.get_stats()["total_chunks"]
        assert count_after == chunks_before_add, (
            f"Após delete, total deve voltar a {chunks_before_add} "
            f"(tinha {count_before_delete}, removeu {removed}, restaram {count_after})"
        )

    # ── Registro ─────────────────────────────────────────────

    runner.run("Inicialização do DocumentRetriever",    t_init)
    runner.run("Chunking de tamanho fixo",              t_chunk_fixed)
    runner.run("Chunking semântico (parágrafos)",       t_chunk_semantic)
    runner.run("Chunking por sentença",                 t_chunk_sentence)
    runner.run("Normalização de texto",                 t_normalize)
    runner.run("Adição de documento ao ChromaDB",       t_add_document)
    runner.run("Busca retorna resultados",              t_search_returns_results)
    runner.run("Resultados ordenados por relevância",   t_search_relevance_order)
    runner.run("Query vazia não lança exceção",         t_search_empty_query)
    runner.run("get_context retorna string formatada",  t_get_context)
    runner.run("get_stats retorna estatísticas",        t_get_stats)
    runner.run("Deleção de documento",                  t_delete_document)

    return runner.summary()