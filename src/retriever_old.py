"""
eval/evaluator.py
─────────────────
RAGEvaluator — avalia a qualidade do pipeline RAG para qualquer PDF.

Métricas de recuperação (por query):
  • Hit Rate          — ao menos 1 chunk relevante recuperado no top-K
  • MRR               — Mean Reciprocal Rank
  • Precision@K       — proporção de chunks relevantes no top-K
  • NDCG              — Normalized Discounted Cumulative Gain
  • Keyword Coverage  — % das keywords encontradas nos chunks recuperados

Métricas de geração (evaluate_answer):
  • Semantic Score    — similaridade semântica via embeddings (cosine)
  • Keyword Coverage  — % das keywords da expected_answer na predicted
  • Is Correct        — limiar combinado de semantic + keyword

Geração de Q&A:
  • Dataset estruturado (qa_pairs.py) — recomendado
  • LLM Groq                          — geração automática
  • Heurística                        — sem API
"""

import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Garantir que src/ está no path
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ingester import extract_text_from_pdf
from utils.display import _info, _ok, _warn, YELLOW, GREEN, RED, BOLD, RESET
from eval.models import EvalReport, EvalResult
from llm_chain import LLMChain


def _load_env() -> None:
    try:
        env_path = _SRC.parent / ".env"
        load_dotenv(dotenv_path=str(env_path) if env_path.exists() else None)
    except ImportError:
        pass

_load_env()


# ══════════════════════════════════════════════════════════════════════════════
# evaluate_answer — avaliação de qualidade da resposta gerada
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_answer(
    predicted_answer: str,
    expected_answer: str,
    keywords: Optional[List[str]] = None,
    model_name: str = "intfloat/multilingual-e5-small",
    semantic_threshold: float = 0.55,
    keyword_threshold: float = 0.40,
) -> Dict:
    """
    Avalia a qualidade de uma resposta gerada comparando com a resposta esperada.

    Args:
        predicted_answer:   Resposta gerada pelo sistema RAG + LLM.
        expected_answer:    Resposta correta de referência.
        keywords:           Lista de palavras-chave adicionais para cobertura.
                            Se None, as keywords são extraídas da expected_answer.
        model_name:         Modelo de embeddings para similaridade semântica.
        semantic_threshold: Limiar mínimo de semantic_score para is_correct.
        keyword_threshold:  Limiar mínimo de keyword_coverage para is_correct.

    Returns:
        {
            "semantic_score":    float  — similaridade cosine (0–1),
            "keyword_coverage":  float  — fração de keywords cobertas (0–1),
            "is_correct":        bool   — True se ambos os limiares são atingidos,
            "details": {
                "model":              str,
                "found_keywords":     List[str],
                "missing_keywords":   List[str],
                "total_keywords":     int,
                "semantic_threshold": float,
                "keyword_threshold":  float,
            }
        }
    """
    if not predicted_answer or not predicted_answer.strip():
        return {
            "semantic_score": 0.0,
            "keyword_coverage": 0.0,
            "is_correct": False,
            "details": {
                "model": model_name,
                "found_keywords": [],
                "missing_keywords": keywords or [],
                "total_keywords": len(keywords) if keywords else 0,
                "semantic_threshold": semantic_threshold,
                "keyword_threshold": keyword_threshold,
            },
        }

    # ── 1. Similaridade semântica via embeddings ──────────────
    semantic_score = _compute_semantic_similarity(
        predicted_answer, expected_answer, model_name
    )

    # ── 2. Cobertura de keywords ──────────────────────────────
    if not keywords:
        keywords = _extract_keywords_from_text(expected_answer)

    found, missing = _compute_keyword_coverage(predicted_answer, keywords)
    keyword_coverage = len(found) / len(keywords) if keywords else 1.0

    # ── 3. Veredicto ─────────────────────────────────────────
    is_correct = (
        semantic_score >= semantic_threshold
        and keyword_coverage >= keyword_threshold
    )

    return {
        "semantic_score": round(semantic_score, 4),
        "keyword_coverage": round(keyword_coverage, 4),
        "is_correct": is_correct,
        "details": {
            "model": model_name,
            "found_keywords": found,
            "missing_keywords": missing,
            "total_keywords": len(keywords),
            "semantic_threshold": semantic_threshold,
            "keyword_threshold": keyword_threshold,
        },
    }


def _compute_semantic_similarity(text_a: str, text_b: str, model_name: str) -> float:
    """Calcula similaridade cosine entre dois textos via sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Cache simples em memória para não recarregar o modelo a cada chamada
        cache_key = f"_st_model_{model_name}"
        model = globals().get(cache_key)
        if model is None:
            model = SentenceTransformer(model_name)
            globals()[cache_key] = model

        # Prefixo obrigatório para e5
        prefix = "query: " if "e5" in model_name else ""
        embs = model.encode(
            [f"{prefix}{text_a}", f"{prefix}{text_b}"],
            normalize_embeddings=True,
        )
        # Cosine similarity = dot product quando normalizado
        similarity = float(np.dot(embs[0], embs[1]))
        return max(0.0, min(1.0, similarity))

    except Exception:
        # Fallback: similaridade léxica (Jaccard sobre tokens)
        return _jaccard_similarity(text_a, text_b)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Similaridade de Jaccard sobre conjuntos de tokens (fallback sem embeddings)."""
    def tokenize(t: str) -> set:
        return set(re.findall(r"\b\w{3,}\b", t.lower()))

    set_a, set_b = tokenize(text_a), tokenize(text_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _extract_keywords_from_text(text: str) -> List[str]:
    """Extrai palavras significativas do texto (ignora stopwords comuns)."""
    STOPWORDS = {
        "de", "da", "do", "a", "o", "e", "em", "um", "uma", "que", "para",
        "com", "os", "as", "na", "no", "se", "por", "mais", "ao", "às",
        "dos", "das", "nos", "nas", "sua", "seu", "ser", "são", "foi",
        "que", "não", "ou", "mas", "também", "este", "esta", "esse", "essa",
    }
    words = re.findall(r"\b[a-záéíóúâêîôûãõàèìòùçA-Z][a-záéíóúâêîôûãõàèìòùç]{3,}\b", text)
    keywords = [w.lower() for w in words if w.lower() not in STOPWORDS]
    # Deduplicar preservando ordem
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:10]  # máximo de 10 keywords extraídas automaticamente


def _compute_keyword_coverage(
    text: str, keywords: List[str]
) -> Tuple[List[str], List[str]]:
    """Retorna (found, missing) para a lista de keywords no texto."""
    text_lower = text.lower()
    found = [kw for kw in keywords if kw.lower() in text_lower]
    missing = [kw for kw in keywords if kw.lower() not in text_lower]
    return found, missing


# ══════════════════════════════════════════════════════════════════════════════
# RAGEvaluator
# ══════════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    """
    Avaliador profissional de pipelines RAG.

    Suporta dois modos de avaliação:
      1. Recuperação (retrieval)  — sem chamada à LLM, rápido
      2. Geração (answer quality) — usa evaluate_answer() para cada query
    """

    def __init__(
        self,
        top_k: int = 5,
        min_keyword_match: float = 0.3,
        answer_semantic_threshold: float = 0.55,
        answer_keyword_threshold: float = 0.40,
    ):
        self.top_k = top_k
        self.min_keyword_match = min_keyword_match
        self.answer_semantic_threshold = answer_semantic_threshold
        self.answer_keyword_threshold = answer_keyword_threshold
        self._retriever: Optional[DocumentRetriever] = None

    # ── Retriever singleton ──────────────────────────────────

    def _get_retriever(self) -> DocumentRetriever:
        if self._retriever is None:
            self._retriever = DocumentRetriever()
        return self._retriever

    # ── Ingestão ─────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str) -> Tuple[str, int]:
        print(_info(f"Ingerindo PDF: {Path(pdf_path).name}"))
        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("PDF não contém texto extraível.")

        retriever = self._get_retriever()
        doc_id = hashlib.md5(Path(pdf_path).name.encode()).hexdigest()[:16]
        result = retriever.add_document(
            document_id=doc_id,
            text=text,
            source=Path(pdf_path).name,
        )
        print(_ok(f"{result['count']} chunks criados"))
        return doc_id, result["count"]

    # ── Geração de Q&A ───────────────────────────────────────

    def generate_qa_pairs(
        self,
        pdf_path: str,
        n_pairs: int = 8,
        use_llm: bool = True,
    ) -> List[Dict]:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("PDF não contém texto extraível.")

        if use_llm and os.getenv("GROQ_API_KEY"):
            return self._generate_qa_llm(text, n_pairs)

        print(_warn("GROQ_API_KEY ausente — usando heurística para gerar Q&A"))
        return self._generate_qa_heuristic(text, n_pairs)

    @staticmethod
    def _sample_document_sections(
        text: str,
        n_sections: int = 5,
        section_size: int = 700,
    ) -> str:
        total        = len(text)
        start_offset = max(int(total * 0.10), 500)
        end_offset   = int(total * 0.95)
        usable       = text[start_offset:end_offset]

        if len(usable) < section_size:
            return usable

        step = max(1, (len(usable) - section_size) // max(n_sections - 1, 1))
        sections = []
        for i in range(n_sections):
            snippet = usable[i * step: i * step + section_size].strip()
            if snippet:
                sections.append(f"[Seção {i + 1}]\n{snippet}")

        return "\n\n".join(sections)

    def _generate_qa_llm(self, text: str, n_pairs: int) -> List[Dict]:
        print(_info("Gerando Q&A com LLM (amostrando seções do documento)..."))
        sampled = self._sample_document_sections(text)

        prompt = f"""Você é um avaliador de sistemas RAG. Analise os trechos abaixo e gere \
exatamente {n_pairs} pares pergunta/keywords para avaliar a recuperação semântica.

REGRAS:
1. Perguntas sobre CONTEÚDO SUBSTANTIVO. NUNCA sobre metadados.
2. Inclua expected_answer (resposta correta e objetiva, máx. 3 frases).
3. Inclua source_article (ex: "Art. 5º").
4. Inclua difficulty: easy | medium | hard.
5. Use 4-6 keywords discriminativas por par.
6. Retorne SOMENTE JSON puro, sem markdown.

FORMATO:
[
  {{
    "query": "pergunta objetiva",
    "keywords": ["termo1", "termo2", "termo3"],
    "expected_answer": "resposta correta e objetiva",
    "source_article": "Art. N",
    "difficulty": "easy|medium|hard"
  }}
]

TRECHOS:
{sampled}"""

        chain = LLMChain(temperature=0.1, max_tokens=2000, stream=False)
        try:
            response = chain.generate_response(prompt)
        except Exception as e:
            print(_warn(f"Erro na LLM ao gerar Q&A: {e} — usando heurística"))
            return self._generate_qa_heuristic(text, n_pairs)

        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if not match:
            print(_warn("LLM não retornou JSON válido — usando heurística"))
            return self._generate_qa_heuristic(text, n_pairs)

        try:
            pairs = json.loads(match.group())
            valid = [
                p for p in pairs
                if isinstance(p.get("query"), str)
                and isinstance(p.get("keywords"), list)
                and len(p["keywords"]) >= 2
            ]
            # Garante campos mínimos nos pares gerados por LLM
            for p in valid:
                p.setdefault("expected_answer", "")
                p.setdefault("source_article", "N/A")
                p.setdefault("difficulty", "medium")
            print(_ok(f"{len(valid)} pares Q&A gerados pela LLM"))
            return valid[:n_pairs]
        except json.JSONDecodeError:
            print(_warn("Erro ao parsear JSON — usando heurística"))
            return self._generate_qa_heuristic(text, n_pairs)

    def _generate_qa_heuristic(self, text: str, n_pairs: int) -> List[Dict]:
        stop = {
            "de", "da", "do", "a", "o", "e", "em", "um", "uma", "que",
            "para", "com", "os", "as", "na", "no", "se", "por", "mais",
        }
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 40]
        step  = max(1, len(sentences) // n_pairs)
        pairs: List[Dict] = []

        for i in range(0, min(len(sentences), n_pairs * step), step):
            words    = re.findall(r'\b[a-záéíóúâêîôûãõàèìòùçA-Z][a-záéíóúâêîôûãõàèìòùç]{3,}\b', sentences[i])
            keywords = [w.lower() for w in words if w.lower() not in stop][:5]
            if len(keywords) < 2:
                continue
            pairs.append({
                "query": f"O que o documento diz sobre {keywords[0]}?",
                "keywords": keywords,
                "expected_answer": sentences[i],
                "source_article": "N/A",
                "difficulty": "medium",
            })
            if len(pairs) >= n_pairs:
                break

        print(_ok(f"{len(pairs)} pares Q&A gerados por heurística"))
        return pairs

    # ── Métricas de recuperação ───────────────────────────────

    def _is_chunk_relevant(
        self, text: str, keywords: List[str], threshold: float
    ) -> bool:
        tl   = text.lower()
        hits = sum(1 for kw in keywords if kw.lower() in tl)
        return (hits / len(keywords)) >= threshold if keywords else False

    @staticmethod
    def _keyword_coverage_chunks(chunks: List[str], keywords: List[str]) -> float:
        if not keywords:
            return 0.0
        all_text = " ".join(chunks).lower()
        return sum(1 for kw in keywords if kw.lower() in all_text) / len(keywords)

    @staticmethod
    def _compute_ndcg(relevances: List[bool]) -> float:
        if not relevances:
            return 0.0
        dcg   = sum(int(r) / math.log2(i + 2) for i, r in enumerate(relevances))
        ideal = sum(1 / math.log2(i + 2) for i in range(sum(int(r) for r in relevances)))
        return dcg / ideal if ideal > 0 else 0.0

    # ── Avaliação principal ───────────────────────────────────

    def evaluate(
        self,
        pdf_path: str,
        qa_pairs: Optional[List[Dict]] = None,
        n_pairs: int = 8,
        use_llm_qa: bool = True,
        evaluate_answer_quality: bool = False,
    ) -> EvalReport:
        """
        Executa a avaliação completa do pipeline RAG.

        Args:
            pdf_path:                Caminho do PDF.
            qa_pairs:                Pares Q&A (se None, gera automaticamente).
            n_pairs:                 Quantidade de pares a gerar.
            use_llm_qa:              Usar LLM para geração de Q&A.
            evaluate_answer_quality: Gerar resposta e avaliá-la com evaluate_answer().
        """
        print(_info(f"\n{'─'*60}"))
        print(_info(f"AVALIAÇÃO RAG: {Path(pdf_path).name}"))
        print(_info(f"{'─'*60}"))

        doc_id, _ = self.ingest_pdf(pdf_path)

        if qa_pairs is None:
            qa_pairs = self.generate_qa_pairs(pdf_path, n_pairs, use_llm_qa)
        if not qa_pairs:
            raise ValueError("Nenhum par Q&A disponível para avaliação.")

        print(_info(f"Avaliando {len(qa_pairs)} queries | top_k={self.top_k}\n"))
        retriever    = self._get_retriever()
        results: List[EvalResult] = []
        total_latency = 0.0

        for i, pair in enumerate(qa_pairs, 1):
            query            = pair["query"]
            keywords         = pair.get("keywords", [])
            expected_answer  = pair.get("expected_answer", "")
            source_article   = pair.get("source_article", "N/A")
            difficulty       = pair.get("difficulty", "medium")

            diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")
            print(f"  {BOLD}[{i}/{len(qa_pairs)}]{RESET} {diff_icon} [{difficulty}] {query}")

            t0     = time.perf_counter()
            chunks = retriever.search(query, top_k=self.top_k, rerank=True)
            total_latency += (time.perf_counter() - t0) * 1000

            texts      = [c["text"]      for c in chunks]
            scores     = [c["relevance"] for c in chunks]
            relevances = [
                self._is_chunk_relevant(t, keywords, self.min_keyword_match)
                for t in texts
            ]

            hit    = any(relevances)
            rr     = next((1 / (j + 1) for j, r in enumerate(relevances) if r), 0.0)
            prec   = sum(relevances) / len(relevances) if relevances else 0.0
            ndcg   = self._compute_ndcg(relevances)
            kw_cov = self._keyword_coverage_chunks(texts, keywords)

            print(
                f"     {'🎯' if hit else '❌'}  hit={hit}  "
                f"RR={rr:.2f}  P@K={prec:.2f}  "
                f"NDCG={ndcg:.2f}  KW={kw_cov:.1%}  "
                f"[{source_article}]"
            )

            # ── Avaliação de qualidade da resposta ────────────
            answer           = ""
            answer_eval      = {}
            if evaluate_answer_quality and os.getenv("GROQ_API_KEY"):
                raw_ctx = retriever.get_context(query, top_k=3)
                if len(raw_ctx) > 2500:
                    raw_ctx = raw_ctx[:2500] + "\n[contexto truncado]"
                chain = LLMChain(max_tokens=400, stream=False)
                try:
                    answer = chain.generate_response(
                        user_message=query,
                        system_prompt=(
                            "Responda em 1-3 frases objetivas usando apenas o contexto. "
                            "Se não estiver no contexto, diga 'Não encontrado no contexto.'"
                        ),
                        context=raw_ctx,
                    )
                    answer_eval = evaluate_answer(
                        predicted_answer=answer,
                        expected_answer=expected_answer,
                        keywords=keywords,
                        semantic_threshold=self.answer_semantic_threshold,
                        keyword_threshold=self.answer_keyword_threshold,
                    )
                    sem  = answer_eval["semantic_score"]
                    kw   = answer_eval["keyword_coverage"]
                    corr = answer_eval["is_correct"]
                    icon = "✅" if corr else "❌"
                    print(
                        f"     {icon}  semantic={sem:.2f}  "
                        f"kw_cov={kw:.1%}  correct={corr}"
                    )
                    print(f"     💬 {answer[:120]}{'…' if len(answer) > 120 else ''}")
                except Exception as e:
                    answer = f"Erro: {e}"
                    print(_warn(f"     Erro na LLM: {e}"))

            results.append(EvalResult(
                query=query,
                expected_keywords=keywords,
                retrieved_texts=texts,
                retrieved_scores=scores,
                hit=hit,
                reciprocal_rank=rr,
                precision_at_k=prec,
                ndcg=ndcg,
                keyword_coverage=kw_cov,
                answer=answer,
                # Novos campos (backward-compat: usa .get)
                expected_answer=expected_answer,
                source_article=source_article,
                difficulty=difficulty,
                answer_semantic_score=answer_eval.get("semantic_score", 0.0),
                answer_keyword_coverage=answer_eval.get("keyword_coverage", 0.0),
                answer_is_correct=answer_eval.get("is_correct", False),
            ))

        retriever.delete_document(doc_id)

        n = len(results)
        answered = [r for r in results if r.answer_semantic_score > 0]

        report = EvalReport(
            document=Path(pdf_path).name,
            total_queries=n,
            hit_rate=sum(r.hit              for r in results) / n,
            mrr     =sum(r.reciprocal_rank  for r in results) / n,
            mean_precision_at_k=sum(r.precision_at_k   for r in results) / n,
            mean_ndcg          =sum(r.ndcg              for r in results) / n,
            mean_keyword_coverage=sum(r.keyword_coverage for r in results) / n,
            latency_ms=total_latency / n,
            # Métricas de geração (0 se evaluate_answer_quality=False)
            mean_answer_semantic=sum(r.answer_semantic_score    for r in results) / n,
            mean_answer_keyword =sum(r.answer_keyword_coverage  for r in results) / n,
            answer_accuracy     =sum(r.answer_is_correct        for r in results) / n,
            results=results,
        )
        self._print_report(report)
        return report

    # ── Exibição ─────────────────────────────────────────────

    @staticmethod
    def _bar(value: float, width: int = 20) -> str:
        filled = round(value * width)
        color  = GREEN if value >= 0.7 else (YELLOW if value >= 0.4 else RED)
        return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET} {value:.1%}"

    def _print_report(self, report: EvalReport) -> None:
        print(f"\n{BOLD}{'-'*60}\n  RELATÓRIO FINAL\n{'-'*60}{RESET}")
        print(f"  Documento   : {BOLD}{report.document}{RESET}")
        print(
            f"  Queries     : {report.total_queries}  |  "
            f"Top-K: {self.top_k}  |  "
            f"Latência: {report.latency_ms:.1f}ms/query"
        )
        print()
        print("  ── Recuperação ──────────────────────────────────")
        print(f"  Hit Rate    : {self._bar(report.hit_rate)}")
        print(f"  MRR         : {self._bar(report.mrr)}")
        print(f"  Precision@K : {self._bar(report.mean_precision_at_k)}")
        print(f"  NDCG        : {self._bar(report.mean_ndcg)}")
        print(f"  KW Coverage : {self._bar(report.mean_keyword_coverage)}")

        if report.mean_answer_semantic > 0:
            print()
            print("  ── Qualidade das Respostas ───────────────────────")
            print(f"  Sem. Score  : {self._bar(report.mean_answer_semantic)}")
            print(f"  KW Coverage : {self._bar(report.mean_answer_keyword)}")
            print(f"  Accuracy    : {self._bar(report.answer_accuracy)}")

        print()
        retrieval_score = (report.hit_rate + report.mrr + report.mean_ndcg) / 3
        if retrieval_score >= 0.7:
            verdict = f"{GREEN}✔ Excelente — RAG recupera bem o conteúdo.{RESET}"
        elif retrieval_score >= 0.4:
            verdict = f"{YELLOW}⚠ Razoável — ajuste chunking ou embeddings.{RESET}"
        else:
            verdict = f"{RED}✗ Fraco — revise CHUNK_SIZE, modelo ou queries.{RESET}"
        print(f"  Veredicto   : {verdict}")

        # Breakdown por dificuldade
        for diff in ("easy", "medium", "hard"):
            subset = [r for r in report.results if r.difficulty == diff]
            if subset:
                hr = sum(r.hit for r in subset) / len(subset)
                icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}[diff]
                print(f"  {icon} {diff:6s}     : hit={hr:.1%}  n={len(subset)}")

    def save_report(self, report: EvalReport, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(_ok(f"Relatório salvo em: {output_path}"))