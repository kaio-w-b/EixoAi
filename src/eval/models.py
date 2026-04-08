"""
eval/models.py
──────────────
Modelos de dados para avaliação do pipeline RAG.

EvalResult  — resultado de uma única query (recuperação + qualidade da resposta)
EvalReport  — relatório agregado de uma execução completa
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalResult:
    # ── Query ─────────────────────────────────────────────────
    query:             str
    expected_keywords: List[str]

    # ── Recuperação ───────────────────────────────────────────
    retrieved_texts:   List[str]
    retrieved_scores:  List[float]
    hit:               bool
    reciprocal_rank:   float
    precision_at_k:    float
    ndcg:              float
    keyword_coverage:  float

    # ── Resposta gerada ───────────────────────────────────────
    answer:            str = ""

    # ── Metadados do par Q&A ──────────────────────────────────
    expected_answer:   str   = ""
    source_article:    str   = "N/A"
    difficulty:        str   = "medium"   # easy | medium | hard

    # ── Métricas de qualidade da resposta (evaluate_answer) ───
    answer_semantic_score:    float = 0.0
    answer_keyword_coverage:  float = 0.0
    answer_is_correct:        bool  = False


@dataclass
class EvalReport:
    # ── Identificação ─────────────────────────────────────────
    document:      str
    total_queries: int

    # ── Métricas de recuperação ───────────────────────────────
    hit_rate:             float
    mrr:                  float
    mean_precision_at_k:  float
    mean_ndcg:            float
    mean_keyword_coverage: float
    latency_ms:           float

    # ── Métricas de qualidade das respostas ───────────────────
    mean_answer_semantic: float = 0.0
    mean_answer_keyword:  float = 0.0
    answer_accuracy:      float = 0.0

    # ── Resultados individuais ────────────────────────────────
    results: List[EvalResult] = field(default_factory=list)