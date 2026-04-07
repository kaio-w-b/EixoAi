from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EvalResult:
    query: str
    expected_keywords: List[str]
    retrieved_texts: List[str]
    retrieved_scores: List[float]
    hit: bool
    reciprocal_rank: float
    precision_at_k: float
    ndcg: float
    keyword_coverage: float
    answer: str


@dataclass
class EvalReport:
    document: str
    total_queries: int
    hit_rate: float
    mrr: float
    mean_precision_at_k: float
    mean_ndcg: float
    mean_keyword_coverage: float
    latency_ms: float
    results: List[EvalResult]
