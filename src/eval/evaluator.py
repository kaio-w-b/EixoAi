"""
eval/evaluator.py
─────────────────
RAGEvaluator — avalia a qualidade do pipeline RAG para qualquer PDF.

Métricas calculadas por query:
  • Hit Rate       — ao menos 1 chunk relevante recuperado no top-K
  • MRR            — Mean Reciprocal Rank (1/posição do primeiro hit)
  • Precision@K    — proporção de chunks relevantes no top-K
  • NDCG           — Normalized Discounted Cumulative Gain
  • Keyword Cover. — % das keywords encontradas nos chunks recuperados

Geração de Q&A:
  • Via LLM Groq   — pares contextualizados gerados automaticamente
  • Via heurística — extração de sentenças/keywords sem API
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


from ingester import extract_text_from_pdf
from retriever import DocumentRetriever
from dotenv import load_dotenv
from utils.display import _info, _ok, _warn, YELLOW, GREEN, RED, BOLD, RESET
from eval.models import EvalReport, EvalResult
from llm_chain import LLMChain



# Garantir que src/ esta no path independente de como o arquivo e chamado
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Carregar .env da raiz do projeto
def _load_env() -> None:
    try:
        env_path = _SRC.parent / ".env"
        load_dotenv(dotenv_path=str(env_path) if env_path.exists() else None)
    except ImportError:
        pass

_load_env()



class RAGEvaluator:

    def __init__(self, top_k: int = 5, min_keyword_match: float = 0.3):
        """
        Args:
            top_k:             Chunks recuperados por query.
            min_keyword_match: Fração mínima de keywords para considerar um chunk relevante.
        """
        self.top_k = top_k
        self.min_keyword_match = min_keyword_match
        self._retriever = None

    # ── Retriever singleton ──────────────────────────────────

    def _get_retriever(self):
        if self._retriever is None:
            self._retriever = DocumentRetriever()
        return self._retriever

    # ── Ingestão ─────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str) -> Tuple[str, int]:
        """Ingere o PDF no banco vetorial. Retorna (doc_id, n_chunks)."""

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
        """
        Gera pares pergunta/keywords a partir do PDF.
        Usa LLM se GROQ_API_KEY estiver configurada e use_llm=True,
        caso contrário aplica heurística.
        """

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
        """
        Amostra seções distribuídas pelo miolo do documento (10%–95%),
        evitando capa, ficha catalográfica e índice final.
        """
        total        = len(text)
        start_offset = max(int(total * 0.10), 500)
        end_offset   = int(total * 0.95)
        usable       = text[start_offset:end_offset]

        if len(usable) < section_size:
            return usable

        step = max(1, (len(usable) - section_size) // max(n_sections - 1, 1))
        sections = []
        for i in range(n_sections):
            snippet = usable[i * step : i * step + section_size].strip()
            if snippet:
                sections.append(f"[Seção {i + 1}]\n{snippet}")

        return "\n\n".join(sections)

    def _generate_qa_llm(self, text: str, n_pairs: int) -> List[Dict]:
        """Usa a LLM Groq para gerar pares Q&A sobre conteúdo substantivo."""

        print(_info("Gerando Q&A com LLM (amostrando seções do documento)..."))
        sampled = self._sample_document_sections(text)

        prompt = f"""Você é um avaliador de sistemas RAG. Analise os trechos abaixo e gere \
exatamente {n_pairs} pares pergunta/keywords para avaliar a recuperação semântica.

REGRAS OBRIGATÓRIAS:
1. Perguntas sobre CONTEÚDO SUBSTANTIVO (artigos, normas, direitos, deveres, procedimentos).
   NUNCA sobre metadados: nomes de autores, ISBN, cargo editorial, data de publicação.
2. Cada pergunta deve ser respondível por um trecho específico do texto.
3. Keywords são termos que DEVEM aparecer no trecho relevante (não na pergunta).
4. Use 4-6 keywords por par, escolhendo termos técnicos e específicos.
5. Varie os tópicos: cubra diferentes seções/artigos/temas.
6. Retorne SOMENTE JSON puro, sem markdown, sem texto extra.

FORMATO:
[
  {{
    "query": "pergunta objetiva sobre conteúdo",
    "keywords": ["termo1", "termo2", "termo3", "termo4"]
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
            print(_ok(f"{len(valid)} pares Q&A gerados pela LLM"))
            return valid[:n_pairs]
        except json.JSONDecodeError:
            print(_warn("Erro ao parsear JSON — usando heurística"))
            return self._generate_qa_heuristic(text, n_pairs)

    def _generate_qa_heuristic(self, text: str, n_pairs: int) -> List[Dict]:
        """Gera Q&A por heurística sem API: extrai sentenças e keywords."""
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
            pairs.append({"query": f"O que o documento diz sobre {keywords[0]}?", "keywords": keywords})
            if len(pairs) >= n_pairs:
                break

        print(_ok(f"{len(pairs)} pares Q&A gerados por heurística"))
        return pairs

    # ── Métricas ─────────────────────────────────────────────

    @staticmethod
    def _is_chunk_relevant(text: str, keywords: List[str], threshold: float) -> bool:
        tl  = text.lower()
        hits = sum(1 for kw in keywords if kw.lower() in tl)
        return (hits / len(keywords)) >= threshold if keywords else False

    @staticmethod
    def _keyword_coverage(chunks: List[str], keywords: List[str]) -> float:
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

    # ── Avaliação principal ──────────────────────────────────

    def evaluate(
        self,
        pdf_path: str,
        qa_pairs: Optional[List[Dict]] = None,
        n_pairs: int = 8,
        use_llm_qa: bool = True,
        evaluate_answer: bool = False,
    ) -> EvalReport:
        """
        Executa a avaliação completa.

        Args:
            pdf_path:        Caminho do PDF.
            qa_pairs:        Pares Q&A pré-definidos (se None, gera automaticamente).
            n_pairs:         Quantidade de pares a gerar.
            use_llm_qa:      Usar LLM para geração de Q&A.
            evaluate_answer: Gerar e exibir resposta da LLM por query.
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
        retriever   = self._get_retriever()
        results: List[EvalResult] = []
        total_latency = 0.0

        for i, pair in enumerate(qa_pairs, 1):
            query    = pair["query"]
            keywords = pair.get("keywords", [])

            print(f"  {BOLD}[{i}/{len(qa_pairs)}]{RESET} {query}")

            t0     = time.perf_counter()
            chunks = retriever.search(query, top_k=self.top_k, rerank=True)
            total_latency += (time.perf_counter() - t0) * 1000

            texts      = [c["text"]      for c in chunks]
            scores     = [c["relevance"] for c in chunks]
            relevances = [self._is_chunk_relevant(t, keywords, self.min_keyword_match) for t in texts]

            hit    = any(relevances)
            rr     = next((1 / (j + 1) for j, r in enumerate(relevances) if r), 0.0)
            prec   = sum(relevances) / len(relevances) if relevances else 0.0
            ndcg   = self._compute_ndcg(relevances)
            kw_cov = self._keyword_coverage(texts, keywords)

            print(f"     {'🎯' if hit else '❌'}  hit={hit}  RR={rr:.2f}  P@K={prec:.2f}  NDCG={ndcg:.2f}  KW={kw_cov:.1%}")

            answer = ""
            if evaluate_answer and os.getenv("GROQ_API_KEY"):
                raw_ctx = retriever.get_context(query, top_k=3)
                if len(raw_ctx) > 2500:
                    raw_ctx = raw_ctx[:2500] + "\n[contexto truncado]"
                chain = LLMChain(max_tokens=300, stream=False)
                try:
                    answer = chain.generate_response(
                        user_message=query,
                        system_prompt=(
                            "Responda em 1-3 frases objetivas usando apenas o contexto. "
                            "Se não estiver no contexto, diga 'Não encontrado no contexto.'"
                        ),
                        context=raw_ctx,
                    )
                    print(f"     💬 {answer[:120]}{'…' if len(answer) > 120 else ''}")
                except Exception as e:
                    answer = f"Erro: {e}"
                    print(_warn(f"     Erro na LLM: {e}"))

            results.append(EvalResult(
                query=query, expected_keywords=keywords,
                retrieved_texts=texts, retrieved_scores=scores,
                hit=hit, reciprocal_rank=rr, precision_at_k=prec,
                ndcg=ndcg, keyword_coverage=kw_cov, answer=answer,
            ))

        # Limpar banco após avaliação
        retriever.delete_document(doc_id)

        n = len(results)
        report = EvalReport(
            document=Path(pdf_path).name,
            total_queries=n,
            hit_rate=sum(r.hit              for r in results) / n,
            mrr     =sum(r.reciprocal_rank  for r in results) / n,
            mean_precision_at_k=sum(r.precision_at_k  for r in results) / n,
            mean_ndcg          =sum(r.ndcg             for r in results) / n,
            mean_keyword_coverage=sum(r.keyword_coverage for r in results) / n,
            latency_ms=total_latency / n,
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
        print(f"  Queries     : {report.total_queries}  |  Top-K: {self.top_k}  |  Latência: {report.latency_ms:.1f}ms/query")
        print()
        print(f"  Hit Rate    : {self._bar(report.hit_rate)}")
        print(f"  MRR         : {self._bar(report.mrr)}")
        print(f"  Precision@K : {self._bar(report.mean_precision_at_k)}")
        print(f"  NDCG        : {self._bar(report.mean_ndcg)}")
        print(f"  KW Coverage : {self._bar(report.mean_keyword_coverage)}")
        print()
        score = (report.hit_rate + report.mrr + report.mean_ndcg) / 3
        if score >= 0.7:
            verdict = f"{GREEN}✔ Excelente — RAG recupera bem o conteúdo.{RESET}"
        elif score >= 0.4:
            verdict = f"{YELLOW}⚠ Razoável — ajuste chunking ou embeddings.{RESET}"
        else:
            verdict = f"{RED}✗ Fraco — revise CHUNK_SIZE, modelo ou queries.{RESET}"
        print(f"  Veredicto   : {verdict}")

    def save_report(self, report: EvalReport, output_path: str) -> None:
        """Serializa o relatório completo em JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(_ok(f"Relatório salvo em: {output_path}"))