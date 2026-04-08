import logging
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
import re

logger = logging.getLogger(__name__)

# ── Configurações ─────────────────────────────────────────────
#
# Modelo: multilingual-e5-small
#   • Treinado em português vs all-MiniLM que é inglês puro
#   • Entende termos jurídicos como "inviolabilidade", "mandado de segurança"
#   • Requer prefixos "passage: " ao indexar e "query: " ao buscar
#
# Chunk: 1024 chars com overlap 200
#   • Mantém artigos completos com seus incisos
#
# Estratégia: "legal"
#   • Respeita a estrutura Art. N da CF/88
#
MODEL_NAME     = "intfloat/multilingual-e5-small"
CHUNK_SIZE     = 1024
CHUNK_OVERLAP  = 200
CHUNK_STRATEGY = "legal"   # 'legal' | 'semantic' | 'fixed' | 'sentence'

DB_PATH = Path(os.environ.get("CHROMADB_PATH", "../vector_db"))
DB_PATH.mkdir(parents=True, exist_ok=True)


class DocumentRetriever:
    """
    Gerenciador de embeddings e recuperação de documentos com ChromaDB.

    Melhorias v2:
      • Modelo multilingual-e5-small (melhor PT-BR jurídico)
      • Prefixos "passage:" / "query:" exigidos pelo e5
      • Chunker jurídico que respeita Art. N da CF/88
      • Metadados de artigo extraídos automaticamente
      • get_context formata o número do artigo para o LLM
    """

    def __init__(self, model_name: str = MODEL_NAME):
        try:
            logger.info(f"Carregando modelo: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name

            self.client = chromadb.PersistentClient(path=str(DB_PATH))
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("✅ DocumentRetriever inicializado")
        except Exception as e:
            logger.error(f"Erro ao inicializar retriever: {str(e)}")
            raise

    # ── Normalização ─────────────────────────────────────────

    def _normalize_text(self, text: str) -> str:
        """Normaliza espaços por linha, preservando quebras (essencial para chunking)."""
        lines = text.split("\n")
        lines = [" ".join(line.split()) for line in lines]
        return "\n".join(lines)

    # ── Extração de metadados ─────────────────────────────────

    @staticmethod
    def _extract_article(text: str) -> str:
        """Retorna o número do primeiro artigo encontrado no chunk, ou 'N/A'."""
        m = re.search(r"Art\.\s*(\d+[\w\-]*)", text, re.IGNORECASE)
        return m.group(1) if m else "N/A"

    # ── Chunker jurídico (principal) ──────────────────────────

    def _chunk_text_legal(self, text: str, chunk_size: int = CHUNK_SIZE) -> List[Dict]:
        """
        Divide o texto respeitando a estrutura de artigos (Art. N).

        Lógica:
          1. Separa em unidades que começam em cada "Art."
          2. Agrupa artigos pequenos consecutivos até encher chunk_size
          3. Artigos grandes (> chunk_size) são divididos por inciso/alínea
          4. Fallback semântico para texto sem marcadores de artigo
        """
        has_articles = bool(re.search(r"\bArt\.\s*\d+", text, re.IGNORECASE))

        if not has_articles:
            logger.info("Texto sem marcadores de artigo — usando chunker semântico")
            return self._chunk_text_semantic(text, chunk_size, CHUNK_OVERLAP)

        parts = re.split(r"(?=\bArt\.\s*\d+)", text)
        units = [p.strip() for p in parts if p.strip()]

        chunks: List[Dict] = []
        current = ""
        idx = 0

        def _save(t: str) -> None:
            nonlocal idx
            if t.strip():
                chunks.append({
                    "id": hashlib.md5(f"{idx}_{t[:50]}".encode()).hexdigest()[:16],
                    "text": t.strip(),
                    "chunk_num": idx,
                })
                idx += 1

        for unit in units:
            if len(unit) > chunk_size:
                if current:
                    _save(current)
                    current = ""

                sub_pattern = r"(?=\b(?:[IVX]+\s*[-\u2013\u2014]|\u00a7\s*\d+|[a-z]\)\s))"
                sub_parts = re.split(sub_pattern, unit)
                sub_units = [s.strip() for s in sub_parts if s.strip()]

                sub_current = ""
                for sub in sub_units:
                    if sub_current and len(sub_current) + len(sub) + 1 > chunk_size:
                        _save(sub_current)
                        art_header = re.match(r"(Art\.\s*\d+[\w\-]*[^.]*\.)", sub_current)
                        sub_current = (art_header.group(1) + " " if art_header else "") + sub
                    else:
                        sub_current += (" " if sub_current else "") + sub

                if sub_current:
                    _save(sub_current)

            elif current and len(current) + len(unit) + 2 <= chunk_size:
                current += "\n\n" + unit

            else:
                if current:
                    _save(current)
                current = unit

        if current:
            _save(current)

        logger.info(f"📦 {len(chunks)} chunks jurídicos ({len(units)} artigos fonte)")
        return chunks

    # ── Chunker semântico (fallback) ──────────────────────────

    def _chunk_text_semantic(
        self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
    ) -> List[Dict]:
        """Chunker adaptativo: parágrafos → linhas → sentenças → fatias fixas."""
        units = [u.strip() for u in re.split(r"\n\n+", text) if u.strip()]
        if len(units) <= 1 and len(text) > chunk_size:
            units = [u.strip() for u in re.split(r"\n", text) if u.strip()]

        sentences: List[str] = []
        for unit in units:
            if len(unit) <= chunk_size:
                sentences.append(unit)
            else:
                sub = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", unit) if s.strip()]
                sentences.extend(sub) if sub else None
                if not sub:
                    for i in range(0, len(unit), chunk_size - overlap):
                        piece = unit[i: i + chunk_size].strip()
                        if piece:
                            sentences.append(piece)

        chunks: List[Dict] = []
        current = ""
        idx = 0

        def _save_chunk(t: str) -> None:
            nonlocal idx
            chunks.append({
                "id": hashlib.md5(f"{idx}_{t[:50]}".encode()).hexdigest()[:16],
                "text": t,
                "chunk_num": idx,
            })
            idx += 1

        for sent in sentences:
            if not current and len(sent) >= chunk_size:
                _save_chunk(sent)
                continue
            if current and len(current) + 1 + len(sent) > chunk_size:
                _save_chunk(current.strip())
                prev_sents = re.split(r"(?<=[.!?;])\s+", current)
                overlap_text = " ".join(prev_sents[-2:]) if len(prev_sents) > 1 else prev_sents[-1]
                current = overlap_text + " " + sent
            else:
                current += (" " if current else "") + sent

        if current.strip():
            _save_chunk(current.strip())

        logger.info(f"📦 {len(chunks)} chunks semânticos")
        return chunks

    def _chunk_text_fixed(
        self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
    ) -> List[Dict]:
        chunks, start, idx = [], 0, 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if end < len(text):
                last_space = chunk_text.rfind(" ")
                if last_space > 0:
                    end = start + last_space
                    chunk_text = text[start:end]
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({
                    "id": hashlib.md5(chunk_text.encode()).hexdigest()[:16],
                    "text": chunk_text,
                    "chunk_num": idx,
                })
                idx += 1
            start = end - overlap
            if start >= len(text):
                break
        logger.info(f"📦 {len(chunks)} chunks fixos")
        return chunks

    def _chunk_text_sentence(
        self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 1
    ) -> List[Dict]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_text = " ".join(sentences[i: i + chunk_size]).strip()
            if chunk_text:
                chunks.append({
                    "id": hashlib.md5(f"{i}_{chunk_text[:50]}".encode()).hexdigest()[:16],
                    "text": chunk_text,
                    "chunk_num": len(chunks),
                })
        logger.info(f"📦 {len(chunks)} chunks por sentença")
        return chunks

    def _chunk_text(
        self,
        text: str,
        strategy: str = CHUNK_STRATEGY,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[Dict]:
        if strategy == "legal":
            return self._chunk_text_legal(text, chunk_size)
        elif strategy == "semantic":
            return self._chunk_text_semantic(text, chunk_size, overlap)
        elif strategy == "sentence":
            return self._chunk_text_sentence(text, chunk_size, overlap)
        else:
            return self._chunk_text_fixed(text, chunk_size, overlap)

    # ── CRUD ─────────────────────────────────────────────────

    def add_document(
        self,
        document_id: str,
        text: str,
        source: str = "unknown",
        page: int = 0,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        try:
            logger.info(f"📥 Adicionando documento: {source}")
            text = self._normalize_text(text)

            chunks = self._chunk_text(text)
            if not chunks:
                logger.warning(f"Nenhum chunk gerado para {source}")
                return {"count": 0, "ids": []}

            logger.info(f"🔄 Gerando {len(chunks)} embeddings (modelo: {self.model_name})…")

            BATCH = 256
            all_ids, all_docs, all_metas, all_embeddings = [], [], [], []

            for i in range(0, len(chunks), BATCH):
                batch = chunks[i: i + BATCH]

                # Prefixo "passage:" obrigatório para o multilingual-e5
                texts_for_embedding = [f"passage: {c['text']}" for c in batch]
                embeddings_batch = self.model.encode(
                    texts_for_embedding,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for j, (chunk, emb) in enumerate(zip(batch, embeddings_batch)):
                    chunk_id = f"{document_id}_{i + j}"
                    meta = dict(metadata or {})
                    meta.update({
                        "source": source,
                        "page": page,
                        "chunk": chunk["chunk_num"],
                        "article": self._extract_article(chunk["text"]),
                        "model": self.model_name,
                        "timestamp": datetime.now().isoformat(),
                        "document_id": document_id,
                    })
                    all_ids.append(chunk_id)
                    all_docs.append(chunk["text"])
                    all_metas.append(meta)
                    all_embeddings.append(emb.tolist())

            self.collection.add(
                ids=all_ids,
                embeddings=all_embeddings,
                documents=all_docs,
                metadatas=all_metas,
            )

            logger.info(f"✅ {len(all_ids)} chunks adicionados")
            return {"count": len(all_ids), "ids": all_ids}

        except Exception as e:
            logger.error(f"Erro ao adicionar documento: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> List[Dict]:
        try:
            query = self._normalize_text(query)
            search_k = top_k * 2 if rerank else top_k

            # Prefixo "query:" obrigatório para o multilingual-e5
            query_embedding = self.model.encode(
                f"query: {query}",
                normalize_embeddings=True,
            )

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(search_k, self.collection.count() or 1),
            )

            formatted = []
            if results["ids"] and results["ids"][0]:
                for doc_id, document, distance, meta in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                ):
                    formatted.append({
                        "id": doc_id,
                        "text": document,
                        "distance": float(distance),
                        "relevance": 1 - float(distance),
                        "source": meta.get("source"),
                        "page": meta.get("page"),
                        "chunk": meta.get("chunk"),
                        "article": meta.get("article", "N/A"),
                        "rank": len(formatted) + 1,
                    })

            if rerank and formatted:
                seen, unique = set(), []
                for r in formatted:
                    h = hashlib.md5(r["text"][:100].encode()).hexdigest()
                    if h not in seen:
                        unique.append(r)
                        seen.add(h)
                formatted = sorted(unique, key=lambda x: x["relevance"], reverse=True)[:top_k]

            return formatted

        except Exception as e:
            logger.error(f"Erro ao buscar: {str(e)}")
            return []

    def get_context(self, query: str, top_k: int = 5, min_relevance: float = 0.0) -> str:
        try:
            results = self.search(query, top_k=top_k, rerank=True)
            relevant = [r for r in results if r["relevance"] >= min_relevance]
            if not relevant:
                return ""

            parts = ["=== CONTEXTO RELEVANTE ===\n"]
            for i, r in enumerate(relevant, 1):
                art = r.get("article", "N/A")
                art_label = f"Art. {art} — " if art != "N/A" else ""
                parts.append(
                    f"[{i}] {art_label}{r['source']} "
                    f"(pág. {r['page']}, relevância: {r['relevance']:.2%})"
                )
                parts.append(f"{r['text']}\n")

            return "\n".join(parts)

        except Exception as e:
            logger.error(f"Erro ao preparar contexto: {str(e)}")
            return ""

    def delete_document(self, document_id: str) -> int:
        try:
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}}
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"🗑️  {len(results['ids'])} chunks removidos ({document_id})")
                return len(results["ids"])
            return 0
        except Exception as e:
            logger.error(f"Erro ao deletar documento: {str(e)}")
            return 0

    def clear_all(self) -> None:
        try:
            self.client.delete_collection(name="documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("🗑️  Banco de dados limpo")
        except Exception as e:
            logger.error(f"Erro ao limpar banco: {str(e)}")

    def get_stats(self) -> Dict:
        try:
            return {
                "total_chunks": self.collection.count(),
                "model": self.model_name,
                "db_path": str(DB_PATH),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {str(e)}")
            return {}


# ── Funções auxiliares ───────────────────────────────────────

def quick_add_document(text: str, source: str = "documento", page: int = 0) -> Tuple[str, int]:
    retriever = DocumentRetriever()
    doc_id = hashlib.md5(source.encode()).hexdigest()[:16]
    result = retriever.add_document(doc_id, text, source=source, page=page)
    return doc_id, result["count"]


def quick_search(query: str, top_k: int = 5) -> str:
    retriever = DocumentRetriever()
    return retriever.get_context(query, top_k=top_k)