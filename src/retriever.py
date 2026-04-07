import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
import re

logger = logging.getLogger(__name__)

# Configurações
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
CHUNK_STRATEGY = "semantic"  # 'semantic' | 'fixed' | 'sentence'
DB_PATH = Path("../vector_db")
DB_PATH.mkdir(parents=True, exist_ok=True)


class DocumentRetriever:
    """
    Gerenciador de embeddings e recuperação de documentos com ChromaDB.
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
        """
        Normaliza espaços dentro de cada linha, preservando as quebras de
        linha — essenciais para o chunker semântico detectar parágrafos.
        """
        lines = text.split("\n")
        lines = [" ".join(line.split()) for line in lines]
        return "\n".join(lines)

    # ── Estratégias de chunking ──────────────────────────────

    def _chunk_text_semantic(
        self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
    ) -> List[Dict]:
        """
        Divide o texto em chunks respeitando chunk_size.

        Pipeline:
          1. Divide por parágrafos (\n\n) → fallback \n → fallback sentença.
          2. Unidades maiores que chunk_size são subdivididas em sentenças.
          3. Agrupa sentenças/unidades pequenas até encher chunk_size.
          4. Aplica overlap de sentenças entre chunks consecutivos.
        """
        # ── Passo 1: unidades primárias ──────────────────────
        units = [u.strip() for u in re.split(r"\n\n+", text) if u.strip()]
        if len(units) <= 1 and len(text) > chunk_size:
            units = [u.strip() for u in re.split(r"\n", text) if u.strip()]

        # ── Passo 2: explode unidades maiores que chunk_size em sentenças ──
        sentences: List[str] = []
        for unit in units:
            if len(unit) <= chunk_size:
                sentences.append(unit)
            else:
                # Divide a unidade grande em sentenças individuais
                sub = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", unit) if s.strip()]
                if sub:
                    sentences.extend(sub)
                else:
                    # Última saída: fatias fixas de chunk_size
                    for i in range(0, len(unit), chunk_size - overlap):
                        piece = unit[i : i + chunk_size].strip()
                        if piece:
                            sentences.append(piece)

        # ── Passo 3: agrupa sentenças até chunk_size ─────────
        chunks: List[Dict] = []
        current = ""
        idx = 0

        def _save_chunk(text_: str) -> None:
            nonlocal idx
            chunks.append({
                "id": hashlib.md5(f"{idx}_{text_[:50]}".encode()).hexdigest()[:16],
                "text": text_,
                "chunk_num": idx,
            })
            idx += 1

        for sent in sentences:
            # Sentença sozinha já passa do limite → salva como chunk próprio
            if not current and len(sent) >= chunk_size:
                _save_chunk(sent)
                continue

            if current and len(current) + 1 + len(sent) > chunk_size:
                _save_chunk(current.strip())

                # ── Passo 4: overlap ──────────────────────────
                prev_sents = re.split(r"(?<=[.!?;])\s+", current)
                overlap_text = " ".join(prev_sents[-2:]) if len(prev_sents) > 1 else prev_sents[-1]
                current = overlap_text + " " + sent
            else:
                current += (" " if current else "") + sent

        if current.strip():
            _save_chunk(current.strip())

        logger.info(
            f"📦 {len(chunks)} chunks semânticos "
            f"(unidades: {len(units)} → sentenças: {len(sentences)})"
        )
        return chunks

    def _chunk_text_sentence(
        self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 1
    ) -> List[Dict]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[Dict] = []

        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_text = " ".join(sentences[i : i + chunk_size]).strip()
            if chunk_text:
                chunks.append({
                    "id": hashlib.md5(f"{i}_{chunk_text[:50]}".encode()).hexdigest()[:16],
                    "text": chunk_text,
                    "chunk_num": len(chunks),
                })

        logger.info(f"📦 {len(chunks)} chunks por sentença")
        return chunks

    def _chunk_text_fixed(
        self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
    ) -> List[Dict]:
        chunks: List[Dict] = []
        start = 0
        idx = 0

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

    def _chunk_text(
        self,
        text: str,
        strategy: str = CHUNK_STRATEGY,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[Dict]:
        if strategy == "semantic":
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

            # Normaliza preservando quebras de linha (necessário para chunking semântico)
            text = self._normalize_text(text)

            chunks = self._chunk_text(text)
            if not chunks:
                logger.warning(f"Nenhum chunk gerado para {source}")
                return {"count": 0, "ids": []}

            logger.info(f"🔄 Gerando {len(chunks)} embeddings…")

            # Processa em lotes para economizar memória em documentos grandes
            BATCH = 256
            all_ids, all_docs, all_metas, all_embeddings = [], [], [], []

            for i in range(0, len(chunks), BATCH):
                batch = chunks[i : i + BATCH]
                texts_batch = [c["text"] for c in batch]
                embeddings_batch = self.model.encode(texts_batch)

                for j, (chunk, emb) in enumerate(zip(batch, embeddings_batch)):
                    chunk_id = f"{document_id}_{i + j}"
                    meta = dict(metadata or {})
                    meta.update({
                        "source": source,
                        "page": page,
                        "chunk": chunk["chunk_num"],
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

            query_embedding = self.model.encode(query)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(search_k, self.collection.count() or 1),
            )

            formatted = []
            if results["ids"] and results["ids"][0]:
                for doc_id, document, distance, metadata in zip(
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
                        "source": metadata.get("source"),
                        "page": metadata.get("page"),
                        "chunk": metadata.get("chunk"),
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
                parts.append(
                    f"[{i}] {r['source']} (pág. {r['page']}, relevância: {r['relevance']:.2%})"
                )
                parts.append(f"{r['text']}\n")

            return "\n".join(parts)

        except Exception as e:
            logger.error(f"Erro ao preparar contexto: {str(e)}")
            return ""

    def get_expanded_context(
        self, query: str, top_k: int = 3, include_neighbors: bool = True
    ) -> str:
        try:
            results = self.search(query, top_k=top_k, rerank=True)
            if not results:
                return ""

            if include_neighbors:
                all_chunks = self.collection.get()
                all_data = {
                    cid: {"text": doc, "metadata": meta}
                    for cid, doc, meta in zip(
                        all_chunks["ids"],
                        all_chunks["documents"],
                        all_chunks["metadatas"],
                    )
                }
                expanded = []
                for result in results:
                    expanded.append(result)
                    curr_chunk_num = result.get("chunk", 0)
                    curr_doc_id = result["id"].rsplit("_", 1)[0]
                    for meta in all_chunks["metadatas"]:
                        if (
                            meta.get("document_id") == curr_doc_id
                            and meta.get("chunk") == curr_chunk_num - 1
                        ):
                            idx = all_chunks["metadatas"].index(meta)
                            expanded.append({
                                "text": all_chunks["documents"][idx],
                                "source": meta.get("source"),
                                "page": meta.get("page"),
                                "chunk": meta.get("chunk"),
                                "is_neighbor": True,
                                "neighbor_type": "anterior",
                                "relevance": 0,
                            })
                results = expanded

            parts = ["=== CONTEXTO RELEVANTE ===\n"]
            count = 0
            for r in results:
                is_neighbor = r.get("is_neighbor", False)
                prefix = (
                    f"  [Chunk vizinho {r.get('neighbor_type', '')}]"
                    if is_neighbor
                    else f"  [{count + 1}]"
                )
                rel = f" (relevância: {r['relevance']:.2%})" if "relevance" in r and not is_neighbor else ""
                parts.append(f"{prefix} {r['source']} (pág. {r['page']}){rel}")
                parts.append(f"{r['text']}\n")
                if not is_neighbor:
                    count += 1

            return "\n".join(parts)

        except Exception as e:
            logger.error(f"Erro ao preparar contexto expandido: {str(e)}")
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