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
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
CHUNK_STRATEGY = "semantic"  # 'semantic' (parágrafo), 'fixed' (caracteres), 'sentence' (sentença)
DB_PATH = Path("../vector_db")
DB_PATH.mkdir(parents=True, exist_ok=True)


class DocumentRetriever:
    """
    Gerenciador de embeddings e recuperação de documentos com ChromaDB.
    Gera embeddings consistentes usando sentence-transformers.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Inicializa o retriever com ChromaDB e modelo de embeddings.
        
        Args:
            model_name: Modelo sentence-transformers a usar
        """
        try:
            logger.info(f"Carregando modelo: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            
            # Inicializar ChromaDB
            self.client = chromadb.PersistentClient(path=str(DB_PATH))
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("✅ DocumentRetriever inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar retriever: {str(e)}")
            raise
    
    def _chunk_text_semantic(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, str]]:
        """
        Divide texto em chunks semânticos (por parágrafos/seções).
        Mantém continuidade natural do conteúdo.
        
        Args:
            text: Texto a dividir
            chunk_size: Tamanho máximo por chunk
            overlap: Sobreposição entre chunks
            
        Returns:
            Lista de chunks semanticamente coerentes
        """
        # Dividir por parágrafos
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Se adicionar parágrafo ultrapassa tamanho, salva chunk atual
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunk_text = current_chunk.strip()
                if chunk_text:
                    chunks.append({
                        "id": hashlib.md5(f"{chunk_index}_{chunk_text[:50]}".encode()).hexdigest()[:16],
                        "text": chunk_text,
                        "chunk_num": chunk_index
                    })
                    chunk_index += 1
                    
                    # Adicionar overlap: últimas frases do chunk anterior
                    sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
                    overlap_text = " ".join(sentences[-2:]) if len(sentences) > 1 else sentences[-1]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
        
        # Adicionar último chunk
        if current_chunk.strip():
            chunks.append({
                "id": hashlib.md5(f"{chunk_index}_{current_chunk[:50]}".encode()).hexdigest()[:16],
                "text": current_chunk.strip(),
                "chunk_num": chunk_index
            })
        
        logger.info(f"📦 Texto dividido semanticamente em {len(chunks)} chunks")
        return chunks
    
    def _chunk_text_sentence(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 1) -> List[Dict[str, str]]:
        """
        Divide texto em chunks por sentença (mais preciso para análise).
        
        Args:
            text: Texto a dividir
            chunk_size: Número de sentenças por chunk
            overlap: Número de sentenças de sobreposição
            
        Returns:
            Lista de chunks por sentença
        """
        # Dividir por sentença
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size - overlap):
            end_idx = min(i + chunk_size, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            chunk_text = " ".join(chunk_sentences).strip()
            
            if chunk_text:
                chunks.append({
                    "id": hashlib.md5(f"{i}_{chunk_text[:50]}".encode()).hexdigest()[:16],
                    "text": chunk_text,
                    "chunk_num": len(chunks)
                })
        
        logger.info(f"📦 Texto dividido em {len(chunks)} chunks por sentença")
        return chunks
    
    def _chunk_text_fixed(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, str]]:
        """
        Divide texto em chunks de tamanho fixo (método original).
        
        Args:
            text: Texto a dividir
            chunk_size: Tamanho de cada chunk
            overlap: Sobreposição entre chunks
            
        Returns:
            Lista de chunks de tamanho fixo
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Evitar cortar no meio de palavra
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
                    "chunk_num": chunk_index
                })
                chunk_index += 1
            
            # Mover para próximo chunk com overlap
            start = end - overlap
            if start >= len(text):
                break
        
        logger.info(f"📦 Texto dividido em {len(chunks)} chunks fixos")
        return chunks
    
    def _chunk_text(self, text: str, strategy: str = CHUNK_STRATEGY, 
                    chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, str]]:
        """
        Divide texto usando estratégia escolhida.
        
        Args:
            text: Texto a dividir
            strategy: 'semantic', 'sentence' ou 'fixed'
            chunk_size: Tamanho de cada chunk
            overlap: Sobreposição entre chunks
            
        Returns:
            Lista de chunks
        """
        if strategy == "semantic":
            return self._chunk_text_semantic(text, chunk_size, overlap)
        elif strategy == "sentence":
            return self._chunk_text_sentence(text, chunk_size, overlap)
        else:  # fixed
            return self._chunk_text_fixed(text, chunk_size, overlap)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normaliza texto para consistência.
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        # Remover espaços extras
        text = " ".join(text.split())
        return text
    
    def add_document(
        self,
        document_id: str,
        text: str,
        source: str = "unknown",
        page: int = 0,
        metadata: Optional[Dict] = None
    ) -> Dict[str, int]:
        """
        Adiciona documento ao banco de embeddings.
        
        Args:
            document_id: ID único do documento
            text: Conteúdo do documento
            source: Fonte/nome do arquivo
            page: Número da página
            metadata: Metadados adicionais
            
        Returns:
            Dicionário com contagem de chunks e IDs
        """
        try:
            logger.info(f"📥 Adicionando documento: {source}")
            
            # Normalizar texto
            text = self._normalize_text(text)
            
            # Dividir em chunks
            chunks = self._chunk_text(text)
            
            if not chunks:
                logger.warning(f"Nenhum chunk gerado para {source}")
                return {"count": 0, "ids": []}
            
            # Gerar embeddings
            logger.info(f"🔄 Gerando {len(chunks)} embeddings...")
            embeddings = self.model.encode([c["text"] for c in chunks])
            
            # Preparar dados para ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}_{i}"
                ids.append(chunk_id)
                documents.append(chunk["text"])
                
                meta = metadata or {}
                meta.update({
                    "source": source,
                    "page": page,
                    "chunk": i,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "document_id": document_id
                })
                metadatas.append(meta)
            
            # Adicionar ao ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"✅ {len(ids)} chunks adicionados com sucesso")
            return {"count": len(ids), "ids": ids}
            
        except Exception as e:
            logger.error(f"Erro ao adicionar documento: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> List[Dict[str, any]]:
        """
        Busca documentos relevantes para a query com opção de reranking.
        
        Args:
            query: Texto da pergunta/busca
            top_k: Número de resultados a retornar
            rerank: Se True, reordena resultados por relevância semântica
            
        Returns:
            Lista com resultados ordenados por relevância
        """
        try:
            logger.info(f"🔍 Buscando: {query[:50]}...")
            
            # Normalizar query
            query = self._normalize_text(query)
            
            # Buscar mais resultados se rerank ativado
            search_k = top_k * 2 if rerank else top_k
            
            # Gerar embedding da query
            query_embedding = self.model.encode(query)
            
            # Buscar no ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=search_k
            )
            
            # Formatar resultados
            formatted_results = []
            
            if results["ids"] and len(results["ids"]) > 0:
                for i, (doc_id, document, distance, metadata) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0]
                )):
                    formatted_results.append({
                        "id": doc_id,
                        "text": document,
                        "distance": float(distance),
                        "relevance": 1 - float(distance),
                        "source": metadata.get("source"),
                        "page": metadata.get("page"),
                        "chunk": metadata.get("chunk"),
                        "rank": i + 1
                    })
            
            # Reranking: filtrar duplicatas e refinar ordem
            if rerank and formatted_results:
                # Remover duplicatas por similaridade de conteúdo
                unique_results = []
                seen_texts = set()
                
                for result in formatted_results:
                    text_hash = hashlib.md5(result["text"][:100].encode()).hexdigest()
                    if text_hash not in seen_texts:
                        unique_results.append(result)
                        seen_texts.add(text_hash)
                
                # Ordenar por relevância e aplicar cutoff
                unique_results = sorted(unique_results, key=lambda x: x["relevance"], reverse=True)[:top_k]
                formatted_results = unique_results
            
            logger.info(f"📊 {len(formatted_results)} resultados encontrados (rerank={rerank})")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erro ao buscar: {str(e)}")
            return []
    
    def get_expanded_context(self, query: str, top_k: int = 3, include_neighbors: bool = True) -> str:
        """
        Retorna contexto expandido incluindo chunks vizinhos para melhor continuidade.
        
        Args:
            query: Pergunta do usuário
            top_k: Número de chunks principais
            include_neighbors: Se True, inclui chunks anteriores e posteriores
            
        Returns:
            Contexto formatado com contexto expandido
        """
        try:
            results = self.search(query, top_k=top_k, rerank=True)
            
            if not results:
                return ""
            
            # Se include_neighbors, buscar chunks vizinhos
            if include_neighbors:
                all_chunks = self.collection.get()
                all_data = {
                    chunk_id: {
                        "text": doc,
                        "metadata": meta
                    }
                    for chunk_id, doc, meta in zip(
                        all_chunks["ids"],
                        all_chunks["documents"],
                        all_chunks["metadatas"]
                    )
                }
                
                # Expandir com vizinhos
                expanded_results = []
                for result in results:
                    expanded_results.append(result)
                    
                    # Procurar chunks vizinhos (mesmo documento, número consecutivo)
                    curr_chunk_num = result.get("chunk", 0)
                    curr_doc_id = result["id"].split("_")[0] if "_" in result["id"] else None
                    
                    # Adicionar chunk anterior se existir
                    for candidate in all_chunks["metadatas"]:
                        if (candidate.get("document_id") == curr_doc_id and 
                            candidate.get("chunk") == curr_chunk_num - 1):
                            # Encontrar o texto correspondente
                            idx = all_chunks["metadatas"].index(candidate)
                            expanded_results.append({
                                "text": all_chunks["documents"][idx],
                                "source": candidate.get("source"),
                                "page": candidate.get("page"),
                                "chunk": candidate.get("chunk"),
                                "is_neighbor": True,
                                "neighbor_type": "anterior"
                            })
                
                results = expanded_results
            
            # Formatar contexto
            context_parts = []
            context_parts.append("=== CONTEXTO RELEVANTE ===\n")
            
            chunk_count = 0
            for result in results:
                is_neighbor = result.get("is_neighbor", False)
                neighbor_type = result.get("neighbor_type", "")
                
                prefix = f"  [Chunk vizinho {neighbor_type}]" if is_neighbor else f"  [{chunk_count + 1}]"
                relevance = f" (relevância: {result['relevance']:.2%})" if "relevance" in result else ""
                
                context_parts.append(f"{prefix} {result['source']} (pág. {result['page']}){relevance}")
                context_parts.append(f"{result['text']}\n")
                
                if not is_neighbor:
                    chunk_count += 1
            
            context = "\n".join(context_parts)
            logger.info(f"📄 Contexto expandido com {chunk_count} chunks principais")
            return context
            
        except Exception as e:
            logger.error(f"Erro ao preparar contexto expandido: {str(e)}")
            return ""
    
    def get_context(self, query: str, top_k: int = 5, min_relevance: float = 0.0) -> str:
        """
        Retorna contexto formatado para passar ao LLM.
        
        Args:
            query: Pergunta do usuário
            top_k: Número de chunks a recuperar
            min_relevance: Relevância mínima (0-1)
            
        Returns:
            Contexto formatado como string
        """
        try:
            results = self.search(query, top_k=top_k, rerank=True)
            
            # Filtrar por relevância mínima
            relevant_results = [
                r for r in results 
                if r["relevance"] >= min_relevance
            ]
            
            if not relevant_results:
                return ""
            
            # Formatar contexto
            context_parts = []
            context_parts.append("=== CONTEXTO RELEVANTE ===\n")
            
            for i, result in enumerate(relevant_results, 1):
                context_parts.append(f"[{i}] {result['source']} (pág. {result['page']}, relevância: {result['relevance']:.2%})")
                context_parts.append(f"{result['text']}\n")
            
            context = "\n".join(context_parts)
            logger.info(f"📄 Contexto preparado com {len(relevant_results)} chunks")
            return context
            
        except Exception as e:
            logger.error(f"Erro ao preparar contexto: {str(e)}")
            return ""
    
    def delete_document(self, document_id: str) -> int:
        """
        Remove todos os chunks de um documento.
        
        Args:
            document_id: ID do documento a remover
            
        Returns:
            Número de chunks removidos
        """
        try:
            # Buscar todos os chunks do documento
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}}
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"🗑️  {len(results['ids'])} chunks removidos")
                return len(results["ids"])
            
            return 0
            
        except Exception as e:
            logger.error(f"Erro ao deletar documento: {str(e)}")
            return 0
    
    def clear_all(self) -> None:
        """Remove todos os documentos da coleção."""
        try:
            # Deletar e recriar a coleção
            self.client.delete_collection(name="documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("🗑️  Banco de dados limpo")
        except Exception as e:
            logger.error(f"Erro ao limpar banco: {str(e)}")
    
    def get_stats(self) -> Dict[str, any]:
        """Retorna estatísticas do banco de dados."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "model": self.model_name,
                "db_path": str(DB_PATH),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {str(e)}")
            return {}


# Função auxiliar para uso simples
def quick_add_document(text: str, source: str = "documento", page: int = 0) -> Tuple[str, int]:
    """
    Adiciona um documento rapidamente ao banco de embeddings.
    
    Args:
        text: Conteúdo do documento
        source: Nome/fonte do documento
        page: Página do documento
        
    Returns:
        Tupla (document_id, número de chunks)
    """
    retriever = DocumentRetriever()
    doc_id = hashlib.md5(source.encode()).hexdigest()[:16]
    result = retriever.add_document(doc_id, text, source=source, page=page)
    return doc_id, result["count"]


def quick_search(query: str, top_k: int = 5) -> str:
    """
    Busca contexto relevante para uma query.
    
    Args:
        query: Pergunta do usuário
        top_k: Número de resultados
        
    Returns:
        Contexto formatado
    """
    retriever = DocumentRetriever()
    return retriever.get_context(query, top_k=top_k)
