import gradio as gr
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from ingester import extract_text_from_pdf
from llm_chain import LLMChain
from retriever import DocumentRetriever
import hashlib

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Diretório para armazenar PDFs
UPLOAD_DIR = Path("../data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Variáveis globais
current_context = ""
current_document_id = ""
chat_history = []
llm_chain = None
retriever = None


def process_pdf(file) -> str:
    """
    Processa um arquivo PDF, extrai texto e gera embeddings.
    """
    global current_context, current_document_id, retriever
    
    try:
        if file is None:
            return "❌ Nenhum arquivo selecionado"
        
        # Inicializar retriever se necessário
        if retriever is None:
            retriever = DocumentRetriever()
        
        # Gradio retorna um dicionário com 'name' e 'size'
        if isinstance(file, dict):
            file_path = file.get('name')
            if not file_path:
                return "❌ Erro: arquivo inválido"
        else:
            file_path = file
        
        # Extrair texto
        text = extract_text_from_pdf(str(file_path))
        
        if not text:
            return "❌ Erro: PDF não contém texto ou está corrompido"
        
        # Extrair nome do arquivo
        file_name = Path(file_path).name if isinstance(file_path, str) else "documento.pdf"
        
        # Gerar ID único do documento
        current_document_id = hashlib.md5(file_name.encode()).hexdigest()[:16]
        
        # Armazenar contexto completo
        current_context = text
        
        # Adicionar ao banco de embeddings
        result = retriever.add_document(
            document_id=current_document_id,
            text=text,
            source=file_name
        )
        
        # Informações do arquivo
        file_size = len(text)
        preview = text[:500] + "..." if len(text) > 500 else text
        
        info = f"""✅ **Arquivo processado com sucesso!**

📄 **Nome**: {file_name}
📊 **Caracteres**: {file_size}
🔍 **Chunks gerados**: {result['count']}

**Preview**:
```
{preview}
```

💡 Agora você pode fazer perguntas sobre o documento usando busca semântica!"""
        
        logger.info(f"PDF processado: {file_name} ({result['count']} chunks)")
        return info
        
    except Exception as e:
        logger.error(f"Erro ao processar PDF: {str(e)}")
        return f"❌ Erro: {str(e)}"


def send_message(user_message: str, history: List[Dict]) -> Tuple[str, List[Dict], str]:
    """
    Processa mensagem do usuário com busca semântica em embeddings.
    Retorna histórico no formato Gradio v6+ e informações completas de rastreamento.
    """
    global llm_chain, retriever, current_document_id
    
    if not user_message.strip():
        return "", history, ""
    
    chunk_info = ""
    
    try:
        # Inicializar LLMChain se necessário
        if llm_chain is None:
            llm_chain = LLMChain(stream=False)
        
        # Buscar contexto relevante se houver documento carregado
        context = ""
        if current_document_id and retriever:
            try:
                # Buscar chunks relevantes COM RERANKING
                results = retriever.search(user_message, top_k=5, rerank=True)
                
                if results:
                    # Formatar informações completas dos chunks
                    chunk_parts = ["🔍 **CHUNKS RECUPERADOS - RASTREAMENTO COMPLETO**\n"]
                    chunk_parts.append("=" * 100)
                    
                    for i, result in enumerate(results, 1):
                        # Cabeçalho do chunk com informações essenciais
                        chunk_parts.append(f"\n### **[{i}] CHUNK #{result['chunk']}**")
                        chunk_parts.append(f"📄 **Arquivo:** `{result['source']}`")
                        chunk_parts.append(f"📍 **Página:** {result['page']}")
                        chunk_parts.append(f"🎯 **Relevância:** {result['relevance']:.1%}")
                        chunk_parts.append(f"🆔 **ID:** `{result['id']}`")
                        chunk_parts.append(f"📊 **Distância:** {result['distance']:.4f}")
                        
                        # Conteúdo completo
                        chunk_parts.append(f"\n**CONTEÚDO COMPLETO:**")
                        chunk_parts.append("```")
                        chunk_parts.append(result['text'])
                        chunk_parts.append("```")
                        chunk_parts.append("-" * 100)
                    
                    chunk_info = "\n".join(chunk_parts)
                    
                    # Usar contexto formatado para LLM
                    context = retriever.get_context(user_message, top_k=5, min_relevance=0.0)
                    logger.info("🔍 Contexto semântico recuperado com reranking")
                else:
                    chunk_info = "⚠️ Nenhum chunk relevante encontrado para esta pergunta."
                    
            except Exception as e:
                logger.warning(f"Erro ao recuperar contexto: {str(e)}")
                chunk_info = f"❌ Erro ao recuperar chunks: {str(e)}"
        else:
            chunk_info = "📭 Nenhum PDF carregado. Faça upload de um PDF para usar busca semântica."
        
        # Tentar gerar resposta com a LLM
        system_prompt = (
            "Você é um assistente útil que responde perguntas sobre documentos. "
            "Use o contexto fornecido para responder com precisão. "
            "Se a informação não estiver no contexto, diga claramente."
        )
        
        response = llm_chain.generate_response(
            user_message=user_message,
            system_prompt=system_prompt,
            context=context if context else current_context
        )
        
        logger.info(f"Resposta gerada com sucesso")
        
    except Exception as e:
        # Se houver erro na API, informar o usuário
        error_msg = f"❌ Erro ao processar mensagem:\n{str(e)}"
        response = error_msg
        logger.error(f"Erro ao chamar LLM: {str(e)}")
        chunk_info = f"❌ Erro: {str(e)}"
    
    # Adicionar mensagem e resposta ao histórico no formato Gradio v6+
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    
    return "", history, chunk_info


def clear_chat() -> List[Dict]:
    """Limpa histórico do chat e o contexto."""
    global current_context, current_document_id, llm_chain, retriever
    current_context = ""
    
    # Remover documento do banco de embeddings
    if current_document_id and retriever:
        try:
            retriever.delete_document(current_document_id)
        except Exception as e:
            logger.warning(f"Erro ao deletar documento: {str(e)}")
    
    current_document_id = ""
    
    if llm_chain is not None:
        llm_chain.clear_history()
    
    return []


# Criar interface Gradio
with gr.Blocks(title="EixoAI") as demo:
    gr.Markdown("# 📚 EixoAI - Chat com PDFs")
    gr.Markdown("Interface com busca semântica e rastreamento de chunks")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Upload PDF")
            pdf_file = gr.File(
                label="Selecione um PDF",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("Processar PDF", variant="primary")
            pdf_info = gr.Markdown("Aguardando arquivo...")
            
            gr.Markdown("---")
            gr.Markdown("### 🧹 Ações")
            clear_btn = gr.Button("Limpar Tudo")
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(label="Histórico", height=400)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Mensagem",
                    placeholder="Digite sua mensagem...",
                    scale=4
                )
                send_btn = gr.Button("Enviar", scale=1)
            
            gr.Markdown("---")
            gr.Markdown("### 🔍 Rastreamento de Chunks")
            chunk_tracker = gr.Markdown(
                "Nenhum chunk recuperado ainda",
                label="Chunks Usados"
            )
    
    # Event handlers
    upload_btn.click(
        fn=process_pdf,
        inputs=[pdf_file],
        outputs=[pdf_info]
    )
    
    send_btn.click(
        fn=send_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, chunk_tracker]
    )
    
    msg_input.submit(
        fn=send_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, chunk_tracker]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)