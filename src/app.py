import gradio as gr
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from ingester import extract_text_from_pdf
from llm_chain import LLMChain
from retriever import DocumentRetriever
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("../data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

current_context = ""
current_document_id = ""
chat_history = []
llm_chain = None
retriever = None

# ─── HTML dos estados de status ────────────────────────────────────────────────

def _status_idle() -> str:
    return """
    <div style="
        display: flex; align-items: center; gap: 10px;
        padding: 14px 18px; border-radius: 10px;
        background: #1e1e2e; border: 1.5px dashed #3b3b58;
        font-family: 'JetBrains Mono', monospace; font-size: 13px;
    ">
        <div style="
            width: 10px; height: 10px; border-radius: 50%;
            background: #45455a; flex-shrink: 0;
        "></div>
        <span style="color: #6c6c8a;">Nenhum PDF carregado — faça upload para começar</span>
    </div>
    """

def _status_loading() -> str:
    return """
    <style>
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pulse-text { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
    <div style="
        display: flex; align-items: center; gap: 12px;
        padding: 14px 18px; border-radius: 10px;
        background: #1a1a2e; border: 1.5px solid #f59e0b;
        font-family: 'JetBrains Mono', monospace; font-size: 13px;
    ">
        <div style="
            width: 16px; height: 16px; border-radius: 50%;
            border: 2.5px solid #f59e0b44;
            border-top-color: #f59e0b;
            animation: spin 0.8s linear infinite;
            flex-shrink: 0;
        "></div>
        <span style="color: #f59e0b; animation: pulse-text 1.4s ease-in-out infinite;">
            Processando PDF e gerando embeddings…
        </span>
    </div>
    """

def _status_ready(filename: str, chunks: int, chars: int) -> str:
    return f"""
    <div style="
        display: flex; flex-direction: column; gap: 6px;
        padding: 14px 18px; border-radius: 10px;
        background: #0d1f17; border: 1.5px solid #22c55e;
        font-family: 'JetBrains Mono', monospace; font-size: 13px;
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="
                width: 10px; height: 10px; border-radius: 50%;
                background: #22c55e; flex-shrink: 0;
                box-shadow: 0 0 6px #22c55e88;
            "></div>
            <span style="color: #22c55e; font-weight: 600;">PDF pronto — pode enviar mensagens</span>
        </div>
        <div style="
            display: flex; gap: 16px; padding-left: 20px;
            color: #4ade8088; font-size: 11px;
        ">
            <span>📄 {filename}</span>
            <span>·</span>
            <span>🧩 {chunks} chunks</span>
            <span>·</span>
            <span>📝 {chars:,} caracteres</span>
        </div>
    </div>
    """

def _status_error(msg: str) -> str:
    return f"""
    <div style="
        display: flex; align-items: center; gap: 10px;
        padding: 14px 18px; border-radius: 10px;
        background: #1f0d0d; border: 1.5px solid #ef4444;
        font-family: 'JetBrains Mono', monospace; font-size: 13px;
    ">
        <div style="
            width: 10px; height: 10px; border-radius: 50%;
            background: #ef4444; flex-shrink: 0;
        "></div>
        <span style="color: #ef4444;">{msg}</span>
    </div>
    """

# ─── Lógica de negócio ─────────────────────────────────────────────────────────

def start_processing(file):
    """
    Chamado imediatamente ao clicar em 'Processar PDF'.
    Retorna o estado de carregamento e bloqueia o chat.
    """
    if file is None:
        return (
            _status_error("Selecione um arquivo PDF antes de processar."),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),   # clear_btn permanece ativo
        )
    return (
        _status_loading(),
        gr.update(interactive=False),  # bloqueia input de texto
        gr.update(interactive=False),  # bloqueia botão enviar
        gr.update(interactive=False),  # bloqueia limpar tudo
    )


def process_pdf(file):
    """
    Processa o PDF, extrai texto e gera embeddings.
    Retorna o status final e libera (ou não) o chat.
    """
    global current_context, current_document_id, retriever

    if file is None:
        return (
            _status_error("Nenhum arquivo selecionado."),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
        )

    try:
        if retriever is None:
            retriever = DocumentRetriever()

        file_path = file.get("name") if isinstance(file, dict) else file
        if not file_path:
            return (
                _status_error("Arquivo inválido."),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )

        text = extract_text_from_pdf(str(file_path))

        if not text:
            return (
                _status_error("PDF sem texto — pode ser um arquivo escaneado ou corrompido."),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),   # libera clear_btn mesmo com erro
            )

        file_name = Path(file_path).name if isinstance(file_path, str) else "documento.pdf"
        new_doc_id = hashlib.md5(file_name.encode()).hexdigest()[:16]

        # Limpa TODO o banco antes de inserir — garante que nunca existam
        # chunks de sessões anteriores ou de uploads anteriores na mesma sessão.
        retriever.clear_all()
        logger.info("🗑️  Banco limpo antes do novo upload")

        current_document_id = new_doc_id
        current_context = text

        result = retriever.add_document(
            document_id=current_document_id,
            text=text,
            source=file_name,
        )

        logger.info(f"PDF processado: {file_name} ({result['count']} chunks)")
        return (
            _status_ready(file_name, result["count"], len(text)),
            gr.update(interactive=True),   # libera input
            gr.update(interactive=True),   # libera botão enviar
            gr.update(interactive=True),   # libera limpar tudo
        )

    except Exception as e:
        logger.error(f"Erro ao processar PDF: {str(e)}")
        return (
            _status_error(f"Erro: {str(e)}"),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),   # libera limpar tudo mesmo com erro
        )


def send_message(user_message: str, history: List[Dict]) -> Tuple[str, List[Dict], str]:
    global llm_chain, retriever, current_document_id

    if not user_message.strip():
        return "", history, ""

    chunk_info = ""

    try:
        if llm_chain is None:
            llm_chain = LLMChain(stream=False)

        context = ""
        if current_document_id and retriever:
            try:
                results = retriever.search(user_message, top_k=5, rerank=True)
                if results:
                    chunk_parts = ["🔍 **CHUNKS RECUPERADOS — RASTREAMENTO COMPLETO**\n"]
                    chunk_parts.append("=" * 100)
                    for i, result in enumerate(results, 1):
                        chunk_parts.append(f"\n### **[{i}] CHUNK #{result['chunk']}**")
                        chunk_parts.append(f"📄 **Arquivo:** `{result['source']}`")
                        chunk_parts.append(f"📍 **Página:** {result['page']}")
                        chunk_parts.append(f"🎯 **Relevância:** {result['relevance']:.1%}")
                        chunk_parts.append(f"🆔 **ID:** `{result['id']}`")
                        chunk_parts.append(f"📊 **Distância:** {result['distance']:.4f}")
                        chunk_parts.append(f"\n**CONTEÚDO COMPLETO:**")
                        chunk_parts.append("```")
                        chunk_parts.append(result["text"])
                        chunk_parts.append("```")
                        chunk_parts.append("-" * 100)
                    chunk_info = "\n".join(chunk_parts)
                    context = retriever.get_context(user_message, top_k=5, min_relevance=0.0)
                else:
                    chunk_info = "⚠️ Nenhum chunk relevante encontrado para esta pergunta."
            except Exception as e:
                logger.warning(f"Erro ao recuperar contexto: {str(e)}")
                chunk_info = f"❌ Erro ao recuperar chunks: {str(e)}"
        else:
            chunk_info = "📭 Nenhum PDF carregado."

        system_prompt = (
            "Você é um assistente útil que responde perguntas sobre documentos. "
            "Use o contexto fornecido para responder com precisão. "
            "Se a informação não estiver no contexto, diga claramente."
        )

        # Limite de segurança: nunca enviar mais do que ~12 000 caracteres de contexto
        # (~3 000 tokens), deixando espaço para o histórico e a resposta.
        MAX_CONTEXT_CHARS = 12_000
        safe_context = context[:MAX_CONTEXT_CHARS] if context else ""

        response = llm_chain.generate_response(
            user_message=user_message,
            system_prompt=system_prompt,
            context=safe_context,
        )

    except Exception as e:
        response = f"❌ Erro ao processar mensagem:\n{str(e)}"
        chunk_info = f"❌ Erro: {str(e)}"
        logger.error(f"Erro ao chamar LLM: {str(e)}")

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return "", history, chunk_info


def clear_chat():
    global current_context, current_document_id, llm_chain, retriever
    current_context = ""

    if current_document_id and retriever:
        try:
            retriever.delete_document(current_document_id)
        except Exception as e:
            logger.warning(f"Erro ao deletar documento: {str(e)}")

    current_document_id = ""

    if llm_chain is not None:
        llm_chain.clear_history()

    return (
        [],
        _status_idle(),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

# ─── CSS customizado ───────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Chat bloqueado — feedback visual claro */
.chat-blocked textarea {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}
.chat-blocked button {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}

/* Remove borda padrão do Gradio no status */
#pdf-status {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* Área de chunk com fonte mono */
#chunk-tracker .prose {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

/* Botão primário */
.primary-btn {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}
"""

# ─── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(title="EixoAI") as demo:

    gr.Markdown("# 📚 EixoAI — Chat com PDFs")
    gr.Markdown("Busca semântica com rastreamento de chunks em tempo real")

    with gr.Row():

        # ── Coluna esquerda: upload + status ──────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Documento")

            pdf_file = gr.File(
                label="Selecione um PDF",
                file_types=[".pdf"],
            )

            upload_btn = gr.Button(
                "⚙️  Processar PDF",
                variant="primary",
                elem_classes=["primary-btn"],
            )

            # Painel de status — muda conforme o estado
            pdf_status = gr.HTML(
                value=_status_idle(),
                elem_id="pdf-status",
            )

            gr.Markdown("---")
            clear_btn = gr.Button("🧹  Limpar tudo", variant="secondary")

        # ── Coluna direita: chat ───────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")

            chatbot = gr.Chatbot(
                label="Histórico",
                height=420,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="Mensagem",
                    placeholder="Carregue um PDF para habilitar o chat…",
                    scale=4,
                    interactive=False,   # bloqueado por padrão
                    elem_id="msg-input",
                )
                send_btn = gr.Button(
                    "Enviar",
                    scale=1,
                    interactive=False,   # bloqueado por padrão
                    variant="primary",
                )

            gr.Markdown("---")
            gr.Markdown("### 🔍 Rastreamento de Chunks")
            chunk_tracker = gr.Markdown(
                "Nenhum chunk recuperado ainda.",
                elem_id="chunk-tracker",
            )

    # ── Event handlers ─────────────────────────────────────────────────────────

    # 1. Clicar em "Processar": imediatamente mostra spinner e bloqueia chat
    upload_btn.click(
        fn=start_processing,
        inputs=[pdf_file],
        outputs=[pdf_status, msg_input, send_btn, clear_btn],
        queue=False,          # resposta imediata, sem fila
    ).then(
        # 2. Em seguida, executa o processamento real e atualiza o status
        fn=process_pdf,
        inputs=[pdf_file],
        outputs=[pdf_status, msg_input, send_btn, clear_btn],
    )

    # Enviar mensagem
    send_btn.click(
        fn=send_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, chunk_tracker],
    )

    msg_input.submit(
        fn=send_message,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, chunk_tracker],
    )

    # Limpar tudo: reseta status para idle e bloqueia o chat
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, pdf_status, msg_input, send_btn],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=CUSTOM_CSS)