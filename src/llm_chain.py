import logging
from typing import List, Dict, Optional, Generator
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

# Limite conservador: modelos Groq têm context window de 8 192 tokens para alguns
# e 32 768 para outros. Usamos 24 000 caracteres (~6 000 tokens) como teto total
# para histórico + contexto + pergunta, deixando ~1 024 tokens para a resposta.
MAX_HISTORY_TURNS = 10          # pares (user + assistant) mantidos no histórico
MAX_CONTEXT_CHARS = 12_000      # caracteres máximos do bloco de contexto RAG
MAX_TOTAL_MSG_CHARS = 24_000    # teto total de todos os conteúdos somados


class LLMChain:
    """Gerenciador de cadeia de LLM com Groq."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = True,
    ):
        self.client = Groq()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.conversation_history: List[Dict[str, str]] = []
        logger.info(f"LLMChain inicializado com modelo: {model}")

    # ── Histórico ────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        if not content.strip():
            raise ValueError("Conteúdo da mensagem não pode estar vazio")
        self.conversation_history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        self.conversation_history.clear()
        logger.info("Histórico de conversa limpo")

    def _trimmed_history(self) -> List[Dict[str, str]]:
        """
        Retorna os últimos MAX_HISTORY_TURNS pares do histórico.
        Garante que o histórico retornado sempre começa com uma mensagem
        de 'user' (nunca de 'assistant' isolada).
        """
        # Cada par = 2 mensagens (user + assistant)
        max_msgs = MAX_HISTORY_TURNS * 2
        trimmed = self.conversation_history[-max_msgs:]

        # Se começar com assistant, descarta a primeira entrada
        if trimmed and trimmed[0]["role"] == "assistant":
            trimmed = trimmed[1:]

        return trimmed

    # ── Preparação de mensagens ──────────────────────────────

    def _prepare_messages(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Histórico aparado
        messages.extend(self._trimmed_history())

        # Contexto truncado + pergunta
        safe_context = (context or "")[:MAX_CONTEXT_CHARS]
        content = (
            f"Contexto:\n{safe_context}\n\nPergunta:\n{user_message}"
            if safe_context
            else user_message
        )

        # Verificação de segurança: se o total ainda for grande demais,
        # descarta histórico mais antigo até caber
        while len(messages) > 1:  # preserva ao menos o system prompt
            total_chars = sum(len(m["content"]) for m in messages) + len(content)
            if total_chars <= MAX_TOTAL_MSG_CHARS:
                break
            # Remove o par mais antigo do histórico (user+assistant após system)
            start = 1 if messages and messages[0]["role"] == "system" else 0
            if start < len(messages) - 1:
                messages.pop(start + 1)  # assistant
                messages.pop(start)      # user
            else:
                break

        messages.append({"role": "user", "content": content})
        self.add_message("user", user_message)
        return messages

    # ── Geração ──────────────────────────────────────────────

    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        try:
            messages = self._prepare_messages(user_message, system_prompt, context)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=False,
            )

            assistant_message = response.choices[0].message.content
            self.add_message("assistant", assistant_message)
            logger.info(f"Resposta gerada ({len(assistant_message)} chars)")
            return assistant_message

        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            raise

    def generate_response_stream(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Generator[str, None, None]:
        try:
            messages = self._prepare_messages(user_message, system_prompt, context)
            full_response = ""

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=True,
            )

            for chunk in completion:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    full_response += delta_content
                    yield delta_content

            self.add_message("assistant", full_response)
            logger.info(f"Streaming concluído ({len(full_response)} chars)")

        except Exception as e:
            logger.error(f"Erro ao fazer streaming: {str(e)}")
            raise


# ── Função auxiliar ──────────────────────────────────────────

def quick_response(
    message: str,
    system_prompt: str = "Você é um assistente útil.",
    stream: bool = False,
) -> str | Generator[str, None, None]:
    chain = LLMChain(stream=stream)
    if stream:
        return chain.generate_response_stream(message, system_prompt)
    return chain.generate_response(message, system_prompt)