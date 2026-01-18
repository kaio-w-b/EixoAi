import logging
from typing import List, Dict, Optional, Generator
from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionMessage

# Carregar variáveis de ambiente
load_dotenv()

logger = logging.getLogger(__name__)

class LLMChain:
    """
    Gerenciador de cadeia de LLM com Groq.
    """
    
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = True
    ):
        """
        Inicializa a cadeia de LLM.
        
        Args:
            model: Modelo a usar
            temperature: Criatividade (0-1)
            max_tokens: Máximo de tokens na resposta
            top_p: Probabilidade acumulada para nucleus sampling
            stream: Se deve usar streaming
        """
        self.client = Groq()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"LLMChain inicializado com modelo: {model}")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Adiciona mensagem ao histórico.
        
        Args:
            role: "user", "assistant", "system"
            content: Conteúdo da mensagem
        """
        if not content.strip():
            raise ValueError("Conteúdo da mensagem não pode estar vazio")
        
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def clear_history(self) -> None:
        """Limpa o histórico da conversa."""
        self.conversation_history.clear()
        logger.info("Histórico de conversa limpo")
    
    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Gera resposta sem streaming.
        
        Args:
            user_message: Mensagem do usuário
            system_prompt: Instruções do sistema (opcional)
            context: Contexto do retriever (opcional)
            
        Returns:
            Resposta do LLM
        """
        try:
            messages = self._prepare_messages(
                user_message, 
                system_prompt, 
                context
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=False
            )
            
            assistant_message = response.choices[0].message.content
            self.add_message("assistant", assistant_message)
            
            logger.info(f"Resposta gerada com sucesso ({len(assistant_message)} chars)")
            return assistant_message
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            raise
    
    def generate_response_stream(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Gera resposta com streaming.
        
        Args:
            user_message: Mensagem do usuário
            system_prompt: Instruções do sistema (opcional)
            context: Contexto do retriever (opcional)
            
        Yields:
            Chunks de texto da resposta
        """
        try:
            messages = self._prepare_messages(
                user_message,
                system_prompt,
                context
            )
            
            full_response = ""
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=True
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
    
    def _prepare_messages(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Prepara lista de mensagens para a API.
        
        Args:
            user_message: Mensagem do usuário
            system_prompt: Prompt do sistema
            context: Contexto adicional
            
        Returns:
            Lista de mensagens formatada
        """
        messages = []
        
        # Adicionar system prompt se fornecido
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Adicionar histórico da conversa
        messages.extend(self.conversation_history)
        
        # Construir mensagem do usuário com contexto
        content = user_message
        if context:
            content = f"Contexto:\n{context}\n\nPergunta:\n{user_message}"
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        self.add_message("user", user_message)
        
        return messages


# Função auxiliar para uso simples
def quick_response(
    message: str,
    system_prompt: str = "Você é um assistente útil.",
    stream: bool = False
) -> str | Generator[str, None, None]:
    """
    Gera resposta rápida sem gerenciar histórico.
    
    Args:
        message: Mensagem do usuário
        system_prompt: Instruções do sistema
        stream: Se deve usar streaming
        
    Returns:
        Resposta do LLM (ou generator se stream=True)
    """
    chain = LLMChain(stream=stream)
    
    if stream:
        return chain.generate_response_stream(message, system_prompt)
    else:
        return chain.generate_response(message, system_prompt)
