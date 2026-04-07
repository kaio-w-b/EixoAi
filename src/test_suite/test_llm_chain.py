"""
tests/test_llm_chain.py
───────────────────────
Testes unitários para src/llm_chain.py.
Cobre: inicialização, gerenciamento de histórico, preparação de mensagens
e (opcionalmente) uma chamada real à API Groq.
"""

import os
from typing import Tuple

from test_suite.conftest import _title, _warn
from test_suite.runner import TestRunner
from llm_chain import LLMChain


def test_llm_chain() -> Tuple[int, int]:
    print(_title("MÓDULO: llm_chain.py"))


    runner = TestRunner()

    # ── Inicialização ────────────────────────────────────────

    def t_init_default():
        chain = LLMChain()
        assert chain.model == "llama-3.3-70b-versatile"
        assert chain.temperature == 0.7
        assert chain.max_tokens  == 1024
        assert chain.conversation_history == []

    def t_init_custom():
        chain = LLMChain(model="gemma2-9b-it", temperature=0.3, max_tokens=512)
        assert chain.model       == "gemma2-9b-it"
        assert chain.temperature == 0.3
        assert chain.max_tokens  == 512

    # ── Gerenciamento de histórico ───────────────────────────

    def t_add_message():
        chain = LLMChain()
        chain.add_message("user", "Olá!")
        assert len(chain.conversation_history) == 1
        assert chain.conversation_history[0]["role"]    == "user"
        assert chain.conversation_history[0]["content"] == "Olá!"

    def t_add_message_empty_raises():
        chain = LLMChain()
        try:
            chain.add_message("user", "   ")
            assert False, "Deveria lançar ValueError para conteúdo vazio"
        except ValueError:
            pass

    def t_add_multiple_messages():
        chain = LLMChain()
        chain.add_message("user",      "Pergunta 1")
        chain.add_message("assistant", "Resposta 1")
        chain.add_message("user",      "Pergunta 2")
        assert len(chain.conversation_history) == 3
        assert chain.conversation_history[0]["role"] == "user"
        assert chain.conversation_history[1]["role"] == "assistant"

    def t_clear_history():
        chain = LLMChain()
        chain.add_message("user",      "Mensagem teste")
        chain.add_message("assistant", "Resposta teste")
        chain.clear_history()
        assert chain.conversation_history == [], "Histórico deve estar vazio após clear"

    # ── Preparação de mensagens ──────────────────────────────

    def t_prepare_messages_no_system():
        chain = LLMChain()
        msgs = chain._prepare_messages("Oi")
        assert any(m["role"] == "user"   for m in msgs), "Deve ter mensagem de user"
        assert all(m["role"] != "system" for m in msgs), "Não deve ter system prompt"

    def t_prepare_messages_with_system():
        chain = LLMChain()
        msgs = chain._prepare_messages("Oi", system_prompt="Seja útil.")
        assert msgs[0]["role"]    == "system", "System deve ser a primeira mensagem"
        assert msgs[0]["content"] == "Seja útil."

    def t_prepare_messages_with_context():
        chain = LLMChain()
        msgs = chain._prepare_messages("O que é IA?", context="IA é inteligência artificial.")
        last = msgs[-1]
        assert "contexto" in last["content"].lower(), "Deve incluir bloco de contexto"
        assert "O que é IA?" in last["content"],      "Deve incluir a pergunta original"

    def t_prepare_messages_history_included():
        chain = LLMChain()
        chain.add_message("user",      "Primeira mensagem")
        chain.add_message("assistant", "Primeira resposta")
        msgs = chain._prepare_messages("Segunda mensagem")
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) >= 2, "Histórico deve estar incluído nas mensagens"

    # ── Chamada real à API (opcional) ────────────────────────

    def t_generate_response_live():
        """Pula silenciosamente se GROQ_API_KEY não estiver configurada."""
        if not os.getenv("GROQ_API_KEY"):
            print(_warn("   GROQ_API_KEY não configurada — pulando teste de API real"))
            return
        chain = LLMChain(max_tokens=50, stream=False)
        response = chain.generate_response(
            "Diga apenas: 'Teste OK'",
            system_prompt="Responda exatamente o que pedir, nada mais."
        )
        assert isinstance(response, str), "Resposta deve ser string"
        assert len(response) > 0,         "Resposta não pode ser vazia"

    # ── Registro ─────────────────────────────────────────────

    runner.run("Inicialização com valores padrão",           t_init_default)
    runner.run("Inicialização com valores customizados",     t_init_custom)
    runner.run("add_message adiciona corretamente",          t_add_message)
    runner.run("add_message vazia levanta ValueError",       t_add_message_empty_raises)
    runner.run("Múltiplas mensagens no histórico",           t_add_multiple_messages)
    runner.run("clear_history limpa o histórico",            t_clear_history)
    runner.run("_prepare_messages sem system prompt",        t_prepare_messages_no_system)
    runner.run("_prepare_messages com system prompt",        t_prepare_messages_with_system)
    runner.run("_prepare_messages inclui contexto",          t_prepare_messages_with_context)
    runner.run("_prepare_messages inclui histórico",         t_prepare_messages_history_included)
    runner.run("generate_response (API real)",               t_generate_response_live)

    return runner.summary()