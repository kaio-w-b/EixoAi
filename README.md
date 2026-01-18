# EixoAI - Chat Inteligente com PDFs

**Uma aplicação RAG (Retrieval-Augmented Generation) completa com busca semântica, LLM e interface web.**

---

## 📚 Sobre o Projeto

**EixoAI** é um sistema inteligente que combina:
- 🤖 **LLM (Large Language Models)** via Groq API
- 🔍 **RAG (Retrieval-Augmented Generation)** com embeddings semânticos
- 💾 **ChromaDB** para persistência de vetores
- 🎛️ **Interface Web** com Gradio
- 📄 **Processamento de PDFs** automático

O objetivo é criar um **chatbot contextualizado** que entende documentos e responde perguntas com precisão usando busca semântica.

### ✨ Características Principais

| Recurso | Descrição |
|---------|-----------|
| 📤 **Upload de PDFs** | Carregue PDFs e o sistema processa automaticamente |
| 🧠 **Embeddings Semânticos** | Gera embeddings consistentes com `sentence-transformers` |
| 🔎 **Busca Inteligente** | Encontra o contexto mais relevante para cada pergunta |
| 💬 **Chat com IA** | Conversa natural com a LLM usando contexto do PDF |
| 💾 **Persistência** | Armazena embeddings em ChromaDB para reutilização |
| ⚡ **Rápido e Leve** | Usa modelo all-MiniLM-L6-v2 (eficiente) |
| 🔧 **Tratamento de Erros** | Mostra erros como mensagens no chat |

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                     EixoAI - Fluxo Completo                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. UPLOAD PDF                                              │
│     ↓                                                        │
│  2. EXTRAÇÃO (ingester.py)                                  │
│     → Extrai texto do PDF                                  │
│     ↓                                                        │
│  3. CHUNKING (retriever.py)                                 │
│     → Divide em pedaços consistentes (512 chars)           │
│     ↓                                                        │
│  4. EMBEDDINGS (sentence-transformers)                      │
│     → Converte para vetores semânticos                     │
│     ↓                                                        │
│  5. ARMAZENAMENTO (ChromaDB)                                │
│     → Salva vetores persistentemente                        │
│     ↓                                                        │
│  6. CHAT                                                    │
│     → Pergunta do usuário                                  │
│     → Busca semântica por contexto relevante               │
│     → Envia para LLM com contexto                          │
│     → Retorna resposta contextualizada                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Estrutura do Repositório

```
EixoAi/
├── .env                      # Configuração (GROQ_API_KEY)
├── .gitignore               # Arquivos ignorados
├── requirements.txt         # Dependências Python
├── README.md                # Este arquivo
│
├── src/                     # Código-fonte principal
│   ├── __init__.py
│   ├── app.py               # Interface Gradio (INÍCIO AQUI)
│   ├── ingester.py          # Extração de PDFs
│   ├── llm_chain.py         # Integração com Groq LLM
│   └── retriever.py         # Busca semântica com ChromaDB
│
├── data/
│   └── uploads/             # PDFs carregados pelos usuários
│
└── vector_db/               # Banco de dados de embeddings (ChromaDB)
    ├── chroma.parquet
    └── index/
```

### 📝 Descrição dos Módulos

| Arquivo | Função |
|---------|--------|
| **`app.py`** | Interface Gradio com upload de PDF e chat em tempo real |
| **`ingester.py`** | Lê PDFs e extrai texto usando PyPDF |
| **`llm_chain.py`** | Gerencia conversa com Groq API, histórico e prompts |
| **`retriever.py`** | **Core do RAG**: chunking, embeddings, busca semântica |
| **`test.py`** | Menu de testes para validar cada componente |

---

## 🚀 Quick Start

### Pré-requisitos
- Python 3.10+
- pip ou uv
- Chave Groq (gratuita em https://console.groq.com)

### 1️⃣ Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/EixoAi.git
cd EixoAi

# Crie ambiente virtual (opcional)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instale dependências
pip install -r requirements.txt
```

### 2️⃣ Configure Groq API

```bash
# Edite .env e adicione sua chave:
# GROQ_API_KEY=gsk_seu_token_aqui
```

Obtenha em: https://console.groq.com

### 3️⃣ Execute a Aplicação

**Interface Web (Gradio):**
```bash
python src/app.py
# Acesse: http://localhost:7860
```

---

## 💡 Como Usar

### Via Interface Web

1. **Faça Upload de um PDF**
   - Clique em "Selecione um PDF"
   - O sistema extrai e processa automaticamente
   - Gera embeddings para busca semântica

2. **Converse com a IA**
   - Digite sua pergunta
   - O sistema busca contexto relevante automaticamente
   - A LLM responde com base no PDF

3. **Limpe e Recomece**
   - "Limpar Tudo" remove o PDF e histórico

---

## 🔧 Configuração Avançada

### Ajustar Tamanho de Chunks

Em `src/retriever.py`:
```python
CHUNK_SIZE = 512        # Aumentar = contexto maior
CHUNK_OVERLAP = 100     # Aumentar = melhor continuidade
```

### Mudar Modelo LLM

Em `src/llm_chain.py`:
```python
model = "llama-3.1-70b-versatile"  # Modelos disponíveis
temperature = 0.7                  # 0 = determinístico, 1 = criativo
max_tokens = 1024                  # Comprimento da resposta
```

### Mudar Modelo de Embeddings

Em `src/retriever.py`:
```python
MODEL_NAME = "all-MiniLM-L6-v2"  # Leve e rápido
# Alternativas:
# "all-mpnet-base-v2" - Mais preciso (lento)
# "multilingual-e5-small" - Para múltiplos idiomas
```

---

## 📊 Exemplo de Funcionamento

```
Usuario: "Qual é o objetivo principal?"

Sistema:
1. Busca semântica no PDF → Encontra 3 chunks relevantes
2. Monta contexto com chunks + pergunta
3. Envia para LLM com instrução
4. LLM responde contextualizadamente

Resposta: "O objetivo principal é..."
```

---

## 🔗 Dependências

```
pypdf==6.6.0                    # Leitura de PDFs
gradio==6.3.0                   # Interface web
chromadb==1.4.1                 # Banco vetorial
sentence-transformers==5.2.0    # Embeddings
langchain==1.2.4                # Orquestração LLM
langchain-groq==1.1.1           # Plugin Groq
python-dotenv==1.2.1            # Variáveis de ambiente
```

---

## 🚨 Troubleshooting

### Erro: "GROQ_API_KEY não definida"
```bash
# Verifique se .env existe
cat .env
# Deve ter: GROQ_API_KEY=gsk_...
```

### Erro: "PDF não contém texto"
- PDFs escaneados precisam de OCR (não suportado ainda)
- Tente com PDFs com texto selecionável

### Erro: "Modelo de embeddings não encontrado"
```bash
# Baixa o modelo automaticamente na primeira execução
# Se travar, download manual:
python -m sentence_transformers.models.download all-MiniLM-L6-v2
```

### ChromaDB com erro de permissão
```bash
# Limpe o banco:
rm -rf vector_db/
# Será recriado automaticamente
```

---

## 📈 Roadmap

- [ ] Suporte a múltiplos PDFs simultâneos
- [ ] Histórico persistente de conversas
- [ ] OCR para PDFs escaneados
- [ ] Dashboard de analytics
- [ ] Exportar conversas (JSON/PDF)
- [ ] Interface mobile
- [ ] Deploy em produção (Docker)

---

## 🤝 Contribuindo

Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Fazer pull requests
- Melhorar documentação

---

## 📄 Licença

MIT License - veja LICENSE para detalhes

---

## 👨‍💻 Autor

**Kaio W. B.** - [GitHub](https://github.com/kaio-w-b)

---

## 📚 Recursos Úteis

- [Documentação Groq](https://console.groq.com/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [Gradio](https://gradio.app/)
- [RAG Explainer](https://docs.langchain.com/docs/modules/chains/popular/qa_with_sources)

---

**Última atualização:** 17 de Janeiro de 2026
**Versão:** 1.0.0