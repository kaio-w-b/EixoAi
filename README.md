# EixoAi

## 📚 Sobre o Projeto

EixoAi é um projeto de estudo focado em **RAG (Retrieval-Augmented Generation)** e **LLMs (Large Language Models)**. O objetivo principal é desenvolver um **chatbot inteligente com arquitetura RAG** que permite ao usuário escolher quais documentos serão utilizados como contexto para as respostas geradas pela IA.

### Características Principais

- 🤖 Chatbot baseado em LLM com capacidade de RAG
- 📄 Seleção dinâmica de documentos pelo usuário
- 🎛️ Interface construída com Gradio
- 🔍 Recuperação inteligente de informações relevantes
- 💾 Persistência de embeddings em banco de vetores

## 📁 Estrutura do Repositório

```
EixoAi/
├── README.md                 # Este arquivo - documentação principal
├── requirements.txt          # Dependências do projeto
├── data/                     # Pasta para armazenar documentos de entrada
├── src/                      # Código-fonte do projeto
│   ├── __init__.py          # Inicialização do pacote
│   ├── app.py               # Aplicação principal (interface Gradio)
│   ├── ingester.py          # Módulo para processar e ingerir documentos
│   ├── llm_chain.py         # Configuração da cadeia LLM com RAG
│   └── retriever.py         # Módulo de recuperação de documentos relevantes
└── vector_db/               # Banco de dados vetorial (armazena embeddings)
```

### Descrição dos Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `app.py` | Aplicação principal que executa a interface Gradio. Gerencia a seleção de documentos e orquestra a comunicação entre o usuário e o chatbot. |
| `ingester.py` | Responsável por carregar, processar e preparar documentos. Converte os dados em chunks e gera embeddings para armazenamento no banco vetorial. |
| `llm_chain.py` | Define a cadeia de processamento que integra o LLM com o sistema de RAG. Combina as informações recuperadas com as capacidades generativas do modelo. |
| `retriever.py` | Módulo que implementa a lógica de busca e recuperação de documentos relevantes do banco de vetores baseado na query do usuário. |
| `vector_db/` | Diretório que armazena o banco de dados vetorial com os embeddings dos documentos para recuperação eficiente. |
| `data/` | Pasta para armazenar os documentos que serão utilizados como fonte de conhecimento para o RAG. |

## 🚀 Como Começar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute a aplicação:
   ```bash
    run src/app.py
   ```

3. Faa o upload dos documentos desejados e inicie uma conversa com o chatbot!