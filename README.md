# EixoAI - Sistema de QA com LLM e RAG aplicado a documentos jurídicos

**Sistema de Question Answering baseado em LLM + RAG (Retrieval-Augmented Generation), com avaliação quantitativa utilizando a Constituição Federal de 1988 como base de conhecimento.**

---

## 🚀 Demonstração

Interface interativa disponível via Gradio (execução local):

```bash
python src/app.py
# http://localhost:7860
```

---

## 📚 Sobre o Projeto

A consulta a documentos jurídicos extensos, como a Constituição Federal de 1988, é um processo manual, demorado e sujeito a erros de interpretação.

O EixoAI resolve esse problema reduzindo consultas de minutos para segundos, utilizando um sistema de RAG (Retrieval-Augmented Generation) que:

* Recupera trechos relevantes via busca semântica
* Gera respostas contextualizadas com LLM
* Fundamenta respostas com base no documento original

O sistema foi avaliado quantitativamente com dataset próprio, garantindo confiabilidade e transparência nos resultados.

---

## 🎯 Objetivo

Construir um sistema modular de IA capaz de:

* Interpretar documentos extensos automaticamente
* Recuperar contexto relevante com busca semântica
* Gerar respostas fundamentadas usando LLM
* Reduzir alucinações através de RAG
* Permitir adaptação para diferentes domínios

---

## ✨ Principais Características

| Recurso                       | Descrição                                          |
| ----------------------------- | -------------------------------------------------- |
| 📄 **Processamento de PDFs**  | Extração automática de texto                       |
| 🧠 **Embeddings Semânticos**  | Representação vetorial com `sentence-transformers` |
| 🔎 **Busca Inteligente**      | Recuperação de contexto relevante via similaridade |
| 🤖 **Geração com LLM**        | Respostas contextualizadas com Groq API            |
| 💾 **Persistência Vetorial**  | Armazenamento com ChromaDB                         |
| 🎛️ **Interface Web**          | Interação via Gradio                               |
| 📊 **Avaliação de Respostas** | Testes com perguntas reais sobre a Constituição    |
| ⚙️ **Arquitetura Modular**    | Fácil extensão para novos domínios                 |

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                     EixoAI - Fluxo Completo                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. UPLOAD PDF                                              │
│     ↓                                                       │
│  2. EXTRAÇÃO (ingester.py)                                  │
│     → Extrai texto do PDF                                   │
│     ↓                                                       │
│  3. CHUNKING (retriever.py)                                 │
│     → Divide em pedaços consistentes                        │
│     ↓                                                       │
│  4. EMBEDDINGS (sentence-transformers)                      │
│     → Converte para vetores semânticos                      │
│     ↓                                                       │
│  5. ARMAZENAMENTO (ChromaDB)                                │
│     → Salva vetores persistentemente                        │
│     ↓                                                       │
│  6. CHAT                                                    │
│     → Pergunta do usuário                                   │
│     → Busca semântica por contexto relevante                │
│     → Envia para LLM com contexto                           │
│     → Retorna resposta contextualizada                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Avaliação

O sistema foi avaliado utilizando um dataset próprio com 48 queries baseadas na Constituição Federal de 1988, incluindo níveis de dificuldade (easy, medium, hard).

### Critérios analisados:

* Relevância da resposta
* Uso correto do contexto recuperado
* Coerência com o documento original

### ⚙️ Configuração:

* Chunks gerados: 1499
* Top-K retrieval: 5
* Latência média: 14.3 ms por query

### 📈 Métricas de Recuperação:

| Métrica                | Descrição  |
| ---------------------- | ---------- |
|  **Hit Rate**          | 79.2%      |
|  **MRR**               | 64.7%      |
|  **Precision@K**      | 30.8%      |
|  **NDCG**              | 67.7%      |
|  **Keyword Coverage**  | 65.3%      |

### 📊 Performance por Dificuldade:

| Nível         | Hit Rate  |
| ------------- | --------- |
| 🟢 **Easy**   | 92.9%     |
| 🟡 **Medium** | 73.9%     |
| 🔴 **Hard**   | 72.7%     |

### 🧠 Interpretação dos Resultados:

* Alta capacidade de recuperação semântica (Hit Rate ~80%)
* Excelente desempenho em perguntas diretas (easy)
* Boa generalização em perguntas complexas (hard ~72%)

### Pontos de melhoria:

* Queries jurídicas específicas (ex: habeas corpus, cláusulas pétreas)
* Aumento de precisão no Top-K (Precision@K)

### 🏁 Veredicto

* ✔ Excelente desempenho em recuperação de contexto
* ✔ Sistema robusto para consultas jurídicas reais
* ⚠️ Oportunidade de melhoria em queries altamente específicas

### 🧪 Dataset de Avaliação

O projeto utiliza um dataset próprio estruturado com:

* 48 perguntas baseadas na CF/88
* Classificação por dificuldade
* Ground truth (respostas esperadas)
* Keywords para validação de contexto
* Variações semânticas

Isso permite avaliar:

* Qualidade da recuperação (retrieval)
* Robustez semântica
* Cobertura do conhecimento
---


## 💬 Exemplo de Uso

**Pergunta:**

> Quais são os direitos fundamentais garantidos pela Constituição?

**Resposta:**

> De acordo com o contexto fornecido, os direitos fundamentais garantidos pela Constituição incluem:
>
> 1. **Direitos Sociais**: 
>    - Educação
>    - Saúde
>    - Alimentação
>    - Trabalho
>    - Moradia
>    - Transporte
>    - Lazer
>    - Segurança
>    - Previdência social
>    - Proteção à maternidade e à infância
>    - Assistência aos desamparados
> 
> 2. **Direitos Humanos e Liberdades Fundamentais**: 
>    - Todos os direitos humanos e liberdades fundamentais para pessoas com deficiência, sem discriminação.
>    - Direito à não discriminação por causa de deficiência.
>    - Direito à igualdade de condições para pessoas com deficiência.
> 
> 3. **Direitos Específicos para Pessoas com Deficiência**:
>    - Pleno exercício de todos os direitos humanos e liberdades fundamentais.
>    - Proteção e promoção dos direitos humanos.
>    - Acesso a programas e políticas que levem em conta a proteção e promoção dos direitos humanos das pessoas com deficiência.
> 
> 4. **Direitos Econômicos, Sociais e Culturais**:
>    - Medidas para assegurar progressivamente o pleno exercício desses direitos, com base nos recursos disponíveis e, quando necessário, no âmbito da cooperação internacional.
> 
> 5. **Direitos Relacionados à Família e à Criança**:
>    - Direito à fertilidade em igualdade de condições.
>    - Direitos e responsabilidades relacionados à guarda, custódia, curatela e adoção de crianças.
>   - Igualdade de direitos em relação à vida familiar para crianças com deficiência.
> 
> Esses direitos estão previstos na Constituição da República Federativa do Brasil e em tratados internacionais sobre direitos humanos dos quais o Brasil é signatário.

---

## 🧱 Estrutura do Projeto

```
EixoAi/
├── .env                         # Configuração (GROQ_API_KEY)
├── .gitignore                   # Arquivos ignorados
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
│
├── src/                         # Código-fonte principal
│   ├── __init__.py
│   ├── app.py                   # Interface Gradio (INÍCIO AQUI)
│   ├── ingester.py              # Extração de PDFs
│   ├── llm_chain.py             # Integração com Groq LLM
│   ├── retriever.py             # Busca semântica com ChromaDB
│   ├── test.py                  # Menu de testes e validação manual
│   ├── eval/                    # Pacote de avaliação automatizada (testes unitários + métricas)
│   │   ├── __init__.py
│   │   ├── evaluator.py         # Avaliação de qualidade RAG (RAGEvaluator)
│   │   ├── models.py            # Modelos de dados para avaliação (EvalReport/EvalResult)
│   │   └── qa_pairs.py          # Geração e manipulação de pares Q&A
│   │  
│   └── test_suite/              # Framework interno de testes integrados
│       ├── __init__.py          
│       ├── conftest.py          # Utilitários compartilhados: .env, texto fixture, fábrica de PDFs
│       ├── runner.py            # Executor de testes: TestResult e TestRunner para validação
│       ├── test_ingester.py     # Testes unitários para extração de texto de PDFs
│       ├── test_integration.py  # Testes de integração: pipeline completo PDF→busca→contexto
│       ├── test_llm_chain.py    # Testes unitários para LLMChain: histórico, mensagens, API
│       └── test_retriever.py    # Testes unitários para DocumentRetriever: chunking, busca, ChromaDB
│
├── tests/                   # Diretório para pytest
│
├── docs/                    # Documentação adicional
├── data/
│   └── uploads/             # PDFs carregados pelos usuários
│
└── vector_db/               # Banco de dados de embeddings (ChromaDB)
    ├── chroma.parquet
    └── index/
```

---

## ⚙️ Tecnologias Utilizadas

* Python
* LangChain
* ChromaDB
* Sentence Transformers
* Groq API (LLM)
* Gradio

---

## 🚀 Como Executar

```bash
git clone https://github.com/kaio-w-b/EixoAi.git
cd EixoAi

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

Configure sua chave:

```bash
GROQ_API_KEY=seu_token
```

Execute:

```bash
python src/app.py
```

Acesse:

```
http://localhost:7860
```

---

## 🔧 Configurações

### Chunking

```python
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
```

### Modelo LLM

```python
model = "llama-3.1-70b-versatile"
```

### Embeddings

```python
MODEL_NAME = "all-MiniLM-L6-v2"
```

---

## 🔮 Próximos Passos

* [ ] API REST com FastAPI
* [ ] Suporte a múltiplos documentos
* [ ] Avaliação automatizada de respostas
* [ ] Deploy em cloud (AWS/Render)
* [ ] Interface web aprimorada
* [ ] Suporte a OCR

---

## 🧠 Diferenciais Técnicos

* Implementação completa de pipeline RAG
* Uso de embeddings semânticos para recuperação de contexto
* Avaliação prática com dados reais
* Arquitetura modular e extensível
* Integração com LLM em ambiente real

---

## 👨‍💻 Autor

**Kaio Wellinghton Batista e Silva**
[GitHub](https://github.com/kaio-w-b)

---

## 📄 Licença

MIT License

---
