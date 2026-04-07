# EixoAI - Sistema de QA com LLM e RAG aplicado a documentos jurídicos

**Sistema de Question Answering baseado em LLM + RAG (Retrieval-Augmented Generation), avaliado utilizando a Constituição Federal de 1988 como base de conhecimento.**

---

## 🚀 Demonstração

Interface interativa disponível via Gradio (execução local):

```bash
python src/app.py
# http://localhost:7860
```

---

## 📚 Sobre o Projeto

O acesso a documentos extensos e complexos, como a Constituição Federal de 1988, é pouco eficiente e exige busca manual e interpretação especializada.

O **EixoAI** resolve esse problema utilizando um sistema de **Question Answering com LLM e RAG**, permitindo consultas inteligentes com respostas contextualizadas e fundamentadas em documentos oficiais.

Diferente de aplicações genéricas, o sistema foi **testado e avaliado com dados reais**, utilizando a Constituição como base, garantindo maior confiabilidade nas respostas.

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
| 🎛️ **Interface Web**         | Interação via Gradio                               |
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

O sistema foi avaliado utilizando perguntas reais sobre a Constituição Federal de 1988.

### Critérios analisados:

* Relevância da resposta
* Uso correto do contexto recuperado
* Coerência com o documento original

### Exemplo:

**Pergunta:**

> O que garante o artigo 5º da Constituição?

**Resposta do sistema:**

> O artigo 5º da Constituição garante, aos brasileiros e estrangeiros residentes no País, a inviolabilidade do direito à:
>
> * Vida
> * Liberdade
> * Igualdade
> * Segurança
> * Propriedade

> Além disso, ele também estabelece alguns princípios fundamentais, como:
>
> * Igualdade entre homens e mulheres em direitos e obrigações
> * Ninguém será obrigado a fazer ou deixar de fazer alguma coisa senão em virtude de lei
> * Proibição de tortura e tratamento desumano ou degradante
> * Liberdade de manifestação do pensamento, desde que não seja anônima
> * Direito de resposta e indenização por dano material, moral ou à imagem
> * Liberdade de consciência e de crença, com garantia de livre exercício dos cultos religiosos e proteção aos locais de culto e suas liturgias.

**Trecho recuperado:**

> Art. 5o Todos são iguais perante a lei, sem distinção de qualquer natureza, garantindo-se
> aos brasileiros e aos estrangeiros residentes no País a inviolabilidade do direito à vida, à
> liberdade, à igualdade, à segurança e à propriedade, nos termos seguintes: I – homens e mulheres são iguais em direitos e obrigações, nos termos desta Cons-
> tituição; II – ninguém será obrigado a fazer ou deixar de fazer alguma coisa senão em virtude
> de lei; III – ninguém será submetido a tortura nem a tratamento desumano ou degradante; IV – é livre a manifestação do pensamento, sendo vedado o anonimato; V – é assegurado o direito de resposta, proporcional ao > agravo, além da indenização
> por dano material, moral ou à imagem; VI – é inviolável a liberdade de consciência e de crença, sendo assegurado o livre
> exercício dos cultos religiosos e garantida, na forma da lei, a proteção aos locais de culto
> e a suas liturgias;

**Resultado:** ✔️ Correta
**Motivo:** A resposta está alinhada com o trecho recuperado e cobre os principais pontos do artigo.

---

⚠️ **Observação:**
Em alguns casos, o modelo pode incluir informações adicionais não diretamente presentes no trecho recuperado, indicando oportunidades de melhoria no pipeline de recuperação (RAG).

---


## 💬 Exemplo de Uso

**Pergunta:**

> Quais são os direitos fundamentais garantidos pela Constituição?

**Resposta:**

> De acordo com o contexto fornecido, os direitos fundamentais garantidos pela Constituição incluem:

1. **Direitos Sociais**: 
   - Educação
   - Saúde
   - Alimentação
   - Trabalho
   - Moradia
   - Transporte
   - Lazer
   - Segurança
   - Previdência social
   - Proteção à maternidade e à infância
   - Assistência aos desamparados

2. **Direitos Humanos e Liberdades Fundamentais**: 
   - Todos os direitos humanos e liberdades fundamentais para pessoas com deficiência, sem discriminação.
   - Direito à não discriminação por causa de deficiência.
   - Direito à igualdade de condições para pessoas com deficiência.

3. **Direitos Específicos para Pessoas com Deficiência**:
   - Pleno exercício de todos os direitos humanos e liberdades fundamentais.
   - Proteção e promoção dos direitos humanos.
   - Acesso a programas e políticas que levem em conta a proteção e promoção dos direitos humanos das pessoas com deficiência.

4. **Direitos Econômicos, Sociais e Culturais**:
   - Medidas para assegurar progressivamente o pleno exercício desses direitos, com base nos recursos disponíveis e, quando necessário, no âmbito da cooperação internacional.

5. **Direitos Relacionados à Família e à Criança**:
   - Direito à fertilidade em igualdade de condições.
   - Direitos e responsabilidades relacionados à guarda, custódia, curatela e adoção de crianças.
   - Igualdade de direitos em relação à vida familiar para crianças com deficiência.

Esses direitos estão previstos na Constituição da República Federativa do Brasil e em tratados internacionais sobre direitos humanos dos quais o Brasil é signatário.

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
