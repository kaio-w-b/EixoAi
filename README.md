# EixoAI - Sistema de QA com LLM e RAG aplicado a documentos jurídicos

**Sistema de Question Answering baseado em LLM + RAG (Retrieval-Augmented Generation), avaliado utilizando a Constituição Federal de 1988 como base de conhecimento.**

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
│     → Divide em pedaços consistentes (512 chars)            │
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

> O Art. 5º da Constituição Brasileira de 1988 é um dos mais importantes da Carta Magna, pois estabelece os direitos fundamentais e garantias individuais e coletivas. O trecho específico que você mencionou:

"Art. 5º Todos são iguais perante a lei, sem distinção de qualquer natureza, garantindo-se aos brasileiros e aos estrangeiros residentes no País a inviolabilidade do direito à vida, à liberdade, à igualdade, à segurança e à propriedade, nos termos seguintes:"

Significa que:

1. **Igualdade perante a lei**: Todos os indivíduos, sejam brasileiros ou estrangeiros residentes no Brasil, são considerados iguais perante a lei, sem distinção de qualquer natureza, como raça, gênero, religião, origem, condição social, etc.
2. **Inviolabilidade do direito à vida**: O direito à vida é considerado inviolável, ou seja, não pode ser violado ou ameaçado por ninguém, incluindo o Estado. Isso significa que o Estado tem o dever de proteger a vida dos indivíduos e garantir que eles não sejam submetidos a situações que possam colocar em risco sua vida.
3. **Direitos fundamentais**: Além do direito à vida, o Art. 5º também garante outros direitos fundamentais, como:
 * **Liberdade**: O direito de fazer escolhas e agir de acordo com a própria vontade, desde que não viole os direitos de outrem.
 * **Igualdade**: O direito de ser tratado de forma igualitária e não sofrer discriminação.
 * **Segurança**: O direito de sentir-se seguro e protegido contra ameaças ou violências.
 * **Propriedade**: O direito de possuir e gozar de bens e propriedades, desde que não viole os direitos de outrem.

Em resumo, o Art. 5º da Constituição Brasileira de 1988 estabelece que todos os indivíduos são iguais perante a lei e têm direito à vida, liberdade, igualdade, segurança e propriedade, e que o Estado tem o dever de proteger e garantir esses direitos.

**Trecho recuperado:**

> [INSERIR TRECHO DO DOCUMENTO]

**Análise:**
Resposta considerada **correta/parcial/incorreta** com base na aderência ao contexto.

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
git clone https://github.com/seu-usuario/EixoAi.git
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