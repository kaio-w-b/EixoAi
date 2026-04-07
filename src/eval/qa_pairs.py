"""
eval/qa_pairs.py
────────────────
Pares pergunta / keywords pré-definidos para avaliar o RAG
sobre o PDF da Constituição Federal do Brasil (CF/88).

Cada item tem:
  • "query"    — pergunta objetiva sobre o conteúdo constitucional
  • "keywords" — termos que DEVEM aparecer nos chunks relevantes recuperados
"""

from typing import Dict, List

CF88_QA_PAIRS: List[Dict] = [
    {
        "query": "Quais são os fundamentos da República Federativa do Brasil?",
        "keywords": ["soberania", "cidadania", "dignidade", "valores sociais", "pluralismo político"],
    },
    {
        "query": "Quais são os objetivos fundamentais da República?",
        "keywords": ["sociedade livre", "justa", "solidária", "desenvolvimento", "pobreza", "marginalização"],
    },
    {
        "query": "Quais são os direitos e deveres individuais e coletivos previstos na Constituição?",
        "keywords": ["igualdade", "liberdade", "segurança", "propriedade", "brasileiros", "estrangeiros"],
    },
    {
        "query": "O que a Constituição prevê sobre a inviolabilidade do domicílio?",
        "keywords": ["domicílio", "inviolável", "asilo", "consentimento", "flagrante"],
    },
    {
        "query": "Como a Constituição define a organização do Estado brasileiro?",
        "keywords": ["União", "Estados", "Distrito Federal", "Municípios", "federativa"],
    },
    {
        "query": "Quais são os poderes da União segundo a Constituição?",
        "keywords": ["Legislativo", "Executivo", "Judiciário", "independentes", "harmônicos"],
    },
    {
        "query": "O que a Constituição estabelece sobre o direito à saúde?",
        "keywords": ["saúde", "direito", "Estado", "políticas sociais", "econômicas"],
    },
    {
        "query": "O que a Constituição diz sobre a educação?",
        "keywords": ["educação", "direito", "dever", "Estado", "família", "desenvolvimento"],
    },
    {
        "query": "Como a Constituição trata dos direitos trabalhistas?",
        "keywords": ["salário mínimo", "jornada", "trabalho", "décimo terceiro", "férias"],
    },
    {
        "query": "O que a Constituição prevê sobre o processo legislativo?",
        "keywords": ["emendas", "leis complementares", "ordinárias", "delegadas", "medidas provisórias"],
    },
]