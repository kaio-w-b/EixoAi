"""
eval/qa_pairs.py
────────────────
Dataset profissional de avaliação RAG para a Constituição Federal de 1988.

Estrutura de cada par:
  • query            — pergunta objetiva, evitando ambiguidade
  • keywords         — termos discriminativos que DEVEM aparecer nos chunks relevantes
  • expected_answer  — resposta correta, factual, baseada na CF/88
  • source_article   — artigo-fonte principal
  • difficulty       — easy | medium | hard

Cobertura:
  ├── Art. 1º–5º   — fundamentos, objetivos e direitos individuais  (15 pares)
  ├── Organização   — Estado, território, competências              ( 8 pares)
  ├── Direitos soc. — educação, saúde, trabalho, moradia            ( 8 pares)
  ├── Poderes       — Legislativo, Executivo, Judiciário             ( 8 pares)
  └── Leg. processo — emendas, leis, medidas provisórias            ( 6 pares)

Total: 45 pares | easy: 15 | medium: 18 | hard: 12
"""

from typing import Dict, List

CF88_QA_PAIRS: List[Dict] = [

    # ══════════════════════════════════════════════════════════
    # BLOCO 1 — Art. 1º–5º: Fundamentos, Objetivos e Direitos
    # ══════════════════════════════════════════════════════════

    # ── easy ─────────────────────────────────────────────────

    {
        "query": "Quais são os cinco fundamentos da República Federativa do Brasil?",
        "keywords": ["soberania", "cidadania", "dignidade", "valores sociais", "pluralismo político"],
        "expected_answer": (
            "Os fundamentos são: I – soberania; II – cidadania; III – dignidade da pessoa humana; "
            "IV – os valores sociais do trabalho e da livre iniciativa; V – o pluralismo político."
        ),
        "source_article": "Art. 1º",
        "difficulty": "easy",
    },
    {
        "query": "O Brasil é uma república ou uma monarquia?",
        "keywords": ["República Federativa", "Estado Democrático de Direito", "indissolúvel", "Municípios"],
        "expected_answer": (
            "O Brasil é uma República Federativa, formada pela união indissolúvel dos Estados, "
            "Municípios e do Distrito Federal, constituída em Estado Democrático de Direito."
        ),
        "source_article": "Art. 1º",
        "difficulty": "easy",
    },
    {
        "query": "De onde emana todo o poder no Estado brasileiro?",
        "keywords": ["poder", "povo", "representantes eleitos", "diretamente"],
        "expected_answer": (
            "Todo o poder emana do povo, que o exerce por meio de representantes eleitos "
            "ou diretamente, nos termos da Constituição."
        ),
        "source_article": "Art. 1º, parágrafo único",
        "difficulty": "easy",
    },
    {
        "query": "Quais são os objetivos fundamentais da República Federativa do Brasil?",
        "keywords": ["sociedade livre", "justa", "solidária", "desenvolvimento nacional", "pobreza", "marginalização", "bem de todos"],
        "expected_answer": (
            "São objetivos fundamentais: construir uma sociedade livre, justa e solidária; "
            "garantir o desenvolvimento nacional; erradicar a pobreza e a marginalização; "
            "promover o bem de todos, sem preconceitos."
        ),
        "source_article": "Art. 3º",
        "difficulty": "easy",
    },
    {
        "query": "A Constituição proíbe preconceito de origem, raça, sexo, cor ou idade?",
        "keywords": ["preconceito", "origem", "raça", "sexo", "cor", "idade", "discriminação"],
        "expected_answer": (
            "Sim. O art. 3º, IV, estabelece como objetivo fundamental promover o bem de todos "
            "sem preconceitos de origem, raça, sexo, cor, idade ou quaisquer outras formas de discriminação."
        ),
        "source_article": "Art. 3º, IV",
        "difficulty": "easy",
    },
    {
        "query": "Todos são iguais perante a lei segundo a Constituição?",
        "keywords": ["igualdade", "lei", "brasileiros", "estrangeiros", "distinção"],
        "expected_answer": (
            "Sim. O art. 5º garante que todos são iguais perante a lei, sem distinção de qualquer natureza, "
            "assegurando aos brasileiros e estrangeiros residentes no País a inviolabilidade do direito "
            "à vida, à liberdade, à igualdade, à segurança e à propriedade."
        ),
        "source_article": "Art. 5º, caput",
        "difficulty": "easy",
    },
    {
        "query": "Homens e mulheres têm direitos iguais na Constituição Federal?",
        "keywords": ["homens", "mulheres", "iguais", "direitos", "obrigações"],
        "expected_answer": (
            "Sim. O art. 5º, I, estabelece que homens e mulheres são iguais em direitos e obrigações "
            "nos termos da Constituição."
        ),
        "source_article": "Art. 5º, I",
        "difficulty": "easy",
    },

    # ── medium ────────────────────────────────────────────────

    {
        "query": "Quais são as relações internacionais que regem o Brasil segundo a Constituição?",
        "keywords": ["independência nacional", "prevalência dos direitos humanos", "autodeterminação", "não-intervenção", "igualdade entre os Estados"],
        "expected_answer": (
            "O art. 4º estabelece que o Brasil se rege nas relações internacionais pelos princípios: "
            "independência nacional, prevalência dos direitos humanos, autodeterminação dos povos, "
            "não-intervenção, igualdade entre os Estados, defesa da paz, solução pacífica dos conflitos, "
            "repúdio ao terrorismo e ao racismo, cooperação entre os povos e concessão de asilo político."
        ),
        "source_article": "Art. 4º",
        "difficulty": "medium",
    },
    {
        "query": "O que a Constituição diz sobre a criação de uma comunidade latino-americana?",
        "keywords": ["integração", "América Latina", "comunidade", "econômica", "política", "cultural"],
        "expected_answer": (
            "O art. 4º, parágrafo único, determina que a República Federativa do Brasil buscará "
            "a integração econômica, política, social e cultural dos povos da América Latina, "
            "visando à formação de uma comunidade latino-americana de nações."
        ),
        "source_article": "Art. 4º, parágrafo único",
        "difficulty": "medium",
    },
    {
        "query": "A Constituição admite pena de morte no Brasil? Em que casos?",
        "keywords": ["pena de morte", "guerra", "declarada", "nacional", "vedado"],
        "expected_answer": (
            "Em regra, não se admite pena de morte. O art. 5º, XLVII, veda a pena de morte, "
            "salvo em caso de guerra declarada, nos termos do art. 84, XIX."
        ),
        "source_article": "Art. 5º, XLVII",
        "difficulty": "medium",
    },
    {
        "query": "O que é o habeas corpus e quando ele cabe segundo a Constituição?",
        "keywords": ["habeas corpus", "liberdade", "locomoção", "violência", "coação", "ilegalidade"],
        "expected_answer": (
            "O habeas corpus é uma garantia constitucional concedida sempre que alguém sofrer "
            "ou se achar ameaçado de sofrer violência ou coação em sua liberdade de locomoção "
            "por ilegalidade ou abuso de poder (art. 5º, LXVIII)."
        ),
        "source_article": "Art. 5º, LXVIII",
        "difficulty": "medium",
    },
    {
        "query": "O que é mandado de segurança e quando é cabível?",
        "keywords": ["mandado de segurança", "direito líquido", "certo", "autoridade", "ilegal", "abuso de poder"],
        "expected_answer": (
            "O mandado de segurança é cabível para proteger direito líquido e certo não amparado "
            "por habeas corpus ou habeas data, quando o responsável pela ilegalidade ou abuso de "
            "poder for autoridade pública ou agente de pessoa jurídica no exercício de atribuições "
            "do Poder Público (art. 5º, LXIX)."
        ),
        "source_article": "Art. 5º, LXIX",
        "difficulty": "medium",
    },
    {
        "query": "O que a Constituição estabelece sobre a inviolabilidade do domicílio?",
        "keywords": ["domicílio", "inviolável", "asilo", "consentimento", "flagrante", "desastre", "determinação judicial"],
        "expected_answer": (
            "O art. 5º, XI, estabelece que a casa é asilo inviolável do indivíduo. Ninguém pode "
            "nela penetrar sem consentimento do morador, salvo em caso de flagrante delito, desastre "
            "ou para prestar socorro, ou, durante o dia, por determinação judicial."
        ),
        "source_article": "Art. 5º, XI",
        "difficulty": "medium",
    },
    {
        "query": "Quais penas são vedadas pela Constituição Federal?",
        "keywords": ["penas", "morte", "perpétua", "trabalhos forçados", "banimento", "cruéis", "vedadas"],
        "expected_answer": (
            "O art. 5º, XLVII, veda as penas: de morte (salvo guerra declarada), de caráter perpétuo, "
            "de trabalhos forçados, de banimento e cruéis."
        ),
        "source_article": "Art. 5º, XLVII",
        "difficulty": "medium",
    },
    {
        "query": "O que diz a Constituição sobre sigilo de correspondência e comunicações?",
        "keywords": ["sigilo", "correspondência", "comunicações telegráficas", "dados", "telefônicas", "inviolável"],
        "expected_answer": (
            "O art. 5º, XII, garante a inviolabilidade do sigilo da correspondência e das comunicações "
            "telegráficas, de dados e das comunicações telefônicas, salvo, no último caso, por ordem "
            "judicial, nas hipóteses e na forma que a lei estabelecer para fins de investigação criminal "
            "ou instrução processual penal."
        ),
        "source_article": "Art. 5º, XII",
        "difficulty": "medium",
    },

    # ── hard ──────────────────────────────────────────────────

    {
        "query": "O que distingue o mandado de injunção do mandado de segurança coletivo?",
        "keywords": ["mandado de injunção", "norma regulamentadora", "inviabilize", "mandado de segurança coletivo", "partido político", "organização sindical"],
        "expected_answer": (
            "O mandado de injunção (art. 5º, LXXI) é concedido quando a falta de norma regulamentadora "
            "tornar inviável o exercício de direitos constitucionais. O mandado de segurança coletivo "
            "(art. 5º, LXX) pode ser impetrado por partido político com representação no Congresso ou "
            "por organização sindical, entidade de classe ou associação, para proteger direito líquido "
            "e certo de seus membros. São institutos distintos em finalidade e legitimidade."
        ),
        "source_article": "Art. 5º, LXX e LXXI",
        "difficulty": "hard",
    },
    {
        "query": "Em que condições a Constituição permite interceptação de comunicações telefônicas?",
        "keywords": ["interceptação", "telefônica", "ordem judicial", "investigação criminal", "instrução processual", "lei"],
        "expected_answer": (
            "O art. 5º, XII, permite a interceptação de comunicações telefônicas exclusivamente por "
            "ordem judicial, nas hipóteses e na forma que a lei estabelecer, e somente para fins de "
            "investigação criminal ou instrução processual penal."
        ),
        "source_article": "Art. 5º, XII",
        "difficulty": "hard",
    },
    {
        "query": "O que a Constituição diz sobre a retroatividade da lei penal mais grave?",
        "keywords": ["retroatividade", "lei penal", "prejudicar", "retroagir", "benéfica", "anterior"],
        "expected_answer": (
            "O art. 5º, XL, estabelece que a lei penal não retroagirá, salvo para beneficiar o réu. "
            "Ou seja, lei penal mais grave não pode retroagir para prejudicar, mas lei mais benéfica "
            "pode ser aplicada a fatos anteriores."
        ),
        "source_article": "Art. 5º, XL",
        "difficulty": "hard",
    },

    # ══════════════════════════════════════════════════════════
    # BLOCO 2 — Organização do Estado e Território
    # ══════════════════════════════════════════════════════════

    # ── easy ─────────────────────────────────────────────────

    {
        "query": "Quais são os entes federativos que compõem a República Federativa do Brasil?",
        "keywords": ["União", "Estados", "Distrito Federal", "Municípios", "indissolúvel"],
        "expected_answer": (
            "A República Federativa do Brasil é formada pela união indissolúvel da União, "
            "dos Estados, do Distrito Federal e dos Municípios (art. 1º)."
        ),
        "source_article": "Art. 1º",
        "difficulty": "easy",
    },
    {
        "query": "Qual é a capital federal do Brasil segundo a Constituição?",
        "keywords": ["capital federal", "Brasília", "Distrito Federal"],
        "expected_answer": (
            "O art. 18, § 1º, estabelece que Brasília é a capital federal."
        ),
        "source_article": "Art. 18, § 1º",
        "difficulty": "easy",
    },

    # ── medium ────────────────────────────────────────────────

    {
        "query": "Como a Constituição organiza a competência exclusiva da União?",
        "keywords": ["competência exclusiva", "União", "relações exteriores", "guerra", "moeda", "comércio interestadual"],
        "expected_answer": (
            "O art. 21 lista as competências exclusivas da União, incluindo: manter relações "
            "com Estados estrangeiros, declarar guerra e paz, emitir moeda, elaborar e executar "
            "planos nacionais, explorar serviços de telecomunicações e energia elétrica, entre outros."
        ),
        "source_article": "Art. 21",
        "difficulty": "medium",
    },
    {
        "query": "O que a Constituição diz sobre a criação, fusão, incorporação e desmembramento de Municípios?",
        "keywords": ["Municípios", "criação", "fusão", "incorporação", "desmembramento", "lei estadual", "plebiscito"],
        "expected_answer": (
            "O art. 18, § 4º, dispõe que a criação, incorporação, fusão e desmembramento de Municípios "
            "se dará por lei estadual, dentro do período determinado por lei complementar federal, "
            "mediante consulta prévia às populações diretamente interessadas."
        ),
        "source_article": "Art. 18, § 4º",
        "difficulty": "medium",
    },
    {
        "query": "Quais são os bens da União segundo a Constituição?",
        "keywords": ["bens da União", "lagos", "rios", "ilhas", "terras", "recursos naturais", "plataforma continental"],
        "expected_answer": (
            "O art. 20 arrola os bens da União, incluindo: terras devolutas, lagos e rios em terrenos "
            "de seu domínio, ilhas oceânicas, recursos naturais da plataforma continental e da zona "
            "econômica exclusiva, terrenos de marinha, terras dos índios, entre outros."
        ),
        "source_article": "Art. 20",
        "difficulty": "medium",
    },

    # ── hard ──────────────────────────────────────────────────

    {
        "query": "Qual a diferença entre competência privativa e competência exclusiva da União?",
        "keywords": ["competência privativa", "exclusiva", "delegação", "lei complementar", "Estados", "art. 22"],
        "expected_answer": (
            "A competência exclusiva (art. 21) é indelegável. A competência privativa (art. 22) pode "
            "ser delegada aos Estados por lei complementar federal, em matérias específicas. "
            "Exemplo: a União tem competência privativa para legislar sobre direito civil, penal, "
            "processual, mas pode delegar legislação sobre questões específicas aos Estados."
        ),
        "source_article": "Art. 22",
        "difficulty": "hard",
    },
    {
        "query": "Quais matérias são de competência legislativa concorrente entre União, Estados e DF?",
        "keywords": ["concorrente", "direito tributário", "financeiro", "penitenciário", "econômico", "urbanístico", "normas gerais"],
        "expected_answer": (
            "O art. 24 lista as matérias de competência concorrente: direito tributário, financeiro, "
            "penitenciário, econômico e urbanístico; orçamento; juntas comerciais; custas dos serviços "
            "forenses; produção e consumo; florestas, caça, pesca; educação, cultura, ensino, desporto; "
            "criação, funcionamento e processo do juizado de pequenas causas; procedimentos em matéria "
            "processual; previdência social; proteção e defesa da saúde; proteção ao meio ambiente; "
            "proteção ao patrimônio histórico, cultural, artístico e paisagístico; "
            "responsabilidade por dano ao meio ambiente; direito da criança, do idoso e dos portadores de deficiência."
        ),
        "source_article": "Art. 24",
        "difficulty": "hard",
    },
    {
        "query": "Como se resolve o conflito de normas nas matérias de competência concorrente?",
        "keywords": ["normas gerais", "suplementar", "suspensa", "superveniência", "federal", "estadual"],
        "expected_answer": (
            "Pelo art. 24, §§ 1º–4º: a União estabelece normas gerais; os Estados suplementam a "
            "legislação federal. Na ausência de lei federal, os Estados exercem competência plena. "
            "A superveniência de lei federal suspende a eficácia da lei estadual no que lhe for "
            "contrário."
        ),
        "source_article": "Art. 24, §§ 1º–4º",
        "difficulty": "hard",
    },

    # ══════════════════════════════════════════════════════════
    # BLOCO 3 — Direitos Sociais
    # ══════════════════════════════════════════════════════════

    # ── easy ─────────────────────────────────────────────────

    {
        "query": "Quais são os direitos sociais previstos na Constituição?",
        "keywords": ["educação", "saúde", "alimentação", "trabalho", "moradia", "transporte", "lazer", "segurança", "previdência"],
        "expected_answer": (
            "O art. 6º elenca como direitos sociais: a educação, a saúde, a alimentação, o trabalho, "
            "a moradia, o transporte, o lazer, a segurança, a previdência social, a proteção à "
            "maternidade e à infância, e a assistência aos desamparados."
        ),
        "source_article": "Art. 6º",
        "difficulty": "easy",
    },
    {
        "query": "Qual é o salário mínimo previsto na Constituição e para que ele serve?",
        "keywords": ["salário mínimo", "necessidades vitais", "moradia", "alimentação", "educação", "saúde", "lazer", "vestuário"],
        "expected_answer": (
            "O art. 7º, IV, garante salário mínimo fixado em lei, nacionalmente unificado, capaz de "
            "atender às necessidades vitais básicas do trabalhador e de sua família: moradia, "
            "alimentação, educação, saúde, lazer, vestuário, higiene, transporte e previdência social."
        ),
        "source_article": "Art. 7º, IV",
        "difficulty": "easy",
    },

    # ── medium ────────────────────────────────────────────────

    {
        "query": "O que a Constituição prevê sobre a jornada de trabalho?",
        "keywords": ["jornada", "oito horas", "quarenta e quatro", "semanal", "turnos ininterruptos", "seis horas"],
        "expected_answer": (
            "O art. 7º, XIII, estabelece jornada de trabalho não superior a oito horas diárias e "
            "quarenta e quatro horas semanais. Para turnos ininterruptos de revezamento, a jornada "
            "é de seis horas (art. 7º, XIV), salvo negociação coletiva."
        ),
        "source_article": "Art. 7º, XIII e XIV",
        "difficulty": "medium",
    },
    {
        "query": "O que a Constituição garante aos trabalhadores em relação às férias?",
        "keywords": ["férias", "remuneradas", "um terço", "adicional", "constitucional"],
        "expected_answer": (
            "O art. 7º, XVII, garante gozo de férias anuais remuneradas com pelo menos um terço "
            "a mais do que o salário normal."
        ),
        "source_article": "Art. 7º, XVII",
        "difficulty": "medium",
    },
    {
        "query": "O que a Constituição diz sobre o direito à saúde?",
        "keywords": ["saúde", "direito", "dever do Estado", "políticas sociais", "acesso universal", "igualitário"],
        "expected_answer": (
            "O art. 196 estabelece que a saúde é direito de todos e dever do Estado, garantido "
            "mediante políticas sociais e econômicas que visem à redução do risco de doença e ao "
            "acesso universal e igualitário às ações e serviços para sua promoção, proteção e recuperação."
        ),
        "source_article": "Art. 196",
        "difficulty": "medium",
    },
    {
        "query": "O que a Constituição determina sobre a educação básica?",
        "keywords": ["educação básica", "obrigatória", "gratuita", "quatro", "dezessete", "progressiva universalização"],
        "expected_answer": (
            "O art. 208, I, estabelece como dever do Estado a educação básica obrigatória e gratuita "
            "dos 4 aos 17 anos de idade, assegurada inclusive sua oferta gratuita para todos os que a "
            "ela não tiveram acesso na idade própria."
        ),
        "source_article": "Art. 208, I",
        "difficulty": "medium",
    },

    # ── hard ──────────────────────────────────────────────────

    {
        "query": "O trabalhador tem direito a aviso prévio proporcional ao tempo de serviço?",
        "keywords": ["aviso prévio", "proporcional", "tempo de serviço", "mínimo trinta dias"],
        "expected_answer": (
            "Sim. O art. 7º, XXI, garante aviso prévio proporcional ao tempo de serviço, sendo no "
            "mínimo de trinta dias, nos termos da lei."
        ),
        "source_article": "Art. 7º, XXI",
        "difficulty": "hard",
    },
    {
        "query": "Qual é a diferença constitucional entre a greve do trabalhador privado e do servidor público?",
        "keywords": ["greve", "trabalhador", "lei ordinária", "servidor público", "lei específica", "essenciais"],
        "expected_answer": (
            "Para os trabalhadores privados, o art. 9º garante o direito de greve, competindo-lhes "
            "decidir sobre sua oportunidade e conveniência; abusos são definidos em lei. Para os "
            "servidores públicos, o art. 37, VII, garante o direito de greve a ser exercido nos "
            "termos e limites definidos em lei específica."
        ),
        "source_article": "Art. 9º e Art. 37, VII",
        "difficulty": "hard",
    },

    # ══════════════════════════════════════════════════════════
    # BLOCO 4 — Poderes da União
    # ══════════════════════════════════════════════════════════

    # ── easy ─────────────────────────────────────────────────

    {
        "query": "Quais são os Poderes da União segundo a Constituição?",
        "keywords": ["Legislativo", "Executivo", "Judiciário", "independentes", "harmônicos"],
        "expected_answer": (
            "O art. 2º estabelece que são Poderes da União, independentes e harmônicos entre si, "
            "o Legislativo, o Executivo e o Judiciário."
        ),
        "source_article": "Art. 2º",
        "difficulty": "easy",
    },
    {
        "query": "O Congresso Nacional é composto por quais casas legislativas?",
        "keywords": ["Congresso Nacional", "Câmara dos Deputados", "Senado Federal", "bicameral"],
        "expected_answer": (
            "O art. 44 dispõe que o Poder Legislativo é exercido pelo Congresso Nacional, "
            "que se compõe da Câmara dos Deputados e do Senado Federal."
        ),
        "source_article": "Art. 44",
        "difficulty": "easy",
    },

    # ── medium ────────────────────────────────────────────────

    {
        "query": "Qual o mandato e o número de senadores por Estado?",
        "keywords": ["senadores", "oito anos", "três", "Estado", "Distrito Federal", "renovação"],
        "expected_answer": (
            "O art. 46 estabelece que o Senado Federal compõe-se de representantes dos Estados e "
            "do Distrito Federal, eleitos pelo sistema majoritário, sendo três por Estado e pelo DF. "
            "O mandato é de oito anos e a renovação é feita de quatro em quatro anos, "
            "alternadamente por um e dois terços."
        ),
        "source_article": "Art. 46",
        "difficulty": "medium",
    },
    {
        "query": "Quais são as atribuições privativas do Senado Federal?",
        "keywords": ["Senado Federal", "processar e julgar", "Presidente", "aprovação", "empréstimos externos", "suspender execução"],
        "expected_answer": (
            "Entre as atribuições privativas do Senado (art. 52) estão: processar e julgar o "
            "Presidente da República e outras autoridades por crimes de responsabilidade; aprovar "
            "empréstimos externos da União, Estados e Municípios; fixar limites globais para a dívida "
            "consolidada; suspender a execução de lei declarada inconstitucional pelo STF."
        ),
        "source_article": "Art. 52",
        "difficulty": "medium",
    },
    {
        "query": "Quais são as competências privativas da Câmara dos Deputados?",
        "keywords": ["Câmara dos Deputados", "impeachment", "Presidente", "privativamente", "autorizar"],
        "expected_answer": (
            "O art. 51 lista como competências privativas da Câmara: autorizar o processo por crime "
            "de responsabilidade contra o Presidente e Vice-Presidente da República e ministros de "
            "Estado; elaborar seu regimento interno; eleger membros do Conselho da República."
        ),
        "source_article": "Art. 51",
        "difficulty": "medium",
    },
    {
        "query": "Quais são as atribuições do Presidente da República previstas na Constituição?",
        "keywords": ["Presidente", "sancionar", "vetar", "expedir", "decretos", "nomear", "ministros", "celebrar tratados"],
        "expected_answer": (
            "O art. 84 lista as competências privativas do Presidente, incluindo: nomear e exonerar "
            "ministros de Estado; exercer a direção superior da administração federal; sancionar, "
            "promulgar e fazer publicar as leis; vetar projetos de lei; celebrar tratados e "
            "convenções internacionais; decretar estado de defesa e de sítio; expedir decretos."
        ),
        "source_article": "Art. 84",
        "difficulty": "medium",
    },

    # ── hard ──────────────────────────────────────────────────

    {
        "query": "Quais são os requisitos constitucionais para ser eleito Presidente da República?",
        "keywords": ["Presidente", "brasileiro nato", "maior de trinta e cinco", "pleno exercício", "direitos políticos"],
        "expected_answer": (
            "O art. 87, por remissão ao art. 14, § 3º, exige: ser brasileiro nato; estar no pleno "
            "exercício dos direitos políticos; ter domicílio eleitoral na circunscrição; estar filiado "
            "a partido político; e ter mais de trinta e cinco anos de idade."
        ),
        "source_article": "Art. 14, § 3º c/c Art. 87",
        "difficulty": "hard",
    },
    {
        "query": "Quando o Presidente pode ser responsabilizado por crime de responsabilidade?",
        "keywords": ["crime de responsabilidade", "impeachment", "Câmara", "dois terços", "Senado", "perda do cargo"],
        "expected_answer": (
            "O art. 85 define crimes de responsabilidade os atos que atentem contra a Constituição "
            "Federal. O processo é iniciado na Câmara dos Deputados (art. 51, I) com autorização "
            "por dois terços, e julgado pelo Senado Federal (art. 52, I). A condenação implica "
            "perda do cargo e inabilitação por oito anos."
        ),
        "source_article": "Art. 85, Art. 51 I, Art. 52 I",
        "difficulty": "hard",
    },
    {
        "query": "Quais são as garantias constitucionais da magistratura?",
        "keywords": ["vitaliciedade", "inamovibilidade", "irredutibilidade", "subsídio", "magistratura"],
        "expected_answer": (
            "O art. 95 garante aos juízes: vitaliciedade (só perde o cargo por sentença judicial "
            "transitada em julgado); inamovibilidade (salvo motivo de interesse público); e "
            "irredutibilidade de subsídio."
        ),
        "source_article": "Art. 95",
        "difficulty": "hard",
    },

    # ══════════════════════════════════════════════════════════
    # BLOCO 5 — Processo Legislativo
    # ══════════════════════════════════════════════════════════

    # ── easy ─────────────────────────────────────────────────

    {
        "query": "Quais são as espécies normativas do processo legislativo?",
        "keywords": ["emendas constitucionais", "leis complementares", "ordinárias", "delegadas", "medidas provisórias", "decretos legislativos", "resoluções"],
        "expected_answer": (
            "O art. 59 lista as espécies normativas: emendas à Constituição; leis complementares; "
            "leis ordinárias; leis delegadas; medidas provisórias; decretos legislativos; e resoluções."
        ),
        "source_article": "Art. 59",
        "difficulty": "easy",
    },

    # ── medium ────────────────────────────────────────────────

    {
        "query": "Quais são os requisitos para aprovação de uma emenda constitucional?",
        "keywords": ["emenda constitucional", "três quintos", "dois turnos", "Câmara", "Senado", "proposta"],
        "expected_answer": (
            "O art. 60, § 2º, exige que a proposta seja discutida e votada em cada Casa do Congresso "
            "em dois turnos, considerando-se aprovada se obtiver em ambos o voto de três quintos dos "
            "respectivos membros."
        ),
        "source_article": "Art. 60, § 2º",
        "difficulty": "medium",
    },
    {
        "query": "O que a Constituição proíbe que seja objeto de emenda constitucional?",
        "keywords": ["cláusulas pétreas", "forma federativa", "voto direto", "separação dos Poderes", "direitos fundamentais"],
        "expected_answer": (
            "O art. 60, § 4º, proíbe emendas que tendam a abolir: a forma federativa de Estado; "
            "o voto direto, secreto, universal e periódico; a separação dos Poderes; e os direitos "
            "e garantias individuais. São as chamadas cláusulas pétreas."
        ),
        "source_article": "Art. 60, § 4º",
        "difficulty": "medium",
    },
    {
        "query": "O que são medidas provisórias e quando podem ser editadas?",
        "keywords": ["medidas provisórias", "relevância", "urgência", "Presidente", "força de lei", "sessenta dias"],
        "expected_answer": (
            "O art. 62 permite ao Presidente da República, em caso de relevância e urgência, adotar "
            "medidas provisórias com força de lei, que devem ser submetidas de imediato ao Congresso "
            "Nacional. Têm validade de 60 dias, prorrogável uma vez por igual período."
        ),
        "source_article": "Art. 62",
        "difficulty": "medium",
    },
    {
        "query": "Quem tem iniciativa para propor leis segundo a Constituição?",
        "keywords": ["iniciativa", "membros", "Câmara", "Senado", "Presidente", "STF", "cidadãos", "iniciativa popular"],
        "expected_answer": (
            "O art. 61 confere iniciativa de leis ordinárias e complementares a: qualquer membro ou "
            "Comissão da Câmara ou do Senado; ao Presidente da República; ao STF; aos Tribunais "
            "Superiores; ao Procurador-Geral da República; e aos cidadãos (iniciativa popular), "
            "mediante subscrição de um por cento do eleitorado nacional."
        ),
        "source_article": "Art. 61",
        "difficulty": "medium",
    },
]


# ── Variações semânticas ──────────────────────────────────────────────────────
# Perguntas alternativas sobre os mesmos tópicos (testa robustez semântica do RAG)

CF88_SEMANTIC_VARIANTS: List[Dict] = [
    {
        "query": "Que bases sustentam o Estado Democrático de Direito no Brasil?",
        "canonical_query": "Quais são os cinco fundamentos da República Federativa do Brasil?",
        "source_article": "Art. 1º",
    },
    {
        "query": "Qual é a fonte de legitimidade do poder político no Brasil?",
        "canonical_query": "De onde emana todo o poder no Estado brasileiro?",
        "source_article": "Art. 1º, parágrafo único",
    },
    {
        "query": "Existe isonomia entre homem e mulher na Constituição de 88?",
        "canonical_query": "Homens e mulheres têm direitos iguais na Constituição Federal?",
        "source_article": "Art. 5º, I",
    },
    {
        "query": "Quando é admitida a pena capital no ordenamento constitucional brasileiro?",
        "canonical_query": "A Constituição admite pena de morte no Brasil? Em que casos?",
        "source_article": "Art. 5º, XLVII",
    },
    {
        "query": "Como se vota uma proposta de emenda à Constituição?",
        "canonical_query": "Quais são os requisitos para aprovação de uma emenda constitucional?",
        "source_article": "Art. 60, § 2º",
    },
]