from pathlib import Path

# 数据集路径
DATASET_PATHS = {
    'cwq': Path('../data/cwq.json'),
    'webqsp': Path('../data/WebQSP.json'),
    'webqsp_sampled': Path('../data/WebQSP_sampled_600.json'),
    'noisy_webqsp': Path('../data/noisy_WebQSP_sampled_600.json')
    # 其他数据集...
}
OUTPUT_PATH = '../outputs/{method}_{dataset}_{suffix}.jsonl'

JSON_PATH = '../outputs/{method}_{dataset}_{suffix}.json'

# SPARQL 配置
SPARQL_ENDPOINT = "http://172.18.34.27:8890/sparql"

# 模型配置
MODEL = {
    'qwen': "Qwen/Qwen3-8B",
    'llama': "meta-llama/Llama-3.1-8B-Instruct",
    'minilm': "sentence-transformers/all-MiniLM-L6-v2"
}

# 其他配置
UNKNOWN_ENTITY = "UnName_Entity"
FINISH_ID = "[FINISH_ID]"
FINISH_ENTITY = "[FINISH]"

EMBEDDING = {
    'index_path': '../Freebase/index/L6_dedup_embeddings.index',
    'embeddings_path': '../Freebase/index/L6_dedup_embeddings.npy',
    'names_path': '../Freebase/index/L6_dedup_names.json'
}

# 提示词配置
GENERATE_TOPIC_ENTITY = """Given a multi-step reasoning question, identify and output the unique Anchor entity present in the question to initiate the reasoning process. Refrain from including any reasoning steps or the final answer.
Question: {}
Anchor Entity:
"""

EXTRACT_RELATION_PROMPT = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
A: 1. {language.human_language.main_country (Score: 0.5))}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {language.human_language.countries_spoken_in (Score: 0.3)}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {base.rosetta.languoid.parent (Score: 0.2)}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

Q: """

SCORE_ENTITY_CANDIDATES_PROMPT = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Relation: film.producer.film
Entites: The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones
Score: 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
The movie that matches the given criteria is "So Undercover" with Miley Cyrus and produced by Tobin Armbrust. Therefore, the score for "So Undercover" would be 1, and the scores for all other entities would be 0.

Q: {}
Relation: {}
Entites: """

ANSWER_PROMPT = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.

"""

PROMPT_EVALUATE = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: {No}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: {Yes}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: {No}. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: {Yes}. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: {No}. Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.

"""

GENERATE_DIRECTLY = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}.

"""

IO_PROMPT = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: {Heroin}."""

COT_PROMPT = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}.

Follow the template above and only answer the questions asked, without generating extra text.

"""

MULTITOPIC_ENTITIES_PROMPT = """Extract all topic entities from the given multi-hop question. Topic entities are proper nouns, named entities, or specific concepts that are crucial for retrieving external knowledge. They may come from different sub-questions that contribute to the final answer. If there are multiple topic entities, separate them with commas.

Examples:

Q: Who directed the movie in which Leonardo DiCaprio played Jordan Belfort?
A: Leonardo DiCaprio, Jordan Belfort

Q: Which team won the Champions League in the same year that Spain won the FIFA World Cup?
A: Champions League, FIFA World Cup, Spain

Q: Who was the US president when the Berlin Wall fell?
A: US president, Berlin Wall

Q: Which book was written by the author of "Pride and Prejudice" and published after 1800?
A: Pride and Prejudice

Q: What is the capital of the country where Mount Everest is located?
A: Mount Everest

Q: Who won the Nobel Prize in Physics in the same year that Albert Einstein died?
A: Nobel Prize in Physics, Albert Einstein

Q: In which city was the founder of Tesla Motors born?
A: Tesla Motors

Q: {}
A: 
"""

REASONING_PROMPT = """Given a question and a summary text, you are asked to answer whether it's sufficient for you to answer the question with both the triplets and the summary.

Please consider:
- The original knowledge triplets.
- The coherent summary text derived from the triplets.

Only use the information provided in the triplets and the summary. Do not add external knowledge unless explicitly allowed.

Q: {}
Summary Text:
{}

Answer strictly in the following format:
A: {{Yes or No}}. Based on the given knowledge triplets and the summary text, [explanation].
"""

IS_RELEVANT = """
Please determine whether the following knowledge triples are relevant to the question: {}

Knowledge Triplets:
{}

Instructions:
- If relevant, summarize the information contained in the following triples and present it as a coherent text.
- Only extract information from the triples provided, and do not add any extra content or commentary.
- If not relevant, please explain the reason.

Respond strictly in the following JSON format:
{{"is_relevant": true or false, "summary": ...}}
"""

EXTRACT_USEFUL_INFORMATION = """You are a knowledge extraction and text-generation assistant. Your job is to identify and transform relevant information from a set of structured knowledge triples into a coherent natural‑language answer for a given question. If no useful information is found, you must provide a brief explanation why.

Input format:
Question:
{}

Knowledge triples (one per line, in the format `(subject, predicate, object)`):
{}

Instructions:
1. Scan the provided triples and select those most relevant to the question.  
2. If at least one relevant triple exists, merge and rephrase their content into a single, fluent paragraph of English that directly answers the question.  
3. If no triples are relevant, output exactly one sentence stating “No relevant information found,” followed by a brief reason (e.g. “triples do not cover the question topic” or “keywords not matched”).  
4. Your output must contain **only** the generated answer or the single explanatory sentence—do **not** include the original triples or any additional commentary.

Example 1 (useful information)  
Question: When did Einstein win the Nobel Prize in Physics?  
Triples: 
(Einstein, birth_year, "1879")
(Einstein, award_received, "Nobel Prize in Physics")
(Einstein, award_year, "1921")
(Einstein, death_date, "1955-04-18")
Expected output:  
{{"is_relevant": true, "information": "Einstein was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect."}}

Example 2 (no useful information)  
Question: Who invented the light bulb?  
Triples:  
(Edison, birth_year, "1847")
(Edison, nationality, "American")
Expected output:  
{{"is_relevant": false, "information": "No relevant information found, because the provided triples do not include the inventor of the light bulb."}}
"""

READ_AND_SUMMARIZE = """You are a text-generation assistant. Your job is to take the provided structured knowledge triples and convert their content directly into a coherent natural-language paragraph. Do NOT add any information beyond what appears in the triples.

Input format:
Question:
{}

Knowledge triples (one per line, in the format `(subject, predicate, object)`):
{}

Instructions:
1. Read all the provided triples.
2. Merge and rephrase their combined content into a single fluent paragraph of English that directly conveys the information.
3. Do NOT introduce any facts, explanations, or commentary that are not explicitly contained in the triples.
4. If the list of triples is empty, output exactly: “No information provided.”

Output:
Your output must contain only the generated paragraph with no additional sections, labels, or commentary.
"""

FILTER_TRIPLES = """Given the question and a list of knowledge triples, identify and return only the triples that are directly relevant to answering the question.
Do not change the format of the triples. Do not generate any explanations or extra text. Return only the filtered triples as-is.

Question: {}
Triples:
{}
"""