from tqdm import tqdm
from core.freebase_client import FreebaseClient
from core.llm_handler import LLMHandler
from core.semantic_search import SemanticSearch
from core.data_processor import DataProcessor
from config.settings import MULTITOPIC_ENTITIES_PROMPT, NOISY_PROMPT
import os, sys
import argparse
import time
from SPARQLWrapper import SPARQLWrapper, JSON

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.chdir(sys.path[0])
SPARQLPATH = "http://172.18.34.27:8890/sparql"
sparql_name2id = """PREFIX ns: <http://rdf.freebase.com/ns/>\n SELECT ?entity WHERE {\n ?entity ns:type.object.name ?name .\n FILTER(?name = "%s"@en)\n}\n"""


def construct_gen_prompt(question):
    return MULTITOPIC_ENTITIES_PROMPT.format(question)

def execurte_sparql(sparql_query, retries=3, delay=2):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    for attempt in range(retries):
        try:
            results =  sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print(f"SPARQL error: {e}. Retrying {attempt+1}/{retries}...")
            time.sleep(delay)
    return []

def get_entity_id(entity):
    query = sparql_name2id % entity
    results = execurte_sparql(query)
    return [result['entity']['value'].replace("http://rdf.freebase.com/ns/", "") for result in results]

def main():
    parser = argparse.ArgumentParser(description="RAGE")

    parser.add_argument("--dataset", type=str, default="noisy_webqsp", help="Select the dataset")
    parser.add_argument("--LLM", type=str, default='gpt-4o-mini', help="LLM model name")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum output length for the LLM")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the large language model")
    parser.add_argument("--Sbert", type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Sentence-BERT model name or path")
    parser.add_argument("--openai_api_keys", type=str, default=os.getenv("DASHSCOPE_API_KEY"), help="Your own OpenAI API keys.")
    parser.add_argument("--url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL.")
    parser.add_argument("--engine", type=str, default="azure_openai", help="Which platform you choose.")
    parser.add_argument("--width", type=int, default=3, help="Search width")
    parser.add_argument("--depth", type=int, default=3, help="Search depth")
    parser.add_argument("--num_retain_entity", type=int, default=10, help="Number of entities to retain")
    parser.add_argument("--keyword_num", type=int, default=5, help="Number of keywords for retrieval")
    parser.add_argument("--relation_num", type=int, default=5, help="Top-K relations per MID for similarity calculation")
    parser.add_argument("--prune_tools", type=str, default="llm", choices=["llm", "sentencebert"], help="Pruning tool")
    parser.add_argument("--no-remove_unnecessary_rel", action="store_false", dest="remove_unnecessary_rel", help="Do not remove unnecessary relations")
    parser.add_argument("--method", type=str, default="cot", choices=['io', 'cot', 'sc', 'rage'], help="Method for experimental comparison")
    parser.add_argument("--agent_count", type=int, default=3, help="Number of agents")
    args = parser.parse_args()

    fb_client = FreebaseClient()
    llm_handler = LLMHandler(args.LLM, args.Sbert)
    data_processor = DataProcessor(llm_handler)
    semantic_searcher = SemanticSearch()

    datas, question_field = data_processor.load_dataset(args.dataset)
    correct_count = 0

    for data in tqdm(datas, desc="Calculate the accuracy rate of mid..."):
        question = data[question_field]
        ground_truth = set(data["topic_entity"].keys())
        mid_scores = []
        topic_entity = {}

        # 1. 获取关键词对应的实体及其所有关系
        prompt = construct_gen_prompt(question)
        response = llm_handler.run_llm(prompt, args)
        words = [e.strip() for e in response.split(",")]
        if not words:
            continue

        # baseline
        # for word in words:
        #     eids = get_entity_id(word)
        #     for eid in eids:
        #         mid_scores.append(eid)
        # if any(mid in ground_truth for mid in mid_scores):
        #     correct_count += 1

        # our method
        keywords = set(words)
        for w in words:
            query_embedding = llm_handler.sbert.encode([w], normalize_embeddings=True, show_progress_bar=False)
            _, indices = semantic_searcher.faiss_index.search(query_embedding, args.keyword_num)
            keywords.update(semantic_searcher.names[idx] for idx in indices[0])

        keywords = list(keywords)
        for k in keywords:
            eids = fb_client.get_entity_id(k)
            for eid in eids:
                predicates = fb_client.get_all_relations(eid)
                if not predicates:
                    continue

                # 去重（可选）
                predicates = list(set(predicates))

                # 2. 计算每个 predicate 与问题的相似度
                sim_scores = llm_handler.compute_similarity_batch(question, predicates)

                # 取 Top-K 平均（代替所有平均）
                top_k = min(args.relation_num, len(sim_scores))
                top_scores = sorted(sim_scores, reverse=True)[:top_k]
                avg_top_score = sum(top_scores) / len(top_scores)
                mid_scores.append((eid, k, avg_top_score))

        # 3. 选出 top-3 相似度的 mid
        predicted = sorted(mid_scores, key=lambda x: x[2], reverse=True)[:args.agent_count]

        # 4. 计算准确率
        if any(p[0] in ground_truth for p in predicted):
            correct_count += 1
    
    acc = correct_count / len(datas)
    print(f"核心实体的命中个数为：{correct_count}")
    print(f"top{args.agent_count} mid 准确率为： {acc*100:.2f}%")


if __name__ == '__main__':
    main()