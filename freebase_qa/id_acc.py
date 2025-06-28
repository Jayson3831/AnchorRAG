from tqdm import tqdm
from core.freebase_client import FreebaseClient
from core.llm_handler import LLMHandler
from core.semantic_search import SemanticSearch
from core.data_processor import DataProcessor
from config.settings import (
    MODEL, MULTITOPIC_ENTITIES_PROMPT
)
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.chdir(sys.path[0])


def construct_gen_prompt(question):
    return MULTITOPIC_ENTITIES_PROMPT.format(question)

def main():
    llm = "gpt"
    Sbert = MODEL['minilm']
    dataset = "webqsp"
    top_k = 5  # 可调参数：取每个mid的Top-K关系相似度来平均
    keywords_num = 3
    agent_count = 1
    correct_count = 0

    fb_client = FreebaseClient()
    llm_handler = LLMHandler(llm, Sbert)
    data_processor = DataProcessor(llm_handler)
    semantic_searcher = SemanticSearch()

    datas, question_field = data_processor.load_dataset(dataset)

    for data in tqdm(datas, desc="Calculate the accuracy rate of mid..."):
        question = data[question_field]
        ground_truth = set(data["topic_entity"].keys())
        mid_scores = []

        # 1. 获取关键词对应的实体及其所有关系
        prompt = construct_gen_prompt(question)
        response = llm_handler.run_llm(prompt)
        words = [e.strip() for e in response.split(",")]
        if not words:
            continue

        keywords = set()
        for w in words:
            keywords.add(w)
            query_embedding = llm_handler.sbert.encode([w], normalize_embeddings=True, show_progress_bar=False)
            _, indices = semantic_searcher.faiss_index.search(query_embedding, keywords_num)
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
                top_k = min(top_k, len(sim_scores))
                top_scores = sorted(sim_scores, reverse=True)[:top_k]
                avg_top_score = sum(top_scores) / len(top_scores)
                mid_scores.append((eid, k, avg_top_score))

        # 3. 选出 top-3 相似度的 mid
        predicted = sorted(mid_scores, key=lambda x: x[2], reverse=True)[:agent_count]

        # 4. 计算准确率
        if any(p[0] in ground_truth for p in predicted):
            correct_count += 1
    
    acc = correct_count / len(datas)
    print(f"核心实体的命中个数为：{correct_count}")
    print(f"top{agent_count} mid 准确率为： {acc*100:.2f}%")


if __name__ == '__main__':
    main()