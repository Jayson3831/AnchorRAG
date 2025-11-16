from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from config.settings import (
    UNKNOWN_ENTITY, 
    FINISH_ID, 
    FINISH_ENTITY,
    GENERATE_TOPIC_ENTITY,
    REASONING_PROMPT,
    SCORE_ENTITY_CANDIDATES_PROMPT,
    ANSWER_PROMPT,
    PROMPT_EVALUATE,
    COT_PROMPT,
    MULTITOPIC_ENTITIES_PROMPT,
    EXTRACT_RELATION_PROMPT,
    IS_RELEVANT,
    EXTRACT_USEFUL_INFORMATION,
    READ_AND_SUMMARIZE,
    FILTER_TRIPLES,
    NOISY_PROMPT,
    EVALUATE_AND_SYNTHESIZE,
    ANSWER_FROM_KNOWLEDGE
)
from core.freebase_client import FreebaseClient
from core.llm_handler import LLMHandler
from core.semantic_search import SemanticSearch
from utils.text_utils import TextUtils
from utils.logging_utils import logger
import random
import re
import faiss
import json
import numpy as np


class ReasoningEngine:
    def __init__(self, freebase_client: FreebaseClient, llm_handler: LLMHandler, semantic_searcher: SemanticSearch):
        self.fb = freebase_client
        self.llm = llm_handler
        self.text_utils = TextUtils()
        self.semantic_searcher = semantic_searcher

    def generate_topic_entity(self, question: str, args: Dict) -> str:
        prompt = self._construct_generated_entity_prompt(question)
        gen_entity = self.llm.run_llm(prompt, args)
        return gen_entity

    def generate_keywords(self, question: str, args: Dict) -> List[str]:
        prompt = self._construct_gen_prompt(question)
        response = self.llm.run_llm(prompt, args)
        keywords = [e.strip() for e in response.split(",")] if isinstance(response, str) else []
        return keywords

    def process_question(self, question: str, args: Dict, output_file: str) -> bool:
        """处理单个问题"""
        mid_scores = []
        keywords = self.retrieve_keyword(question, args)
        for k in keywords:
            eids = self.fb.get_entity_id(k)
            for eid in eids:
                predicates = self.fb.get_all_relations(eid)
                if not predicates:
                    continue

                # 去重（可选）
                predicates = list(set(predicates))

                # 2. 计算每个 predicate 与问题的相似度
                sim_scores = self.llm.compute_similarity_batch(question, predicates)

                # 取 Top-K 平均（代替所有平均）
                top_scores = sorted(sim_scores, reverse=True)[:args.relation_num]
                avg_top_score = sum(top_scores) / len(top_scores)
                mid_scores.append((eid, k, avg_top_score))

        top_mids = sorted(mid_scores, key=lambda x: x[2], reverse=True)
        mid, name, _ = top_mids[0]
        topic_entity = {mid: name}

        cluster_chain = []
        pre_relations = []
        pre_heads = [-1] * len(topic_entity)
        flag_printed = False

        if not topic_entity:
            results = self.generate_without_explored_paths(question, args)
            self.save_results(question, results, [], output_file)
            return

        for depth in range(1, args.depth + 1):
            # 1. 搜索和剪枝关系
            current_relations = self._search_and_prune_relations(
                question, topic_entity, pre_relations, pre_heads, args
            )
            
            if not current_relations:
                self.half_stop(question, cluster_chain, depth, args, output_file)
                flag_printed = True
                break

            # 2. 收集和评分候选实体
            candidates = self._collect_and_score_entities(
                question, current_relations, args
            )
            
            if not candidates:
                self.half_stop(question, cluster_chain, depth, args, output_file)
                flag_printed = True
                break
            
            # 3. 剪枝实体
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = self.entity_prune(
                *candidates, args
            )
            
            cluster_chain.append(chain_of_entities)
            
            if flag:
                # 4. 推理答案
                stop, results = self.reasoning(question, cluster_chain, args)
                logger.info(f"Depth {depth} results:\n{results}")
                
                if stop:
                    logger.info(f"Stopped at depth {depth}")
                    self.save_results(question, results, cluster_chain, output_file)
                    flag_printed =  True
                    break
                else:
                    logger.info(f"depth %d still not find the answer." % depth)
                    finish, entities_id = self.text_utils.check_finish(entities_id)
                    if finish:
                        self.half_stop(question, cluster_chain, depth, args, output_file)
                        flag_printed =  True
                        break
                    else:
                        topic_entity = {e: self.fb.get_entity_info(e) for e in entities_id}
            else:
                self.half_stop(question, cluster_chain, depth, args, output_file)
                flag_printed =  True
                break
        
        if not flag_printed:
            # 深度耗尽仍未找到答案
            results = self.generate_without_explored_paths(question, args)
            self.save_results(question, results, [], output_file)

    def _search_and_prune_relations(self, question: str, topic_entity: Dict[str, str], 
                                  pre_relations: List[str], pre_heads: List[int], 
                                  args: Dict) -> List[Dict]:
        """搜索并剪枝关系"""
        current_relations = []
        
        for i, entity in enumerate(topic_entity):
            if entity == FINISH_ID:  # 如果有特殊标记（如FINISH_ID），跳过
                continue
                
            relations = self.relation_search_prune(
                entity, topic_entity[entity], pre_relations, 
                pre_heads[i], question, args
            )
            current_relations.extend(relations)
        
        return current_relations

    def _collect_and_score_entities(self, question: str, relations: List[Dict], 
                                  args: Dict):
        """收集并评分候选实体"""
        total_candidates = []
        total_scores = []
        total_relations = []
        total_entities_id = []
        total_topic_entities = []
        total_head = []
        
        for entity in relations:
            relation = entity['relation']
            if not re.match(r'^[a-z_]+\.[a-z_]+(\.[a-z_]+)?$', relation):
                continue
            entity_candidates_id = self.entity_search(
                entity['id'], 
                relation, 
                entity['head']
            )
            
            if not entity_candidates_id:
                continue
                
            if len(entity_candidates_id) >= 1000:
                entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)
            
            scores, entity_candidates, entity_candidates_id = self.entity_score(question, entity, entity_candidates_id, args)
            
            total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = self.update_history(
                entity_candidates, entity, scores, entity_candidates_id,
                total_candidates, total_scores, total_relations,
                total_entities_id, total_topic_entities, total_head
            )
        
        if not total_candidates:
            return None
            
        return (total_entities_id, total_relations, total_candidates, 
                total_topic_entities, total_head, total_scores)

    def relation_search_prune(self, entity_id: str, entity_name: str, pre_relations: List[str], 
                            pre_head: bool, question: str, args: Dict) -> List[Dict]:
        """搜索并剪枝关系"""
        docs = []
        # 获取头关系和尾关系
        head_relations = self.fb.get_relations(entity_id, 'head')
        tail_relations = self.fb.get_relations(entity_id, 'tail')
        
        # 过滤不需要的关系
        if args.remove_unnecessary_rel:
            head_relations = [r for r in head_relations if not self._abandon_relation(r)]
            tail_relations = [r for r in tail_relations if not self._abandon_relation(r)]
        
        # 去除已探索的关系
        if pre_head:
            tail_relations = list(set(tail_relations) - set(pre_relations))
        else:
            head_relations = list(set(head_relations) - set(pre_relations))
        
        total_relations = sorted(list(set(head_relations + tail_relations)))
        
        if not total_relations:
            return []
        
        if args.prune_tools == "llm":
            prompt = self._construct_relation_prompt(question, entity_name, total_relations, args)

            result = self.llm.run_llm(prompt, args)
            flag, relations = self.clean_relations(result, entity_id, head_relations)
        else:
            # beam-search
            # docs = [
            #     (rel, f"{entity_name} [SEP] {rel}" if pre_head else f"{rel} [SEP] {entity_name}")
            #     for rel in total_relations
            # ]
            beam_relations, beam_scores = self.retrieve_top_docs(question, total_relations, args.width)

            # Sparse Retrieval
            sparse_relations, sparse_scores = self.compute_bm25_similarity(question, total_relations, args.width)

            relations = beam_relations + sparse_relations
            scores = beam_scores + sparse_scores
            flag, relations = self._format_relations(entity_id, entity_name, relations, scores, head_relations)
        
        return relations if flag else []

    def entity_search(self, entity: str, relation: str, head: bool = True) -> List[str]:
        """搜索相关实体"""
        return self.fb.get_related_entities(entity, relation, head)

    def entity_score(self, question: str, entity_info:Dict, entity_candidates_id: List[str], 
                    args: Dict) -> Tuple[List[float], List[str], List[str]]:
        """评分实体"""
        entity_names = [self.fb.get_entity_info(eid) for eid in entity_candidates_id]
        
        if all(name == UNKNOWN_ENTITY for name in entity_names):
            scores = [1/len(entity_names) * entity_info["score"]] * len(entity_names)
            return scores, entity_names, entity_candidates_id
            
        entity_names, entity_candidates_id = self.text_utils.filter_unknown_entities(entity_names, entity_candidates_id)
        
        if len(entity_names) == 1:
            return [entity_info["score"]], entity_names, entity_candidates_id
        if not entity_names:
            return [0.0], entity_names, entity_candidates_id
        
        # 对齐名称和 ID
        zipped = list(zip(entity_names, entity_candidates_id))

        if args.prune_tools == "llm":
            prompt = self._construct_entity_prompt(question, entity_info['relation'], entity_names)
            result = self.llm.run_llm(prompt, args)
            return [float(x) * entity_info['score'] for x in self.clean_scores(result, entity_names)], entity_names, entity_candidates_id
        else:
            docs = [
                (entity, f"{entity_info['name']} [SEP] {entity_info['relation']} [SEP] {entity}" if entity_info['head'] 
                else f"{entity} [SEP] {entity_info['relation']} [SEP] {entity_info['name']}")
                for entity in entity_names
            ]
            beam_entities, beam_scores = self.retrieve_top_docs(question, docs, args.width)
            sparse_entities, sparse_scores = self.compute_bm25_similarity(question, entity_names, args.width)

            topn_entities = beam_entities + sparse_entities
            scores = beam_scores + sparse_scores
        
        if all(s == 0 for s in scores):
            scores = [1/len(scores)] * len(scores)
            
        name2id = {name: eid for name, eid in zipped}
        topn_ids = [name2id[name] for name in topn_entities]

        return [s for s in scores], topn_entities, topn_ids

    def update_history(self, entity_candidates, entity, scores, entity_candidates_id,
                      total_candidates, total_scores, total_relations, 
                      total_entities_id, total_topic_entities, total_head):
        """更新历史记录"""
        if not entity_candidates:
            entity_candidates = [FINISH_ENTITY]
            entity_candidates_id = [FINISH_ID]
            
        candidates_relation = [entity['relation']] * len(entity_candidates)
        topic_entities = [entity['id']] * len(entity_candidates)
        head_flags = [entity['head']] * len(entity_candidates)
        
        total_candidates.extend(entity_candidates)
        total_scores.extend(scores)
        total_relations.extend(candidates_relation)
        total_entities_id.extend(entity_candidates_id)
        total_topic_entities.extend(topic_entities)
        total_head.extend(head_flags)
        
        return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head

    def entity_prune(self, total_entities_id, total_relations, total_candidates, 
                    total_topic_entities, total_head, total_scores, args):
        """剪枝实体"""
        combined = list(zip(total_entities_id, total_relations, total_candidates, 
                          total_topic_entities, total_head, total_scores))
        combined_sorted = sorted(combined, key=lambda x: x[5], reverse=True)
        
        # 解压排序后的数据
        entities_id, relations, candidates, topics, heads, scores = zip(*combined_sorted)
        
        # 取前width个
        width = min(len(candidates), args.width)
        entities_id = list(entities_id[:width])
        relations = list(relations[:width])
        candidates = list(candidates[:width])
        topics = list(topics[:width])
        heads = list(heads[:width])
        scores = list(scores[:width])
        
        # 过滤掉分数为0的
        filtered = [(eid, rel, cand, top, head, score) 
                   for eid, rel, cand, top, head, score 
                   in zip(entities_id, relations, candidates, topics, heads, scores)
                   if score != 0]
        
        if not filtered:
            return False, [], [], [], []
            
        entities_id, relations, candidates, topics, heads, scores = zip(*filtered)
        
        # 获取主题实体名称
        topic_names = [self.fb.get_entity_info(tid) for tid in topics]
        
        # 构建实体链
        chain = [[(topic_names[i], relations[i], candidates[i]) 
                for i in range(len(candidates))]]
        
        return True, chain, list(entities_id), list(relations), list(heads)

    def triples_prune(self, question, eids, rels, cans, tids, heads, scores, args):
        topic_names = [self.fb.get_entity_info(tid) for tid in tids]
        infos = [{"subject": tids[i], 
                   "relation": rels[i], 
                   "object_id": eids[i], 
                   "object": cans[i], 
                   "head": heads[i], 
                   "triple": (topic_names[i], rels[i], cans[i]) if heads[i]
                                else (cans[i], rels[i], topic_names[i])
                } for i in range(len(cans))]
        origin_triples = [f"({subj}, {rel}, {obj})" for subj, rel, obj in [info["triple"] for info in infos]]

        # 大模型筛选三元组
        origin_triples = list(set(origin_triples))
        candidate_chains = "\n".join(origin_triples)
        prompt = FILTER_TRIPLES.format(question, candidate_chains)
        response = self.llm.run_llm(prompt, args)

        # 使用正则表达式提取符合格式的三元组
        pattern = r"\(([^,]+?),\s*([^,]+?),\s*([^)]+?)\)"
        matches = re.findall(pattern, response) if isinstance(response, str) else []
        filtered_triples = [tuple(m.strip() for m in match) for match in matches]

        candidate_entities = {}
        pre_relations = []
        pre_heads = []
        reasoning_chains = set()
        for info in infos:
            triple = info["triple"]
            if triple in filtered_triples:
                candidate_entities[info["object_id"]] = info["object"]
                pre_relations.append(info["relation"])
                pre_heads.append(info["head"])
                reasoning_chains.add(triple)

        return list(reasoning_chains), candidate_entities, pre_relations, pre_heads

    def reasoning(self, question: str, cluster_chain: List, args: Dict) -> Tuple[bool, str]:
        """推理答案"""
        prompt = self._construct_reasoning_prompt(question, cluster_chain)
        response = self.llm.run_llm(prompt, args)
        
        result = self.text_utils.extract_answer(response)
        return self.text_utils.is_true(result), response

    def reasoning_with_summary(self, question: str, summary: str, args: Dict) -> Tuple[bool, str]:
        """推理答案：基于问题、三元组和摘要判断是否能回答问题"""
        prompt = self._construct_reasoning_prompt_with_summary(question, summary)
        
        # 调用 LLM
        response = self.llm.run_llm(prompt, args)
        
        # 解析结果
        return self.text_utils.is_yes_in_response(response), response

    def generate_without_explored_paths(self, question: str, args: Dict) -> str:
        """无探索路径生成答案"""
        prompt = self._construct_cot_prompt(question)
        # return self.llm.qwen_generate_text(
        #     prompt,
        #     args.temperature_reasoning
        # )

        return self.llm.run_llm(prompt, args)

    def half_stop(self, question: str, cluster_chain: List, depth: int, args: Dict, file_name: str):
        """中途停止处理"""
        logger.info(f"No new knowledge added at depth {depth}, stopping search.")
        answer = self.generate_answer(question, cluster_chain, args)
        self.save_results(question, answer, cluster_chain, file_name)

    def generate_answer(self, question: str, cluster_chain: List, args: Dict) -> str:
        """生成最终答案"""
        prompt = self._construct_answer_prompt(question, cluster_chain)
        # return self.llm.qwen_generate_text(
        #     prompt,
        #     args.temperature_reasoning
        # )
        return self.llm.run_llm(prompt, args)

    def save_results(self, question: str, results: str, cluster_chain: List, file_name: str):
        """保存结果"""
        data = {
            "question": question,
            "results": results,
            "reasoning_chains": cluster_chain
        }
        from utils.file_utils import FileUtils
        FileUtils.save_to_jsonl(data, file_name)

    def _abandon_relation(self, relation: str) -> bool:
        """判断是否应该丢弃关系"""
        return (relation == "type.object.type" or
                relation == "type.object.name" or 
                relation.startswith("common.") or 
                relation.startswith("freebase.") or 
                "sameAs" in relation)

    def _construct_gen_prompt(self, question: str) -> str:
        return MULTITOPIC_ENTITIES_PROMPT.format(question)
        # return NOISY_PROMPT.format(question)

    def _construct_generated_entity_prompt(self, question: str) -> str:
        return GENERATE_TOPIC_ENTITY.format(question)

    def _construct_relation_prompt(self, question: str, entity_name, relations: List[Dict], args) -> str:
        """构建关系剪枝提示"""
        return EXTRACT_RELATION_PROMPT % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(relations)

    def _construct_entity_prompt(self, question: str, relation: str, entity_candidates: str) -> str:
        """构建实体评分提示"""
        return SCORE_ENTITY_CANDIDATES_PROMPT.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

    def _construct_reasoning_prompt(self, question: str, cluster_chain: List) -> str:
        """构建推理提示"""
        chain_text = '\n'.join(
            ', '.join(str(item) for item in chain)
            for sublist in cluster_chain
            for chain in (sublist if isinstance(sublist, list) else [])
        )
        prompt_evaluate = PROMPT_EVALUATE + '\n' + f"Q: {question}\n" + f"Knowledge Triplets: {chain_text}"
        return prompt_evaluate  # 推荐命名参数

    def _construct_reasoning_prompt_with_summary(self, question: str, summary: str = None) -> str:
        """构建推理提示，包含问题、知识三元组和摘要文本"""
        summary_text = summary if summary is not None else ""

        # 使用 REASONING_PROMPT 模板填充内容
        prompt = REASONING_PROMPT.format(question, summary_text)
        return prompt

    def _construct_answer_prompt(self, question: str, cluster_chain: List) -> str:
        """构建答案生成提示"""
        chain_text = '\n'.join([', '.join(map(str, chain)) 
                              for sublist in cluster_chain 
                              for chain in sublist])
        answer_prompt = ANSWER_PROMPT + f"Q: {question}\n" + f"Knowledge Triplets: {chain_text}"
        return answer_prompt

    def _construct_cot_prompt(self, question: str) -> str:
        """构建思维链提示"""
        cot_prompt = COT_PROMPT + f"\n\nQ: {question}\n"
        return cot_prompt

    def clean_relations(self, text: str, entity_id: str, head_relations: List[str]) -> Tuple[bool, List[Dict]]:
        """清理关系文本"""
        pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
        relations = []
        
        matches = re.finditer(pattern, text) if isinstance(text, str) else []
        for match in matches:
            relation = match.group("relation").strip()
            if ';' in relation:
                continue
                
            score = match.group("score")
            if not relation or not score:
                return False, "output uncompleted.."
            
            try:
                score = float(score)
                relations.append({
                    "id": entity_id,
                    "relation": relation,
                    "score": score,
                    "head": relation in head_relations
                })
            except ValueError:
                return False, "Invalid score"
        
        return (True, relations) if relations else (False, "No relations found")

    def clean_scores(self, text: str, candidates: List[str]) -> List[float]:
        """清理分数文本"""
        scores = [float(num) for num in re.findall(r'\d+\.\d+', text)]
        return scores if len(scores) == len(candidates) else [1/len(candidates)] * len(candidates)

    def compute_bm25_similarity(self, query: str, corpus: List[str], width: int) -> Tuple[List[str], List[float]]:
        """计算BM25相似度"""
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        
        doc_scores = bm25.get_scores(tokenized_query)
        corpus_with_scores = list(zip(corpus, doc_scores))
        sorted_corpus = sorted(corpus_with_scores, key=lambda item: item[1], reverse=True)
        top_n = sorted_corpus[:width]

        relations, scores = zip(*top_n)
        return list(relations), list(scores)

    def question_retrieve_entity(self, question: str, args) -> List[str]:
        query_embedding = self.llm.sbert.encode([question], normalize_embeddings=True, show_progress_bar=False)
        _, indices = self.semantic_searcher.faiss_index.search(query_embedding, args.width)

        return [self.semantic_searcher.names[idx] for idx in indices[0]]

    def retrieve_top_entity(self, question: str, args) -> List[str]:
        query_entity = self.generate_topic_entity(question, args)
        if not query_entity:
            return ""

        query_embedding = self.llm.sbert.encode([query_entity], normalize_embeddings=True, show_progress_bar=False)
        _, indices = self.semantic_searcher.faiss_index.search(query_embedding, args.width)

        return query_entity, [self.semantic_searcher.names[idx] for idx in indices[0]]

    def retrieve_keyword(self, question: str, args) -> List[str]:
        words = self.generate_keywords(question, args)
        if not words:
            return ""
        keywords = set(words)
        for w in words:
            query_embedding = self.llm.sbert.encode([w], normalize_embeddings=True, show_progress_bar=False)
            _, indices = self.semantic_searcher.faiss_index.search(query_embedding, args.keyword_num)
            keywords.update(self.semantic_searcher.names[idx] for idx in indices[0])

        keywords = list(keywords)
        return keywords

    def retrieve_query_entity(self, question: str, args):
        query_embedding = self.llm.sbert.encode([question], normalize_embeddings=True, show_progress_bar=False)
        _, indices = self.semantic_searcher.faiss_index.search(query_embedding, 20)

        query_entities = [self.semantic_searcher.names[idx] for idx in indices[0]]
        
        return query_entities

    def retrieve_top_docs(self, query: str, relation_pairs: List[str], width: int) -> Tuple[List[str], List[float]]:
        """检索相关文档"""
        try:
            elements, docs = zip(*relation_pairs)
        except:
            elements = relation_pairs
            docs = relation_pairs
        doc_embeddings = self.llm.sbert.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(doc_embeddings)

        query_embedding = self.llm.sbert.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        width = min(width, len(docs))
        scores, indices = index.search(query_embedding, width)
        
        top_docs = [elements[i] for i in indices[0]]
        top_scores = scores[0].tolist()
        
        return top_docs, top_scores
    
    def _format_relations(self, entity_id: str, entity_name: str, relations: List[str], scores: List[float], 
                        head_relations: List[str]) -> Tuple[bool, List[Dict]]:
        """格式化关系数据"""
        if all(score == 0 for score in scores):
            scores = [1/len(scores)] * len(scores)
        
        return True, [{
            "id": entity_id,
            "name": entity_name,
            "relation": rel,
            "score": scores[i],
            "head": rel in head_relations
        } for i, rel in enumerate(relations) if scores[i] > 0]

    def is_relevant_to_question(self, question: str, triplets: List, args) -> Tuple[bool, str]:
        """调用 LLM 判断三元组是否与问题相关"""
        chain_text = '\n'.join(
            '('
            + ', '.join(
                f'"{item}"' if isinstance(item, str) else str(item)
                for item in chain
            )
            + ')'
            for sublist in triplets
            for chain in (sublist if isinstance(sublist, list) else [])
        )

        prompt = EXTRACT_USEFUL_INFORMATION.format(question, chain_text)

        response = self.llm.run_llm(prompt, args)
        is_rel, info = self.extract_fields(response)
        return is_rel, info

    def evaluate_and_synthesize_knowledge(self, question: str, memory: List[str], new_chains: List, args) -> Dict:
        """
        调用 LLM 对新知识进行事实核查、纠正，判断信息是否充足，并提取关键信息。
        """
        memory_text = "\n".join(f"- {m}" for m in memory) if memory else "none"
        new_chains_text = "\n".join(f"- {chain}" for chain in new_chains) if new_chains else "none"

        prompt = EVALUATE_AND_SYNTHESIZE.format(
            question=question,
            memory=memory_text,
            new_chains=new_chains_text
        )
        
        response_text = self.llm.run_llm(prompt, args)
        
        # 解析 LLM 返回的 JSON 字符串
        try:
            # 尝试直接解析
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError:
            # 如果失败，使用正则表达式提取 JSON 部分
            match = re.search(r'```json\n({.*?})\n```', response_text, re.DOTALL)
            if match:
                try:
                    parsed_response = json.loads(match.group(1))
                except json.JSONDecodeError:
                    # 如果仍然失败，返回一个表示失败的默认结构
                    return {"is_sufficient": False, "new_knowledge": "Failed to parse LLM response."}
            else:
                return {"is_sufficient": False, "new_knowledge": "No valid JSON found in LLM response."}

        return parsed_response

    def answer_from_knowledge(self, question: str, memory: List[str], args) -> str:
        """
        当达到最大深度时，根据所有累积的记忆生成答案。
        """
        memory_text = "\n".join(f"- {m}" for m in memory) if memory else "无"
        prompt = ANSWER_FROM_KNOWLEDGE.format(
            question=question,
            memory=memory_text
        )
        return self.llm.run_llm(prompt, args)

    def read_and_summarize(self, question: str, triplets: List, args) -> Tuple[bool, str]:
        """调用 LLM 判断三元组是否与问题相关"""
        chain_text = '\n'.join(
            '('
            + ', '.join(
                f'"{item}"' if isinstance(item, str) else str(item)
                for item in chain
            )
            + ')'
            for sublist in triplets
            for chain in (sublist if isinstance(sublist, list) else [])
        )

        prompt = READ_AND_SUMMARIZE.format(question, chain_text)
        response = self.llm.run_llm(prompt, args)

        return response

    def is_valid_predicate(self, predicate: str) -> bool:
        return not any(predicate.startswith(prefix) for prefix in ["atom.feed", "freebase.", "dataworld", "common.document", "type.object.type", "type.object.permission","type.type."])
    
    def safe_extract_model_output(self, text):
        try:
            # 优先尝试标准 JSON
            return json.loads(text)
        except:
            # Fallback: 用正则提取
            is_relevant, information = self.extract_fields(text)
            return {"is_relevant": is_relevant, "information": information}

    def extract_fields(self, text):
        # 提取 is_relevant 值（大小写不敏感）
        is_relevant_match = re.search(r'"?is_relevant"?\s*:\s*(true|false)', text, re.IGNORECASE)
        is_relevant = is_relevant_match.group(1).lower() == 'true' if is_relevant_match else False

        # 提取 information 值（支持中间有嵌套标点或空格）
        info_match = re.search(r'"?information"?\s*:\s*"(.*?)"', text, re.DOTALL)
        information = info_match.group(1).strip() if info_match else "No information extracted."

        return is_relevant, information