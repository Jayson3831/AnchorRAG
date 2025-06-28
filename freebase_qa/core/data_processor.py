import json
import re
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from config.settings import DATASET_PATHS
from core.llm_handler import LLMHandler
from utils.logging_utils import logger


class DataProcessor:
    def __init__(self, llm_handler: LLMHandler):
        self.model = llm_handler
    
    def load_dataset(self, dataset_name: str) -> Tuple[List[Dict], str]:
        """加载数据集"""
        if dataset_name not in DATASET_PATHS:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        path = DATASET_PATHS[dataset_name]
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 根据数据集确定问题字段
        question_field = {
            'webqsp': 'RawQuestion',
            'cwq': 'question',
            'webqsp_sampled': 'RawQuestion',
            'noisy_webqsp': 'NoisyQuestion'
            # 其他数据集映射...
        }.get(dataset_name, 'question')
        
        return data, question_field
    
    def retrieve_top_docs(self, query: str, docs: List[str], width: int) -> Tuple[List[str], List[float]]:
        """使用SBERT检索相关文档"""
        query_emb = self.model.sbert.encode(query)
        doc_emb = self.model.sbert.encode(docs)
        
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        doc_score_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        return [pair[0] for pair in doc_score_pairs[:width]], [pair[1] for pair in doc_score_pairs[:width]]
    
    def compute_bm25_similarity(self, query: str, corpus: List[str], width: int) -> Tuple[List[str], List[float]]:
        """计算BM25相似度"""
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        
        doc_scores = bm25.get_scores(tokenized_query)
        relations = bm25.get_top_n(tokenized_query, corpus, n=width)
        scores = sorted(doc_scores, reverse=True)[:width]
        
        return relations, scores
    
    def clean_scores(self, text: str, candidates: List[str]) -> List[float]:
        """从文本中提取分数"""
        scores = [float(num) for num in re.findall(r'\d+\.\d+', text)]
        return scores if len(scores) == len(candidates) else [1/len(candidates)] * len(candidates)