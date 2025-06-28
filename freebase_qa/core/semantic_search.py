import numpy as np
from typing import Dict
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING
from utils.logging_utils import logger
from pathlib import Path
import faiss
import json
import os
import torch

class SemanticSearch:
    def __init__(self):
        """初始化语义搜索引擎"""
        self.embeddings_path = EMBEDDING
        self.faiss_index = self._load_index()
        logger.info("Semantic search engine initialized")

    def _load_index(self) -> faiss.Index:
        """
        从 .npy 和 .json 加载实体嵌入，若已有索引则直接加载；否则创建并保存。
        """
        try:
            embeddings_path = self.embeddings_path['embeddings_path']
            names_path = self.embeddings_path['names_path']
            index_path = self.embeddings_path['index_path']

            # 加载嵌入向量 & 实体名
            self.embeddings = np.load(embeddings_path)
            with open(names_path, 'r', encoding='utf-8') as f:
                self.names = json.load(f)

            if len(self.names) != len(self.embeddings):
                raise ValueError(f"数据不一致: names({len(self.names)}) != embeddings({len(self.embeddings)})")

            # 归一化嵌入向量（单位化，适用于 Inner Product）
            faiss.normalize_L2(self.embeddings)

            dimension = self.embeddings.shape[1]

            # 优先尝试加载已存在的索引文件
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
                logger.info(f"Loaded existing FAISS index from: {index_path}")
            else:
                if len(self.embeddings) > 100_000:
                    # 大规模：用 IVF + PQ 结构
                    nlist = min(100, len(self.embeddings) // 1000)
                    quantizer = faiss.IndexFlatIP(dimension)
                    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
                    index.train(self.embeddings)
                    index.add(self.embeddings)
                    index.nprobe = min(10, nlist)
                    logger.info(f"Trained new IndexIVFPQ (nlist={nlist})")
                else:
                    # 小规模：用全量暴力索引
                    index = faiss.IndexFlatIP(dimension)
                    index.add(self.embeddings)
                    logger.info(f"Created flat index (n={len(self.embeddings)})")

                # 保存索引
                faiss.write_index(index, index_path)
                logger.info(f"Saved FAISS index to: {index_path}")

            self.entity_data = {
                "names": self.names,
                "embeddings": self.embeddings,
                "index": index
            }
            return index

        except Exception as e:
            logger.error(f"Index loading failed: {e}")
            raise

