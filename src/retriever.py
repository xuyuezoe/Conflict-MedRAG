#!/usr/bin/env python3
"""
混合检索器

提供 BM25 + Dense 向量检索，RRF（Reciprocal Rank Fusion）分数融合。

接口设计：
  HybridRetriever.retrieve(query, top_k, method) → List[RetrievalResult]
  调用方（DCR、SCSR、Baseline 等）使用统一接口，不需要了解底层实现。

数学基础：
  RRF(d) = Σ_{m∈{bm25,dense}} 1 / (rank_m(d) + k)，k=60（标准 RRF 常数）
  BM25 分数 → 按查询结果排名（不直接用原始 BM25 分数，因为 BM25 分数无上界）
  Dense 余弦相似度 → 直接作为排名分数（已在索引构建时 L2 归一化）
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

from src.types import TextChunk


# ── 检索结果 ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """
    单次检索结果，Stage 1 的输出单元。

    参数：
        chunk:            文本块
        score:            归一化分数（RRF 分数，或单一方法的原始分）
        retrieval_method: 检索方法标识
        bm25_rank:        BM25 排名（RRF 中间结果，调试用）
        dense_rank:       Dense 排名（RRF 中间结果，调试用）
    """
    chunk: TextChunk
    score: float
    retrieval_method: str
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None


# ── 混合检索器 ────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    混合检索器：BM25 + Dense 向量检索，RRF 分数融合。

    初始化时加载索引到内存（BM25 pickle + FAISS index）。
    线程安全：只读，无状态变更。

    关键参数：
        rrf_k:       RRF 常数（默认 60，TREC 标准值）
        bm25_weight: BM25 在 RRF 中的权重（0~1，剩余给 Dense）
    """

    RRF_K = 60    # 标准 RRF 常数

    def __init__(
        self,
        index_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight: float = 0.5,
    ) -> None:
        """
        初始化：加载 BM25 索引、FAISS 索引、文本块元数据。

        参数：
            index_dir:       索引目录（data/index）
            embedding_model: Dense embedding 模型（需与建索引时一致）
            bm25_weight:     RRF 中 BM25 的权重（默认 0.5，即等权融合）
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight

        # 第一步：加载文本块元数据
        chunks_path = index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"[Retriever] 文本块元数据不存在: {chunks_path}。"
                f"请先运行 python3 scripts/index_textbooks.py"
            )
        self._chunks: Dict[str, TextChunk] = {}
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                chunk = TextChunk(
                    chunk_id=row["chunk_id"],
                    source_book=row["source_book"],
                    text=row["text"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    token_count=row.get("token_count", 0),
                )
                self._chunks[chunk.chunk_id] = chunk
        self._chunk_list: List[TextChunk] = list(self._chunks.values())

        # 第二步：加载 BM25 索引
        bm25_path = index_dir / "bm25" / "bm25_index.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"[Retriever] BM25 索引不存在: {bm25_path}")
        with bm25_path.open("rb") as f:
            bm25_data = pickle.load(f)
        self._bm25 = bm25_data["bm25"]
        self._bm25_chunk_ids: List[str] = bm25_data["chunk_ids"]

        # 第三步：加载 FAISS 索引（可选，若不存在则只用 BM25）
        faiss_path = index_dir / "dense" / "faiss.index"
        chunk_ids_path = index_dir / "dense" / "chunk_ids.json"
        self._faiss_index = None
        self._dense_chunk_ids: List[str] = []
        self._embedding_model = None

        if faiss_path.exists() and chunk_ids_path.exists():
            import faiss
            from sentence_transformers import SentenceTransformer
            self._faiss_index = faiss.read_index(str(faiss_path))
            with chunk_ids_path.open("r", encoding="utf-8") as f:
                self._dense_chunk_ids = json.load(f)
            self._embedding_model = SentenceTransformer(embedding_model)
        else:
            print(
                f"[Retriever] 未找到 Dense 索引（{faiss_path}），"
                f"将仅使用 BM25 检索。"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        method: Literal["hybrid", "bm25", "dense"] = "hybrid",
    ) -> List[RetrievalResult]:
        """
        执行检索，返回按分数降序排列的文本块列表。

        参数：
            query:  检索查询（D_q 或完整 q，由调用方决定语义）
            top_k:  返回结果数量上限
            method: 检索方法（hybrid/bm25/dense）

        返回：
            RetrievalResult 列表，按 score 降序，最多 top_k 条。

        异常：
            ValueError: method="dense" 但 Dense 索引不存在
        """
        if method == "dense" and self._faiss_index is None:
            raise ValueError(
                "[Retriever] 请求 Dense 检索但 Dense 索引未加载。"
                "请先构建 Dense 索引或使用 method='bm25'。"
            )

        if method == "bm25":
            return self._bm25_search(query, top_k)
        if method == "dense":
            return self._dense_search(query, top_k)
        return self._hybrid_search(query, top_k)

    def _bm25_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        BM25 检索。

        词条化方式与建索引时保持一致（小写+空格分词）。
        """
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # 取 top_k 个最高分的索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices):
            chunk_id = self._bm25_chunk_ids[idx]
            if chunk_id not in self._chunks:
                continue
            results.append(RetrievalResult(
                chunk=self._chunks[chunk_id],
                score=float(scores[idx]),
                retrieval_method="bm25",
                bm25_rank=rank + 1,
            ))
        return results

    def _dense_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Dense 向量检索（FAISS + SentenceTransformer）。

        查询 embedding 同样 L2 归一化，与 IndexFlatIP 配合实现余弦相似度。
        """
        query_embedding = self._embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        distances, indices = self._faiss_index.search(query_embedding, top_k)

        results: List[RetrievalResult] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:    # FAISS 返回 -1 表示不足 top_k 个结果
                continue
            chunk_id = self._dense_chunk_ids[idx]
            if chunk_id not in self._chunks:
                continue
            results.append(RetrievalResult(
                chunk=self._chunks[chunk_id],
                score=float(dist),          # 余弦相似度（[0,1]，已归一化）
                retrieval_method="dense",
                dense_rank=rank + 1,
            ))
        return results

    def _hybrid_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        混合检索：RRF 融合 BM25 和 Dense 排名。

        RRF 公式：score(d) = bm25_weight × 1/(rank_bm25(d)+k)
                            + dense_weight × 1/(rank_dense(d)+k)
        未在某方法排名中出现的文档，按排名 = top_k + 1 处理。
        """
        # 获取两个方法的结果（扩大检索量，保证 RRF 融合后仍有足够候选）
        pool_k = min(top_k * 3, 60)
        bm25_results = self._bm25_search(query, pool_k)
        dense_results = self._dense_search(query, pool_k) if self._faiss_index else []

        # 建立排名映射
        bm25_rank_map: Dict[str, int] = {r.chunk.chunk_id: r.bm25_rank for r in bm25_results}
        dense_rank_map: Dict[str, int] = {r.chunk.chunk_id: r.dense_rank for r in dense_results}

        # 合并所有候选 chunk_id
        all_ids = set(bm25_rank_map.keys()) | set(dense_rank_map.keys())

        rrf_scores: Dict[str, float] = {}
        fallback_rank = pool_k + 1
        for cid in all_ids:
            bm25_r = bm25_rank_map.get(cid, fallback_rank)
            dense_r = dense_rank_map.get(cid, fallback_rank)
            rrf_scores[cid] = (
                self.bm25_weight * 1.0 / (bm25_r + self.RRF_K)
                + self.dense_weight * 1.0 / (dense_r + self.RRF_K)
            )

        # 按 RRF 分数降序排列，取 top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: -rrf_scores[cid])[:top_k]

        results: List[RetrievalResult] = []
        for cid in sorted_ids:
            if cid not in self._chunks:
                continue
            results.append(RetrievalResult(
                chunk=self._chunks[cid],
                score=rrf_scores[cid],
                retrieval_method="hybrid",
                bm25_rank=bm25_rank_map.get(cid),
                dense_rank=dense_rank_map.get(cid),
            ))
        return results
