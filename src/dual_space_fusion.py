#!/usr/bin/env python3
"""
Stage 2：双空间融合（Dual-Space Fusion）

数学核心：正交双空间检索的条件概率融合（Product of Experts / Bayesian fusion）

理论基础：
  由于 D_q ⊥ C_q（正交分解），文档对 query 的贡献满足独立性：
    P(d is ideal evidence | q)
    = P(d | D_q, C_q)
    = P(d | D_q) × P(d safe | C_q)   [由正交独立性]
    = S_joint(d) × κ(C_q, π_d)       [实例化]

  这是 Hinton (1999) Product of Experts 的直接应用：
    每个"专家"（疾病维度 + 约束维度）独立给出概率评估，乘积为联合概率。

三信号 RRF 融合公式：
  S_joint(d) = alpha * [w_bm25/(rank_BM25+60) + w_dense/(rank_Dense+60)]
             + beta  * [1/(rank_Constraint+60)]
             = alpha * RRF_D(d) + beta * RRF_C(d)

  其中：
    RRF_D(d)：疾病空间（BM25 + Dense）的 RRF 分数（来自 Stage 1A）
    RRF_C(d)：约束空间的 RRF 分数（来自 Stage 1B）
    α = disease_weight（默认 0.7）
    β = constraint_weight（默认 0.3）

  当 d ∉ R_D 时：rank_BM25 = rank_Dense = ∞ → RRF_D(d) = 0
  当 d ∉ R_C 时：rank_Constraint = ∞ → RRF_C(d) = 0

DCR 条件概率评分（不变）：
  score(d) = S_joint(d) × κ(C_q, π_d)
  κ = 0 → score = 0，物理排除（不可补偿性保证，Theorem 3.5）

与旧版 DCRReranker 的区别：
  旧：只接受 R_D，S_joint = S_D（疾病相关性）
  新：接受 R_D ∪ R_C，S_joint 同时包含疾病和约束维度的贡献
  数学等价关系：当 R_C = ∅ 时，DualSpaceFusion 退化为 DCRReranker
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.types import QueryDecomposition, RetrievedDoc
from src.kappa_scorer import KappaScorer
from src.retriever import RetrievalResult


class DualSpaceFusion:
    """
    双空间融合：Triple-RRF + DCR 条件概率评分。

    输入：
      R_D = Stage 1A 的疾病空间检索结果（含 BM25 和 Dense 排名）
      R_C = Stage 1B 的约束空间检索结果

    输出：
      E(q) = {d ∈ R_D ∪ R_C | κ(d) > 0}，按 score = S_joint × κ 降序排列

    关键性质：
      d ∈ R_D ∩ R_C → 两维度贡献叠加，得分最高
      d ∈ R_D，不在 R_C → 约束空间贡献为 0，S_joint = alpha * RRF_D(d)
      d ∈ R_C，不在 R_D → 疾病空间贡献为 0，S_joint = beta * RRF_C(d)
      任何 d：若 κ(d) = 0，则 score(d) = 0，物理排除
    """

    RRF_K = 60   # 标准 RRF 常数（与 HybridRetriever 保持一致）

    def __init__(
        self,
        kappa_scorer: KappaScorer,
        disease_weight: float = 0.7,
        constraint_weight: float = 0.3,
    ) -> None:
        """
        参数：
            kappa_scorer:       κ 计算器（复用 Stage 1A 使用的同一实例）
            disease_weight:     疾病空间的 RRF 权重 α（默认 0.7）
            constraint_weight:  约束空间的 RRF 权重 β（默认 0.3）

        约束：
            disease_weight + constraint_weight 无需等于 1（RRF 本身不要求归一化）
        """
        self._kappa_scorer = kappa_scorer
        self._disease_weight = disease_weight
        self._constraint_weight = constraint_weight

    def fuse_and_score(
        self,
        disease_results: List[RetrievalResult],
        constraint_results: List[RetrievalResult],
        decomposition: QueryDecomposition,
        top_k: int = 5,
    ) -> Tuple[List[RetrievedDoc], List[RetrievedDoc]]:
        """
        执行双空间融合，返回精准证据集 E(q)。

        参数：
            disease_results:    Stage 1A 的疾病空间检索结果（R_D，按分数降序）
            constraint_results: Stage 1B 的约束空间检索结果（R_C，按排名排序）
            decomposition:      Module 0 分解结果（含 C_q，用于 κ 计算）
            top_k:              E(q) 大小上限（默认 5）

        返回：
            (admissible_docs, all_scored_pool)
            admissible_docs:  E(q)，κ > 0 的文档，按 dcr_score 降序，最多 top_k 条
            all_scored_pool:  R_D ∪ R_C 中所有文档（含 κ=0 的），供 SLR 计算使用

        数学保证：
            admissible_docs 中所有文档的 κ > 0（Theorem 3.5 的实现）。
            all_scored_pool 包含全部 κ=0 文档，用于 inadmissible_chunk_ids 集合。
        """
        # 第一步：构建排名映射
        # disease_rank_map[chunk_id] = 在 R_D 中的排名（1-indexed）
        disease_rank_map: Dict[str, int] = {
            r.chunk.chunk_id: idx + 1
            for idx, r in enumerate(disease_results)
        }
        # constraint_rank_map[chunk_id] = 在 R_C 中的排名（1-indexed）
        constraint_rank_map: Dict[str, int] = {
            r.chunk.chunk_id: idx + 1
            for idx, r in enumerate(constraint_results)
        }

        # 第二步：合并所有候选 chunk_id（R_D ∪ R_C 去重）
        all_chunk_ids = set(disease_rank_map.keys()) | set(constraint_rank_map.keys())

        # 第三步：为每个候选计算 S_joint（Triple-RRF）
        # 使用 math.inf 作为"未出现"的排名，使该信号贡献为 0
        joint_scores: Dict[str, float] = {}
        for cid in all_chunk_ids:
            d_rank = float(disease_rank_map.get(cid, math.inf))
            c_rank = float(constraint_rank_map.get(cid, math.inf))

            # RRF_D：疾病空间贡献（来自 Stage 1A 的 BM25+Dense RRF 排名）
            rrf_d = self._disease_weight / (d_rank + self.RRF_K)

            # RRF_C：约束空间贡献（来自 Stage 1B 的约束排名）
            rrf_c = self._constraint_weight / (c_rank + self.RRF_K)

            joint_scores[cid] = rrf_d + rrf_c

        # 第四步：构建合并候选池的 RetrievalResult 列表
        # 以 S_joint 作为 score 传入 KappaScorer，KappaScorer 用它计算 dcr_score = score × κ
        chunk_map: Dict[str, "TextChunk"] = {}  # type: ignore[name-defined]
        for r in disease_results:
            chunk_map[r.chunk.chunk_id] = r.chunk
        for r in constraint_results:
            chunk_map[r.chunk.chunk_id] = r.chunk

        pool_results: List[RetrievalResult] = [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=joint_scores[cid],
                retrieval_method="dual_space",
            )
            for cid in all_chunk_ids
        ]

        # 第五步：批量计算 κ（复用 KappaScorer，含批量 π_d 提取和磁盘缓存）
        all_scored: List[RetrievedDoc] = self._kappa_scorer.score_documents(
            retrieval_results=pool_results,
            decomposition=decomposition,
        )
        # all_scored 已按 dcr_score = S_joint × κ 降序排列

        # 第六步：物理排除 κ=0 文档，返回 E(q)（前 top_k 条）
        admissible_docs: List[RetrievedDoc] = [
            d for d in all_scored if d.kappa > 0.0
        ][:top_k]

        return admissible_docs, all_scored
