#!/usr/bin/env python3
"""
Stage 2：DCR（Decomposed Conditional Retrieval）重排序器

数学公式（research.md §3.5）：
  score_MARC(q, d) = sim(D_q, d) · κ(C_q, π_d)

  其中：
    sim(D_q, d)    = Stage 1 的疾病相关性分数（已计算）
    κ(C_q, π_d)   = KappaScorer 计算的适用范围相容性

不可补偿性保证（Theorem 3.5 直接体现）：
  κ=0 时，score = 0 × sim = 0，无论 sim 多高。
  INADMISSIBLE_ABS 文档得分永远为 0，物理排除于检索 top-K 之外。
  这是乘法结构相对于任何加法 scope penalty 的关键区别。
"""
from __future__ import annotations

from typing import List

from src.types import QueryDecomposition, RetrievedDoc
from src.kappa_scorer import KappaScorer
from src.retriever import RetrievalResult


class DCRReranker:
    """
    Stage 2：DCR 重排序器。

    将 Stage 1 的检索结果按 score = sim × κ 重排，
    κ=0 的文档物理排除（不进入下游 context）。

    设计要点：
      1. 调用 KappaScorer.score_documents() 批量计算 κ
      2. 按 dcr_score 降序排列
      3. 物理排除 κ=0 文档（不降权，不保留，直接过滤）
      4. 返回前 top_k_out 条（κ > 0 的文档）
    """

    def __init__(self, kappa_scorer: KappaScorer) -> None:
        """
        参数：
            kappa_scorer: 已初始化的 KappaScorer 实例
        """
        self._kappa_scorer = kappa_scorer

    def rerank(
        self,
        stage1_results: List[RetrievalResult],
        decomposition: QueryDecomposition,
        top_k_out: int = 5,
    ) -> List[RetrievedDoc]:
        """
        对 Stage 1 检索结果执行 DCR 重排序。

        参数：
            stage1_results: Stage 1 的全部检索结果（含 sim_score）
            decomposition:  Module 0 分解结果（含 D_q 和 C_q）
            top_k_out:      最终保留的文档数（κ > 0 的文档中，取前 top_k_out 条）

        返回：
            RetrievedDoc 列表（按 dcr_score 降序，仅包含 κ > 0 的文档）
            可能少于 top_k_out 条（若大量文档被 κ=0 排除）

        数学保证：
            任何 κ=0 的文档不出现在返回列表中——这是 Theorem 3.5 的直接实现。
        """
        # 第一步：批量计算 κ，生成含 dcr_score 的 RetrievedDoc
        all_docs: List[RetrievedDoc] = self._kappa_scorer.score_documents(
            retrieval_results=stage1_results,
            decomposition=decomposition,
        )

        # 第二步：物理排除 κ=0 的文档（不降权，直接过滤）
        admissible_docs = [d for d in all_docs if d.kappa > 0.0]

        # 第三步：取前 top_k_out 条（已按 dcr_score 降序，score_documents 保证顺序）
        return admissible_docs[:top_k_out]

    def get_excluded_docs(
        self,
        stage1_results: List[RetrievalResult],
        decomposition: QueryDecomposition,
    ) -> List[RetrievedDoc]:
        """
        获取被 κ=0 排除的文档列表（用于 SLR 验证和调试）。

        返回所有 κ=0 的文档（INADMISSIBLE_ABS）。
        """
        all_docs = self._kappa_scorer.score_documents(
            retrieval_results=stage1_results,
            decomposition=decomposition,
        )
        return [d for d in all_docs if d.kappa == 0.0]
