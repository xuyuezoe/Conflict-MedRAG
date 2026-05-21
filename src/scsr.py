#!/usr/bin/env python3
"""
Stage 3：SCSR（Scope-Constrained Supplementary Retrieval）

理论定位（research.md §3.4）：
  SCSR 不是 SC 的主要处理机制——DCR 已在检索层通过 κ=0 处理绝对禁忌。
  SCSR 的作用是 A(q) gap-filling：
    当 DCR 排除 INADMISSIBLE 文献后，若 A(q) 内替代证据不足，
    主动构建新的检索 query，补充查找 A(q) 内的替代方案证据。

触发条件：
  len(admissible_docs) < MIN_ADMISSIBLE_DOCS（默认 2）

两种子类型：
  scsr_absolute: 主 action 绝对禁忌 → 寻找替代方案（domain shift）
  scsr_relative: 主 action 相对禁忌 → 寻找参数调整证据（precision retrieval）
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from src.llm_client import LLMClient

from src.types import PatientConstraint, QueryDecomposition, RetrievedDoc
from src.kappa_scorer import KappaScorer
from src.retriever import HybridRetriever, RetrievalResult


# ── SCSR Query 生成 Prompt ───────────────────────────────────────────────────

SCSR_ABSOLUTE_PROMPT = """\
A patient with the following contraindication needs treatment for a medical condition.
The primary standard treatment is contraindicated.

Condition: {disease}
Contraindication: {constraint_text}
Contraindicated treatment: {inadmissible_action}

Generate a medical search query (1-2 sentences) to find ALTERNATIVE treatments \
that would be safe for this patient. Focus on alternatives within the same condition, \
not the contraindicated drug.

Return only the query text."""

SCSR_RELATIVE_PROMPT = """\
A patient needs treatment for a medical condition, but requires dose adjustment \
due to organ impairment.

Condition: {disease}
Constraint: {constraint_text} (actual value: {param_value}, safety threshold: {param_threshold})
Treatment requiring adjustment: {restricted_action}

Generate a medical search query (1-2 sentences) to find dosing guidelines \
or dose-adjusted protocols for this drug in patients with this condition.

Return only the query text."""


class SCSRRetriever:
    """
    Stage 3：SCSR 补充检索器。

    核心逻辑：
      1. 判断是否触发（admissible_docs 数量是否低于阈值）
      2. 生成 SCSR 检索 query（LLM，按 absolute/relative 子类型）
      3. 执行补充检索
      4. 对补充文档再次进行 κ 验证（确保补充结果也在 A(q) 内）

    注意：补充检索的文档同样需要通过 κ > 0 验证，
    以确保 SCSR 不会意外引入新的 INADMISSIBLE 文档。
    """

    def __init__(
        self,
        client: LLMClient,
        kappa_scorer: KappaScorer,
        model: str = "claude-haiku-4-5-20251001",
        min_admissible_docs: int = 2,
    ) -> None:
        """
        参数：
            client:             Anthropic 客户端
            kappa_scorer:       κ 计算器（用于验证补充文档）
            model:              SCSR query 生成模型
            min_admissible_docs: 触发 SCSR 的最小 admissible 文档数阈值
        """
        self._client = client
        self._kappa_scorer = kappa_scorer
        self._model = model
        self._min_admissible = min_admissible_docs

    def should_trigger(self, stage2_docs: List[RetrievedDoc]) -> bool:
        """
        判断是否需要触发 SCSR。

        参数：
            stage2_docs: Stage 2 DCR 的输出（已过滤 κ>0）

        返回：
            True 若 admissible 文档数 < 阈值
        """
        return len(stage2_docs) < self._min_admissible

    def retrieve(
        self,
        decomposition: QueryDecomposition,
        stage2_docs: List[RetrievedDoc],
        retriever: HybridRetriever,
        top_k: int = 5,
    ) -> Tuple[List[RetrievedDoc], str]:
        """
        执行 SCSR 补充检索。

        参数：
            decomposition: Module 0 分解结果
            stage2_docs:   Stage 2 输出（admissible 文档，用于判断触发和类型）
            retriever:     混合检索器
            top_k:         补充检索返回数量上限

        返回：
            (supplementary_docs, scsr_query)
            supplementary_docs: κ>0 的补充文档列表
            scsr_query:         生成的 SCSR 检索查询文本
        """
        # 第一步：确定 SCSR 子类型（absolute 还是 relative）
        scsr_type = self._determine_type(decomposition)

        # 第二步：生成 SCSR 检索 query
        scsr_query = self._generate_query(decomposition, scsr_type)

        # 第三步：执行检索
        raw_results: List[RetrievalResult] = retriever.retrieve(
            query=scsr_query,
            top_k=top_k * 2,    # 扩大召回，因为 κ 验证会过滤一部分
        )

        # 第四步：κ 验证（确保补充结果在 A(q) 内）
        verified_docs: List[RetrievedDoc] = self._kappa_scorer.score_documents(
            retrieval_results=raw_results,
            decomposition=decomposition,
        )
        admissible = [d for d in verified_docs if d.kappa > 0.0][:top_k]

        return admissible, scsr_query

    def _determine_type(
        self,
        decomposition: QueryDecomposition,
    ) -> Literal["absolute", "relative"]:
        """
        根据患者约束类型确定 SCSR 子类型。

        有绝对禁忌 → absolute（寻找替代方案）
        只有相对禁忌 → relative（寻找参数调整证据）
        """
        if decomposition.has_absolute_constraint:
            return "absolute"
        return "relative"

    def _generate_query(
        self,
        decomposition: QueryDecomposition,
        scsr_type: Literal["absolute", "relative"],
    ) -> str:
        """
        调用 LLM 生成 SCSR 检索 query。

        参数：
            decomposition: 包含 D_q 和 C_q 的分解结果
            scsr_type:     absolute 或 relative

        返回：
            SCSR 检索 query 文本（1-2 句）
        """
        if scsr_type == "absolute":
            # 找到绝对禁忌约束
            abs_constraints = [
                c for c in decomposition.constraints
                if c.constraint_type == "ABSOLUTE"
            ]
            if not abs_constraints:
                raise ValueError(
                    f"[SCSRRetriever] scsr_type=absolute 但 constraints 中没有 ABSOLUTE 约束。"
                    f"decomposition: {decomposition.original_query[:100]}"
                )
            constraint = abs_constraints[0]
            prompt = SCSR_ABSOLUTE_PROMPT.format(
                disease=decomposition.disease_query,
                constraint_text=constraint.raw_text,
                inadmissible_action=constraint.target_action,
            )
        else:
            # 找到相对禁忌约束
            rel_constraints = [
                c for c in decomposition.constraints
                if c.constraint_type == "RELATIVE"
            ]
            if not rel_constraints:
                # 无任何约束时，直接用疾病查询作为补充
                return decomposition.disease_query
            constraint = rel_constraints[0]
            prompt = SCSR_RELATIVE_PROMPT.format(
                disease=decomposition.disease_query,
                constraint_text=constraint.raw_text,
                restricted_action=constraint.target_action,
                param_value=constraint.parameter_value or "unknown",
                param_threshold=constraint.parameter_threshold or "unknown",
            )

        return self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
        )
