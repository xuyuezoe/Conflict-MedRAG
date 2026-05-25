#!/usr/bin/env python3
"""
Stage 1B：约束空间主动检索器（Constraint-Space Retrieval）

数学动机（正交双空间检索理论）：
  Module 0 将 query 正交分解为 D_q ⊕ C_q。
  正交性意味着两个维度应各自独立检索：
    Stage 1A（src/pipeline.py）：从 D_q 维度检索 R_D
    Stage 1B（本模块）        ：从 C_q 维度检索 R_C

  R_C 的作用：
    对于 ABSOLUTE 约束：主动检索"在该约束条件下的替代治疗方案"
    对于 RELATIVE 约束：主动检索"在该约束条件下的剂量调整协议"
  这些文档在疾病空间 R_D 中可能排名靠后（专业人群用语不同），
  但携带了高 κ 值（专为约束患者设计），是精准证据集 E(q) 的关键组成。

与旧版 SCSR 的本质区别：
  SCSR（Stage 3）：被动触发（当 admissible_docs < 阈值时），gap-filling 修补
  Stage 1B（本模块）：主动运行（每次推理均执行），正交完备检索
  理论上 Stage 1B ⊇ SCSR：覆盖 SCSR 的所有场景，同时消除触发条件依赖
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.llm_client import LLMClient
from src.types import PatientConstraint, QueryDecomposition
from src.retriever import HybridRetriever, RetrievalResult


# ── 约束查询生成 Prompt ───────────────────────────────────────────────────────

CONSTRAINT_ABSOLUTE_PROMPT = """\
A patient with the following contraindication needs treatment for a medical condition.
The primary standard treatment is contraindicated.

Condition: {disease}
Contraindication: {constraint_text}
Contraindicated treatment: {inadmissible_action}

Generate a medical search query (1-2 sentences) to find ALTERNATIVE treatments \
that would be safe for this patient. Focus on alternatives within the same condition, \
not the contraindicated drug.

Return only the query text."""

CONSTRAINT_RELATIVE_PROMPT = """\
A patient needs treatment for a medical condition, but requires dose adjustment \
due to organ impairment.

Condition: {disease}
Constraint: {constraint_text} (actual value: {param_value}, safety threshold: {param_threshold})
Treatment requiring adjustment: {restricted_action}

Generate a medical search query (1-2 sentences) to find dosing guidelines \
or dose-adjusted protocols for this drug in patients with this condition.

Return only the query text."""


# ── Stage 1B 主类 ─────────────────────────────────────────────────────────────

class ConstraintRetriever:
    """
    Stage 1B：约束空间主动检索器。

    数学目标：
      从 C_q 维度独立检索 R_C，补全 Stage 1A 的 R_D。
      两路检索结果在 DualSpaceFusion（Stage 2）中通过条件概率乘积融合：
        P(d ∈ E(q)) = P(d|D_q) × P(d safe|C_q) = S_joint(d) × κ(d)

    无论患者约束类型（ABSOLUTE/RELATIVE），均主动生成专属检索 query：
      ABSOLUTE → 寻找 C_q 安全的替代方案
      RELATIVE → 寻找 C_q 条件下的剂量调整依据

    检索结果合并策略：
      多个约束各自检索，按 chunk_id 去重，保留最佳排名（最低排名号）。

    缓存策略：
      约束 query 生成结果以 MD5(disease_query + constraint.raw_text) 为键缓存到磁盘，
      避免同一约束+疾病组合重复调用 LLM。
    """

    def __init__(
        self,
        client: LLMClient,
        model: str = "claude-haiku-4-5-20251001",
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        参数：
            client:    Anthropic API 客户端
            model:     约束 query 生成模型（默认 Haiku，成本低）
            cache_dir: 缓存目录（None 表示不缓存）
        """
        self._client = client
        self._model = model
        self._cache_dir = cache_dir
        if cache_dir:
            (cache_dir / "constraint_queries").mkdir(parents=True, exist_ok=True)

    def retrieve(
        self,
        decomposition: QueryDecomposition,
        retriever: HybridRetriever,
        top_k_per_constraint: int = 10,
    ) -> Tuple[List[RetrievalResult], Dict[str, str]]:
        """
        执行约束空间检索，返回 R_C。

        参数：
            decomposition:          Module 0 分解结果（含 C_q 约束列表和 D_q）
            retriever:              混合检索器实例（与 Stage 1A 共用）
            top_k_per_constraint:   每个约束检索的最大文档数（默认 10）

        返回：
            (R_C, constraint_queries)
            R_C:               约束空间候选文档列表（按约束排名去重合并）
                               检索方法标记为 "constraint"
            constraint_queries: {constraint.target_action → 生成的检索 query}
                               供 MARCOutput 记录（可解释性 + 调试）

        特殊情况：
            若 C_q 为空（无任何约束），直接返回 ([], {})。
            若某约束 query 生成失败，跳过该约束（抛出异常而非兜底）。
        """
        if not decomposition.constraints:
            return [], {}

        # 第一步：为每个约束生成检索 query
        constraint_queries: Dict[str, str] = {}
        constraint_result_lists: List[List[RetrievalResult]] = []

        for constraint in decomposition.constraints:
            query_text = self._get_or_generate_query(decomposition, constraint)
            constraint_queries[constraint.target_action] = query_text

            # 第二步：执行检索
            raw: List[RetrievalResult] = retriever.retrieve(
                query=query_text,
                top_k=top_k_per_constraint,
            )
            # 标记来源为 constraint（区别于 hybrid/bm25/dense）
            for r in raw:
                r.retrieval_method = "constraint"
            constraint_result_lists.append(raw)

        # 第三步：按 chunk_id 去重合并，保留最佳排名（最低排名号）
        merged: Dict[str, RetrievalResult] = {}
        for results in constraint_result_lists:
            for rank, r in enumerate(results, start=1):
                cid = r.chunk.chunk_id
                if cid not in merged:
                    # 第一次遇到该文档，记录排名和结果
                    merged[cid] = r
                    merged[cid]._constraint_rank = rank  # type: ignore[attr-defined]
                else:
                    # 已存在，保留排名更好（更小）的
                    existing_rank = getattr(merged[cid], "_constraint_rank", rank + 1)
                    if rank < existing_rank:
                        merged[cid] = r
                        merged[cid]._constraint_rank = rank  # type: ignore[attr-defined]

        # 按内部排名排序（合并后的统一排序）
        r_c: List[RetrievalResult] = sorted(
            merged.values(),
            key=lambda r: getattr(r, "_constraint_rank", 9999),
        )

        return r_c, constraint_queries

    def _get_or_generate_query(
        self,
        decomposition: QueryDecomposition,
        constraint: PatientConstraint,
    ) -> str:
        """
        获取或生成约束检索 query（带磁盘缓存）。

        缓存键：MD5(disease_query + "|" + constraint.raw_text)
        不同疾病+约束组合使用不同 query，缓存精确匹配。

        参数：
            decomposition: 包含 D_q 的分解结果
            constraint:    单个患者约束

        返回：
            约束检索 query 文本（1-2 句）

        异常：
            anthropic.APIError: API 调用失败（不兜底）
        """
        # 第一步：检查缓存
        cache_key = hashlib.md5(
            (decomposition.disease_query + "|" + constraint.raw_text).encode("utf-8")
        ).hexdigest()

        if self._cache_dir:
            cache_file = self._cache_dir / "constraint_queries" / f"{cache_key}.txt"
            if cache_file.exists():
                return cache_file.read_text(encoding="utf-8").strip()

        # 第二步：LLM 生成 query
        query_text = self._generate_constraint_query(decomposition, constraint)

        # 第三步：写入缓存
        if self._cache_dir:
            cache_file = self._cache_dir / "constraint_queries" / f"{cache_key}.txt"
            cache_file.write_text(query_text, encoding="utf-8")

        return query_text

    def _generate_constraint_query(
        self,
        decomposition: QueryDecomposition,
        constraint: PatientConstraint,
    ) -> str:
        """
        调用 LLM 为单个约束生成约束空间检索 query。

        ABSOLUTE → 替代方案查询（覆盖 E(q) 中的 κ > 0 替代治疗）
        RELATIVE → 剂量调整查询（覆盖 E(q) 中的调整剂量证据）

        参数：
            decomposition: 包含 D_q 和 C_q 的分解结果
            constraint:    需要生成 query 的约束

        返回：
            检索 query 文本

        异常：
            ValueError: 约束类型非法（NONE 约束不应调用此方法）
        """
        if constraint.constraint_type == "ABSOLUTE":
            prompt = CONSTRAINT_ABSOLUTE_PROMPT.format(
                disease=decomposition.disease_query,
                constraint_text=constraint.raw_text,
                inadmissible_action=constraint.target_action,
            )
        elif constraint.constraint_type == "RELATIVE":
            prompt = CONSTRAINT_RELATIVE_PROMPT.format(
                disease=decomposition.disease_query,
                constraint_text=constraint.raw_text,
                restricted_action=constraint.target_action,
                param_value=constraint.parameter_value if constraint.parameter_value is not None else "unknown",
                param_threshold=constraint.parameter_threshold if constraint.parameter_threshold is not None else "unknown",
            )
        else:
            raise ValueError(
                f"[ConstraintRetriever] 约束类型 '{constraint.constraint_type}' 不支持 query 生成。"
                f"仅 ABSOLUTE/RELATIVE 约束需要约束空间检索。"
                f"target_action: {constraint.target_action}"
            )

        return self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
        ).strip()


class _DisabledConstraintRetriever(ConstraintRetriever):
    """
    永不检索的约束空间检索器（消融实验专用）。

    用于 marc_no_scsr 基线：禁用 Stage 1B，仅保留 Stage 1A。
    retrieve() 始终返回空结果，DualSpaceFusion 退化为纯疾病空间 DCR。

    消融目的：
      量化 Stage 1B（主动约束空间检索）的独立贡献。
      marc_no_scsr 与 marc 的差异 = Stage 1B 对 AEC/SDR 的贡献。
    """

    def retrieve(
        self,
        decomposition: QueryDecomposition,
        retriever: HybridRetriever,
        top_k_per_constraint: int = 10,
    ) -> Tuple[List[RetrievalResult], Dict[str, str]]:
        """
        不执行任何检索，始终返回空结果。

        参数（均未使用）：
            decomposition:          分解结果
            retriever:              检索器
            top_k_per_constraint:   每约束检索数

        返回：
            ([], {})：空的约束空间检索结果
        """
        return [], {}
