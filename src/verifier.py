#!/usr/bin/env python3
"""
归因校验器（Attribution Verifier）

目标：
  验证生成答案未引用 κ=0 的文献，计算 SLR（Source Leakage Rate）。

SLR 定义（research.md 评估指标）：
  SLR = #{claims 引用了 κ=0 文献} / #{总 claims}
  理想值 SLR = 0

理论基础：
  DCR 在检索层物理排除 INADMISSIBLE 文献后，Generator 的 context 中不应含 κ=0 文档。
  若 SLR > 0，说明 Generator 通过参数记忆引入了 INADMISSIBLE 证据
  （context-memory 冲突的残留，即 P_LLM 偏置未被 SCOPE BIAS WARNING 完全抑制）。

在 MARC 完整系统中，SLR 理论上应接近 0，
但在 Standard RAG 等基线系统中，SLR 可能较高（无 scope 过滤）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


class AttributionVerifier:
    """
    归因校验器。

    从 Generator 的 attribution 列表中，
    找出引用了 INADMISSIBLE 文献（κ=0 chunk）的 claim，
    计算 SLR 并返回违规列表。

    注意：
      κ=0 的文档 chunk_id 集合由 DCRReranker.get_excluded_docs() 提供。
      Verifier 不需要知道文档内容，只做 ID 集合查找。
    """

    def verify(
        self,
        attribution: List[Dict[str, Any]],
        inadmissible_chunk_ids: Set[str],
    ) -> Tuple[float, List[str]]:
        """
        计算 SLR 并返回违规 claim 列表。

        参数：
            attribution:           Generator 输出的归因列表
                                   格式: [{"claim": "...", "source_chunk_ids": [...]}]
            inadmissible_chunk_ids: DCR 排除的 κ=0 文档 chunk_id 集合

        返回：
            (slr_value, violation_claims)
            slr_value:        SLR ∈ [0, 1]，0 为理想
            violation_claims: 引用了 INADMISSIBLE 文献的 claim 文本列表

        若 attribution 为空，返回 (0.0, [])。
        """
        if not attribution:
            return 0.0, []

        total_claims = len(attribution)
        violation_claims: List[str] = []

        for item in attribution:
            claim_text = item.get("claim", "")
            source_ids = item.get("source_chunk_ids", [])

            if not isinstance(source_ids, list):
                raise ValueError(
                    f"[Verifier] attribution 中 source_chunk_ids 类型非法："
                    f"期望 list，实际 {type(source_ids)}。claim: {claim_text[:80]}"
                )

            # 检查该 claim 是否引用了任何 INADMISSIBLE 文献
            leaked = [sid for sid in source_ids if sid in inadmissible_chunk_ids]
            if leaked:
                violation_claims.append(
                    f"[LEAK] claim='{claim_text[:80]}' 引用了 INADMISSIBLE 文献: {leaked}"
                )

        slr = len(violation_claims) / total_claims
        return slr, violation_claims
