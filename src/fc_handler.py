#!/usr/bin/env python3
"""
FC Handler：事实性冲突（Factual Conflict）仲裁器

数学位置（research.md §1.2）：
  FC 操作在支撑集 A(q) 已确定后的值域层进行。
  本模块只处理 κ>0 的文档，不重新引入 INADMISSIBLE 文献。
  FC 是对"在 A(q) 内，多个证据对同一 action 的效果估计是否矛盾"的处理。

对 MedQA 基准的说明：
  MedQA 题目大多数是 FC（答案在 A(q) 内的最佳选择）。
  FC Handler 对 MedQA 的主要作用：
    1. 确保 MARC 在 FC 场景下不退化（FC-AA 指标测量）
    2. 证明 MARC 的两层架构（SC 层 + FC 层）在实际系统中可以共存

FC 检测策略：
  轻量：基于同 action 关键词的文档对聚类 + LLM 判断矛盾关系
  重量限制：最多检测前 N 对（默认 N=5），避免过多 API 调用
"""
from __future__ import annotations

import itertools
import re
from typing import List, Optional, Tuple

from src.llm_client import LLMClient

from src.types import FCConflict, QueryDecomposition, RetrievedDoc


# ── FC 检测 Prompt ────────────────────────────────────────────────────────────

FC_DETECTION_PROMPT = """\
You are a medical evidence analyst. Determine if two medical text passages \
contain a factual conflict (contradictory claims about the same treatment).

Passage A (chunk_id: {id_a}):
{text_a}

Passage B (chunk_id: {id_b}):
{text_b}

Is there a factual conflict between these two passages about the same medical action?

Return JSON only:
{{
  "has_conflict": true or false,
  "conflict_type": "contradict" or "update" or "population_diff" or "none",
  "claim_a": "<key claim from passage A>",
  "claim_b": "<key claim from passage B>",
  "resolution": "prefer_a" or "prefer_b" or "uncertain",
  "resolution_reason": "<why you prefer one over the other, or why uncertain>"
}}

conflict_type definitions:
- "contradict": passages directly contradict each other on same claim
- "update": passage B is a more recent update superseding passage A
- "population_diff": passages apply to different populations (not a real conflict)
- "none": no conflict
"""


class FCHandler:
    """
    FC 值域冲突仲裁器。

    接收 Stage 2 + Stage 3 输出的 admissible 文档（全部 κ>0），
    检测其中是否存在事实性冲突（同 action，相反效果主张），
    返回冲突列表和仲裁后的精简文档列表。

    复杂度控制：
      - 最多检测 max_pairs 对文档（默认 5）
      - 每对只调用一次 LLM
      - 无冲突时直接返回原始文档（无 API 成本）
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        model: str = "claude-haiku-4-5-20251001",
        max_pairs: int = 5,
    ) -> None:
        """
        参数：
            client:    Anthropic 客户端（None 表示跳过 LLM 检测，仅做关键词粗筛）
            model:     FC 检测模型
            max_pairs: 最大检测文档对数（防止 API 调用过多）
        """
        self._client = client
        self._model = model
        self._max_pairs = max_pairs

    def detect_and_resolve(
        self,
        docs: List[RetrievedDoc],
        decomposition: QueryDecomposition,
    ) -> Tuple[List[FCConflict], List[RetrievedDoc]]:
        """
        检测 FC 冲突并返回仲裁结果和精简后的文档列表。

        参数：
            docs:          admissible 文档列表（全部 κ>0）
            decomposition: Module 0 的分解结果（用于确定 action 关键词）

        返回：
            (conflicts, resolved_docs)
            conflicts:     检测到的所有 FCConflict
            resolved_docs: 仲裁后保留的文档（冲突对中较弱的一方被移除）

        若无 LLM 客户端，直接返回 ([], docs)（跳过检测）。
        """
        if self._client is None or len(docs) < 2:
            return [], docs

        # 第一步：候选对筛选（同 action 关键词的文档对，减少 LLM 调用次数）
        candidate_pairs = self._find_candidate_pairs(docs)
        if not candidate_pairs:
            return [], docs

        # 第二步：LLM 检测每对
        conflicts: List[FCConflict] = []
        excluded_ids = set()

        for doc_a, doc_b in candidate_pairs[:self._max_pairs]:
            conflict = self._detect_pair(doc_a, doc_b)
            if conflict is not None:
                conflicts.append(conflict)
                # 仲裁：prefer_a → 排除 b；prefer_b → 排除 a；uncertain → 保留两者
                if conflict.resolution == "prefer_a":
                    excluded_ids.add(doc_b.chunk.chunk_id)
                elif conflict.resolution == "prefer_b":
                    excluded_ids.add(doc_a.chunk.chunk_id)

        # 第三步：过滤被排除的文档
        resolved_docs = [d for d in docs if d.chunk.chunk_id not in excluded_ids]

        return conflicts, resolved_docs

    def _find_candidate_pairs(
        self,
        docs: List[RetrievedDoc],
    ) -> List[Tuple[RetrievedDoc, RetrievedDoc]]:
        """
        找出可能存在 FC 冲突的文档对（同 action 关键词粗筛）。

        策略：提取每篇文档中的主要药物/治疗关键词，若两篇文档共享关键词则为候选对。
        """
        # 提取关键词（医学药物名：小写，去停用词，取 3-4 字母以上单词）
        def extract_keywords(text: str) -> set:
            words = re.findall(r"\b[a-z]{4,}\b", text.lower())
            stopwords = {"with", "that", "this", "from", "have", "been", "were", "also",
                         "other", "more", "most", "used", "when", "than", "then", "they"}
            return set(words) - stopwords

        doc_keywords = [(d, extract_keywords(d.chunk.text)) for d in docs]
        pairs = []
        for (d_a, kw_a), (d_b, kw_b) in itertools.combinations(doc_keywords, 2):
            if kw_a & kw_b:    # 共享关键词
                pairs.append((d_a, d_b))
        return pairs

    def _detect_pair(
        self,
        doc_a: RetrievedDoc,
        doc_b: RetrievedDoc,
    ) -> Optional[FCConflict]:
        """
        对单对文档调用 LLM 判断 FC 冲突。

        返回：
            FCConflict（若有冲突）或 None（无冲突）
        """
        prompt = FC_DETECTION_PROMPT.format(
            id_a=doc_a.chunk.chunk_id,
            text_a=doc_a.chunk.text[:800],
            id_b=doc_b.chunk.chunk_id,
            text_b=doc_b.chunk.text[:800],
        )
        raw = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32000,
        )

        # 解析 JSON（失败则视为无冲突，记录警告）
        import json
        try:
            json_str = raw
            if "```" in raw:
                start = raw.index("```") + 3
                if raw[start:start+4] == "json":
                    start += 4
                end = raw.rindex("```")
                json_str = raw[start:end].strip()
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # FC 检测失败不影响系统运行，记录日志但不抛出
            import sys
            print(
                f"[FCHandler] JSON 解析失败，跳过此对："
                f"{doc_a.chunk.chunk_id} vs {doc_b.chunk.chunk_id}",
                file=sys.stderr,
            )
            return None

        if not data.get("has_conflict", False):
            return None

        return FCConflict(
            action=data.get("claim_a", "")[:50],
            doc_a_id=doc_a.chunk.chunk_id,
            doc_a_claim=data.get("claim_a", ""),
            doc_b_id=doc_b.chunk.chunk_id,
            doc_b_claim=data.get("claim_b", ""),
            conflict_type=data.get("conflict_type", "contradict"),  # type: ignore[arg-type]
            resolution=data.get("resolution", "uncertain"),          # type: ignore[arg-type]
            resolution_reason=data.get("resolution_reason", ""),
        )
