#!/usr/bin/env python3
"""
消融 D7：MARC-noDCR

MARC 的消融变体：保留 scope-anchored 生成（含 SCOPE BIAS WARNING），
但跳过 DCR 重排序（不计算 κ，不物理排除 INADMISSIBLE 文献）。

测量目标：
  DCR 对 CRR 的贡献。
  若 MARC-noDCR 的 CRR 显著高于 MARC，说明 DCR 是关键的检索层保障。
  仅靠 Prompt（SCOPE BIAS WARNING）不足以阻止 INADMISSIBLE 文献影响生成。

预期结果：
  CRR(MARC-noDCR) > CRR(MARC)：SCOPE BIAS WARNING 有效但不充分
  SLR(MARC-noDCR) > SLR(MARC)：无 DCR 时 INADMISSIBLE 文献进入 context
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.llm_client import LLMClient

from baselines.base import BaseRAGSystem
from src.retriever import HybridRetriever
from src.query_decomposer import QueryDecomposer
from src.types import SampleResult


SCOPE_ANCHORED_NO_DCR_PROMPT = """\
You are a clinical decision support system.

Patient information: {patient_info}

Retrieved Evidence (NOTE: some evidence may not apply to this patient):
{context}

SCOPE BIAS WARNING: Based on the patient information above, the following \
treatments may be contraindicated for this patient: {inadmissible_hint}
Do NOT recommend contraindicated treatments even if they appear in evidence.

Question: {query}

Provide a treatment recommendation (1-2 sentences):"""


class MARCNoDCR(BaseRAGSystem):
    """
    MARC 消融变体：无 DCR。

    保留：Query 分解（D_q/C_q）+ scope-anchored 生成 prompt
    去除：κ 计算 + 乘法评分 + INADMISSIBLE 文档物理排除

    关键：INADMISSIBLE 文档仍在 context 中，SCOPE BIAS WARNING 是唯一保护。
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        decomposer: QueryDecomposer,
        client: LLMClient,
        model: str = "claude-haiku-4-5-20251001",
        top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._decomposer = decomposer
        self._client = client
        self._model = model
        self._top_k = top_k

    @property
    def system_name(self) -> str:
        return "marc_no_dcr"

    def run(self, query: str, sample_id: str, patient_profile=None) -> SampleResult:
        """
        MARC-noDCR 推理流程。

        第一步：Query 分解（保留 Module 0）
        第二步：Stage 1 检索（使用 D_q，与 MARC 相同）
        第三步：跳过 DCR（不计算 κ，不排除文档）
        第四步：scope-anchored 生成（含 SCOPE BIAS WARNING）
        """
        # 第一步：Query 分解
        decomposition = self._decomposer.decompose(query)

        # 第二步：用 D_q 检索（与 MARC Stage 1 相同）
        raw_results = self._retriever.retrieve(
            query=decomposition.disease_query,
            top_k=self._top_k,
        )

        # 第三步：无 κ 过滤，全量文档进入 context
        context_parts = []
        context_chunks = []
        for i, r in enumerate(raw_results, start=1):
            context_parts.append(f"[Doc {i}] {r.chunk.source_book}\n{r.chunk.text[:600]}")
            context_chunks.append({
                "chunk_id": r.chunk.chunk_id,
                "source_book": r.chunk.source_book,
                "sim_score": round(r.score, 4),
                "text_snippet": r.chunk.text[:300],
            })
        context_text = "\n\n".join(context_parts)

        # 第四步：scope-anchored 生成（只有 SCOPE BIAS WARNING，无物理排除）
        inadmissible_hint = ", ".join(decomposition.absolute_target_actions) or "none identified"
        patient_info = decomposition.original_query[:200]

        prompt = SCOPE_ANCHORED_NO_DCR_PROMPT.format(
            patient_info=patient_info,
            context=context_text,
            inadmissible_hint=inadmissible_hint,
            query=query,
        )
        predicted_answer = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )

        options_text = query.split("\n")[-1] if "\n" in query else query
        per_action_status = self.extract_action_status(
            predicted_answer=predicted_answer,
            options_text=options_text,
            client=self._client,
        )

        return SampleResult(
            sample_id=sample_id,
            system_name=self.system_name,
            predicted_answer=predicted_answer,
            per_action_status_pred=per_action_status,
            scsr_triggered=False,
            srl_violations=[],
            marc_output=None,
            raw_response=predicted_answer,
            context_chunks=context_chunks,
        )
