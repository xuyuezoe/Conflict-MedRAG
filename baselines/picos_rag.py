#!/usr/bin/env python3
"""
Baseline D6：PICOs-RAG 风格（Query 改写后检索）

参考：PICOs-RAG [arXiv 2510.23998]
将 query 改写为 PICO 结构（Patient/Intervention/Comparison/Outcome），
再执行标准检索。

对比目标：
  MARC 的 DCR 与 PICOs-RAG 的核心区别（research.md §2.1）：
  - PICOs-RAG：修改检索输入 query，检索评分函数仍为 sim(q', d)
    → scope 信息被编码进 query 向量，在高维空间中被疾病语义淹没
  - MARC DCR：乘法结构 sim(D_q, d) × κ(C_q, π_d)
    → κ=0 时，无论 sim 多高，score=0

预期结果：PICOs-RAG 的 CRR 介于 Standard RAG 和 MARC 之间，
          但显著高于 MARC（因为 sim 相似度空间仍受疾病语义主导）。
"""
from __future__ import annotations

from src.llm_client import LLMClient

from baselines.base import BaseRAGSystem
from src.retriever import HybridRetriever
from src.types import SampleResult


PICO_REWRITE_PROMPT = """\
Rewrite this medical question into a structured PICO format for retrieval:
- P (Patient): patient population and key characteristics
- I (Intervention): the treatment or intervention being considered
- C (Comparison): alternative treatments
- O (Outcome): desired outcomes

Question: {query}

Return a single search query string (2-3 sentences) that captures the PICO structure. \
Include patient constraints (allergies, lab values) in the P component. \
Return only the query text."""

PICO_GENERATION_PROMPT = """\
You are a medical advisor. Based on the retrieved evidence below, \
provide a treatment recommendation for the patient described in the question.

Retrieved Evidence:
{context}

Patient Question: {query}

Answer (1-2 sentences, cite evidence):"""


class PICOsRAG(BaseRAGSystem):
    """
    PICOs-RAG 风格基准。

    区别于 Standard RAG：使用 PICO 结构化 query 改写后检索。
    区别于 MARC：改写后仍使用原始相似度评分，无 κ 乘法结构。
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        client: LLMClient,
        model: str = "claude-haiku-4-5-20251001",
        top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._client = client
        self._model = model
        self._top_k = top_k

    @property
    def system_name(self) -> str:
        return "picos_rag"

    def run(self, query: str, sample_id: str, patient_profile=None) -> SampleResult:
        """
        PICOs-RAG 推理流程。

        第一步：PICO 结构化 query 改写
        第二步：使用改写后的 query 检索（原始相似度评分，无 κ）
        第三步：标准 RAG 生成（无 scope-anchored prompt）
        """
        # 第一步：PICO 改写
        rewrite_prompt = PICO_REWRITE_PROMPT.format(query=query)
        pico_query = self._client.chat(
            messages=[{"role": "user", "content": rewrite_prompt}],
            max_tokens=2000,
        )

        # 第二步：检索（使用 PICO query，原始相似度评分）
        raw_results = self._retriever.retrieve(query=pico_query, top_k=self._top_k)

        # 第三步：构造 context 并生成
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

        gen_prompt = PICO_GENERATION_PROMPT.format(context=context_text, query=query)
        predicted_answer = self._client.chat(
            messages=[{"role": "user", "content": gen_prompt}],
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
