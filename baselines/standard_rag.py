#!/usr/bin/env python3
"""
Baseline D1：Standard RAG

标准检索增强生成（无 scope filtering）。

检索：混合检索（BM25 + Dense），使用完整 query（含患者约束）
生成：直接将检索文档作为 context，无 scope-aware prompt
对比目标：展示没有 DCR 时，INADMISSIBLE 文献进入 context 导致 CRR 升高的现象。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from baselines.base import BaseRAGSystem
from src.llm_client import LLMClient
from src.retriever import HybridRetriever
from src.types import MARCOutput, SampleResult


STANDARD_RAG_PROMPT = """\
You are a medical advisor. Based on the retrieved evidence below, \
provide a treatment recommendation.

Retrieved Evidence:
{context}

Question: {query}

Answer with the most appropriate treatment option (1-2 sentences, cite evidence):"""


class StandardRAG(BaseRAGSystem):
    """
    标准 RAG 基准。

    使用完整 query 检索（不分解 D_q/C_q），
    无 scope filtering，无 SCOPE BIAS WARNING。

    对比 MARC 时的主要差异：
      - 检索：完整 query vs. D_q（无分解）
      - 排序：原始相似度 vs. DCR 乘法分
      - 生成：标准 context prompt vs. scope-anchored prompt
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
        return "standard_rag"

    def run(self, query: str, sample_id: str) -> SampleResult:
        """
        标准 RAG 推理流程。

        第一步：混合检索（完整 query，含患者约束）
        第二步：直接 concat context，无 scope filtering
        第三步：生成（无 SCOPE BIAS WARNING）
        """
        # 第一步：检索（完整 query，不分解）
        raw_results = self._retriever.retrieve(query=query, top_k=self._top_k)

        # 第二步：构造 context
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

        # 第三步：生成推荐
        prompt = STANDARD_RAG_PROMPT.format(context=context_text, query=query)
        predicted_answer = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )

        # 第四步：解析 per_action_status（从文本提取）
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
