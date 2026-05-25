#!/usr/bin/env python3
"""
Baseline D2：BM25-Only RAG

纯 BM25 词频检索，无 Dense 向量检索。

测量目标：
  Dense 向量检索对整体性能的贡献。
  BM25 在医学领域（术语精确匹配）具有较强的基线能力，
  但对语义同义词（如 "penicillin allergy" vs "β-lactam hypersensitivity"）处理能力弱。

与 standard_rag 的差异：
  检索方法：method="bm25"（纯 BM25，不使用 Dense）
  其余：完全相同（无 scope filtering，无分解，无 SCOPE BIAS WARNING）
"""
from __future__ import annotations

from typing import Any

from baselines.base import BaseRAGSystem
from src.llm_client import LLMClient
from src.retriever import HybridRetriever
from src.types import SampleResult


BM25_ONLY_PROMPT = """\
You are a medical advisor. Based on the retrieved evidence below, \
provide a treatment recommendation.

Retrieved Evidence:
{context}

Question: {query}

Answer with the most appropriate treatment option (1-2 sentences, cite evidence):"""


class BM25OnlyRAG(BaseRAGSystem):
    """
    BM25-Only RAG 基准。

    使用完整 query + 纯 BM25 检索（无 Dense），
    无 scope filtering，无 SCOPE BIAS WARNING。

    消融目标：
      通过对比 standard_rag（hybrid）与 bm25_only，
      量化 Dense 向量检索对 CRR/SDR 指标的贡献。
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        client: LLMClient,
        model: str = "claude-haiku-4-5-20251001",
        top_k: int = 5,
    ) -> None:
        """
        参数：
            retriever: 混合检索器（实际只调用 BM25 模式）
            client:    Anthropic 客户端
            model:     生成模型
            top_k:     检索文档数量
        """
        self._retriever = retriever
        self._client = client
        self._model = model
        self._top_k = top_k

    @property
    def system_name(self) -> str:
        return "bm25_only"

    def run(self, query: str, sample_id: str, patient_profile=None) -> SampleResult:
        """
        BM25-Only RAG 推理流程。

        第一步：BM25 检索（完整 query，method="bm25"）
        第二步：构造 context（无 scope filtering）
        第三步：生成推荐（无 SCOPE BIAS WARNING）
        第四步：解析 per_action_status（从文本提取）
        """
        # 第一步：纯 BM25 检索
        raw_results = self._retriever.retrieve(
            query=query,
            top_k=self._top_k,
            method="bm25",
        )

        # 第二步：构造 context（无过滤）
        context_parts = []
        context_chunks = []
        for i, r in enumerate(raw_results, start=1):
            context_parts.append(
                f"[Doc {i}] {r.chunk.source_book}\n{r.chunk.text[:600]}"
            )
            context_chunks.append({
                "chunk_id": r.chunk.chunk_id,
                "source_book": r.chunk.source_book,
                "sim_score": round(r.score, 4),
                "text_snippet": r.chunk.text[:300],
            })
        context_text = "\n\n".join(context_parts)

        # 第三步：生成推荐
        prompt = BM25_ONLY_PROMPT.format(context=context_text, query=query)
        predicted_answer = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )

        # 第四步：解析 per_action_status
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
