#!/usr/bin/env python3
"""
Baseline D3：Dense-Only RAG

纯 Dense 向量检索，无 BM25。

测量目标：
  BM25 词频检索对整体性能的贡献。
  Dense 检索擅长语义泛化（如跨同义词、跨表述），
  但可能在精确术语匹配（如剂量数值、药品名拼写）上弱于 BM25。

与 standard_rag 的差异：
  检索方法：method="dense"（纯 Dense，不使用 BM25）
  其余：完全相同（无 scope filtering，无分解，无 SCOPE BIAS WARNING）

注意：
  Dense 检索需要 FAISS 索引存在（data/index/dense/faiss.index）。
  若 Dense 索引未构建，请先运行：
    python3 scripts/index_textbooks.py  # 不加 --skip-dense
"""
from __future__ import annotations

from baselines.base import BaseRAGSystem
from src.llm_client import LLMClient
from src.retriever import HybridRetriever
from src.types import SampleResult


DENSE_ONLY_PROMPT = """\
You are a medical advisor. Based on the retrieved evidence below, \
provide a treatment recommendation.

Retrieved Evidence:
{context}

Question: {query}

Answer with the most appropriate treatment option (1-2 sentences, cite evidence):"""


class DenseOnlyRAG(BaseRAGSystem):
    """
    Dense-Only RAG 基准。

    使用完整 query + 纯 Dense 向量检索（无 BM25），
    无 scope filtering，无 SCOPE BIAS WARNING。

    消融目标：
      通过对比 standard_rag（hybrid）与 dense_only，
      量化 BM25 对 CRR/SDR 指标的贡献。

    前提条件：
      FAISS 索引必须已构建（data/index/dense/faiss.index）。
      若仅构建了 BM25 索引，此 baseline 不可运行。
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
            retriever: 混合检索器（实际只调用 Dense 模式）
            client:    Anthropic 客户端
            model:     生成模型
            top_k:     检索文档数量

        异常：
            ValueError: Dense 索引未加载（在 retrieve 时抛出）
        """
        self._retriever = retriever
        self._client = client
        self._model = model
        self._top_k = top_k

    @property
    def system_name(self) -> str:
        return "dense_only"

    def run(self, query: str, sample_id: str, patient_profile=None) -> SampleResult:
        """
        Dense-Only RAG 推理流程。

        第一步：Dense 向量检索（完整 query，method="dense"）
        第二步：构造 context（无 scope filtering）
        第三步：生成推荐（无 SCOPE BIAS WARNING）
        第四步：解析 per_action_status（从文本提取）

        异常：
            ValueError: Dense 索引未加载（传播自 retriever.retrieve）
        """
        # 第一步：纯 Dense 向量检索（若索引未加载则抛出 ValueError）
        raw_results = self._retriever.retrieve(
            query=query,
            top_k=self._top_k,
            method="dense",
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
        prompt = DENSE_ONLY_PROMPT.format(context=context_text, query=query)
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
