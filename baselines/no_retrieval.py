#!/usr/bin/env python3
"""
Baseline D4：No Retrieval（直接 LLM 参数记忆基准）

无检索，直接用 LLM 参数记忆回答。

用途：
  1. 测量 P_LLM(a|D) 的参数记忆偏置（Experiment 2 的基准）
  2. 证明 Standard RAG（检索后的 context-memory 冲突）比 No Retrieval 更危险

预期结果：CRR 较高（参数记忆主要针对一般人群，忽视患者特异约束），
          但比 Standard RAG 略低（无 INADMISSIBLE 文献加强偏置的问题）。
"""
from __future__ import annotations

from src.llm_client import LLMClient

from baselines.base import BaseRAGSystem
from src.types import SampleResult


NO_RETRIEVAL_PROMPT = """\
You are a medical advisor. Answer the following medical question based on \
your medical knowledge.

Question: {query}

Provide a concise treatment recommendation (1-2 sentences):"""


class NoRetrievalBaseline(BaseRAGSystem):
    """
    无检索基准：直接 LLM 参数记忆。

    这是 CRR 的上界基准之一（若参数记忆严重偏置向一般人群，CRR 较高）。
    同时也是 AEC-Gain 计算的参照点（AEC_baseline = AEC_noretrieval）。
    """

    def __init__(
        self,
        client: LLMClient,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._client = client
        self._model = model

    @property
    def system_name(self) -> str:
        return "no_retrieval"

    def run(self, query: str, sample_id: str, patient_profile=None) -> SampleResult:
        """
        直接 LLM 推理（无检索步骤）。
        """
        prompt = NO_RETRIEVAL_PROMPT.format(query=query)
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
        )
