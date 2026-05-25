#!/usr/bin/env python3
"""
MARC Pipeline 的 BaseRAGSystem 接口包装器

将 MARCPipeline.run() 的输出转换为 SampleResult 格式，
使 MARC 可以与 Baseline 系统使用相同的评估接口。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from baselines.base import BaseRAGSystem
from src.pipeline import MARCPipeline
from src.types import SampleResult


class MARCSystemWrapper(BaseRAGSystem):
    """
    将 MARCPipeline 包装为 BaseRAGSystem 接口。

    MARCPipeline.run() 返回 MARCOutput（含完整中间结果），
    包装器将其转换为 SampleResult（评估框架要求的格式）。
    """

    def __init__(self, pipeline: MARCPipeline) -> None:
        self._pipeline = pipeline

    @property
    def system_name(self) -> str:
        return "marc"

    def run(
        self,
        query: str,
        sample_id: str,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> SampleResult:
        """
        调用 MARCPipeline.run() 并封装输出。
        """
        marc_output = self._pipeline.run(query=query, patient_profile=patient_profile)

        return SampleResult(
            sample_id=sample_id,
            system_name=self.system_name,
            predicted_answer=marc_output.generated_answer,
            per_action_status_pred=marc_output.per_action_status,
            # scsr_triggered：Stage 1B 是否对 E(q) 产生净贡献（新语义，向后兼容字段）
            scsr_triggered=marc_output.scsr_triggered,
            srl_violations=marc_output.srl_violations,
            marc_output=marc_output,
            raw_response=marc_output.generated_answer,
        )
