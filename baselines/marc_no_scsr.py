#!/usr/bin/env python3
"""
消融 D8：MARC-noCSR（保留 marc_no_scsr 名称以维持论文实验一致性）

MARC 的消融变体：保留完整双空间融合 + DCR 评分 + scope-anchored 生成，
但禁用 Stage 1B（约束空间主动检索，Constraint-Space Retrieval）。

测量目标：
  Stage 1B 对 AEC（替代方案覆盖率）和 SDR（约束检测召回率）的贡献。
  若 MARC 的 AEC 显著高于 MARC-noCSR，说明主动约束空间检索有效填补了
  疾病空间检索遗漏的替代方案证据。

预期结果：
  AEC(MARC) > AEC(MARC-noCSR)：Stage 1B 主动覆盖替代方案
  SDR(MARC) > SDR(MARC-noCSR)：Stage 1B 补全约束检测召回
  CRR(MARC) ≈ CRR(MARC-noCSR)：κ=0 排除由 DualSpaceFusion 保证（不受 Stage 1B 影响）

实现方式：
  注入 _DisabledConstraintRetriever（继承 ConstraintRetriever，retrieve() 始终返回空集）。
  DualSpaceFusion 仅接收 R_D（Stage 1A），退化为等价 DCRReranker 的行为。
  通过工厂函数 build_marc_no_scsr_pipeline() 构建管道，与 build_marc_pipeline() 对称。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines.base import BaseRAGSystem
from src.constraint_retriever import _DisabledConstraintRetriever
from src.types import SampleResult, MARCOutput


def build_marc_no_scsr_pipeline(
    index_dir: str = "data/index",
    cache_dir: str = "data/cache",
    stage1_top_k: int = 20,
    stage2_top_k: int = 5,
) -> Any:
    """
    工厂函数：构建 MARC-noCSR 管道（Stage 1B 禁用）。

    与 build_marc_pipeline() 完全相同，唯一差异是
    将 ConstraintRetriever 替换为 _DisabledConstraintRetriever。
    DualSpaceFusion 接收空 R_C，退化为纯疾病空间 DCR。

    所有 LLM 配置从 src/llm_client 读取（来源为 .env 文件）。

    参数：
        index_dir:    教材索引目录
        cache_dir:    缓存目录（π_d 缓存、query 分解缓存）
        stage1_top_k: Stage 1A 检索数量（默认 20）
        stage2_top_k: E(q) 大小上限（默认 5）

    返回：
        完整初始化的 MARCPipeline 实例（Stage 1B 已禁用）
    """
    from src.llm_client import get_client, get_model, get_embedding_model
    from src.retriever import HybridRetriever
    from src.query_decomposer import QueryDecomposer
    from src.kappa_scorer import KappaScorer
    from src.dual_space_fusion import DualSpaceFusion
    from src.fc_handler import FCHandler
    from src.generator import ScopeAnchoredGenerator
    from src.verifier import AttributionVerifier
    from src.pipeline import MARCPipeline

    client = get_client()
    model = get_model()
    embedding_model = get_embedding_model()

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever(
        index_dir=Path(index_dir),
        embedding_model=embedding_model,
    )
    decomposer = QueryDecomposer(
        client=client,
        model=model,
        cache_dir=cache_path,
    )
    kappa_scorer = KappaScorer(
        client=client,
        model=model,
        cache_dir=cache_path,
    )

    # 关键差异：注入永不检索的约束空间检索器（消融实验）
    # retrieve() 始终返回 ([], {})，DualSpaceFusion 退化为纯疾病空间 DCR
    constraint_retriever = _DisabledConstraintRetriever(
        client=client,
        model=model,
        cache_dir=None,  # 不缓存（不会生成任何 query）
    )

    dual_space_fusion = DualSpaceFusion(
        kappa_scorer=kappa_scorer,
        disease_weight=0.7,
        constraint_weight=0.3,
    )
    fc_handler = FCHandler(
        client=client,
        model=model,
    )
    generator = ScopeAnchoredGenerator(
        client=client,
        model=model,
    )
    verifier = AttributionVerifier()

    return MARCPipeline(
        retriever=retriever,
        decomposer=decomposer,
        kappa_scorer=kappa_scorer,
        constraint_retriever=constraint_retriever,
        dual_space_fusion=dual_space_fusion,
        fc_handler=fc_handler,
        generator=generator,
        verifier=verifier,
        stage1_top_k=stage1_top_k,
        stage1b_top_k=10,    # 实际不生效（_DisabledConstraintRetriever 忽略此参数）
        stage2_top_k=stage2_top_k,
        enable_fc=False,
    )


class MARCNoSCSR(BaseRAGSystem):
    """
    MARC-noCSR：完整 MARC 管道，Stage 1B（约束空间检索）永不执行。

    保留：Module 0（Query 分解）+ Stage 1A（疾病空间检索）
          Stage 2（DualSpaceFusion，但仅含 R_D，无 R_C）
          FC Handler + Scope-Anchored Generator + Verifier

    去除：Stage 1B（ConstraintRetriever，主动约束空间检索）

    与 MARC 的差异 = Stage 1B 的完整贡献（AEC/SDR 的约束覆盖增量）。

    使用方法：
        pipeline = build_marc_no_scsr_pipeline(...)
        system = MARCNoSCSR(pipeline)
    """

    def __init__(self, pipeline: Any) -> None:
        """
        参数：
            pipeline: build_marc_no_scsr_pipeline() 构建的 MARCPipeline 实例
        """
        self._pipeline = pipeline

    @property
    def system_name(self) -> str:
        return "marc_no_scsr"

    def run(self, query: str, sample_id: str, patient_profile=None) -> SampleResult:
        """
        MARC-noCSR 推理流程。

        完全委托给内部 MARCPipeline.run()，
        由于 _DisabledConstraintRetriever 始终返回空集，
        Stage 1B 分支不产生任何结果（scsr_triggered 始终为 False）。

        参数：
            query:          输入查询文本
            sample_id:      样本 ID（用于日志追踪）
            patient_profile: 患者 profile dict（可选）

        返回：
            SampleResult（scsr_triggered 始终为 False）
        """
        marc_output: MARCOutput = self._pipeline.run(
            query=query,
            patient_profile=patient_profile,
        )

        return SampleResult(
            sample_id=sample_id,
            system_name=self.system_name,
            predicted_answer=marc_output.generated_answer,
            per_action_status_pred=marc_output.per_action_status,
            scsr_triggered=marc_output.scsr_triggered,   # 始终 False
            srl_violations=marc_output.srl_violations,
            marc_output=marc_output,
            raw_response=marc_output.generated_answer,
        )
