#!/usr/bin/env python3
"""
消融 D8：MARC-noSCSR

MARC 的消融变体：保留完整 DCR（κ 计算 + INADMISSIBLE 物理排除），
但禁用 Stage 3 SCSR（A(q) gap-filling 补充检索永不触发）。

测量目标：
  SCSR 对 SDR（缺口填充率）和 AEC（答案覆盖率）的贡献。
  若 MARC-noSCSR 的 SDR 显著低于 MARC，说明 SCSR 对 evidence gap-filling 有效。
  若 SDR 差异不大，说明 DCR 已经找到足够的 admissible 证据。

预期结果：
  SDR(MARC-noSCSR) < SDR(MARC)：SCSR 对边缘案例（strong contraindication）有效
  CRR(MARC-noSCSR) ≈ CRR(MARC)：SCSR 不影响 CRR（只填充，不排除）

实现方式：
  注入 _DisabledSCSRRetriever（继承 SCSRRetriever，覆盖 should_trigger 返回 False）。
  通过工厂函数 build_marc_no_scsr_pipeline() 构建管道，与 build_marc_pipeline() 对称。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines.base import BaseRAGSystem
from src.scsr import SCSRRetriever
from src.types import QueryDecomposition, RetrievedDoc, SampleResult, MARCOutput


class _DisabledSCSRRetriever(SCSRRetriever):
    """
    永不触发的 SCSR（消融专用）。

    覆盖 should_trigger() 强制返回 False，
    使 MARCPipeline 的 Stage 3 分支永远不执行。
    retrieve() 方法保持可用但实际不会被调用。
    """

    def should_trigger(self, stage2_docs: List[RetrievedDoc]) -> bool:
        """
        永远返回 False，禁用 SCSR 触发。

        参数：
            stage2_docs: Stage 2 输出（不使用，仅为接口兼容）

        返回：
            False（始终）
        """
        return False


def build_marc_no_scsr_pipeline(
    index_dir: str = "data/index",
    cache_dir: str = "data/cache",
    stage1_top_k: int = 20,
    stage2_top_k: int = 5,
) -> Any:
    """
    工厂函数：构建 MARC-noSCSR 管道。

    与 build_marc_pipeline() 完全相同，唯一差异是
    将 SCSRRetriever 替换为 _DisabledSCSRRetriever。

    所有 LLM 配置从 src/llm_client 读取（来源为 .env 文件）。

    参数：
        index_dir:    教材索引目录
        cache_dir:    缓存目录（π_d 缓存、query 分解缓存）
        stage1_top_k: Stage 1 检索数量（默认 20）
        stage2_top_k: Stage 2 DCR 输出数量（默认 5）

    返回：
        完整初始化的 MARCPipeline 实例（SCSR 已禁用）
    """
    from src.llm_client import get_client, get_model, get_embedding_model
    from src.retriever import HybridRetriever
    from src.query_decomposer import QueryDecomposer
    from src.kappa_scorer import KappaScorer
    from src.dcr import DCRReranker
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
    dcr_reranker = DCRReranker(kappa_scorer=kappa_scorer)

    # 关键差异：注入永不触发的 SCSR（消融实验）
    scsr_retriever = _DisabledSCSRRetriever(
        client=client,
        kappa_scorer=kappa_scorer,
        model=model,
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
        dcr_reranker=dcr_reranker,
        scsr_retriever=scsr_retriever,
        fc_handler=fc_handler,
        generator=generator,
        verifier=verifier,
        stage1_top_k=stage1_top_k,
        stage2_top_k=stage2_top_k,
        enable_fc=False,
    )


class MARCNoSCSR(BaseRAGSystem):
    """
    MARC-noSCSR：完整 MARC 管道，SCSR 永不触发。

    保留：Module 0（Query 分解）+ Stage 1（检索）+ Stage 2（DCR，κ+物理排除）
          FC Handler + Scope-Anchored Generator + Verifier
    去除：Stage 3 SCSR（gap-filling 补充检索）

    通过 _DisabledSCSRRetriever 实现，确保 MARCPipeline 架构不变，
    仅 SCSR 触发判断短路为 False。

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

    def run(self, query: str, sample_id: str) -> SampleResult:
        """
        MARC-noSCSR 推理流程。

        完全委托给内部 MARCPipeline.run()，
        由于 SCSR 已被 _DisabledSCSRRetriever 禁用，
        Stage 3 分支永远不会执行（scsr_triggered 始终为 False）。

        参数：
            query:      输入查询文本
            sample_id:  样本 ID（用于日志追踪）

        返回：
            SampleResult（scsr_triggered 字段始终为 False）
        """
        marc_output: MARCOutput = self._pipeline.run(query=query)

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
