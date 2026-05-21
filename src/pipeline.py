#!/usr/bin/env python3
"""
MARC 端到端推理 Pipeline

组合所有模块：Module 0 → Stage 1 → Stage 2 → Stage 3 → FC → Gen → Verify

设计原则：
  1. 显式数据流：每个模块的输出是下一模块的输入
  2. 富返回：返回完整 MARCOutput（含所有中间结果）
  3. 无兜底：异常直接传播（不静默捕获）
  4. 可观测：MARCOutput.metrics 记录延迟、token 数、触发情况
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


from src.types import MARCOutput, QueryDecomposition, RetrievedDoc
from src.retriever import HybridRetriever
from src.query_decomposer import QueryDecomposer
from src.kappa_scorer import KappaScorer
from src.dcr import DCRReranker
from src.scsr import SCSRRetriever
from src.fc_handler import FCHandler
from src.generator import ScopeAnchoredGenerator
from src.verifier import AttributionVerifier


class MARCPipeline:
    """
    MARC 端到端推理 pipeline。

    架构（research.md §4.1）：
      Module 0: QueryDecomposer → (D_q, C_q)
      Stage 1:  HybridRetriever.retrieve(D_q, top_k=20) → top-K 候选
      Stage 2:  DCRReranker → admissible_docs（κ>0，物理排除 INADMISSIBLE）
      Stage 3:  SCSRRetriever → 补充文档（按需，gap-filling）
      FC:       FCHandler → 值域冲突检测和仲裁
      Gen:      ScopeAnchoredGenerator → 推荐文本 + attribution
      Verify:   AttributionVerifier → SLR 计算

    数学保证：
      DCR 的乘法结构确保 κ=0 文档不进入 Gen 的 context（Theorem 3.5 直接体现）。
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        decomposer: QueryDecomposer,
        kappa_scorer: KappaScorer,
        dcr_reranker: DCRReranker,
        scsr_retriever: SCSRRetriever,
        fc_handler: FCHandler,
        generator: ScopeAnchoredGenerator,
        verifier: AttributionVerifier,
        stage1_top_k: int = 20,
        stage2_top_k: int = 5,
        enable_fc: bool = False,
    ) -> None:
        """
        参数：
            retriever:      混合检索器（BM25 + Dense）
            decomposer:     Query 分解器（Module 0）
            kappa_scorer:   κ 计算器
            dcr_reranker:   DCR 重排序器（Stage 2）
            scsr_retriever: SCSR 补充检索器（Stage 3）
            fc_handler:     FC 冲突仲裁器
            generator:      scope-anchored 生成器
            verifier:       归因校验器
            stage1_top_k:   Stage 1 初始检索数量（默认 20）
            stage2_top_k:   Stage 2 DCR 输出数量（默认 5）
            enable_fc:      是否启用 FC 冲突检测（默认 False）
                            MACB benchmark 无 FC 金标签，实验时关闭以节省延迟
        """
        self._retriever = retriever
        self._decomposer = decomposer
        self._kappa_scorer = kappa_scorer
        self._dcr_reranker = dcr_reranker
        self._scsr_retriever = scsr_retriever
        self._fc_handler = fc_handler
        self._generator = generator
        self._verifier = verifier
        self._stage1_top_k = stage1_top_k
        self._stage2_top_k = stage2_top_k
        self._enable_fc = enable_fc

    def run(
        self,
        query: str,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> MARCOutput:
        """
        执行完整 MARC 推理流程。

        参数：
            query:          自然语言输入（含疾病描述+患者约束）
            patient_profile: 患者 profile dict（可选，用于生成器格式化）

        返回：
            MARCOutput（完整中间结果 + 最终答案 + 评估指标）

        异常：
            任何模块内部异常直接传播（不被静默捕获）。
        """
        metrics: Dict[str, Any] = {}
        t_start = time.time()

        # ── Module 0: Query Decompose ─────────────────────────────────────────
        t0 = time.time()
        decomposition: QueryDecomposition = self._decomposer.decompose(query)
        metrics["module0_latency_s"] = time.time() - t0
        metrics["n_constraints"] = len(decomposition.constraints)

        # ── Stage 1: Disease Retrieval ────────────────────────────────────────
        t0 = time.time()
        stage1_raw = self._retriever.retrieve(
            query=decomposition.disease_query,
            top_k=self._stage1_top_k,
        )
        metrics["stage1_latency_s"] = time.time() - t0
        metrics["stage1_n_docs"] = len(stage1_raw)

        # ── Stage 2: DCR Reranking ────────────────────────────────────────────
        t0 = time.time()
        # 保留全量 κ 计算结果（含 κ=0 的文档，用于 SLR 计算）
        all_stage1_scored: List[RetrievedDoc] = self._kappa_scorer.score_documents(
            retrieval_results=stage1_raw,
            decomposition=decomposition,
        )
        stage2_docs: List[RetrievedDoc] = [
            d for d in all_stage1_scored if d.kappa > 0.0
        ][:self._stage2_top_k]
        metrics["stage2_latency_s"] = time.time() - t0
        metrics["stage2_n_admissible"] = len(stage2_docs)
        metrics["stage2_n_excluded"] = sum(1 for d in all_stage1_scored if d.kappa == 0.0)

        # ── Stage 3: SCSR（按需触发） ─────────────────────────────────────────
        scsr_triggered = self._scsr_retriever.should_trigger(stage2_docs)
        scsr_docs: List[RetrievedDoc] = []
        scsr_query: str = ""

        if scsr_triggered:
            t0 = time.time()
            scsr_docs, scsr_query = self._scsr_retriever.retrieve(
                decomposition=decomposition,
                stage2_docs=stage2_docs,
                retriever=self._retriever,
            )
            metrics["stage3_latency_s"] = time.time() - t0
            metrics["stage3_n_docs"] = len(scsr_docs)
            metrics["stage3_query"] = scsr_query
        else:
            metrics["stage3_latency_s"] = 0.0
            metrics["stage3_n_docs"] = 0

        # ── FC Handler（可选，默认关闭）────────────────────────────────────────
        # MACB benchmark 无 FC 金标签，关闭后节省每样本 20-30 秒推理时间
        all_admissible = stage2_docs + scsr_docs
        if self._enable_fc:
            t0 = time.time()
            fc_conflicts, resolved_docs = self._fc_handler.detect_and_resolve(
                docs=all_admissible,
                decomposition=decomposition,
            )
            metrics["fc_latency_s"] = time.time() - t0
        else:
            fc_conflicts = []
            resolved_docs = all_admissible
            metrics["fc_latency_s"] = 0.0
        metrics["n_fc_conflicts"] = len(fc_conflicts)

        # ── Generator ─────────────────────────────────────────────────────────
        # 从 query 末尾提取选项文本（evaluate.py 构造格式："{question}\n\nOptions: {opts}"）
        options_text = ""
        if "\n\nOptions: " in query:
            options_text = query.split("\n\nOptions: ", 1)[1]

        inadmissible_actions = decomposition.absolute_target_actions
        t0 = time.time()
        answer_text, per_action_status, attribution = self._generator.generate(
            decomposition=decomposition,
            admissible_docs=resolved_docs,
            inadmissible_actions=inadmissible_actions,
            fc_conflicts=fc_conflicts,
            patient_profile=patient_profile,
            options_text=options_text,
        )
        metrics["gen_latency_s"] = time.time() - t0

        # ── Verifier ──────────────────────────────────────────────────────────
        inadmissible_ids = {d.chunk.chunk_id for d in all_stage1_scored if d.kappa == 0.0}
        slr, srl_violations = self._verifier.verify(
            attribution=attribution,
            inadmissible_chunk_ids=inadmissible_ids,
        )
        metrics["slr"] = slr
        metrics["total_latency_s"] = time.time() - t_start

        return MARCOutput(
            query=query,
            decomposition=decomposition,
            stage1_docs=all_stage1_scored,
            stage2_docs=stage2_docs,
            scsr_triggered=scsr_triggered,
            scsr_docs=scsr_docs,
            fc_conflicts=fc_conflicts,
            generated_answer=answer_text,
            per_action_status=per_action_status,
            attribution=attribution,
            srl_violations=srl_violations,
            metrics=metrics,
        )


def build_marc_pipeline(
    index_dir: str = "data/index",
    cache_dir: str = "data/cache",
    stage1_top_k: int = 20,
    stage2_top_k: int = 5,
    enable_fc: bool = False,
) -> MARCPipeline:
    """
    工厂函数：一键构建 MARCPipeline（含所有子模块初始化）。

    所有 LLM 配置（API key、base URL、模型名）从 src/llm_client 读取，
    最终来源是项目根目录的 .env 文件（参考 .env.example）。

    参数：
        index_dir:    教材索引目录（data/index）
        cache_dir:    缓存目录（π_d 缓存、query 分解缓存）
        stage1_top_k: Stage 1 检索数量（默认 20）
        stage2_top_k: Stage 2 DCR 输出数量（默认 5）

    返回：
        完整初始化的 MARCPipeline 实例
    """
    from pathlib import Path
    from src.llm_client import get_client, get_model, get_embedding_model

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
    scsr_retriever = SCSRRetriever(
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
        enable_fc=enable_fc,
    )
