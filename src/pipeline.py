#!/usr/bin/env python3
"""
MARC 端到端推理 Pipeline（正交双空间检索架构）

架构（research.md §3.6）：
  Module 0: QueryDecomposer      → (D_q, C_q)
  Stage 1A: HybridRetriever      → R_D（疾病空间，BM25+Dense）
  Stage 1B: ConstraintRetriever  → R_C（约束空间，主动检索）  ← 新增，替换 SCSR
  Stage 2:  DualSpaceFusion      → E(q)（κ>0，三信号 RRF + DCR 条件概率评分）
  FC:       FCHandler            → 值域冲突检测和仲裁（可选）
  Gen:      ScopeAnchoredGenerator → 推荐文本 + attribution
  Verify:   AttributionVerifier  → SLR 计算

数学保证：
  Stage 2 的乘法结构确保 κ=0 文档不进入 Gen 的 context（Theorem 3.5）。
  双空间检索保证 E(q) 同时覆盖疾病相关性维度和约束安全性维度。

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
from src.diagnostic_refiner import DiagnosticRefiner
from src.constraint_expander import ConstraintExpander
from src.cvfr_query_constructor import CVFRQueryConstructor, CVFRResult
from src.kappa_scorer import KappaScorer
from src.constraint_retriever import ConstraintRetriever
from src.dual_space_fusion import DualSpaceFusion
from src.fc_handler import FCHandler
from src.generator import ScopeAnchoredGenerator
from src.verifier import AttributionVerifier
from src.scope_index import ScopeIndex


class MARCPipeline:
    """
    MARC 端到端推理 pipeline（正交双空间检索架构）。

    核心变化（相对于 DCR+SCSR 旧架构）：
      - Stage 1B（ConstraintRetriever）替代被动 SCSR，主动从 C_q 维度检索
      - Stage 2（DualSpaceFusion）融合 R_D ∪ R_C，应用 Triple-RRF + κ 条件概率评分
      - 消除了 SCSR 的触发阈值条件，检索流程完全对称

    数学等价：当 C_q = ∅ 时，Stage 1B 返回空集，DualSpaceFusion 退化为 DCRReranker。
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        kappa_scorer: KappaScorer,
        constraint_retriever: ConstraintRetriever,
        dual_space_fusion: DualSpaceFusion,
        fc_handler: FCHandler,
        generator: ScopeAnchoredGenerator,
        verifier: AttributionVerifier,
        # CVFR Phase 0（条件查询向量，可选，不提供则退化为旧双空间架构）
        cvfr_constructor: Optional[CVFRQueryConstructor] = None,
        # CVFR Phase 1（scope embedding 索引，可选，提供时计算 CDR/RSI/CAEC 指标）
        scope_index: Optional[ScopeIndex] = None,
        # 新 Module 0（Phase 1 精化架构）
        diagnostic_refiner: Optional[DiagnosticRefiner] = None,
        constraint_expander: Optional[ConstraintExpander] = None,
        # 旧 Module 0（向后兼容，供 marc_no_scsr 等 baseline 使用）
        decomposer: Optional[QueryDecomposer] = None,
        stage1_top_k: int = 20,
        stage1b_top_k: int = 10,
        stage2_top_k: int = 5,
        enable_fc: bool = False,
    ) -> None:
        """
        参数：
            retriever:            混合检索器（BM25 + Dense，Stage 1A 和 Stage 1B 共用）
            kappa_scorer:         κ 计算器（DualSpaceFusion 内部使用，此处保留供外部访问）
            constraint_retriever: 约束空间检索器（Stage 1B）
            dual_space_fusion:    双空间融合器（Stage 2）
            fc_handler:           FC 冲突仲裁器（可选）
            generator:            scope-anchored 生成器
            verifier:             归因校验器
            cvfr_constructor:     CVFR Phase 0 条件查询向量构造器（可选）
                                  提供时：Stage 1A Dense 检索使用 e*(D,C) 替代 e_D
                                  不提供时：退化为标准疾病向量检索（旧行为）
            scope_index:          CVFR Phase 1 scope embedding 索引（可选）
                                  提供时：在 Stage 2 后计算 CDR/RSI/CAEC 并写入 metrics
                                  不提供时：metrics 中无 CDR/RSI/CAEC 字段（不影响推理）
            diagnostic_refiner:   Module 0A — 临床诊断精化器（新架构，优先使用）
            constraint_expander:  Module 0B — 规则化约束展开器（新架构，与 refiner 配合）
            decomposer:           旧 Module 0（QueryDecomposer，向后兼容 fallback）
            stage1_top_k:         Stage 1A 检索数量（默认 20）
            stage1b_top_k:        Stage 1B 每约束检索数量（默认 10）
            stage2_top_k:         E(q) 大小上限（默认 5）
            enable_fc:            是否启用 FC 冲突检测（默认 False）

        Module 0 路径选择逻辑：
            若 diagnostic_refiner + constraint_expander 均已提供 → 新架构路径
            否则若 decomposer 已提供 → 旧路径（向后兼容）
            两者均无 → RuntimeError
        """
        self._retriever = retriever
        self._cvfr_constructor = cvfr_constructor
        self._scope_index = scope_index
        self._diagnostic_refiner = diagnostic_refiner
        self._constraint_expander = constraint_expander
        self._decomposer = decomposer
        self._kappa_scorer = kappa_scorer
        self._constraint_retriever = constraint_retriever
        self._dual_space_fusion = dual_space_fusion
        self._fc_handler = fc_handler
        self._generator = generator
        self._verifier = verifier
        self._stage1_top_k = stage1_top_k
        self._stage1b_top_k = stage1b_top_k
        self._stage2_top_k = stage2_top_k
        self._enable_fc = enable_fc

        if diagnostic_refiner is None and constraint_expander is None and decomposer is None:
            raise ValueError(
                "[MARCPipeline] 必须提供以下之一：\n"
                "  (a) diagnostic_refiner + constraint_expander（新架构，推荐）\n"
                "  (b) decomposer（旧架构，向后兼容）"
            )

        cvfr_status = "CVFR Phase 0 已启用（e*(D,C) 条件查询向量）" if cvfr_constructor else "CVFR 未启用（使用 e_D 标准检索）"
        scope_status = "ScopeIndex 已加载（CDR/RSI/CAEC 指标可用）" if (scope_index and scope_index.is_loaded) else "ScopeIndex 未加载（CDR/RSI/CAEC 指标不可用）"
        print(f"[MARCPipeline] {cvfr_status}")
        print(f"[MARCPipeline] {scope_status}")

    def run(
        self,
        query: str,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> MARCOutput:
        """
        执行完整 MARC 推理流程（正交双空间架构）。

        参数：
            query:           自然语言输入（含疾病描述+患者约束）
            patient_profile: 患者 profile dict（可选，用于生成器格式化）

        返回：
            MARCOutput（完整中间结果 + 最终答案 + 评估指标）

        异常：
            任何模块内部异常直接传播（不被静默捕获）。
        """
        metrics: Dict[str, Any] = {}
        t_start = time.time()

        # ── Module 0: Query Decompose ─────────────────────────────────────────
        # 路径 A（新架构）：DiagnosticRefiner（LLM 临床推理）+ ConstraintExpander（规则）
        # 路径 B（旧架构）：QueryDecomposer（LLM 提取 D_q + C_q，向后兼容）
        t0 = time.time()
        if self._diagnostic_refiner is not None and self._constraint_expander is not None:
            # 新路径：Module 0A（诊断精化）+ Module 0B（规则约束展开）
            diagnostic = self._diagnostic_refiner.refine(query)
            constraints = self._constraint_expander.expand(
                patient_profile=patient_profile,
                narrative_fallback=query,
            )
            decomposition: QueryDecomposition = QueryDecomposition(
                original_query=query,
                disease_query=diagnostic.retrieval_query,
                constraints=constraints,
                decompose_model="diagnostic_refiner+constraint_expander",
                debug={
                    "refined_diagnosis":       diagnostic.refined_diagnosis,
                    "discriminating_features": diagnostic.discriminating_features,
                    "refiner_debug":           diagnostic.debug,
                },
            )
        elif self._decomposer is not None:
            # 旧路径（向后兼容）：QueryDecomposer
            decomposition = self._decomposer.decompose(
                query=query,
                patient_profile=patient_profile,
            )
        else:
            raise RuntimeError(
                "[MARCPipeline] Module 0 未初始化：需要 diagnostic_refiner+constraint_expander 或 decomposer。"
            )
        metrics["module0_latency_s"] = time.time() - t0
        metrics["n_constraints"] = len(decomposition.constraints)

        # ── CVFR Phase 0: 条件查询向量计算 ────────────────────────────────────
        # 将标准疾病查询向量 e_D 修正为条件查询向量 e*(D, C)。
        # BM25 检索不受影响（仍使用 disease_query 文本）。
        # Dense 检索使用 e*(D, C) 替代 e_D，在向量空间中实现条件概率检索。
        t0 = time.time()
        cvfr_result: Optional[CVFRResult] = None
        stage1a_query_vector = None   # None → Dense 使用文本编码（退化为旧行为）

        if self._cvfr_constructor is not None:
            cvfr_result = self._cvfr_constructor.compute_conditional_vector(
                disease_query=decomposition.disease_query,
                constraints=decomposition.constraints,
            )
            stage1a_query_vector = cvfr_result.e_star
            metrics["cvfr_case"] = cvfr_result.case
            metrics["cvfr_lambda_star"] = cvfr_result.lambda_star
            metrics["cvfr_cos_dc"] = cvfr_result.cos_dc
            metrics["cvfr_angular_shift_deg"] = cvfr_result.angular_shift_deg
            metrics["cvfr_n_active_constraints"] = cvfr_result.n_active
        metrics["cvfr_latency_s"] = time.time() - t0

        # ── Stage 1A: 疾病空间检索（Disease-Space Retrieval）─────────────────
        # BM25：使用 disease_query 文本（词项匹配，不受 CVFR 影响）
        # Dense：使用 e*(D, C)（CVFR 条件查询向量）或 e_D（无 CVFR 时）
        t0 = time.time()
        from src.retriever import RetrievalResult
        stage1a_raw: List[RetrievalResult] = self._retriever.retrieve(
            query=decomposition.disease_query,
            top_k=self._stage1_top_k,
            query_vector=stage1a_query_vector,    # CVFR Phase 0 注入点
        )
        metrics["stage1a_latency_s"] = time.time() - t0
        metrics["stage1a_n_docs"] = len(stage1a_raw)

        # ── Stage 1B: 约束空间主动检索（Constraint-Space Retrieval）──────────
        # 从 C_q 维度检索：为每个约束生成专属 query，主动检索替代方案/剂量调整证据
        t0 = time.time()
        stage1b_raw, constraint_queries = self._constraint_retriever.retrieve(
            decomposition=decomposition,
            retriever=self._retriever,
            top_k_per_constraint=self._stage1b_top_k,
        )
        metrics["stage1b_latency_s"] = time.time() - t0
        metrics["stage1b_n_docs"] = len(stage1b_raw)
        metrics["stage1b_triggered"] = len(stage1b_raw) > 0

        # ── Stage 2: 双空间融合（Dual-Space Fusion）+ DCR 条件概率评分 ────────
        # 合并 R_D ∪ R_C，Triple-RRF 计算 S_joint，κ 计算，dcr_score = S_joint × κ
        # E(q) = {d | κ(d) > 0}，按 dcr_score 降序，取 top stage2_top_k
        t0 = time.time()
        stage2_docs, all_scored_pool = self._dual_space_fusion.fuse_and_score(
            disease_results=stage1a_raw,
            constraint_results=stage1b_raw,
            decomposition=decomposition,
            top_k=self._stage2_top_k,
        )
        metrics["stage2_latency_s"] = time.time() - t0
        metrics["stage2_n_admissible"] = len(stage2_docs)
        metrics["stage2_n_excluded"] = sum(1 for d in all_scored_pool if d.kappa == 0.0)
        metrics["stage2_pool_size"] = len(all_scored_pool)

        # 从 all_scored_pool 中提取 Stage 1B 的评分文档（用于 MARCOutput 记录）
        stage1b_chunk_ids = {r.chunk.chunk_id for r in stage1b_raw}
        stage1b_docs: List[RetrievedDoc] = [
            d for d in all_scored_pool
            if d.chunk.chunk_id in stage1b_chunk_ids
        ]

        # ── FC Handler（可选，默认关闭）────────────────────────────────────────
        if self._enable_fc:
            t0 = time.time()
            fc_conflicts, resolved_docs = self._fc_handler.detect_and_resolve(
                docs=stage2_docs,
                decomposition=decomposition,
            )
            metrics["fc_latency_s"] = time.time() - t0
        else:
            fc_conflicts = []
            resolved_docs = stage2_docs
            metrics["fc_latency_s"] = 0.0
        metrics["n_fc_conflicts"] = len(fc_conflicts)

        # ── 从选项文本填充 candidate_actions（供 CAEC 指标使用）────────────────
        # 选项格式："A: desc | B: desc | ..."，将描述文本作为候选 action
        # DiagnosticRefiner 路径不提取 candidate_actions，在此补充
        options_text = ""
        if "\n\nOptions: " in query:
            options_text = query.split("\n\nOptions: ", 1)[1]

        if not decomposition.candidate_actions and options_text:
            decomposition.candidate_actions = [
                part.strip().split(": ", 1)[1].strip()
                for part in options_text.split("|")
                if ": " in part.strip()
            ]

        # ── CVFR Phase 1：CDR / RSI / CAEC（需要 scope_index 已加载）──────────
        # 若 scope_index 未提供或未加载，跳过（不影响推理结果，仅无指标）
        if self._scope_index is not None and self._scope_index.is_loaded:
            constraint_texts = [c.raw_text for c in decomposition.constraints if c.raw_text]
            if constraint_texts:
                try:
                    e_C = self._scope_index.encode_constraint(constraint_texts)
                    cdr_result = self._scope_index.compute_cdr(stage2_docs, e_C)
                    rsi_result = self._scope_index.compute_rsi(stage2_docs, e_C, theta_high=0.6)
                    metrics["cdr"] = cdr_result["cdr"]
                    metrics["rsi"] = rsi_result["rsi"]
                    metrics["cdr_per_doc"] = cdr_result["per_doc"]   # 供 CAEC 在 eval 时使用
                    metrics["rsi_n_specific"] = rsi_result["n_specific"]
                    metrics["rsi_n_total"] = rsi_result["n_total"]
                except Exception as e:
                    # scope_index 计算失败不应中断推理
                    metrics["cdr"] = None
                    metrics["rsi"] = None
                    metrics["scope_index_error"] = str(e)

        # ── Generator ─────────────────────────────────────────────────────────

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
        # inadmissible_chunk_ids：R_D ∪ R_C 融合池中所有 κ=0 的文档
        inadmissible_ids = {d.chunk.chunk_id for d in all_scored_pool if d.kappa == 0.0}
        slr, srl_violations = self._verifier.verify(
            attribution=attribution,
            inadmissible_chunk_ids=inadmissible_ids,
        )
        metrics["slr"] = slr
        metrics["total_latency_s"] = time.time() - t_start

        # Stage 1B 贡献统计（backward compat：scsr_triggered/scsr_docs）
        # scsr_triggered = Stage 1B 是否产生了对 E(q) 有净贡献的新文档
        stage1a_chunk_ids = {r.chunk.chunk_id for r in stage1a_raw}
        stage1b_new_admissible = [
            d for d in stage2_docs
            if d.chunk.chunk_id not in stage1a_chunk_ids
        ]
        scsr_triggered = len(stage1b_new_admissible) > 0

        return MARCOutput(
            query=query,
            decomposition=decomposition,
            # stage1_docs = 双空间融合池全量（供 SLR 计算 inadmissible_chunk_ids）
            stage1_docs=all_scored_pool,
            stage2_docs=stage2_docs,
            scsr_triggered=scsr_triggered,
            scsr_docs=stage1b_new_admissible,  # 向后兼容：Stage 1B 对 E(q) 的净增量
            fc_conflicts=fc_conflicts,
            generated_answer=answer_text,
            per_action_status=per_action_status,
            attribution=attribution,
            srl_violations=srl_violations,
            metrics=metrics,
            # 双空间检索新增字段
            stage1b_docs=stage1b_docs,
            constraint_queries=constraint_queries,
        )


def build_marc_pipeline(
    index_dir: str = "data/index",
    cache_dir: str = "data/cache",
    stage1_top_k: int = 20,
    stage1b_top_k: int = 10,
    stage2_top_k: int = 5,
    enable_fc: bool = False,
    disease_weight: float = 0.7,
    constraint_weight: float = 0.3,
    cvfr_tau: float = 0.5,
    enable_cvfr: bool = True,
) -> MARCPipeline:
    """
    工厂函数：一键构建 MARCPipeline（CVFR Phase 0 + 正交双空间检索架构）。

    所有 LLM 配置（API key、base URL、模型名）从 src/llm_client 读取，
    最终来源是项目根目录的 .env 文件（参考 .env.example）。

    参数：
        index_dir:          教材索引目录（data/index）
        cache_dir:          缓存目录（π_d 缓存、query 分解缓存、约束 query 缓存）
        stage1_top_k:       Stage 1A 检索数量（默认 20）
        stage1b_top_k:      Stage 1B 每约束检索数量（默认 10）
        stage2_top_k:       E(q) 大小上限（默认 5）
        enable_fc:          是否启用 FC 冲突检测（默认 False）
        disease_weight:     双空间融合疾病维度权重 α（默认 0.7）
        constraint_weight:  双空间融合约束维度权重 β（默认 0.3）
        cvfr_tau:           CVFR Phase 0 约束激活阈值 τ（默认 0.5）
                            cos(e_D, ê_C) < τ 时激活修正。
                            标定值（BGE-M3 空间，7个医学约束对测量）：
                              τ=0.3: 无约束激活（BGE-M3 医学术语最低 cos_DC ≈ 0.39）
                              τ=0.5: 大多数患者约束激活（推荐，6/7 覆盖）
                              τ=0.6: 全部约束强制激活
        enable_cvfr:        是否启用 CVFR Phase 0（默认 True）
                            设为 False 等同于旧双空间架构（消融对比用）

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

    # CVFR Phase 0：条件查询向量构造器
    # 复用 retriever 的 embedding_model（SentenceTransformer 实例），避免二次加载
    cvfr_constructor: Optional[CVFRQueryConstructor] = None
    if enable_cvfr and retriever.embedding_model is not None:
        cvfr_constructor = CVFRQueryConstructor(
            embedding_model=retriever.embedding_model,
            tau=cvfr_tau,
        )
        print(f"[build_marc_pipeline] CVFR Phase 0 已初始化（τ={cvfr_tau}）")
    elif enable_cvfr:
        print("[build_marc_pipeline] 警告：Dense 索引未加载，CVFR Phase 0 无法启用")

    # CVFR Phase 1：Scope Embedding 索引（可选，仅当 data/index/scope/ 目录存在时加载）
    # 若索引未构建（需先运行 scripts/build_scope_index.py），跳过加载，CDR/RSI/CAEC 不可用
    scope_index: Optional[ScopeIndex] = None
    scope_index_dir = Path(index_dir) / "scope"
    if scope_index_dir.exists() and retriever.embedding_model is not None:
        try:
            scope_index = ScopeIndex.load(
                scope_index_dir=scope_index_dir,
                embedding_model=retriever.embedding_model,
                scope_predicates_cache_dir=Path(cache_dir) / "scope_predicates",
            )
        except FileNotFoundError:
            print("[build_marc_pipeline] scope index 文件缺失，CDR/RSI/CAEC 指标不可用")
    else:
        print(f"[build_marc_pipeline] scope index 目录不存在（{scope_index_dir}），"
              f"请运行 scripts/build_scope_index.py 构建后重启")

    # Module 0A：临床诊断精化器（LLM，专注诊断推理，无答案泄露）
    diagnostic_refiner = DiagnosticRefiner(
        client=client,
        model=model,
        cache_dir=cache_path,
    )

    # Module 0B：规则化约束展开器（零 LLM，确定性）
    # 注入 QueryDecomposer 作为 fallback（patient_profile 为空时使用）
    decomposer_fallback = QueryDecomposer(
        client=client,
        model=model,
        cache_dir=cache_path,
    )
    constraint_expander = ConstraintExpander(
        decomposer=decomposer_fallback,
    )

    kappa_scorer = KappaScorer(
        client=client,
        model=model,
        cache_dir=cache_path,
    )
    constraint_retriever = ConstraintRetriever(
        client=client,
        model=model,
        cache_dir=cache_path,
    )
    dual_space_fusion = DualSpaceFusion(
        kappa_scorer=kappa_scorer,
        disease_weight=disease_weight,
        constraint_weight=constraint_weight,
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
        cvfr_constructor=cvfr_constructor,
        scope_index=scope_index,
        diagnostic_refiner=diagnostic_refiner,
        constraint_expander=constraint_expander,
        kappa_scorer=kappa_scorer,
        constraint_retriever=constraint_retriever,
        dual_space_fusion=dual_space_fusion,
        fc_handler=fc_handler,
        generator=generator,
        verifier=verifier,
        stage1_top_k=stage1_top_k,
        stage1b_top_k=stage1b_top_k,
        stage2_top_k=stage2_top_k,
        enable_fc=enable_fc,
    )
