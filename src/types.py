#!/usr/bin/env python3
"""
共享数据类型定义

所有 MARC 模块使用的数据类型集中定义于此，避免循环依赖和类型散落。
修改此文件时需同步更新所有引用模块。

类型层级：
  PatientConstraint → QueryDecomposition
  TextChunk + ScopePredicate → RetrievedDoc
  RetrievedDoc + QueryDecomposition → MARCOutput
  EvalSample + MARCOutput → SampleResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set


# ── 患者约束 ──────────────────────────────────────────────────────────────────

@dataclass
class PatientConstraint:
    """
    单个患者约束的结构化表示。

    参数：
        constraint_type:      约束严重程度
                              ABSOLUTE = 绝对禁忌（κ=0，集合排除）
                              RELATIVE = 相对禁忌（κ∈(0,1)，降权+标记）
                              NONE     = 无约束（κ=1，正常保留）
        target_action:        被约束的 action 关键词（药物名、操作名等）
        raw_text:             原始文本描述（保留供调试）
        parameter_value:      相对约束的患者实际参数值（如 eGFR=28）
        parameter_threshold:  相对约束的安全阈值（如 eGFR 阈值=60）
    """
    constraint_type: Literal["ABSOLUTE", "RELATIVE", "NONE"]
    target_action: str
    raw_text: str
    parameter_value: Optional[float] = None
    parameter_threshold: Optional[float] = None

    def compute_kappa_single(self) -> float:
        """
        根据约束类型计算单约束 κ 值。

        数学公式（research.md §3.5.2）：
          ABSOLUTE → κ = 0
          RELATIVE（有参数）→ κ = f(δ) = 1 - max(0, threshold - actual) / threshold
          RELATIVE（无参数）→ κ = 0.5（定性约束保守默认，LLM 未能提取数值）
          NONE     → κ = 1
        """
        if self.constraint_type == "ABSOLUTE":
            return 0.0
        if self.constraint_type == "RELATIVE":
            if self.parameter_value is None or self.parameter_threshold is None:
                # LLM 提取了约束但未获得数值参数（如定性描述"elevated AST/ALT"、
                # 胎龄无阈值等），无法计算精确 f(δ)。
                # 保守默认 κ=0.5：承认相对禁忌存在，施加中等惩罚，同时不物理排除文档。
                return 0.5
            delta = max(0.0, self.parameter_threshold - self.parameter_value)
            return 1.0 - delta / self.parameter_threshold
        return 1.0


@dataclass
class QueryDecomposition:
    """
    Module 0 的输出：自然语言 query 分解为 (D_q, C_q, A_q)。

    数学含义（cvfr_theory.md §2.2）：
      最小充分检索结构为三元组 (D, C, a)：
        D_q 用于 sim(D_q, d)（疾病相关性）
        C_q 用于 κ(C_q, π_d)（适用范围蕴含判断）
        A_q 是候选治疗动作集合（约束 C 作用在 action 上，而非直接作用在疾病上）
      三者正交，约束 C 是 action-specific 的，不是 disease-specific 的。

    两阶段引入 A_q（避免循环依赖）：
      Phase 1：用 D_q 宽泛检索后枚举 A_q（若 LLM 可在分解阶段识别候选动作则直接填入）
      Phase 2：对每个 a ∈ A_q 做 (D, C, a) 条件适用性检索

    参数：
        original_query:     原始输入 query（完整，含疾病+约束）
        disease_query:      D_q（纯疾病查询，去除患者约束）
        constraints:        C_q（患者约束结构化列表）
        decompose_model:    用于分解的 LLM 模型 ID
        candidate_actions:  A_q（候选治疗动作列表，从 query 或检索结果中枚举）
                            空列表表示未枚举（Phase 0），需要在 Phase 1 检索后填充
        debug:              LLM 原始输出和中间状态（供调试）
    """
    original_query: str
    disease_query: str
    constraints: List[PatientConstraint]
    decompose_model: str
    candidate_actions: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_absolute_constraint(self) -> bool:
        """是否存在绝对禁忌约束（κ=0）"""
        return any(c.constraint_type == "ABSOLUTE" for c in self.constraints)

    @property
    def has_relative_constraint(self) -> bool:
        """是否存在相对禁忌约束（κ∈(0,1)）"""
        return any(c.constraint_type == "RELATIVE" for c in self.constraints)

    @property
    def absolute_target_actions(self) -> List[str]:
        """所有绝对禁忌的目标 action 列表"""
        return [c.target_action for c in self.constraints if c.constraint_type == "ABSOLUTE"]


# ── 文档与检索 ────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    教材文本块，检索系统的基本单位。

    ID 格式：{book_slug}_{idx:05d}，如 InternalMed_Harrison_00042
    文本块大小：默认 400 tokens，重叠 50 tokens（见 scripts/index_textbooks.py）

    参数：
        chunk_id:    全局唯一 ID
        source_book: 来源教材名（不含路径和扩展名）
        text:        文本内容
        start_char:  在原文中的起始字符位置（溯源用）
        end_char:    在原文中的结束字符位置
        token_count: 大致 token 数（按空格计算的近似值）
    """
    chunk_id: str
    source_book: str
    text: str
    start_char: int
    end_char: int
    token_count: int = 0


@dataclass
class ScopePredicate:
    """
    文献适用范围谓词 π_d（v2，含结构化适用域）。

    数学含义（cvfr_theory.md §6）：
      π_d(x, a) = True 当且仅当患者 x 在治疗动作 a 的背景下满足文献适用域。
      κ(C_q, π_d) 实现为：C_patient ⊨ Scope(d, a) 的蕴含判断。

    参数：
        chunk_id:              对应文本块 ID
        recommended_action:    文献推荐的主要 action（药物名或操作，可 "none"）
        population:            适用人群自然语言描述
        polarity:              文献对 action 的态度
                               recommended:      积极推荐
                               contraindicated:  绝对禁忌（hard gate 触发 κ=0）
                               caution:          相对禁忌 / 需注意
                               dose_adjustment:  需要剂量调整
                               not_applicable:   文献未涉及具体推荐
        scope_inclusion:       明确适用的人群特征列表（如 ["adult", "pregnancy-compatible"]）
        scope_exclusion:       明确排除的人群特征列表（如 ["pregnancy", "renal impairment"]）
        scope_status:          适用域提取状态
                               explicit:      文献明确声明了 scope 条件
                               inferred:      LLM 从上下文推断（不确定性较高）
                               not_specified: 文献未明确描述 scope（≠ 无禁忌）
        contraindications:     文献明示的禁忌证描述列表（后向兼容，与 scope_exclusion 部分重叠）
        relative_restrictions: 文献明示的相对禁忌（需剂量调整等）
        extraction_model:      提取所用 LLM 模型 ID
        raw_output:            LLM 原始 JSON 输出（供调试和审计）

    重要语义约定：
        scope_status="not_specified" ≠ "无禁忌"。
        未在文本中找到 scope 声明时，不应推断为"适用于所有患者"，
        而应在 κ 计算中以 UNKNOWN 处理（保守默认 κ=0.5，而非 κ=1.0）。
    """
    chunk_id: str
    recommended_action: str
    population: str
    contraindications: List[str]
    relative_restrictions: List[Dict[str, str]]
    extraction_model: str
    raw_output: str
    # v2 新增字段（含默认值以保持对旧缓存的向后兼容）
    polarity: Literal[
        "recommended", "contraindicated", "caution", "dose_adjustment", "not_applicable"
    ] = "not_applicable"
    scope_inclusion: List[str] = field(default_factory=list)
    scope_exclusion: List[str] = field(default_factory=list)
    scope_status: Literal["explicit", "inferred", "not_specified"] = "not_specified"

    @property
    def is_hard_contraindicated(self) -> bool:
        """是否为明确禁忌（polarity=contraindicated → hard gate 直接返回 κ=0）"""
        return self.polarity == "contraindicated"


@dataclass
class RetrievedDoc:
    """
    检索结果，携带 DCR 分数和适用范围信息。

    分数体系（research.md §3.5）：
      sim_score  = sim(D_q, d)：疾病相关性（Stage 1 原始分）
      kappa      = κ(C_q, π_d)：适用范围相容性（Stage 2 计算）
      dcr_score  = sim_score × kappa：DCR 综合分

    scope_status 枚举：
      ADMISSIBLE:       κ=1，进入 context
      INADMISSIBLE_ABS: κ=0，物理排除（绝对禁忌）
      INADMISSIBLE_REL: κ∈(0,1)，降权+标记（相对禁忌）
      UNKNOWN:          无法确定（π_d 提取失败或规则库未覆盖）
    """
    chunk: TextChunk
    sim_score: float
    kappa: float
    dcr_score: float
    scope_predicate: Optional[ScopePredicate]
    scope_status: Literal["ADMISSIBLE", "INADMISSIBLE_ABS", "INADMISSIBLE_REL", "UNKNOWN"]

    @property
    def is_admissible(self) -> bool:
        """是否可进入生成 context（κ > 0）"""
        return self.kappa > 0.0

    @property
    def is_absolutely_inadmissible(self) -> bool:
        """是否绝对禁忌（κ = 0）"""
        return self.scope_status == "INADMISSIBLE_ABS"


# ── FC 冲突 ──────────────────────────────────────────────────────────────────

@dataclass
class FCConflict:
    """
    单个 FC（事实性冲突）的结构化描述。

    FC 操作在 A(q) 已确定后的值域层进行（research.md §1.2）。
    本数据结构只含 κ>0 的文档之间的冲突。

    参数：
        action:            涉及的 action（两篇文档对此 action 存在矛盾）
        doc_a_id:          文档 A 的 chunk_id
        doc_a_claim:       文档 A 的相关陈述摘要
        doc_b_id:          文档 B 的 chunk_id
        doc_b_claim:       文档 B 的相关陈述摘要
        conflict_type:     冲突类型（contradict/update/population_diff）
        resolution:        仲裁结果（prefer_a/prefer_b/uncertain）
        resolution_reason: 仲裁依据
    """
    action: str
    doc_a_id: str
    doc_a_claim: str
    doc_b_id: str
    doc_b_claim: str
    conflict_type: Literal["contradict", "update", "population_diff"]
    resolution: Literal["prefer_a", "prefer_b", "uncertain"]
    resolution_reason: str


# ── MARC 完整输出 ─────────────────────────────────────────────────────────────

@dataclass
class MARCOutput:
    """
    端到端 MARC pipeline 的完整输出（rich return 结构）。

    设计原则（research.md §4.1 + CLAUDE.md 富返回规范）：
      不仅返回最终答案，还携带所有中间阶段的完整信息，
      供评估指标计算（CRR/SDR/AEC/FC-AA/SLR）和论文分析使用。

    双空间检索架构（research.md §3.6）：
      stage1_docs:   Stage 1A（疾病空间）+ Stage 1B（约束空间）融合池的全量评分结果
                     即 R_D ∪ R_C 中所有文档（含 κ=0 的 INADMISSIBLE 文档）
      stage2_docs:   E(q)：融合池中 κ > 0 的精准证据集，按 S_joint × κ 降序

    参数：
        query:               原始查询文本
        decomposition:       Module 0 分解结果（D_q + C_q）
        stage1_docs:         双空间融合池全量评分（含 κ=0 文档，用于 SLR 计算）
        stage2_docs:         E(q)：κ > 0 的精准证据集（生成器上下文来源）
        stage1b_docs:        Stage 1B 约束空间检索的评分文档（R_C 子集，调试用）
        constraint_queries:  Stage 1B 生成的约束检索 query（{target_action → query}）
        scsr_triggered:      Stage 1B 是否产生了新文档（True = 约束空间检索有贡献）
        scsr_docs:           Stage 1B 中 κ > 0 的文档（向后兼容字段，同 stage1b admissible）
        fc_conflicts:        FC 冲突检测结果
        generated_answer:    最终生成的治疗推荐文本
        per_action_status:   各 action 的可行性状态（系统预测值）
        attribution:         每个 claim 的来源 chunk_id 列表
        srl_violations:      引用了 κ=0 文献的 claim（SLR 违规列表）
        metrics:             运行时指标（延迟、token 数、API 调用次数、总成本）
    """
    query: str
    decomposition: QueryDecomposition
    stage1_docs: List[RetrievedDoc]
    stage2_docs: List[RetrievedDoc]
    scsr_triggered: bool
    scsr_docs: List[RetrievedDoc]
    fc_conflicts: List[FCConflict]
    generated_answer: str
    per_action_status: Dict[str, str]
    attribution: List[Dict[str, Any]]
    srl_violations: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    # 双空间检索新增字段（正交双空间架构 research.md §3.6）
    stage1b_docs: List[RetrievedDoc] = field(default_factory=list)
    constraint_queries: Dict[str, str] = field(default_factory=dict)

    @property
    def admissible_docs(self) -> List[RetrievedDoc]:
        """E(q)：κ > 0 的精准证据集（stage2_docs 已过滤，直接返回）"""
        return [d for d in self.stage2_docs if d.is_admissible]

    @property
    def inadmissible_chunk_ids(self) -> Set[str]:
        """双空间融合池中 κ = 0 的文档 ID 集合（用于 SLR 计算）"""
        return {d.chunk.chunk_id for d in self.stage1_docs if d.is_absolutely_inadmissible}


# ── 评估数据类型 ──────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """
    MACB 评测样本的标准格式，与 data/macb_v1.jsonl 中的字段对应。

    参数：
        sample_id:                   样本 ID（MACB-001 格式）
        query:                       输入查询（MedQA question 文本）
        options_text:                选项文本（A: ... | B: ... 格式）
        answer_idx:                  正确答案选项字母
        candidate_tag:               样本类型（SC_ABSOLUTE_CAND/SC_RELATIVE_CAND/FC_CAND/MIXED_CAND）
        patient_profile:             患者 profile JSON（标注者填写）
        gold_admissible_set:         可行 action 集合 A(q)（标注者填写）
        gold_per_action_status:      per-action 金标准状态（标注者填写）
        gold_scsr_needed:            是否需要 SCSR（标注者填写）
        gold_scsr_query:             SCSR 查询（标注者手工构造）
        parametric_prior_conflict_label: LLM 参数记忆是否与患者特异性条件产生先验冲突
                                         取值：CONFLICT / NO_CONFLICT
        parametric_prior_disease_query: 去除患者约束的纯疾病查询（LLM 生成）
        gold_preferred_set:             严格最优 action 集合（仅 ADMISSIBLE，无需任何调整）
                                        gold_admissible_set 的子集
        gold_conflict_types_present:    细粒度冲突类型列表（从 gold_per_action_status 派生）
                                        可含 SC_ABSOLUTE / SC_RELATIVE / NO_CONFLICT
                                        mixed SC 样本含两个元素（如 ["SC_ABSOLUTE","SC_RELATIVE"]）
        task_type:                      任务类型，影响主评估子集选取
                                        treatment_recommendation（默认）: 推荐可行治疗
                                        contraindication_recognition: 识别禁忌药（排除在主评估外）
    """
    sample_id: str
    query: str
    options_text: str
    answer_idx: str
    candidate_tag: str
    patient_profile: Dict[str, Any]
    gold_admissible_set: List[str]
    gold_per_action_status: Dict[str, str]
    gold_scsr_needed: bool
    gold_scsr_query: Optional[str]
    parametric_prior_conflict_label: str
    parametric_prior_disease_query: str
    gold_preferred_set: List[str] = field(default_factory=list)
    gold_conflict_types_present: List[str] = field(default_factory=list)
    task_type: str = "treatment_recommendation"

    @property
    def is_sc_absolute(self) -> bool:
        """是否为 SC_ABSOLUTE 类型样本"""
        return self.candidate_tag == "SC_ABSOLUTE_CAND"

    @property
    def is_sc_relative(self) -> bool:
        """是否为 SC_RELATIVE 类型样本"""
        return self.candidate_tag == "SC_RELATIVE_CAND"

    @property
    def is_fc(self) -> bool:
        """是否为 FC 类型样本（用于 FC-AA 指标计算）"""
        return self.candidate_tag == "FC_CAND"

    @property
    def inadmissible_actions(self) -> List[str]:
        """所有绝对禁忌 action 的列表"""
        return [
            action for action, status in self.gold_per_action_status.items()
            if status == "INADMISSIBLE_ABS"
        ]


@dataclass
class SampleResult:
    """
    单个样本在单个系统上的评测结果。

    用于 eval/metrics.py 计算所有指标。

    参数：
        sample_id:               样本 ID
        system_name:             系统名称（如 "marc"/"standard_rag"）
        predicted_answer:        系统输出的自然语言推荐文本
        per_action_status_pred:  系统对各 action 状态的预测
                                 {"amoxicillin": "AVOIDED", "levofloxacin": "RECOMMENDED"}
        scsr_triggered:          是否触发了 Stage 3（仅 MARC 有效）
        srl_violations:          引用 INADMISSIBLE 文献的 claim 列表
        marc_output:             完整 MARCOutput（仅 MARC 及其消融系统有此字段）
        raw_response:            系统原始 API 响应文本（审计用）
        error:                   若运行出错，记录错误信息（不用于静默兜底，只作审计）
    """
    sample_id: str
    system_name: str
    predicted_answer: str
    per_action_status_pred: Dict[str, str]
    scsr_triggered: bool
    srl_violations: List[str]
    marc_output: Optional[MARCOutput]
    raw_response: str
    error: Optional[str] = None
    context_chunks: List[Dict[str, Any]] = field(default_factory=list)
    # 格式：[{"chunk_id":"...", "source_book":"...", "sim_score":0.87, "text_snippet":"..."}]
    # baselines 填充（用于 run_log 调试）；MARC 系统通过 marc_output 获取检索信息
