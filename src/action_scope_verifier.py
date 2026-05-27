#!/usr/bin/env python3
"""
ActionScopeVerifier：逐选项适用性验证器（per-option admissibility）

第一性原理（cvfr_theory.md §2.2）：
  约束 C 是 action-specific 的，MCQ 的选项正是候选治疗动作 A_q。
  原系统只在"文档维度"计算 κ(C_q, π_d)，从未把约束作用到选项空间，
  导致 SCSR/SRL 空转、禁忌选项仍被推荐。本模块把 κ 的判定显式扩展到
  选项粒度：对每个选项 a_i 计算 κ(C_q, a_i)，产出结构化、可审计的判定。

设计原则（CLAUDE.md）：
  - 确定性优先：规则层（药物 × 约束类别）命中即定论，无 LLM。
  - 无静默兜底：规则层与谓词层均无定论时显式标 UNKNOWN，绝不假装可行。
  - 可解释：每个 verdict 携带 reason、source（rule/predicate/none）与命中约束。
  - 复用既有机制：判定逻辑全部委托给 KappaScorer.check_option，零重复。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.kappa_scorer import KappaScorer
from src.types import ActionScopeReport, OptionVerdict, PatientConstraint


# κ-scorer 返回的 scope_status → OptionVerdict.status 的映射
# （INADMISSIBLE_REL 在选项语义下表示"相对禁忌/需调整"，即有条件可接受）
_SCOPE_STATUS_TO_VERDICT: Dict[str, str] = {
    "ADMISSIBLE": "ADMISSIBLE",
    "INADMISSIBLE_ABS": "INADMISSIBLE_ABS",
    "INADMISSIBLE_REL": "CONDITIONALLY_ADMISSIBLE",
    "UNKNOWN": "UNKNOWN",
}

# 选项描述中需剥离的剂量/给药信息（用于抽取主 action 关键词，仅供展示）
_DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|mg/kg|mg/dl|iu)\b",
    re.IGNORECASE,
)
# 常见给药途径/频次前后缀词（剥离后剩下的主体更接近药名/操作名）
_ROUTE_FREQ_WORDS = {
    "oral", "iv", "intravenous", "im", "intramuscular", "subcutaneous", "topical",
    "therapy", "treatment", "administration", "daily", "twice", "once", "bid",
    "tid", "qid", "infusion", "tablet", "tablets", "capsule", "capsules",
    "perform", "administer", "start", "begin", "continue", "initiate",
}


class ActionScopeVerifier:
    """
    逐选项适用性验证器。

    对每个候选选项调用 KappaScorer.check_option，得到 (κ, scope_status, source)，
    再封装为 OptionVerdict 并汇总为 ActionScopeReport。
    """

    def __init__(self, kappa_scorer: KappaScorer, use_llm_layer: bool = True) -> None:
        """
        参数：
            kappa_scorer:  已构建的 κ-scorer（共享规则集与 LLM client/cache）
            use_llm_layer: 规则层无定论时是否启用 LLM 谓词层兜底
                           （False 用于纯确定性消融实验）
        """
        self._kappa_scorer: KappaScorer = kappa_scorer
        self._use_llm_layer: bool = use_llm_layer

    def verify_options(
        self,
        options: Dict[str, str],
        constraints: List[PatientConstraint],
    ) -> ActionScopeReport:
        """
        对全部候选选项逐一判定适用性。

        参数：
            options:     {选项字母 → 选项描述}（如 {"A": "Oral warfarin therapy", ...}）
            constraints: 患者约束列表 C_q

        返回：
            ActionScopeReport：{选项字母 → OptionVerdict}
        """
        # 第一阶段：输入校验（显式失败，不静默）
        if not options:
            raise ValueError("[ActionScopeVerifier] options 为空，无选项可判定")

        # 第二阶段：逐选项判定
        verdicts: Dict[str, OptionVerdict] = {}
        for letter in sorted(options):
            description = options[letter]
            verdicts[letter] = self._verify_single(letter, description, constraints)

        # 第三阶段：封装富返回
        return ActionScopeReport(verdicts=verdicts)

    def _verify_single(
        self,
        letter: str,
        description: str,
        constraints: List[PatientConstraint],
    ) -> OptionVerdict:
        """
        单个选项的判定：委托 check_option，再映射为 OptionVerdict。

        参数：
            letter:      选项字母
            description: 选项描述文本
            constraints: 患者约束列表

        返回：
            OptionVerdict
        """
        action = self._extract_action(description)

        # 委托 κ-scorer 做两层判定（规则层 → 谓词层）
        kappa, scope_status, source = self._kappa_scorer.check_option(
            letter=letter,
            description=description,
            constraints=constraints,
            use_llm_layer=self._use_llm_layer,
        )

        verdict_status = _SCOPE_STATUS_TO_VERDICT.get(scope_status, "UNKNOWN")

        # 找出触发判定的约束（供审计与人类可读 reason）
        matched_constraint: Optional[str] = self._find_matched_constraint(
            verdict_status, constraints
        )
        reason = self._build_reason(verdict_status, source, kappa, matched_constraint)

        return OptionVerdict(
            letter=letter,
            action=action,
            description=description,
            status=verdict_status,  # type: ignore[arg-type]
            kappa=kappa,
            reason=reason,
            source=source,  # type: ignore[arg-type]
            matched_constraint=matched_constraint,
        )

    @staticmethod
    def _find_matched_constraint(
        verdict_status: str,
        constraints: List[PatientConstraint],
    ) -> Optional[str]:
        """
        根据判定状态选取最可能触发的约束 raw_text（供审计）。

        绝对禁忌 → 取首个 ABSOLUTE 约束；相对/有条件 → 取首个 RELATIVE 约束。
        注意：这是面向可解释的近似归因，非严格逐对匹配（严格匹配需改 rule 层返回值）。
        """
        if verdict_status == "INADMISSIBLE_ABS":
            for c in constraints:
                if c.constraint_type == "ABSOLUTE":
                    return c.raw_text
        elif verdict_status == "CONDITIONALLY_ADMISSIBLE":
            for c in constraints:
                if c.constraint_type == "RELATIVE":
                    return c.raw_text
        return None

    @staticmethod
    def _build_reason(
        verdict_status: str,
        source: str,
        kappa: float,
        matched_constraint: Optional[str],
    ) -> str:
        """构造人类可读判定依据。"""
        base = f"status={verdict_status}, κ={kappa:.2f}, source={source}"
        if matched_constraint:
            return f"{base}; 触发约束: {matched_constraint}"
        if verdict_status == "ADMISSIBLE":
            return f"{base}; 无约束命中，判定可行"
        if verdict_status == "UNKNOWN":
            return f"{base}; 规则层与谓词层均无定论"
        return base

    @staticmethod
    def _extract_action(description: str) -> str:
        """
        从选项描述抽取主 action 关键词（药物/操作名），仅供 verdict 展示。

        策略（best-effort）：去剂量 token → 去给药途径/频次词 → 取剩余首个实词。
        抽不出时回退为原始描述。
        """
        text = _DOSAGE_PATTERN.sub(" ", description).strip().lower()
        tokens = [t for t in re.split(r"[\s,/()]+", text) if t]
        content_tokens = [t for t in tokens if t not in _ROUTE_FREQ_WORDS]
        if content_tokens:
            return content_tokens[0]
        return description.strip()
