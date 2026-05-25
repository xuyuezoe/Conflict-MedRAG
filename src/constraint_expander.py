#!/usr/bin/env python3
"""
Module 0B：ConstraintExpander — 规则化约束展开器

从 patient_profile 结构化字段确定性展开为 PatientConstraint 列表，零 LLM 调用。

设计原理：
  patient_profile 字段在 benchmark 中是结构化金标准（allergies, pregnancy 等）。
  这些字段→约束的映射是纯规则查表，与 LLM 非确定性无关。

  与 QueryDecomposer（LLM 从 narrative 提取约束）的核心差异：
    QueryDecomposer：每次调用可能提取不同数量/内容的约束（LLM 随机性）
    ConstraintExpander：同一 patient_profile 始终产生完全相同的约束列表（确定性）

  当 patient_profile 不可用时，可通过注入 QueryDecomposer 实例作为 fallback。

MACB-008 根因修复：
  原问题：生成器 SCOPE BIAS WARNING 只显示抽象类名（如 "tetracycline"），
          LLM 误将大环内酯类 azithromycin 归入四环素禁忌类别。
  修复：raw_text 展开具体成员药物（"tetracycline class (includes: doxycycline, ...)"），
        使生成器能准确区分四环素类与大环内酯类。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.types import PatientConstraint

if TYPE_CHECKING:
    from src.query_decomposer import QueryDecomposer


# ── 药物类别展开表 ──────────────────────────────────────────────────────────────
#
# 用途双重性：
#   1. 将 allergies 中的类名展开，使 raw_text 包含具体成员药物。
#      → 生成器 SCOPE BIAS WARNING 能列出具体禁忌药物，而非抽象类名。
#   2. 隐式分类信息：表中"大环内酯"不含四环素，反之亦然。
#      → 生成器能正确推断 azithromycin（大环内酯）不属于 tetracycline 禁忌范围。
#
# 与 kappa_scorer.ABSOLUTE_CONTRAINDICATION_RULES 的关系：
#   ABSOLUTE_CONTRAINDICATION_RULES：文档侧匹配（文档推荐了 class 成员 → κ=0）
#   DRUG_CLASS_EXPANSION：约束侧归一化（allergies → raw_text 展开）
#   两者互补，不重叠。

DRUG_CLASS_EXPANSION: Dict[str, List[str]] = {
    # 青霉素类（penicillin class）
    "penicillin": [
        "amoxicillin",
        "amoxicillin-clavulanate",
        "ampicillin",
        "ampicillin-sulbactam",
        "piperacillin",
        "piperacillin-tazobactam",
        "nafcillin",
        "oxacillin",
        "dicloxacillin",
    ],
    # 磺胺类（sulfonamide class）
    "sulfa": ["trimethoprim-sulfamethoxazole", "sulfamethoxazole"],
    "sulfonamide": ["trimethoprim-sulfamethoxazole", "sulfamethoxazole"],
    # 四环素类（tetracycline class）
    # 注意：大环内酯类（azithromycin）不属于四环素类，不在此列表中
    "tetracycline": ["doxycycline", "minocycline", "demeclocycline", "tetracycline"],
    # 大环内酯类（macrolide class）
    # 注意：四环素类（doxycycline）不属于大环内酯类
    "macrolide": ["azithromycin", "erythromycin", "clarithromycin"],
    "macrolides": ["azithromycin", "erythromycin", "clarithromycin"],
    # 氟喹诺酮类（fluoroquinolone class）
    "fluoroquinolone": [
        "ciprofloxacin",
        "levofloxacin",
        "moxifloxacin",
        "ofloxacin",
        "norfloxacin",
    ],
    "fluoroquinolones": [
        "ciprofloxacin",
        "levofloxacin",
        "moxifloxacin",
        "ofloxacin",
        "norfloxacin",
    ],
    "quinolone": ["ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin"],
    # 头孢菌素类（cephalosporin class）
    "cephalosporin": [
        "ceftriaxone",
        "cefixime",
        "cephalexin",
        "cefuroxime",
        "cefazolin",
        "cefdinir",
        "cefepime",
        "ceftazidime",
    ],
    # 氨基糖苷类（aminoglycoside class）
    "aminoglycoside": ["gentamicin", "tobramycin", "amikacin", "streptomycin"],
    # 阿片类（opioid class）
    "opioid": ["morphine", "oxycodone", "hydrocodone", "codeine", "fentanyl", "hydromorphone"],
    # NSAIDs
    "nsaid": ["ibuprofen", "naproxen", "indomethacin", "ketorolac", "celecoxib"],
    "nsaids": ["ibuprofen", "naproxen", "indomethacin", "ketorolac", "celecoxib"],
    # 造影剂（IV contrast）
    "iv contrast": ["iohexol", "iopamidol", "iodixanol"],
    "contrast": ["iohexol", "iopamidol", "iodixanol"],
}


# ── 妊娠绝对禁忌药物列表 ────────────────────────────────────────────────────────
#
# 与 kappa_scorer.ABSOLUTE_CONTRAINDICATION_RULES 中的 ("pregnancy", ...) 对保持一致。
# 仅用于在 raw_text 中展示，帮助生成器明确知晓哪些药物妊娠期禁用。

PREGNANCY_ABSOLUTE_CONTRAINDICATIONS: List[str] = [
    "warfarin",
    "isotretinoin",
    "thalidomide",
    "methotrexate",
    "tetracycline",
    "doxycycline",
    "minocycline",
    "fluoroquinolone",
    "ciprofloxacin",
    "levofloxacin",
    "moxifloxacin",
    "valproic acid",
    "carbamazepine",
    "phenytoin",
    "misoprostol",
    "ACE inhibitors",
    "angiotensin receptor blockers",
]


class ConstraintExpander:
    """
    Module 0B：规则化约束展开器。

    从 patient_profile 结构化字段展开为 PatientConstraint 列表，零 LLM 调用。

    核心设计：
      patient_profile 字段（allergies, pregnancy, renal_impairment 等）是结构化金标准。
      通过规则查表（DRUG_CLASS_EXPANSION + 禁忌列表）转换为 PatientConstraint，
      确保同一输入始终产生相同输出（确定性，可复现）。

    Fallback 链：
      1. patient_profile 结构化字段 → 规则展开（零 LLM，首选）
      2. narrative_fallback → 注入的 QueryDecomposer（有 LLM，兜底）
      3. 两者均无有效信息 → 返回空列表（无约束）

    使用方式：
        expander = ConstraintExpander(decomposer=decomposer)  # decomposer 可选
        constraints = expander.expand(patient_profile, narrative_fallback=query)
    """

    # 药物类别别名归一化表（用于统一 allergies 中的不同写法）
    _CLASS_ALIASES: Dict[str, str] = {
        "macrolides":       "macrolide",
        "fluoroquinolones": "fluoroquinolone",
        "quinolones":       "quinolone",
        "penicillins":      "penicillin",
        "tetracyclines":    "tetracycline",
        "cephalosporins":   "cephalosporin",
        "sulfonamides":     "sulfonamide",
        "aminoglycosides":  "aminoglycoside",
        "opioids":          "opioid",
    }

    def __init__(
        self,
        decomposer: Optional["QueryDecomposer"] = None,
    ) -> None:
        """
        参数：
            decomposer: QueryDecomposer 实例（可选 fallback）。
                        当 patient_profile 无结构化约束时，
                        通过 LLM 从 narrative 文本提取约束。
        """
        self._decomposer = decomposer

    def expand(
        self,
        patient_profile: Optional[Dict[str, Any]],
        narrative_fallback: Optional[str] = None,
    ) -> List[PatientConstraint]:
        """
        从 patient_profile 展开为 PatientConstraint 列表。

        参数：
            patient_profile:    患者 profile dict（allergies, pregnancy, 等字段）
            narrative_fallback: 原始自然语言 query（当 profile 无约束时，触发 LLM fallback）

        返回：
            PatientConstraint 列表，可直接传入 QueryDecomposition.constraints

        字段处理规则：
            allergies:         每项 → ABSOLUTE 约束（含类别展开 raw_text）
            pregnancy=True:    → ABSOLUTE 约束（列出关键妊娠禁忌药物）
            renal_impairment:  → RELATIVE 约束（含 eGFR 参数，若可用）
            hepatic_impairment:→ RELATIVE 约束
        """
        constraints: List[PatientConstraint] = []

        if patient_profile:
            # ── 1. 过敏约束 ──────────────────────────────────────────────────
            for allergy in patient_profile.get("allergies") or []:
                c = self._expand_allergy(allergy)
                if c is not None:
                    constraints.append(c)

            # ── 2. 妊娠禁忌 ──────────────────────────────────────────────────
            if patient_profile.get("pregnancy"):
                constraints.append(self._make_pregnancy_constraint())

            # ── 3. 肾功能损害 ────────────────────────────────────────────────
            if patient_profile.get("renal_impairment"):
                egfr_raw = (
                    patient_profile.get("egfr_value")
                    or patient_profile.get("egfr")
                )
                egfr = float(egfr_raw) if egfr_raw is not None else None
                constraints.append(
                    PatientConstraint(
                        constraint_type="RELATIVE",
                        target_action="renal",
                        raw_text=(
                            "Patient has renal impairment — dose adjustment or "
                            "avoidance needed for renally-cleared drugs"
                        ),
                        parameter_value=egfr,
                        parameter_threshold=30.0 if egfr is not None else None,
                    )
                )

            # ── 4. 肝功能损害 ────────────────────────────────────────────────
            if patient_profile.get("hepatic_impairment"):
                constraints.append(
                    PatientConstraint(
                        constraint_type="RELATIVE",
                        target_action="hepatic",
                        raw_text=(
                            "Patient has hepatic impairment — avoid hepatotoxic "
                            "drugs and adjust doses of hepatically-metabolized medications"
                        ),
                    )
                )

        # ── Fallback：无结构化约束时调 LLM ─────────────────────────────────────
        # 当 patient_profile 为 None 或未产生任何约束，且 narrative_fallback 可用
        if not constraints and narrative_fallback and self._decomposer is not None:
            decomp = self._decomposer.decompose(
                query=narrative_fallback,
                patient_profile=patient_profile,
            )
            return decomp.constraints

        return constraints

    def _normalize_drug_class(self, name: str) -> str:
        """归一化药物类名（统一复数/别名形式）"""
        lower = name.lower().strip()
        return self._CLASS_ALIASES.get(lower, lower)

    def _expand_allergy(self, allergy: str) -> Optional[PatientConstraint]:
        """
        将单个过敏项转换为 PatientConstraint。

        若过敏项是已知药物类别名，在 raw_text 中列出具体成员药物。
        这使生成器 SCOPE BIAS WARNING 能明确区分不同类别
        （如：四环素类 ≠ 大环内酯类，修复 MACB-008 中的 azithromycin 误判问题）。

        参数：
            allergy: 过敏项字符串（如 "tetracycline", "penicillin", "latex"）

        返回：
            PatientConstraint，或 None（若输入为空）
        """
        if not allergy or not allergy.strip():
            return None

        allergy_clean = allergy.strip()
        target = self._normalize_drug_class(allergy_clean)
        members = DRUG_CLASS_EXPANSION.get(target, [])

        if members:
            member_str = ", ".join(members)
            raw_text = (
                f"Patient is allergic to {allergy_clean} class "
                f"(includes: {member_str})"
            )
        else:
            raw_text = f"Patient is allergic to {allergy_clean}"

        return PatientConstraint(
            constraint_type="ABSOLUTE",
            target_action=target,
            raw_text=raw_text,
        )

    def _make_pregnancy_constraint(self) -> PatientConstraint:
        """
        构造妊娠绝对禁忌约束。

        在 raw_text 中明确列出关键禁忌药物，避免生成器依赖模糊的 "pregnancy" 关键词
        而遗漏或混淆具体禁忌药物。

        返回：
            PatientConstraint（ABSOLUTE，target_action="pregnancy"）
        """
        key_drugs = PREGNANCY_ABSOLUTE_CONTRAINDICATIONS[:8]
        drug_list = ", ".join(key_drugs)
        return PatientConstraint(
            constraint_type="ABSOLUTE",
            target_action="pregnancy",
            raw_text=(
                f"Patient is pregnant — absolutely contraindicated: "
                f"{drug_list} (and others)"
            ),
        )
