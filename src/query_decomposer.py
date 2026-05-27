#!/usr/bin/env python3
"""
Module 0：Query Decomposer

将自然语言 query 分解为 (D_q, C_q) 两个正交组件。

数学目标（research.md §3.5.1）：
  DCR 评分函数需要 D_q 和 C_q 独立计算：
    score(q, d) = sim(D_q, d) · κ(C_q, π_d)
  若将 C_q（患者约束）混入 D_q 的 embedding，约束信息在高维空间中被疾病语义稀释。
  因此必须将 query 分解为正交的两部分。

实现：
  使用 LLM（Haiku）进行结构化信息提取，返回 JSON 格式。
  缓存：同一 query 的分解结果用 MD5(query) 作为 key 缓存到磁盘，避免重复 API 调用。
  错误处理：JSON 解析失败直接抛出 ValueError（禁止兜底逻辑）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm_client import LLMClient

from src.types import PatientConstraint, QueryDecomposition


# ── Prompt 模板 ───────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM_PROMPT = """\
You are a clinical scope analyst. Decompose a clinical vignette into two orthogonal \
components used by a scope-aware retrieval system:

1. disease_query (D): a concise description of the disease/condition that seeks the \
   STANDARD treatment, with ALL patient-specific constraints removed (no pregnancy, \
   no organ impairment, no allergy, no comorbidity qualifiers). This is used for \
   evidence retrieval, so it must describe only the condition.

2. constraints (C): the list of patient-specific facts that make certain treatments \
   inadmissible or restricted FOR THIS PATIENT. For each constraint, also name the \
   specific drugs / drug-classes that are contraindicated by that patient state, \
   using established clinical pharmacology knowledge.

You MUST classify each constraint into exactly one category:
  - PREGNANCY                    (current pregnancy; teratogen contraindications)
  - LACTATION                    (breastfeeding; drugs unsafe in breast milk)
  - AGE                          (neonate / infant / pediatric / geriatric dosing or bans)
  - RENAL_IMPAIRMENT             (reduced eGFR; renally-cleared/nephrotoxic drugs)
  - HEPATIC_IMPAIRMENT           (liver disease; hepatotoxic/hepatically-cleared drugs)
  - CARDIAC                      (e.g. heart failure, QT prolongation, arrhythmia bans)
  - ALLERGY                      (drug allergy / hypersensitivity; the allergen class)
  - COMORBIDITY_CONTRAINDICATION (a named disease that contraindicates a drug class,
                                  e.g. Parkinson disease → dopamine antagonists;
                                  asthma → non-selective beta-blockers; G6PD → oxidants)
  - DRUG_INTERACTION             (a current medication that contraindicates another drug)

Each constraint object:
{
  "category": "<one of the categories above>",
  "constraint_type": "ABSOLUTE" | "RELATIVE",
  "patient_state": "<the specific patient fact, e.g. 'first-trimester pregnancy', 'eGFR 25 mL/min', 'Parkinson disease'>",
  "contraindicated_targets": ["<drug or drug-class name>", ...],
  "parameter_value": <number or null>,
  "parameter_threshold": <number or null>
}

constraint_type:
  - ABSOLUTE: a true absolute contraindication (κ=0, the drug must NOT be used).
  - RELATIVE: requires dose adjustment / caution but not an absolute ban (κ in (0,1)).
    For RENAL/HEPATIC with a numeric lab, set parameter_value (patient value, e.g. eGFR 25 → 25.0)
    and parameter_threshold (the safety cutoff, e.g. 30.0). Leave null if no number is stated.

contraindicated_targets RULES:
  - List specific drug names or recognizable drug-class names (e.g. "warfarin",
    "methotrexate", "methimazole", "ACE inhibitors", "NSAIDs", "tetracyclines",
    "dopamine antagonists", "metoclopramide").
  - Include the well-established contraindicated drugs for this patient state even if
    they are not explicitly mentioned in the vignette (use medical knowledge).
  - Do NOT list a drug unless it is genuinely contraindicated/restricted by this state.

STRICT DISCIPLINE (avoid false contraindications):
  - Only extract constraints that are explicitly supported by facts in the vignette.
  - When severity is uncertain, prefer RELATIVE over ABSOLUTE (conservative).
  - Do NOT extract "wrong-treatment-for-this-disease" as a constraint — that is a
    disease-appropriateness issue, NOT a patient scope constraint. Constraints are
    about patient state making an otherwise-reasonable drug unsafe.

Return ONLY valid JSON:
{
  "disease_query": "<standard treatment for the condition, constraint-free>",
  "constraints": [ ... ]
}
If the patient has no scope constraints, return "constraints": [].
"""

DECOMPOSE_USER_TEMPLATE = """\
Clinical vignette:
{query}

Decompose into disease_query (D) and constraints (C) as specified. Return JSON only."""


# ── 主类 ─────────────────────────────────────────────────────────────────────

class QueryDecomposer:
    """
    Module 0：自然语言医学 query 分解器。

    将含有疾病描述和患者约束的混合 query 分解为：
      D_q (disease_query): 纯疾病查询，用于 sim(D_q, d) 计算
      C_q (constraints):   患者约束结构化列表，用于 κ(C_q, π_d) 计算

    设计决策：
      - LLM 调用而非规则：医学 query 结构复杂，规则覆盖率有限
      - 缓存磁盘结果：避免评估阶段重复调用（同一 query 评估多个系统时）
      - 严格 JSON 验证：不接受格式错误输出（禁止兜底降级）
    """

    def __init__(
        self,
        client: LLMClient,
        model: str = "claude-haiku-4-5-20251001",
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        参数：
            client:    Anthropic API 客户端
            model:     分解模型 ID（默认 Haiku，成本低）
            cache_dir: 缓存目录（None 表示不缓存）
        """
        self._client = client
        self._model = model
        self._cache_dir = cache_dir
        if cache_dir:
            (cache_dir / "decompositions").mkdir(parents=True, exist_ok=True)

    def decompose(
        self,
        query: str,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> QueryDecomposition:
        """
        将自然语言 query 分解为 (D_q, C_q)。

        参数：
            query:           原始自然语言医学查询（含疾病描述+患者约束）
            patient_profile: 结构化患者 profile（可选）；若提供，从中提取生理约束，
                             比纯叙事文本提取更精确，避免隐式约束遗漏。
                             只使用 allergies/pregnancy/renal_impairment/hepatic_impairment 字段。

        返回：
            QueryDecomposition（含 disease_query 和 constraints 列表）

        异常：
            ValueError:  LLM 输出不符合预期 JSON 格式（不兜底，直接抛出）
            anthropic.APIError: API 调用失败
        """
        # 第一步：检查缓存（缓存键包含 profile 哈希，不同 profile 不共享缓存）
        cached = self._load_cache(query, patient_profile)
        if cached is not None:
            return cached

        # 第二步：调用 LLM 分解
        raw_output = self._call_llm(query, patient_profile)

        # 第三步：解析 JSON（失败则抛出，不兜底）
        decomposition = self._parse_output(query, raw_output)

        # 第四步：写入缓存
        self._save_cache(query, decomposition, patient_profile)

        return decomposition

    def _build_profile_json(
        self, patient_profile: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        从 patient_profile 中提取生理约束字段，序列化为 JSON 字符串。

        只使用以下字段（生理性约束，可靠地映射为 PatientConstraint）：
          allergies, pregnancy, renal_impairment, hepatic_impairment

        忽略 other_constraints（临床表现，语义模糊，易引发误提取）。
        """
        if not patient_profile:
            return None
        relevant = {
            k: patient_profile.get(k)
            for k in ("allergies", "pregnancy", "renal_impairment", "hepatic_impairment")
        }
        # 所有字段均为 null 或空列表时无需附加 profile
        has_content = any(
            v not in (None, [], False)
            for v in relevant.values()
        )
        if not has_content:
            return None
        return json.dumps(relevant, ensure_ascii=False, indent=2)

    @staticmethod
    def _strip_options(query: str) -> str:
        """
        去除 query 末尾的 MCQ 选项块，只保留临床主诉 stem。

        约束抽取只应基于患者临床事实，不应被候选选项影响（保持 D/C 与选项正交，
        且使设计可迁移到无结构化 profile 的全量 MedQA）。
        """
        if "\n\nOptions: " in query:
            return query.split("\n\nOptions: ", 1)[0]
        return query

    def _call_llm(
        self,
        query: str,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        调用 LLM 执行全上下文结构化分解，返回原始文本响应。

        输入仅为临床主诉 stem（去除 MCQ 选项），不依赖 patient_profile，
        以保证设计可迁移到全量 MedQA（无人工标注 profile 的场景）。
        """
        stem = self._strip_options(query)
        user_content = DECOMPOSE_USER_TEMPLATE.format(query=stem)
        return self._client.chat(
            messages=[{"role": "user", "content": user_content}],
            max_tokens=32000,
            system=DECOMPOSE_SYSTEM_PROMPT,
        )

    def _parse_output(self, original_query: str, raw_output: str) -> QueryDecomposition:
        """
        解析 LLM 输出的 JSON，构造 QueryDecomposition。

        参数：
            original_query: 原始查询（用于错误信息）
            raw_output:     LLM 原始文本输出

        异常：
            ValueError: JSON 格式非法或缺少必要字段
        """
        # 提取 JSON 块（LLM 有时会在 JSON 前后添加说明文字或代码块标记）
        if not raw_output:
            raise ValueError(
                f"[QueryDecomposer] LLM 返回空响应（推理模型 token 耗尽）。\n"
                f"  查询（前 100 字）: {original_query[:100]}"
            )

        json_str = raw_output
        if "```json" in raw_output:
            start = raw_output.index("```json") + 7
            end = raw_output.rindex("```")
            json_str = raw_output[start:end].strip()
        elif "```" in raw_output:
            start = raw_output.index("```") + 3
            end = raw_output.rindex("```")
            json_str = raw_output[start:end].strip()
        else:
            # 无代码块时，使用正则提取最外层 {} 对象（兼容 MiniMax 无 fence 格式）
            import re as _re
            m = _re.search(r"\{.*\}", raw_output, _re.DOTALL)
            if m:
                json_str = m.group()

        try:
            data: Dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[QueryDecomposer] LLM 输出 JSON 解析失败。\n"
                f"  查询（前 100 字）: {original_query[:100]}\n"
                f"  原始输出（前 200 字）: {raw_output[:200]}\n"
                f"  解析错误: {e}"
            )

        # 校验必要字段
        if "disease_query" not in data:
            raise ValueError(
                f"[QueryDecomposer] LLM 输出缺少 'disease_query' 字段。"
                f"  原始输出: {raw_output[:200]}"
            )
        if "constraints" not in data or not isinstance(data["constraints"], list):
            raise ValueError(
                f"[QueryDecomposer] LLM 输出缺少 'constraints' 字段或类型非数组。"
                f"  原始输出: {raw_output[:200]}"
            )

        # 合法约束类别枚举
        valid_categories = {
            "PREGNANCY", "LACTATION", "AGE", "RENAL_IMPAIRMENT", "HEPATIC_IMPAIRMENT",
            "CARDIAC", "ALLERGY", "COMORBIDITY_CONTRAINDICATION", "DRUG_INTERACTION",
            "UNSPECIFIED",
        }

        # 解析约束列表（新 schema：category/constraint_type/patient_state/contraindicated_targets）
        constraints: List[PatientConstraint] = []
        for i, c in enumerate(data["constraints"]):
            # 严重度：兼容旧字段名 type
            ctype = (c.get("constraint_type") or c.get("type") or "NONE").upper()
            if ctype not in {"ABSOLUTE", "RELATIVE", "NONE"}:
                raise ValueError(
                    f"[QueryDecomposer] 约束[{i}] constraint_type 值非法: {repr(ctype)}。"
                    f"仅接受 ABSOLUTE/RELATIVE/NONE。"
                )
            # NONE 约束不参与 κ 计算，直接跳过
            if ctype == "NONE":
                continue

            category = (c.get("category") or "UNSPECIFIED").upper()
            if category not in valid_categories:
                raise ValueError(
                    f"[QueryDecomposer] 约束[{i}] category 值非法: {repr(category)}。"
                    f"  合法值: {sorted(valid_categories)}"
                )

            patient_state = (c.get("patient_state") or c.get("text") or "").strip()
            if not patient_state:
                raise ValueError(
                    f"[QueryDecomposer] 约束[{i}] 缺少 patient_state 字段。约束数据: {c}"
                )

            # 禁忌药物/药类名列表（patient×drug 绑定的显式载体）
            raw_targets = c.get("contraindicated_targets", [])
            contraindicated_targets: List[str] = []
            if isinstance(raw_targets, list):
                contraindicated_targets = [
                    t.strip() for t in raw_targets if isinstance(t, str) and t.strip()
                ]

            # target_action 保留为类别关键词（供规则层快速匹配与向后兼容）
            target_action = category.lower()

            param_val: Optional[float] = None
            param_thr: Optional[float] = None
            if ctype == "RELATIVE":
                raw_val = c.get("parameter_value")
                raw_thr = c.get("parameter_threshold")
                # 数值参数为可选增强字段：缺失时 KappaScorer 回退定性评估（κ=0.5）
                if raw_val is not None:
                    try:
                        param_val = float(raw_val)
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"[QueryDecomposer] RELATIVE 约束[{i}] parameter_value 类型错误: {e}，约束: {c}"
                        )
                if raw_thr is not None:
                    try:
                        param_thr = float(raw_thr)
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"[QueryDecomposer] RELATIVE 约束[{i}] parameter_threshold 类型错误: {e}，约束: {c}"
                        )

            constraints.append(PatientConstraint(
                constraint_type=ctype,  # type: ignore[arg-type]
                target_action=target_action,
                raw_text=patient_state,
                parameter_value=param_val,
                parameter_threshold=param_thr,
                category=category,
                contraindicated_targets=contraindicated_targets,
            ))

        return QueryDecomposition(
            original_query=original_query,
            disease_query=data["disease_query"].strip(),
            constraints=constraints,
            decompose_model=self._model,
            candidate_actions=[],
            debug={"raw_llm_output": raw_output},
        )

    def _cache_key(
        self,
        query: str,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成缓存键（MD5）。

        新架构为 query-only（不依赖 profile），缓存键基于去选项的 stem。
        前缀 schema 版本标签 "v2scope"，使旧 schema 缓存自动失效（键不同 → cache miss）。
        """
        stem = self._strip_options(query)
        return hashlib.md5(("v2scope|" + stem).encode("utf-8")).hexdigest()

    def _load_cache(
        self,
        query: str,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[QueryDecomposition]:
        """
        从磁盘加载缓存的分解结果。
        缓存未命中时返回 None（不抛出异常）。
        """
        if self._cache_dir is None:
            return None
        cache_file = self._cache_dir / "decompositions" / f"{self._cache_key(query, patient_profile)}.json"
        if not cache_file.exists():
            return None

        data = json.loads(cache_file.read_text(encoding="utf-8"))
        constraints = [
            PatientConstraint(
                constraint_type=c["constraint_type"],
                target_action=c["target_action"],
                raw_text=c["raw_text"],
                parameter_value=c.get("parameter_value"),
                parameter_threshold=c.get("parameter_threshold"),
                category=c.get("category", "UNSPECIFIED"),
                contraindicated_targets=c.get("contraindicated_targets", []),
            )
            for c in data["constraints"]
        ]
        return QueryDecomposition(
            original_query=data["original_query"],
            disease_query=data["disease_query"],
            constraints=constraints,
            decompose_model=data["decompose_model"],
            # 向后兼容：旧缓存没有 candidate_actions 字段
            candidate_actions=data.get("candidate_actions", []),
            debug=data.get("debug", {}),
        )

    def _save_cache(
        self,
        query: str,
        decomp: QueryDecomposition,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将分解结果写入磁盘缓存"""
        if self._cache_dir is None:
            return
        cache_file = self._cache_dir / "decompositions" / f"{self._cache_key(query, patient_profile)}.json"
        data = {
            "original_query": decomp.original_query,
            "disease_query": decomp.disease_query,
            "candidate_actions": decomp.candidate_actions,
            "constraints": [
                {
                    "constraint_type": c.constraint_type,
                    "target_action": c.target_action,
                    "raw_text": c.raw_text,
                    "parameter_value": c.parameter_value,
                    "parameter_threshold": c.parameter_threshold,
                    "category": c.category,
                    "contraindicated_targets": c.contraindicated_targets,
                }
                for c in decomp.constraints
            ],
            "decompose_model": decomp.decompose_model,
            "debug": decomp.debug,
        }
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
