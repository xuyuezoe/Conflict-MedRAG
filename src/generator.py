#!/usr/bin/env python3
"""
scope-anchored 生成器

核心设计（research.md §4.6）：
  1. context 只包含 κ>0 的文献（DCR 已物理排除 INADMISSIBLE 文献）
  2. SCOPE BIAS WARNING 显式列出 INADMISSIBLE actions
     → 抑制 P_LLM(a|D) 的参数记忆偏置（Experiment 2 测量的边缘分布偏差）
  3. 要求对每个 claim 给出来源 chunk_id（供 Verifier 计算 SLR）

关键问题（research.md §4.6 中明确）：
  即使 context 已经不含 INADMISSIBLE 文献，
  LLM 的参数记忆仍可能"注入"一般人群推荐（context-memory 冲突）。
  SCOPE BIAS WARNING 是对参数记忆偏置的显式对抗。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from src.llm_client import LLMClient

from src.types import FCConflict, QueryDecomposition, RetrievedDoc


# ── Prompt 模板 ───────────────────────────────────────────────────────────────

SCOPE_ANCHORED_SYSTEM = """\
You are a clinical decision support system that provides \
evidence-based treatment recommendations.

CRITICAL INSTRUCTION: You MUST ONLY recommend treatments that are safe \
and admissible for this specific patient. The evidence provided to you \
has already been filtered for scope compatibility with this patient.

You must:
1. Base your recommendation ONLY on the provided evidence passages
2. Cite the source chunk_id for each key claim
3. Never recommend treatments listed in the SCOPE BIAS WARNING
4. Return a JSON response in the specified format"""

SCOPE_ANCHORED_USER = """\
PATIENT PROFILE:
{patient_profile}

ADMISSIBLE EVIDENCE (scope-verified, κ > 0, safe for this patient):
{admissible_context}

SCOPE BIAS WARNING:
The following treatments are CONTRAINDICATED for this patient. \
Standard medical training may suggest these for the general population, \
but they are NOT applicable to this specific patient and MUST NOT appear \
in your recommendation:
{inadmissible_actions}

FC CONFLICT NOTES (if any):
{fc_conflict_notes}

ANSWER OPTIONS:
{options_section}

TASK: Based ONLY on the admissible evidence above, select the single best \
option for this patient. Classify EVERY option listed above.

Return JSON only:
{{
  "answer": "<option letter, e.g. D>",
  "reasoning": "<1-2 sentence rationale citing evidence>",
  "per_action_status": {{
{per_action_status_template}
  }},
  "attribution": [
    {{
      "claim": "<key claim>",
      "source_chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"]
    }}
  ]
}}"""


class ScopeAnchoredGenerator:
    """
    scope-anchored 生成器。

    接收 DCR+SCSR 过滤后的 admissible 文档，生成范围安全的治疗推荐。

    输出格式（JSON）：
      answer:            最终推荐文本
      reasoning:         推理依据
      per_action_status: 各 action 的状态预测（用于 CRR/SDR 计算）
      attribution:       每个 claim 的来源 chunk_id（用于 SLR 计算）
    """

    def __init__(
        self,
        client: LLMClient,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        """
        参数：
            client: Anthropic 客户端
            model:  主生成模型（默认 Sonnet 4.6，质量最高）
        """
        self._client = client
        self._model = model

    def generate(
        self,
        decomposition: QueryDecomposition,
        admissible_docs: List[RetrievedDoc],
        inadmissible_actions: List[str],
        fc_conflicts: List[FCConflict],
        patient_profile: Optional[Dict[str, Any]] = None,
        options_text: str = "",
    ) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
        """
        生成 scope-anchored 推荐。

        参数：
            decomposition:        Module 0 分解结果（含原始查询和约束）
            admissible_docs:      admissible 文档列表（全部 κ>0）
            inadmissible_actions: 绝对禁忌 action 名称列表（SCOPE BIAS WARNING 内容）
            fc_conflicts:         FC 冲突列表（生成时提供上下文提示）
            patient_profile:      患者 profile 字典（可选，若为 None 从 constraints 构造）
            options_text:         MedQA 选项文本（A: ... | B: ...），用于输出对齐

        返回：
            (answer_text, per_action_status, attribution_list)
            answer_text:        最终推荐文本
            per_action_status:  {选项字母: "RECOMMENDED"/"AVOIDED"/"NOT_MENTIONED"}
            attribution_list:   来源归因列表

        异常：
            ValueError: LLM 输出格式非法（不兜底）
        """
        # 第一步：构造患者 profile 描述
        profile_text = self._format_patient_profile(decomposition, patient_profile)

        # 第二步：构造 context（admissible 文档摘要）
        context_text = self._format_context(admissible_docs)

        # 第三步：构造 SCOPE BIAS WARNING
        warning_text = self._format_scope_warning(inadmissible_actions, decomposition)

        # 第四步：构造 FC conflict notes
        fc_notes = self._format_fc_notes(fc_conflicts)

        # 第五步：解析选项字母并构造 per_action_status 模板（与 gold 对齐）
        options_section, per_action_status_template = self._format_options(options_text)

        # 第六步：调用 LLM 生成
        prompt = SCOPE_ANCHORED_USER.format(
            patient_profile=profile_text,
            admissible_context=context_text,
            inadmissible_actions=warning_text,
            fc_conflict_notes=fc_notes,
            options_section=options_section,
            per_action_status_template=per_action_status_template,
        )
        if not prompt:
            raise ValueError("[Generator] prompt 为空，检查模板变量填充。")

        raw = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32000,
            system=SCOPE_ANCHORED_SYSTEM,
        )

        if not raw:
            raise ValueError(
                f"[Generator] LLM 返回空响应（推理模型 token 耗尽）。\n"
                f"  查询（前 80 字）: {decomposition.original_query[:80]}"
            )

        # 第七步：解析输出 JSON（失败则抛出，不兜底）
        return self._parse_output(raw, decomposition.original_query)

    def _format_patient_profile(
        self,
        decomposition: QueryDecomposition,
        patient_profile: Optional[Dict[str, Any]],
    ) -> str:
        """
        构造患者 profile 描述文本。

        优先使用传入的 patient_profile dict；若为 None，从约束列表构造。
        """
        if patient_profile:
            return json.dumps(patient_profile, ensure_ascii=False, indent=2)

        lines = [f"Query: {decomposition.original_query}"]
        if decomposition.constraints:
            lines.append("Constraints:")
            for c in decomposition.constraints:
                lines.append(f"  - [{c.constraint_type}] {c.raw_text}")
        return "\n".join(lines)

    def _format_context(self, admissible_docs: List[RetrievedDoc]) -> str:
        """
        将 admissible 文档格式化为 context 文本。

        每篇文档附带 chunk_id（用于 LLM 归因）和 κ 值（供审计）。
        """
        if not admissible_docs:
            return "[NO ADMISSIBLE EVIDENCE FOUND - respond based on query only]"

        parts = []
        for i, doc in enumerate(admissible_docs, start=1):
            parts.append(
                f"[Document {i}] chunk_id: {doc.chunk.chunk_id} | "
                f"source: {doc.chunk.source_book} | κ={doc.kappa:.2f}\n"
                f"{doc.chunk.text[:600]}"
            )
        return "\n\n".join(parts)

    def _format_scope_warning(
        self,
        inadmissible_actions: List[str],
        decomposition: QueryDecomposition,
    ) -> str:
        """
        构造 SCOPE BIAS WARNING 文本。

        将绝对禁忌的 action 和约束原文结合，强调不可推荐。
        """
        if not inadmissible_actions and not decomposition.has_absolute_constraint:
            return "None (no absolute contraindications identified)"

        lines = []
        for action in inadmissible_actions:
            lines.append(f"  - {action} (ABSOLUTELY CONTRAINDICATED for this patient)")

        for c in decomposition.constraints:
            if c.constraint_type == "ABSOLUTE":
                lines.append(f"  - Reason: {c.raw_text}")

        return "\n".join(lines) if lines else "None"

    def _format_fc_notes(self, fc_conflicts: List[FCConflict]) -> str:
        """格式化 FC 冲突信息"""
        if not fc_conflicts:
            return "None"
        lines = []
        for cf in fc_conflicts:
            lines.append(
                f"  - Conflict on '{cf.action}': {cf.doc_a_id} vs {cf.doc_b_id} "
                f"→ resolution: {cf.resolution} ({cf.resolution_reason[:100]})"
            )
        return "\n".join(lines)

    def _format_options(self, options_text: str) -> tuple:
        """
        从 "A: desc | B: desc | ..." 格式解析选项，返回：
          (options_section, per_action_status_template)

        options_section:          展示给 LLM 的选项列表文本
        per_action_status_template: JSON 模板中每个选项的 key 行（缩进 4 空格）

        无选项时使用通用占位符，不影响生成，但 per_action_status 无法对齐 gold。
        """
        import re as _re

        options: Dict[str, str] = {}
        for part in options_text.split("|"):
            part = part.strip()
            if ": " in part:
                letter, desc = part.split(": ", 1)
                letter = letter.strip()
                if _re.match(r"^[A-E]$", letter):
                    options[letter] = desc.strip()

        if not options:
            return (
                "(no structured options provided)",
                '    "<option>": "RECOMMENDED" or "AVOIDED" or "NOT_MENTIONED"',
            )

        section_lines = [f"{letter}: {desc}" for letter, desc in sorted(options.items())]
        template_lines = [
            f'    "{letter}": "RECOMMENDED" or "AVOIDED" or "NOT_MENTIONED"'
            for letter in sorted(options)
        ]
        return "\n".join(section_lines), ",\n".join(template_lines)

    def _parse_output(
        self,
        raw_output: str,
        original_query: str,
    ) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
        """
        解析 LLM 生成的 JSON 输出。

        异常：
            ValueError: JSON 格式非法或缺少必要字段（不兜底）
        """
        json_str = raw_output
        if "```json" in raw_output:
            start = raw_output.index("```json") + 7
            end = raw_output.rindex("```")
            json_str = raw_output[start:end].strip()
        elif "```" in raw_output:
            start = raw_output.index("```") + 3
            end = raw_output.rindex("```")
            json_str = raw_output[start:end].strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[Generator] LLM 输出 JSON 解析失败。\n"
                f"  查询（前 80 字）: {original_query[:80]}\n"
                f"  原始输出（前 300 字）: {raw_output[:300]}\n"
                f"  解析错误: {e}"
            )

        if "answer" not in data:
            raise ValueError(
                f"[Generator] LLM 输出缺少 'answer' 字段。"
                f"  原始输出: {raw_output[:300]}"
            )

        answer_text: str = data["answer"]
        per_action_status: Dict[str, str] = data.get("per_action_status", {})
        attribution: List[Dict[str, Any]] = data.get("attribution", [])

        return answer_text, per_action_status, attribution
