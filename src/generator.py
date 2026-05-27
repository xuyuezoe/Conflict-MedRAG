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

from src.types import ActionScopeReport, FCConflict, QueryDecomposition, RetrievedDoc


# ── Prompt 模板 ───────────────────────────────────────────────────────────────

SCOPE_ANCHORED_SYSTEM = """\
You are a clinical decision support system that selects the single best \
evidence-based treatment for a specific patient.

PRIMARY TASK: choose the one option that is the standard, guideline-appropriate \
treatment for this patient's condition AND is supported by the provided evidence.

Scope-safety is handled for you: the PER-OPTION SCOPE VERDICTS and SCOPE BIAS \
WARNING already identify options that are contraindicated for this patient. You \
MUST NOT choose any option flagged INADMISSIBLE_ABS, and you must mark such \
options AVOIDED.

You must:
1. Base your recommendation primarily on the provided evidence; cite the source chunk_id for key claims.
2. Pick the single best option for this patient's condition. Do NOT reject a clinically \
   standard, guideline-appropriate option unless it is flagged INADMISSIBLE_ABS or is clearly \
   contraindicated by the patient's profile.
3. per_action_status: mark exactly ONE option RECOMMENDED (your answer); mark AVOIDED only for \
   options that are flagged INADMISSIBLE_ABS or clearly contraindicated; use NOT_MENTIONED for \
   everything else. When unsure, prefer NOT_MENTIONED over AVOIDED — do NOT over-exclude options.
4. Return a JSON response in the specified format."""

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
{scope_verdict_block}
ANSWER OPTIONS:
{options_section}

DECISION GUIDANCE:
Pick the single option that is the standard, evidence-supported treatment for this \
patient's condition and is NOT flagged INADMISSIBLE_ABS. Options flagged INADMISSIBLE_ABS \
must be marked AVOIDED and must never be chosen. Do NOT mark an option AVOIDED merely \
because it is not your choice — only mark AVOIDED when it is flagged or clearly \
contraindicated; otherwise use NOT_MENTIONED.

TASK: Using the admissible evidence above, select the single best option for this patient.

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
        enforce_admissibility_gate: bool = True,
    ) -> None:
        """
        参数：
            client: Anthropic 客户端
            model:  主生成模型（默认 Sonnet 4.6，质量最高）
            enforce_admissibility_gate:
                    是否对最终答案启用硬门控：当 LLM 选中被判定为
                    INADMISSIBLE_ABS 的选项时，确定性改判为 κ 最高的可行选项，
                    并显式记录 gate_override（不静默）。False 用于"仅告知"消融实验。
        """
        self._client = client
        self._model = model
        self._enforce_admissibility_gate = enforce_admissibility_gate

    def generate(
        self,
        decomposition: QueryDecomposition,
        admissible_docs: List[RetrievedDoc],
        inadmissible_actions: List[str],
        fc_conflicts: List[FCConflict],
        patient_profile: Optional[Dict[str, Any]] = None,
        options_text: str = "",
        action_scope_report: Optional[ActionScopeReport] = None,
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
            action_scope_report:  逐选项适用性判定报告（可选）
                                  提供时：注入 prompt、确定性覆盖 per_action_status
                                  安全维度、并对最终答案做硬门控

        返回：
            (answer_text, per_action_status, attribution_list)
            answer_text:        最终推荐文本（经硬门控后的最终选项字母）
            per_action_status:  {选项字母: "RECOMMENDED"/"AVOIDED"/"NOT_MENTIONED"}
            attribution_list:   来源归因列表（含 gate_override 审计项，若发生改判）

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
        #         若有逐选项判定报告，给每个选项附确定性判定标注
        options_section, per_action_status_template = self._format_options(
            options_text, action_scope_report
        )

        # 第六步：构造逐选项判定块（含硬约束指令）
        scope_verdict_block = self._format_scope_verdict_block(action_scope_report)

        # 第七步：调用 LLM 生成
        prompt = SCOPE_ANCHORED_USER.format(
            patient_profile=profile_text,
            admissible_context=context_text,
            inadmissible_actions=warning_text,
            fc_conflict_notes=fc_notes,
            scope_verdict_block=scope_verdict_block,
            options_section=options_section,
            per_action_status_template=per_action_status_template,
        )
        if not prompt:
            raise ValueError("[Generator] prompt 为空，检查模板变量填充。")

        # 第八步：调用 LLM 并解析 JSON（chat_json 带有界重试，最终失败仍抛错，不兜底）
        data, raw = self._client.chat_json(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32000,
            system=SCOPE_ANCHORED_SYSTEM,
        )
        answer_text, per_action_status, attribution = self._parse_output(
            data, raw, decomposition.original_query
        )

        # 第九步：用判定报告做确定性后处理（安全维度覆盖 + 最终答案硬门控）
        return self._apply_scope_report(
            answer_text, per_action_status, attribution, action_scope_report
        )

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

        优先使用约束的 raw_text（ConstraintExpander 提供完整类别展开信息）：
          - "Patient is allergic to tetracycline class (includes: doxycycline, minocycline, ...)"
          - "Patient is pregnant — absolutely contraindicated: warfarin, isotretinoin, ..."

        此格式帮助生成器精确识别禁忌药物，避免将非同类药（如 azithromycin 是大环内酯，
        不是四环素）误判为禁忌（MACB-008 根因修复）。
        """
        if not inadmissible_actions and not decomposition.has_absolute_constraint:
            return "None (no absolute contraindications identified)"

        lines = []
        covered_actions: set = set()

        # 优先展示约束的完整 raw_text（含类别成员展开）
        for c in decomposition.constraints:
            if c.constraint_type == "ABSOLUTE":
                lines.append(f"  - CONTRAINDICATED: {c.raw_text}")
                covered_actions.add(c.target_action.lower())

        # 补充 inadmissible_actions 中未被约束覆盖的项（向后兼容旧 QueryDecomposer 路径）
        for action in inadmissible_actions:
            if action.lower() not in covered_actions:
                lines.append(f"  - {action} (ABSOLUTELY CONTRAINDICATED for this patient)")

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

    def _format_options(
        self,
        options_text: str,
        action_scope_report: Optional[ActionScopeReport] = None,
    ) -> tuple:
        """
        从 "A: desc | B: desc | ..." 格式解析选项，返回：
          (options_section, per_action_status_template)

        options_section:          展示给 LLM 的选项列表文本
        per_action_status_template: JSON 模板中每个选项的 key 行（缩进 4 空格）

        若提供 action_scope_report，给每个选项行追加确定性判定标注，例如：
          B: warfarin 5 mg  [INADMISSIBLE_ABS κ=0.00 — 触发约束: ...]

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

        verdicts = action_scope_report.verdicts if action_scope_report else {}
        section_lines = []
        for letter, desc in sorted(options.items()):
            verdict = verdicts.get(letter)
            if verdict is not None:
                section_lines.append(
                    f"{letter}: {desc}  "
                    f"[{verdict.status} κ={verdict.kappa:.2f} — {verdict.reason}]"
                )
            else:
                section_lines.append(f"{letter}: {desc}")

        template_lines = [
            f'    "{letter}": "RECOMMENDED" or "AVOIDED" or "NOT_MENTIONED"'
            for letter in sorted(options)
        ]
        return "\n".join(section_lines), ",\n".join(template_lines)

    def _format_scope_verdict_block(
        self,
        action_scope_report: Optional[ActionScopeReport],
    ) -> str:
        """
        构造逐选项判定块（含硬约束指令）。

        无报告时返回空串（向后兼容：prompt 退化为原始无判定形态）。
        """
        if action_scope_report is None or not action_scope_report.verdicts:
            return ""

        lines = [
            "PER-OPTION SCOPE VERDICTS (deterministic, computed from patient constraints):"
        ]
        for letter in sorted(action_scope_report.verdicts):
            verdict = action_scope_report.verdicts[letter]
            lines.append(
                f"  {letter}: {verdict.status} (κ={verdict.kappa:.2f}) — {verdict.reason}"
            )
        lines.append("")
        lines.append(
            "HARD CONSTRAINT: You MUST NOT select an option marked INADMISSIBLE_ABS "
            "as your \"answer\". You MUST mark every INADMISSIBLE_ABS option as "
            "\"AVOIDED\" in per_action_status. For CONDITIONALLY_ADMISSIBLE options, "
            "prefer a fully ADMISSIBLE option when one exists."
        )
        return "\n".join(lines) + "\n"

    def _parse_output(
        self,
        data: Dict[str, Any],
        raw_output: str,
        original_query: str,
    ) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
        """
        校验并提取已解析的 LLM JSON 输出的字段。

        参数：
            data:           已由 chat_json 解析的 dict
            raw_output:     原始输出文本（仅用于错误信息）
            original_query: 原始查询（仅用于错误信息）

        异常：
            ValueError: 缺少必要字段（不兜底）
        """
        if "answer" not in data:
            raise ValueError(
                f"[Generator] LLM 输出缺少 'answer' 字段。"
                f"  原始输出: {raw_output[:300]}"
            )

        answer_text: str = data["answer"]
        per_action_status: Dict[str, str] = data.get("per_action_status", {})
        attribution: List[Dict[str, Any]] = data.get("attribution", [])

        return answer_text, per_action_status, attribution

    @staticmethod
    def _normalize_answer_letter(answer_text: str) -> Optional[str]:
        """
        从答案文本提取选项字母（A–E）。

        兼容 "D"、"D: ..."、"Option D" 等形式；无法提取时返回 None。
        """
        import re as _re

        match = _re.search(r"\b([A-E])\b", answer_text.strip())
        return match.group(1) if match else None

    def _apply_scope_report(
        self,
        answer_text: str,
        per_action_status: Dict[str, str],
        attribution: List[Dict[str, Any]],
        action_scope_report: Optional[ActionScopeReport],
    ) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
        """
        用逐选项判定报告做确定性后处理。

        两步（混合：确定性主导安全维度，LLM 主导可行集内择优）：
          1. per_action_status 安全维度覆盖：把所有 INADMISSIBLE_ABS /
             CONDITIONALLY_ADMISSIBLE 选项强制标为 "AVOIDED"（确定性、可解释）。
             RECOMMENDED / NOT_MENTIONED 仍保留 LLM 对可行选项的判断。
          2. 最终答案硬门控（enforce_admissibility_gate=True 时）：
             若 LLM 答案落在绝对禁忌选项上，改判为 κ 最高的可行选项，
             并在 attribution 追加显式 gate_override 审计项（不静默）。
             若无任何可行选项，显式抛错（不编造）。

        无报告时原样返回（向后兼容）。
        """
        if action_scope_report is None or not action_scope_report.verdicts:
            return answer_text, per_action_status, attribution

        # 第一步：安全维度确定性覆盖
        for letter, verdict in action_scope_report.verdicts.items():
            if verdict.status in {"INADMISSIBLE_ABS", "CONDITIONALLY_ADMISSIBLE"}:
                per_action_status[letter] = "AVOIDED"

        # 第二步：最终答案硬门控
        if not self._enforce_admissibility_gate:
            return answer_text, per_action_status, attribution

        answer_letter = self._normalize_answer_letter(answer_text)
        inadmissible = set(action_scope_report.inadmissible_letters)
        if answer_letter is None or answer_letter not in inadmissible:
            # 未选中禁忌项，无需改判
            return answer_text, per_action_status, attribution

        # LLM 选中了绝对禁忌项 → 改判为最优"非禁忌"选项
        # （优先 ADMISSIBLE，其次 CONDITIONALLY，再次 UNKNOWN/NOT_APPLICABLE）
        redirect_letters = action_scope_report.gate_redirect_letters
        if not redirect_letters:
            raise ValueError(
                f"[Generator] 硬门控：LLM 答案 '{answer_letter}' 为绝对禁忌，"
                f"但全部选项均为绝对禁忌，无可改判项（不编造答案）。"
                f"verdicts={ {l: v.status for l, v in action_scope_report.verdicts.items()} }"
            )

        new_letter = redirect_letters[0]
        new_verdict = action_scope_report.verdicts[new_letter]
        # 显式记录改判审计（不静默）
        attribution.append({
            "gate_override": {
                "from": answer_letter,
                "to": new_letter,
                "reason": (
                    f"LLM 选中绝对禁忌选项 {answer_letter}"
                    f"（{action_scope_report.verdicts[answer_letter].reason}）；"
                    f"改判为 κ 最高的可行选项 {new_letter}（κ={new_verdict.kappa:.2f}）"
                ),
            }
        })
        # 同步 per_action_status：被改判到的选项标 RECOMMENDED
        per_action_status[new_letter] = "RECOMMENDED"
        return new_letter, per_action_status, attribution
