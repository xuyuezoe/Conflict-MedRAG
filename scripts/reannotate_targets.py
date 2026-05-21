#!/usr/bin/env python3
"""
重标注脚本：修复 answer_error、wrong_NA、FC_CAND 三类问题样本。

目标：
  - answer_error (6个)：answer_idx 不在 gold_admissible_set 中
  - wrong_NA (14个)：候选为治疗类但被错误标为 NOT_APPLICABLE
  - FC_CAND (40个)：无患者约束的治疗题，重标以提升质量

使用模型：MiniMax-M2.5（通过 .env 配置）
"""

import csv
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 第一阶段：配置 ──────────────────────────────────────────────────────────

MODEL: str = os.environ["LLM_MODEL"]
API_KEY: str = os.environ["LLM_API_KEY"]
BASE_URL: str = os.environ["LLM_BASE_URL"]
CSV_PATH: str = "annotations/macb_v3_sheet.csv"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── 第二阶段：Prompt 模板 ──────────────────────────────────────────────────

SC_PROMPT_TEMPLATE = """\
You are an expert medical annotator for the MACB benchmark dataset.

## Question
{question}

## Options
{options}

## Correct USMLE Answer
{answer_idx}: {answer}

## Candidate Tag
{candidate_tag}

## Prior Patient Profile (may be incomplete or wrong)
{patient_profile}

## Prior Disease Query
{parametric_prior}

## Your Task
Annotate this question for medical admissibility.

### Step 1: Determine question_type
- "TREATMENT": question asks what to prescribe/administer/give/do as therapy
- "DIAGNOSTIC": question asks for diagnosis, test, workup, cause identification only

### Step 2 (only if TREATMENT): Annotate admissibility
Extract patient constraints from the vignette:
- allergies: list of drug names the patient is allergic to
- pregnancy: true if patient is pregnant
- renal_impairment: "mild"/"moderate"/"severe"/null
- hepatic_impairment: "mild"/"moderate"/"severe"/null
- other_constraints: list of other relevant contraindications

For each option (A, B, C, D, E as applicable), assign:
- "ADMISSIBLE": safe and appropriate for this patient
- "INADMISSIBLE_ABS": absolutely contraindicated (allergy, pregnancy X, absolute CI)
- "INADMISSIBLE_REL": relatively contraindicated (dose reduction needed, caution)
- "NOT_APPLICABLE": not a treatment (is a test, lab, imaging, or non-drug)

### CRITICAL RULE
The correct USMLE answer ({answer_idx}) MUST have status "ADMISSIBLE".
If your analysis would make it inadmissible, re-examine your reasoning.

### gold_memory_conflict_label
- "CONFLICT": the parametric prior query suggests a drug that is INADMISSIBLE for this patient
- "NO_CONFLICT": the suggested drug is ADMISSIBLE or no scope constraints exist
- "NOT_APPLICABLE": question_type is DIAGNOSTIC

Return ONLY valid JSON (no markdown fence, no extra text):
{{
  "question_type": "TREATMENT" or "DIAGNOSTIC",
  "patient_profile": {{"allergies": [], "pregnancy": false, "renal_impairment": null, "hepatic_impairment": null, "age_years": null, "other_constraints": []}},
  "per_action_status": {{}},
  "gold_admissible_set": [],
  "gold_scsr_needed": true or false,
  "gold_memory_conflict_label": "CONFLICT" or "NO_CONFLICT" or "NOT_APPLICABLE",
  "annotation_confidence": "HIGH" or "MEDIUM" or "LOW",
  "annotation_notes": "one sentence"
}}"""

FC_CAND_PROMPT_TEMPLATE = """\
You are an expert medical annotator for the MACB benchmark dataset.

## Question
{question}

## Options
{options}

## Correct USMLE Answer
{answer_idx}: {answer}

## Context
This is an FC_CAND question: a treatment question about a patient WITHOUT any drug allergies, \
pregnancy, or significant organ impairment. There are NO patient-specific scope constraints.

## Your Task
For each option, determine medical correctness for the described condition:
- "ADMISSIBLE": medically valid/acceptable treatment for this condition
- "INADMISSIBLE_ABS": clearly wrong or contraindicated treatment for this condition (in general)
- "NOT_APPLICABLE": not a drug treatment (is a test, procedure, or non-treatment)

### CRITICAL RULE
The correct USMLE answer ({answer_idx}) MUST have status "ADMISSIBLE".

### Fixed fields for FC_CAND
- gold_scsr_needed: always false (no patient constraints)
- gold_memory_conflict_label: always "NO_CONFLICT" (no scope conflict)

Return ONLY valid JSON (no markdown fence, no extra text):
{{
  "question_type": "TREATMENT",
  "patient_profile": {{"allergies": [], "pregnancy": false, "renal_impairment": null, "hepatic_impairment": null, "age_years": null, "other_constraints": []}},
  "per_action_status": {{}},
  "gold_admissible_set": [],
  "gold_scsr_needed": false,
  "gold_memory_conflict_label": "NO_CONFLICT",
  "annotation_confidence": "HIGH" or "MEDIUM" or "LOW",
  "annotation_notes": "one sentence"
}}"""


# ── 第三阶段：辅助函数 ─────────────────────────────────────────────────────

def _call_llm(prompt: str, max_retries: int = 3) -> str:
    """调用 LLM，返回原始文本（含 think chain）。"""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
                temperature=0.1,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"    [重试 {attempt+1}] {exc}")
                time.sleep(3)
            else:
                raise


def _extract_json(raw: str) -> Dict[str, Any]:
    """从含 think chain 的输出中提取 JSON 对象。"""
    # 去除 <think>...</think>
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # 提取 {...} 块
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError(f"未找到 JSON: {raw[:200]}")
    return json.loads(match.group())


def _enforce_answer_admissible(
    result: Dict[str, Any],
    answer_idx: str,
) -> Dict[str, Any]:
    """
    安全保障：确保 answer_idx 出现在 gold_admissible_set 且状态为 ADMISSIBLE。
    若 LLM 遗漏，强制修正。
    """
    per_action = result.get("per_action_status", {})
    admissible_set: List[str] = result.get("gold_admissible_set", [])

    if answer_idx and per_action.get(answer_idx) != "ADMISSIBLE":
        per_action[answer_idx] = "ADMISSIBLE"
        result["per_action_status"] = per_action

    if answer_idx and answer_idx not in admissible_set:
        admissible_set.append(answer_idx)
        result["gold_admissible_set"] = admissible_set

    return result


def _get_options_str(row: Dict[str, str]) -> str:
    """从 options_text 字段获取格式化选项字符串。"""
    return row.get("options_text", "")


def _annotate_one(row: Dict[str, str], is_fc_cand: bool) -> Dict[str, Any]:
    """
    对单个样本调用 LLM 标注，返回解析后的结果。

    参数：
        row: CSV 行数据
        is_fc_cand: True 时使用 FC_CAND 专用 prompt

    返回：
        Dict 包含标注结果字段
    """
    question = row.get("question", "")
    options = _get_options_str(row)
    answer_idx = row.get("answer_idx", "").strip()
    answer = row.get("answer", "").strip()
    candidate_tag = row.get("candidate_tag", "")
    patient_profile = row.get("patient_profile_json", "{}") or "{}"
    parametric_prior = row.get("parametric_prior_disease_query", "")

    if is_fc_cand:
        prompt = FC_CAND_PROMPT_TEMPLATE.format(
            question=question,
            options=options,
            answer_idx=answer_idx,
            answer=answer,
        )
    else:
        prompt = SC_PROMPT_TEMPLATE.format(
            question=question,
            options=options,
            answer_idx=answer_idx,
            answer=answer,
            candidate_tag=candidate_tag,
            patient_profile=patient_profile,
            parametric_prior=parametric_prior,
        )

    raw = _call_llm(prompt)
    result = _extract_json(raw)
    result = _enforce_answer_admissible(result, answer_idx)
    return result


# ── 第四阶段：识别目标样本 ────────────────────────────────────────────────

def _identify_targets(
    rows: List[Dict[str, str]],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    返回三类目标 sample_id 集合：
      (answer_error_ids, wrong_na_ids, fc_cand_ids)
    """
    answer_error_ids: Set[str] = set()
    wrong_na_ids: Set[str] = set()
    fc_cand_ids: Set[str] = set()

    for r in rows:
        sid = r["sample_id"]
        tag = r["candidate_tag"]
        label = r.get("gold_memory_conflict_label", "").strip()
        ans = r.get("answer_idx", "").strip()

        # FC_CAND
        if tag == "FC_CAND":
            fc_cand_ids.add(sid)

        # answer_error（排除 NOT_APPLICABLE）
        if label != "NOT_APPLICABLE":
            try:
                admissible = json.loads(r.get("gold_admissible_set_json", "[]") or "[]")
                if ans and admissible and ans not in admissible:
                    answer_error_ids.add(sid)
            except (json.JSONDecodeError, TypeError):
                pass

        # wrong_NA
        if label == "NOT_APPLICABLE" and tag in (
            "SC_ABSOLUTE_CAND", "SC_RELATIVE_CAND", "TREATMENT_ONLY", "FC_CAND"
        ):
            wrong_na_ids.add(sid)

    return answer_error_ids, wrong_na_ids, fc_cand_ids


# ── 第五阶段：主流程 ───────────────────────────────────────────────────────

def main() -> None:
    """主标注流程。"""

    # 读取 CSV
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = list(reader)
        fieldnames: List[str] = list(reader.fieldnames or [])

    # 识别目标
    answer_error_ids, wrong_na_ids, fc_cand_ids = _identify_targets(rows)
    all_targets = answer_error_ids | wrong_na_ids | fc_cand_ids
    print(f"使用模型: {MODEL}")
    print(f"重标目标: {len(all_targets)} 个")
    print(f"  answer_error: {len(answer_error_ids)}")
    print(f"  wrong_NA:     {len(wrong_na_ids)}")
    print(f"  FC_CAND:      {len(fc_cand_ids)}")
    print()

    # 按 sample_id 建索引
    row_index: Dict[str, int] = {r["sample_id"]: i for i, r in enumerate(rows)}

    success = 0
    failure = 0

    for idx_global, sid in enumerate(sorted(all_targets), 1):
        row_pos = row_index[sid]
        r = rows[row_pos]
        is_fc = sid in fc_cand_ids
        tag_label = "FC_CAND" if is_fc else ("answer_error" if sid in answer_error_ids else "wrong_NA")

        print(f"  [{idx_global}/{len(all_targets)}] {sid} ({tag_label}) ...", end=" ", flush=True)

        try:
            result = _annotate_one(r, is_fc_cand=is_fc)

            q_type = result.get("question_type", "TREATMENT")

            if q_type == "DIAGNOSTIC":
                # 真正的诊断题：保留 NOT_APPLICABLE
                rows[row_pos]["gold_memory_conflict_label"] = "NOT_APPLICABLE"
                rows[row_pos]["gold_per_action_status_json"] = "{}"
                rows[row_pos]["gold_admissible_set_json"] = "[]"
                rows[row_pos]["patient_profile_json"] = json.dumps(
                    result.get("patient_profile", {})
                )
                rows[row_pos]["gold_scsr_needed"] = "false"
                rows[row_pos]["gold_scsr_query"] = ""
                rows[row_pos]["reviewer"] = f"reannotation:{MODEL}"
                rows[row_pos]["notes"] = result.get("annotation_notes", "")
                print(f"DIAGNOSTIC → NOT_APPLICABLE")
            else:
                # 治疗题：更新所有标注字段
                rows[row_pos]["patient_profile_json"] = json.dumps(
                    result.get("patient_profile", {})
                )
                rows[row_pos]["gold_per_action_status_json"] = json.dumps(
                    result.get("per_action_status", {})
                )
                rows[row_pos]["gold_admissible_set_json"] = json.dumps(
                    result.get("gold_admissible_set", [])
                )
                rows[row_pos]["gold_scsr_needed"] = (
                    "true" if result.get("gold_scsr_needed") else "false"
                )
                rows[row_pos]["gold_memory_conflict_label"] = result.get(
                    "gold_memory_conflict_label", "NO_CONFLICT"
                )
                rows[row_pos]["reviewer"] = f"reannotation:{MODEL}"
                rows[row_pos]["notes"] = result.get("annotation_notes", "")
                label = result.get("gold_memory_conflict_label", "?")
                conf = result.get("annotation_confidence", "?")
                print(f"{label} [{conf}]")

            success += 1

        except Exception as exc:
            print(f"FAILED → {exc}")
            failure += 1

        # 速率控制（MiniMax 限流保护）
        time.sleep(1.0)

    print()
    print(f"重标完成: {success} 成功, {failure} 失败")

    # 写回 CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV 已更新 → {CSV_PATH}")


if __name__ == "__main__":
    main()
