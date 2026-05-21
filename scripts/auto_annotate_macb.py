#!/usr/bin/env python3
"""
MACB 自动标注脚本

用 LLM 为每个候选样本生成所有 gold label 字段，
替代人工标注，并将结果写回标注 CSV 和候选 jsonl。

标注逻辑：
  Step 1: 判断题目类型（TREATMENT / DIAGNOSTIC / PROGNOSIS / OTHER）
          只有 TREATMENT 类问题才有实质的 SC 冲突，其余标记 NOT_APPLICABLE。
  Step 2: 提取患者约束（patient_profile）——过敏史、器官功能、妊娠等。
  Step 3: 对每个选项判断 admissibility：
            ADMISSIBLE         对该患者安全可用
            INADMISSIBLE_ABS   绝对禁忌（如青霉素过敏→阿莫西林）
            INADMISSIBLE_REL   相对禁忌（如 eGFR 低→需减量）
            NOT_APPLICABLE     选项不是治疗方案（诊断/病理等）
  Step 4: 判断 gold_scsr_needed（admissible 文档是否不足，需补充检索）
  Step 5: 判断 gold_memory_conflict_label（LLM 一般推荐是否与患者约束冲突）

使用方法：
  python3 scripts/auto_annotate_macb.py
  python3 scripts/auto_annotate_macb.py --limit 5   # 先试跑 5 个
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import LLMClient, get_client


# ── 标注 Prompt ───────────────────────────────────────────────────────────────

ANNOTATION_PROMPT = """\
You are a medical expert annotating a clinical decision benchmark dataset.

Analyze this USMLE question and provide structured annotation.

=== QUESTION ===
{question}

=== OPTIONS ===
{options_text}

=== CORRECT ANSWER ===
{answer_idx}: {answer}

=== LLM PRIOR (general population recommendation without patient constraints) ===
{prior_response}

=== YOUR TASK ===
Return a JSON object with these fields:

1. "question_type": One of:
   - "TREATMENT": Question asks which treatment/drug/intervention to give
   - "DIAGNOSTIC": Question asks for a test, diagnosis, or next diagnostic step
   - "PROGNOSIS": Question asks about complications, outcomes, or prognosis
   - "OTHER": Doesn't fit above categories

2. "patient_profile": Object with patient constraints extracted from the question.
   Only include fields that are explicitly stated or clearly implied:
   {{
     "allergies": [],          // e.g. ["penicillin", "sulfa"]
     "pregnancy": false,       // true if pregnant
     "renal_impairment": null, // e.g. "eGFR 25 mL/min" or null
     "hepatic_impairment": null,
     "age_years": null,
     "other_constraints": []   // any other relevant constraints
   }}

3. "per_action_status": For each option letter, one of:
   - "ADMISSIBLE": Safe and appropriate for this specific patient
   - "INADMISSIBLE_ABS": Absolutely contraindicated (e.g., allergy to drug)
   - "INADMISSIBLE_REL": Relatively contraindicated (e.g., needs dose adjustment)
   - "NOT_APPLICABLE": Option is not a treatment (it's a test, diagnosis, etc.)

   If question_type is not TREATMENT, ALL options should be NOT_APPLICABLE.

4. "gold_admissible_set": List of option letters that are ADMISSIBLE.
   Empty list [] if question_type is not TREATMENT.

5. "gold_scsr_needed": Boolean. True if:
   - question_type is TREATMENT, AND
   - There are INADMISSIBLE options (suggesting standard treatment is blocked), AND
   - The admissible alternatives are not obvious from standard knowledge

6. "gold_memory_conflict_label": One of:
   - "CONFLICT": The LLM prior recommends something that is INADMISSIBLE for this patient
   - "NO_CONFLICT": The LLM prior aligns with what's safe for this patient
   - "NOT_APPLICABLE": question_type is not TREATMENT

7. "annotation_confidence": "HIGH", "MEDIUM", or "LOW"
   HIGH: Clear-cut case with obvious constraints and contraindications
   MEDIUM: Some ambiguity in patient constraints or option classification
   LOW: Uncertain about the SC applicability or option classification

8. "annotation_notes": Brief explanation of your reasoning (1-2 sentences).

Return ONLY valid JSON. No explanation outside the JSON.
"""


def annotate_sample(client: LLMClient, sample: dict) -> Optional[dict]:
    """
    用 LLM 为单个样本生成 gold labels。

    参数：
        client:  LLMClient 实例
        sample:  候选样本 dict

    返回：
        标注结果 dict，失败时返回 None
    """
    options_text = "\n".join(
        f"  {k}: {v}" for k, v in sample["options"].items()
    )
    prior = sample.get("parametric_prior_stub") or {}
    prior_response = prior.get("disease_only_response") or "(not available)"

    prompt = ANNOTATION_PROMPT.format(
        question=sample["question"],
        options_text=options_text,
        answer_idx=sample["answer_idx"],
        answer=sample["answer"],
        prior_response=prior_response,
    )

    raw = client.chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )

    # 提取 JSON（处理模型可能包裹在 ```json ``` 中的情况）
    json_str = raw
    if "```" in raw:
        start = raw.find("```")
        end = raw.rfind("```")
        if start != end:
            json_str = raw[start + 3: end].strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()

    return json.loads(json_str)


def build_csv_row_updates(annotation: dict, sample: dict) -> dict:
    """
    将标注结果转换为 CSV 行更新字段。

    参数：
        annotation: LLM 返回的标注 dict
        sample:     原始候选样本

    返回：
        需要更新的 CSV 字段 dict
    """
    return {
        "patient_profile_json": json.dumps(
            annotation.get("patient_profile", {}), ensure_ascii=False
        ),
        "gold_admissible_set_json": json.dumps(
            annotation.get("gold_admissible_set", []), ensure_ascii=False
        ),
        "gold_per_action_status_json": json.dumps(
            annotation.get("per_action_status", {}), ensure_ascii=False
        ),
        "gold_scsr_needed": str(annotation.get("gold_scsr_needed", False)).lower(),
        "gold_scsr_query": "",   # 留空，由 SCSR 模块在实验时生成
        "gold_memory_conflict_label": annotation.get("gold_memory_conflict_label", "NOT_APPLICABLE"),
        "notes": annotation.get("annotation_notes", ""),
        "reviewer": f"auto:{sample.get('parametric_prior_stub', {}).get('model_used', 'llm')}",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="MACB 自动标注")
    p.add_argument(
        "--candidates",
        default="data/interim/macb_candidates_v2_with_prior.jsonl",
    )
    p.add_argument(
        "--annotation-csv",
        default="annotations/macb_v2_sheet.csv",
    )
    p.add_argument(
        "--output-candidates",
        default="data/interim/macb_candidates_v2_annotated.jsonl",
        help="写出带标注的候选 jsonl",
    )
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 个样本")
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument(
        "--tag-filter",
        nargs="+",
        default=None,
        help="只处理指定 candidate_tag，如 SC_ABSOLUTE_CAND SC_RELATIVE_CAND",
    )
    args = p.parse_args()

    client = get_client()

    # 读取候选池
    candidates = [
        json.loads(l)
        for l in Path(args.candidates).open(encoding="utf-8")
        if l.strip()
    ]

    # 过滤
    to_annotate = candidates
    if args.tag_filter:
        to_annotate = [c for c in candidates if c["candidate_tag"] in args.tag_filter]
    if args.limit:
        to_annotate = to_annotate[: args.limit]

    print(f"候选池总数: {len(candidates)}")
    print(f"待标注: {len(to_annotate)}")
    print(f"模型: {client.model}")
    print()

    # 读取现有 CSV
    csv_path = Path(args.annotation_csv)
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        csv_rows = {r["sample_id"]: r for r in reader}

    # 建立 candidate_id → sample_id 的映射（通过 question 匹配）
    question_to_sample_id = {r["question"]: sid for sid, r in csv_rows.items()}

    # 逐样本标注
    annotated_ids: set = set()
    errors: list = []
    stats = {"TREATMENT": 0, "DIAGNOSTIC": 0, "PROGNOSIS": 0, "OTHER": 0}

    for sample in to_annotate:
        cid = sample["candidate_id"]
        try:
            annotation = annotate_sample(client, sample)

            qtype = annotation.get("question_type", "OTHER")
            stats[qtype] = stats.get(qtype, 0) + 1

            # 更新候选 dict（加入标注结果）
            sample["gold_annotation"] = annotation

            # 更新 CSV
            sid = question_to_sample_id.get(sample["question"])
            if sid and sid in csv_rows:
                updates = build_csv_row_updates(annotation, sample)
                csv_rows[sid].update(updates)

            annotated_ids.add(cid)
            conf = annotation.get("annotation_confidence", "?")
            admissible = annotation.get("gold_admissible_set", [])
            print(
                f"  [OK] {cid} | {qtype:<11} | conf={conf} | "
                f"admissible={admissible} | "
                f"{annotation.get('gold_memory_conflict_label','?')}"
            )
            time.sleep(args.sleep)

        except Exception as e:
            errors.append(f"{cid}: {e}")
            print(f"  [ERROR] {cid}: {e}", file=sys.stderr)

    # 写出带标注的候选 jsonl
    out_path = Path(args.output_candidates)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in candidates:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 写回 CSV
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows.values())

    # 汇总
    print()
    print(f"标注完成: {len(annotated_ids)} 个，失败: {len(errors)} 个")
    print("题目类型分布:")
    for qtype, cnt in stats.items():
        print(f"  {qtype}: {cnt}")
    print(f"候选 jsonl → {out_path}")
    print(f"标注 CSV   → {csv_path}")

    if errors:
        print(f"\n失败列表:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
