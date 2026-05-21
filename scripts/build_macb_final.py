#!/usr/bin/env python3
"""
MACB 最终数据集组装器

将人工标注完成的 CSV 组装为最终 MACB jsonl benchmark。

修正了 GPT v1 的 silent fallback 问题：
  - 所有 JSON 字段解析失败时抛出明确错误，不返回默认值
  - 对必填标注字段做完整性校验，输出缺失报告
  - 只有通过校验的样本才写入最终文件
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ── 必须由标注者填写的字段 ───────────────────────────────────────────────────
REQUIRED_ANNOTATION_FIELDS = [
    "patient_profile_json",
    "gold_admissible_set_json",
    "gold_per_action_status_json",
    "gold_scsr_needed",
]


def parse_json_field(
    sample_id: str,
    field_name: str,
    raw: str,
    allow_empty_collection: bool = False,
) -> Any:
    """
    解析 CSV 中的 JSON 字段。

    规则：
      - 空字符串视为标注未完成，抛出 ValueError
      - allow_empty_collection=False 时，"{}" / "[]" 视为占位符，抛出 ValueError
      - allow_empty_collection=True 时，"{}" / "[]" 视为合法值（非 TREATMENT 问题的正确语义）
      - 解析失败抛出 ValueError（不返回兜底默认值）

    参数：
        sample_id:              样本 ID，用于错误定位
        field_name:             字段名
        raw:                    原始字符串
        allow_empty_collection: 是否允许空集合（[] / {}）作为有效值
    返回：
        解析后的 Python 对象
    """
    stripped = (raw or "").strip()
    if not stripped or stripped == '""':
        raise ValueError(
            f"[{sample_id}] 字段 '{field_name}' 为空或占位符，"
            f"标注尚未完成。"
        )
    if not allow_empty_collection and stripped in ("{}", "[]"):
        raise ValueError(
            f"[{sample_id}] 字段 '{field_name}' 为空集合占位符，"
            f"标注尚未完成。"
        )
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[{sample_id}] 字段 '{field_name}' JSON 解析失败: {e}\n"
            f"  原始值: {repr(raw)}"
        )


def parse_bool_field(sample_id: str, field_name: str, raw: str) -> bool:
    """
    解析布尔型标注字段（接受 true/false/yes/no/1/0）。
    空值抛出 ValueError（不静默返回 False）。
    """
    stripped = (raw or "").strip().lower()
    if not stripped:
        raise ValueError(
            f"[{sample_id}] 字段 '{field_name}' 为空，标注尚未完成。"
        )
    if stripped in {"true", "yes", "1", "y"}:
        return True
    if stripped in {"false", "no", "0", "n"}:
        return False
    raise ValueError(
        f"[{sample_id}] 字段 '{field_name}' 值非法: {repr(raw)}。"
        f"仅接受 true/false/yes/no/1/0。"
    )


def assemble_sample(row: Dict[str, str]) -> Tuple[dict, List[str]]:
    """
    将 CSV 一行组装为最终样本 dict。

    返回：
        (sample_dict, errors_list)
        errors_list 为空表示此样本通过校验。
    """
    sample_id = row.get("sample_id", "UNKNOWN").strip()
    errors: List[str] = []
    sample: dict = {
        "sample_id": sample_id,
        "query": row.get("question", "").strip(),
        "options_text": row.get("options_text", "").strip(),
        "answer_idx": row.get("answer_idx", "").strip(),
        "meta_info": row.get("meta_info", "").strip(),
        "candidate_tag": row.get("candidate_tag", "").strip(),
        "no_keyword_flag": row.get("no_keyword_flag", "false").strip().lower() == "true",
        "parametric_prior_disease_query": row.get("parametric_prior_disease_query", "").strip(),
        "gold_memory_conflict_label": row.get("gold_memory_conflict_label", "").strip(),
        "reviewer": row.get("reviewer", "").strip(),
        "notes": row.get("notes", "").strip(),
    }

    # 确定当前问题类型：NOT_APPLICABLE 说明不是 TREATMENT 问题
    # 非 TREATMENT 问题的 gold_admissible_set = [] 和 gold_per_action_status = {} 是合法值
    is_treatment = (
        sample["gold_memory_conflict_label"].upper()
        in {"CONFLICT", "NO_CONFLICT"}
    )

    # 解析 JSON 字段（强制，不静默兜底）
    for field in ["patient_profile_json", "gold_admissible_set_json", "gold_per_action_status_json"]:
        key = field.replace("_json", "")
        # gold_admissible_set 允许空列表：
        #   - 非 TREATMENT 问题：语义上就是"无可接纳治疗选项"
        #   - TREATMENT 问题但所有选项均不可接纳时：也合法（如 MACB-087）
        # patient_profile 和 per_action_status 不允许为空对象（说明标注未完成）
        allow_empty = (field == "gold_admissible_set_json") or (not is_treatment)
        try:
            sample[key] = parse_json_field(
                sample_id, field, row.get(field, ""),
                allow_empty_collection=allow_empty,
            )
        except ValueError as e:
            errors.append(str(e))
            sample[key] = None  # 标记为未完成，不写入最终文件

    # 解析布尔字段
    try:
        sample["gold_scsr_needed"] = parse_bool_field(
            sample_id, "gold_scsr_needed", row.get("gold_scsr_needed", "")
        )
    except ValueError as e:
        errors.append(str(e))
        sample["gold_scsr_needed"] = None

    # gold_scsr_query 在 gold_scsr_needed=True 时必填
    sample["gold_scsr_query"] = row.get("gold_scsr_query", "").strip()
    if sample.get("gold_scsr_needed") is True and not sample["gold_scsr_query"]:
        errors.append(
            f"[{sample_id}] gold_scsr_needed=True 但 gold_scsr_query 为空。"
        )

    return sample, errors


def main() -> None:
    p = argparse.ArgumentParser(description="MACB 最终数据集组装器")
    p.add_argument("--input",  required=True, help="标注完成的 CSV 路径")
    p.add_argument("--output", required=True, help="最终 benchmark jsonl 路径")
    p.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="严格模式：有任何标注缺失则整体退出（不输出文件）",
    )
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"[输入不存在] {inp}")

    rows: List[Dict[str, str]] = []
    with inp.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    valid_samples: List[dict] = []
    all_errors: List[str] = []

    for row in rows:
        sample, errors = assemble_sample(row)
        if errors:
            all_errors.extend(errors)
        else:
            valid_samples.append(sample)

    # 错误报告
    if all_errors:
        print(f"\n[标注完整性问题] {len(all_errors)} 处：", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
        print(
            f"\n有效样本: {len(valid_samples)} / {len(rows)}，"
            f"跳过不完整样本: {len(rows) - len(valid_samples)}",
            file=sys.stderr,
        )
        if args.strict:
            raise SystemExit("[严格模式] 存在标注错误，终止写出。")

    if not valid_samples:
        raise SystemExit("[错误] 没有通过校验的样本，文件不写出。")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in valid_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 输出分布统计
    from collections import Counter
    tag_dist = Counter(s["candidate_tag"] for s in valid_samples)
    print(f"\n最终 benchmark → {out}  ({len(valid_samples)} 个有效样本)")
    print(f"  tag 分布: {dict(tag_dist)}")
    no_kw = sum(1 for s in valid_samples if s.get("no_keyword_flag"))
    print(f"  no_keyword_flag=True: {no_kw}")
    scsr = sum(1 for s in valid_samples if s.get("gold_scsr_needed"))
    print(f"  gold_scsr_needed=True: {scsr}")


if __name__ == "__main__":
    main()
