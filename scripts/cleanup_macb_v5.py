#!/usr/bin/env python3
"""
MACB Benchmark v4 → v5 targeted cleanup 脚本

修复 GPT 第三轮审查发现的实验定义问题，生成 macb_treatment_v5.jsonl。

G: 全量派生 gold_conflict_types_present（fine-grained 分层字段）
H: 全量写 task_type，MACB-025/040 标为 contraindication_recognition
I: SCSR 4 条异常清洗（MACB-237/096/116/210）
J: MACB-167 notes 文本修正（contraindicated → not indicated）
K: 7 条 reviewer 字段补 "pending"

执行顺序：G → H → I → J → K（各组独立，无依赖）
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ── 路径配置 ──────────────────────────────────────────────────────────────────

INPUT_PATH = Path("data/macb_treatment_v4.jsonl")
OUTPUT_PATH = Path("data/macb_treatment_v5.jsonl")
REPORT_PATH = Path("data/macb_v5_cleanup_report.json")

# ── 合法枚举 ──────────────────────────────────────────────────────────────────

VALID_CONFLICT_TYPES_PRESENT = {"SC_ABSOLUTE", "SC_RELATIVE", "NO_CONFLICT"}
VALID_TASK_TYPES = {"treatment_recommendation", "contraindication_recognition"}
VALID_STATUSES = {
    "ADMISSIBLE", "CONDITIONALLY_ADMISSIBLE", "INADMISSIBLE_ABS",
    "NOT_INDICATED", "NOT_APPLICABLE",
}

# ── H 组：contraindication_recognition 标注 ───────────────────────────────────
#
# MedQA 官方答案（answer_idx）指向 INADMISSIBLE_ABS 选项的样本：
# 这类题目是"识别禁忌药"而非"推荐治疗"，CRR/SDR 不适用。
# 已验证 v4 中仅此 2 条。
#
CONTRAINDICATION_SAMPLES: Set[str] = {
    "MACB-025",   # answer_idx=E, E=INADMISSIBLE_ABS（帕金森+甲氧氯普胺）
    "MACB-040",   # answer_idx=A, A=INADMISSIBLE_ABS（可卡因胸痛+普萘洛尔）
}

# ── I 组：SCSR 异常清洗 ───────────────────────────────────────────────────────
#
# SCSR 仅服务于 SC 排除后的 A(q) gap-filling，不服务 parametric prior conflict。
# gold_scsr_needed=false 时 query 应为空；NO_CONFLICT 样本不应触发 SCSR。
#
SCSR_QUERY_CLEAR: Set[str] = {
    "MACB-237",   # gold_scsr_needed=false 但 gold_scsr_query 非空
}

SCSR_NEEDED_FALSE: Set[str] = {
    "MACB-096",   # NO_CONFLICT + scsr_needed=true（仅有 parametric prior conflict）
    "MACB-116",   # 同上
    "MACB-210",   # 同上
}

# ── J 组：MACB-167 notes 文本修正 ─────────────────────────────────────────────
#
# notes 写 "iron is contraindicated in thalassemia"（暗示绝对禁忌），
# 但 status 是 NOT_INDICATED（无指征，非患者特异性约束），语义冲突。
#
MACB_167_OLD_TEXT = "iron is contraindicated in thalassemia."
MACB_167_NEW_TEXT = (
    "iron supplementation is not indicated in thalassemia "
    "unless iron deficiency is documented."
)

# ── K 组：reviewer 字段补全 ────────────────────────────────────────────────────
#
# 7 条缺失 reviewer 的样本，填入占位符（不伪造标注者身份）。
#
REVIEWER_PENDING: Set[str] = {
    "MACB-004", "MACB-058", "MACB-077", "MACB-102",
    "MACB-122", "MACB-124", "MACB-141",
}


# ── diff log 工具 ─────────────────────────────────────────────────────────────

def _log(
    diff_log: List[Dict[str, Any]],
    sample_id: str,
    group: str,
    field: str,
    old_value: Any,
    new_value: Any,
) -> None:
    """将单个字段变更追加到 diff_log。"""
    diff_log.append({
        "sample_id": sample_id,
        "group": group,
        "field": field,
        "old": old_value,
        "new": new_value,
    })


# ── 各阶段处理函数 ────────────────────────────────────────────────────────────

def apply_group_g(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    G 组：派生 gold_conflict_types_present 字段（全量 231 条）。

    逻辑：
      INADMISSIBLE_ABS 存在 → SC_ABSOLUTE 进列表
      CONDITIONALLY_ADMISSIBLE 存在 → SC_RELATIVE 进列表
      两者均不存在 → NO_CONFLICT 进列表

    返回：(sample, 变更数)
    """
    sid = sample["sample_id"]
    pas = sample["gold_per_action_status"]
    pas_values = set(pas.values())

    types_present = []
    if "INADMISSIBLE_ABS" in pas_values:
        types_present.append("SC_ABSOLUTE")
    if "CONDITIONALLY_ADMISSIBLE" in pas_values:
        types_present.append("SC_RELATIVE")
    if not types_present:
        types_present.append("NO_CONFLICT")

    old_value = sample.get("gold_conflict_types_present")
    if old_value != types_present:
        _log(diff_log, sid, "G:conflict_types_present",
             "gold_conflict_types_present", old_value, types_present)
        sample["gold_conflict_types_present"] = types_present
        return sample, 1

    return sample, 0


def apply_group_h(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    H 组：写入 task_type 字段（全量默认 + 2 条特殊标注）。

    MACB-025/040 的 MedQA 官方答案为 INADMISSIBLE_ABS 选项，
    属于"识别禁忌药"题，应排除在 CRR/SDR/AEC/SLR 主评估之外。

    返回：(sample, 变更数)
    """
    sid = sample["sample_id"]
    new_task_type = (
        "contraindication_recognition"
        if sid in CONTRAINDICATION_SAMPLES
        else "treatment_recommendation"
    )

    old_value = sample.get("task_type")
    if old_value != new_task_type:
        _log(diff_log, sid, "H:task_type", "task_type", old_value, new_task_type)
        sample["task_type"] = new_task_type
        return sample, 1

    return sample, 0


def apply_group_i(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    I 组：清洗 SCSR 字段异常（4 条）。

    SCSR 设计为 SC gap-filling，不服务 parametric prior conflict：
      - gold_scsr_needed=false 时 gold_scsr_query 应为空
      - NO_CONFLICT 样本不应有 gold_scsr_needed=true

    返回：(sample, 变更数)
    """
    sid = sample["sample_id"]
    count = 0

    if sid in SCSR_QUERY_CLEAR:
        old_query = sample.get("gold_scsr_query", "")
        if old_query:
            _log(diff_log, sid, "I:scsr_query_clear", "gold_scsr_query", old_query, "")
            sample["gold_scsr_query"] = ""
            count += 1

    if sid in SCSR_NEEDED_FALSE:
        old_needed = sample.get("gold_scsr_needed")
        if old_needed:
            _log(diff_log, sid, "I:scsr_needed_false",
                 "gold_scsr_needed", old_needed, False)
            sample["gold_scsr_needed"] = False
            count += 1
        # gold_scsr_needed=false 时 query 必须同时清空
        old_query = sample.get("gold_scsr_query", "")
        if old_query:
            _log(diff_log, sid, "I:scsr_query_clear",
                 "gold_scsr_query", old_query, "")
            sample["gold_scsr_query"] = ""
            count += 1

    return sample, count


def apply_group_j(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    J 组：修正 MACB-167 notes 文本。

    将 "iron is contraindicated in thalassemia." 替换为
    "iron supplementation is not indicated in thalassemia unless iron deficiency is documented."
    以与 NOT_INDICATED status 语义保持一致。

    返回：(sample, 变更数)
    """
    sid = sample["sample_id"]
    if sid != "MACB-167":
        return sample, 0

    notes = sample.get("notes", "")
    if MACB_167_OLD_TEXT in notes:
        new_notes = notes.replace(MACB_167_OLD_TEXT, MACB_167_NEW_TEXT)
        _log(diff_log, sid, "J:notes_fix", "notes", notes, new_notes)
        sample["notes"] = new_notes
        return sample, 1

    return sample, 0


def apply_group_k(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    K 组：为缺失 reviewer 的 7 条样本补充占位符。

    填入 "pending" 而非伪造标注者身份，便于后续追溯。

    返回：(sample, 变更数)
    """
    sid = sample["sample_id"]
    if sid not in REVIEWER_PENDING:
        return sample, 0

    old_reviewer = sample.get("reviewer")
    if not old_reviewer:
        _log(diff_log, sid, "K:reviewer_pending", "reviewer", old_reviewer, "pending")
        sample["reviewer"] = "pending"
        return sample, 1

    return sample, 0


# ── 验证 ──────────────────────────────────────────────────────────────────────

def validate_sample(sample: Dict[str, Any]) -> List[str]:
    """
    验证单个样本的新字段完整性与语义一致性。

    规则：
      - gold_conflict_types_present 与 per_action_status 一致
      - task_type 合法值
      - gold_scsr_needed=false 时 gold_scsr_query 为空
      - NO_CONFLICT + gold_scsr_needed=true 不存在
      - MACB-167 notes 不含旧文本

    返回：错误列表（空列表表示通过）
    """
    errors: List[str] = []
    sid = sample.get("sample_id", "UNKNOWN")
    pas = sample.get("gold_per_action_status", {})
    pas_values = set(pas.values())

    # G 组验证：gold_conflict_types_present 一致性
    expected_types = []
    if "INADMISSIBLE_ABS" in pas_values:
        expected_types.append("SC_ABSOLUTE")
    if "CONDITIONALLY_ADMISSIBLE" in pas_values:
        expected_types.append("SC_RELATIVE")
    if not expected_types:
        expected_types.append("NO_CONFLICT")

    actual_types = sample.get("gold_conflict_types_present")
    if actual_types != expected_types:
        errors.append(
            f"{sid}: gold_conflict_types_present={actual_types} 应为 {expected_types}"
        )

    # H 组验证：task_type 合法值
    task_type = sample.get("task_type")
    if task_type not in VALID_TASK_TYPES:
        errors.append(f"{sid}: 非法 task_type='{task_type}'")

    # I 组验证：SCSR 一致性
    scsr_needed = sample.get("gold_scsr_needed", False)
    scsr_query = sample.get("gold_scsr_query", "")
    conflict_type = sample.get("gold_conflict_type", "")

    if not scsr_needed and scsr_query:
        errors.append(f"{sid}: gold_scsr_needed=false 但 gold_scsr_query 非空")

    if conflict_type == "NO_CONFLICT" and scsr_needed:
        errors.append(f"{sid}: NO_CONFLICT + gold_scsr_needed=true 矛盾")

    # J 组验证：MACB-167 notes 文本
    if sid == "MACB-167" and MACB_167_OLD_TEXT in sample.get("notes", ""):
        errors.append(f"{sid}: notes 仍含旧文本 '{MACB_167_OLD_TEXT}'")

    return errors


def compute_distribution(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算 v5 数据分布，用于 diff 报告。

    返回：包含各关键字段分布的字典
    """
    conflict_counter: Counter = Counter()
    task_type_counter: Counter = Counter()
    types_present_counter: Counter = Counter()

    for s in samples:
        conflict_counter[s.get("gold_conflict_type", "UNKNOWN")] += 1
        task_type_counter[s.get("task_type", "MISSING")] += 1
        key = str(sorted(s.get("gold_conflict_types_present", [])))
        types_present_counter[key] += 1

    mixed_count = sum(
        1 for s in samples
        if set(s.get("gold_conflict_types_present", [])) == {"SC_ABSOLUTE", "SC_RELATIVE"}
    )

    return {
        "gold_conflict_type": dict(conflict_counter),
        "task_type": dict(task_type_counter),
        "gold_conflict_types_present_distribution": dict(types_present_counter),
        "mixed_sc_count": mixed_count,
    }


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    主入口：按 G→H→I→J→K 顺序执行清洗，输出 v5 JSONL 和 diff 报告。
    """
    # 第一阶段：加载 v4 数据
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    samples: List[Dict[str, Any]] = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_num} 行 JSON 解析失败: {e}") from e

    print(f"[第一阶段] 加载完成：{len(samples)} 个样本")

    # 第二～六阶段：逐样本应用 G→H→I→J→K
    diff_log: List[Dict[str, Any]] = []
    group_counts: Dict[str, int] = defaultdict(int)

    for sample in samples:
        for group_fn, group_key in [
            (apply_group_g, "G"),
            (apply_group_h, "H"),
            (apply_group_i, "I"),
            (apply_group_j, "J"),
            (apply_group_k, "K"),
        ]:
            sample, n = group_fn(sample, diff_log)
            group_counts[group_key] += n

    print(f"[第二~六阶段] G-K 修复完成")

    # 第七阶段：一致性验证
    all_errors: List[str] = []
    for sample in samples:
        all_errors.extend(validate_sample(sample))

    if all_errors:
        print(f"\n[验证失败] 发现 {len(all_errors)} 个错误：", file=sys.stderr)
        for err in all_errors[:30]:
            print(f"  - {err}", file=sys.stderr)
        if len(all_errors) > 30:
            print(f"  ... 以及另外 {len(all_errors) - 30} 个错误", file=sys.stderr)
        raise ValueError("验证失败，v5 文件未写出。")

    print("[第七阶段] 验证通过：所有样本新字段语义一致")

    # 第八阶段：输出 v5 JSONL 和 diff 报告
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    distribution = compute_distribution(samples)
    total_changes = sum(group_counts.values())
    report = {
        "summary": {
            "input_file": str(INPUT_PATH),
            "output_file": str(OUTPUT_PATH),
            "total_samples": len(samples),
            "total_changes": total_changes,
            "changes_by_group": dict(group_counts),
        },
        "changes": diff_log,
        "new_distribution": distribution,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 输出摘要
    print(f"\n[第八阶段] 输出完成，总变更 {total_changes} 处")
    print(f"\n变更统计：")
    for group in ["G", "H", "I", "J", "K"]:
        print(f"  {group}: {group_counts.get(group, 0)} 处")

    print(f"\n新 gold_conflict_type 分布（应与 v4 完全相同）：")
    for k, v in sorted(distribution["gold_conflict_type"].items()):
        print(f"  {k}: {v}")

    print(f"\ntask_type 分布：")
    for k, v in sorted(distribution["task_type"].items()):
        print(f"  {k}: {v}")

    print(f"\nmixed SC 样本（同时含 ABS+REL）：{distribution['mixed_sc_count']} 条")
    print(f"\n输出文件：{OUTPUT_PATH}")
    print(f"Diff 报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
