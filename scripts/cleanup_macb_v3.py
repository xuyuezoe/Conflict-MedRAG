#!/usr/bin/env python3
"""
MACB Benchmark v2 → v3 targeted cleanup 脚本

修复 GPT 专家审查识别的 8 类语义问题，生成 macb_treatment_v3.jsonl。

P0: MACB-085 补 C 选项（schema 硬错误）
P1: INADMISSIBLE_REL → CONDITIONALLY_ADMISSIBLE（全量重命名）
P2a: TREATMENT_ONLY+NO_CONFLICT 样本的 INADMISSIBLE_* → NOT_INDICATED
P2b: MACB-231 gold_conflict_type 升级 NO_CONFLICT → SC_ABSOLUTE
P3: FC 40条逐条重分类（医学判断，硬编码决策表）
P4: gold_memory_conflict_label → parametric_prior_conflict_label（字段重命名）
P5: options_text 清洗（\n 和多余引号）
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── 路径配置 ──────────────────────────────────────────────────────────────────

INPUT_PATH = Path("data/macb_treatment_v2.jsonl")
OUTPUT_PATH = Path("data/macb_treatment_v3.jsonl")
REPORT_PATH = Path("data/macb_v3_cleanup_report.json")

# ── 合法枚举值（清洗后的新标准） ───────────────────────────────────────────────

VALID_STATUSES = {
    "ADMISSIBLE",
    "CONDITIONALLY_ADMISSIBLE",
    "INADMISSIBLE_ABS",
    "NOT_INDICATED",
    "NOT_APPLICABLE",
}
VALID_CONFLICT_TYPES = {"SC_ABSOLUTE", "SC_RELATIVE", "FC", "NO_CONFLICT"}
VALID_OPTIONS = {"A", "B", "C", "D", "E"}

# ── P3：FC 样本重分类决策表 ───────────────────────────────────────────────────
#
# 判断标准：真实 FC = 至少 2 个权威指南对同一 action 给出相反推荐。
# 普通错误答案、副作用识别题、首选-替代偏好题均不构成 FC。
#
FC_RECLASSIFICATION: Dict[str, str] = {
    # ── 保留 FC：真实多指南冲突 ──────────────────────────────────────────────
    "MACB-167": "FC",   # β-thal：欧血学会推荐叶酸 vs AAP 认为无需
    "MACB-185": "FC",   # 大规模 PE：AHA 溶栓 vs ESC 手术 vs ACCP 抗凝 三路冲突
    "MACB-193": "FC",   # 狗咬伤：USMLE 首选阿莫西林-克拉维酸 vs CDC/WHO 克林/多西等效
    # ── 改为 SC_ABSOLUTE：患者特异性绝对禁忌 ─────────────────────────────────
    "MACB-177": "SC_ABSOLUTE",  # ACS 患者 NSAIDs 绝对禁忌（急性冠脉综合征期间）
    "MACB-189": "SC_ABSOLUTE",  # HIT 患者肝素绝对禁忌（必须换用直接凝血酶抑制剂）
    # ── 改为 NO_CONFLICT：普通错误/偏好/副作用识别题 ──────────────────────────
    "MACB-160": "NO_CONFLICT",  # scabies：permethrin vs ivermectin 为首选-替代关系
    "MACB-161": "NO_CONFLICT",
    "MACB-162": "NO_CONFLICT",  # beta-blocker OD：解救药识别题
    "MACB-163": "NO_CONFLICT",
    "MACB-164": "NO_CONFLICT",
    "MACB-165": "NO_CONFLICT",
    "MACB-166": "NO_CONFLICT",
    "MACB-168": "NO_CONFLICT",
    "MACB-169": "NO_CONFLICT",
    "MACB-170": "NO_CONFLICT",  # metoclopramide：药物副作用识别题
    "MACB-171": "NO_CONFLICT",  # 直立性低血压：最不易致病的药物比较题
    "MACB-172": "NO_CONFLICT",
    "MACB-173": "NO_CONFLICT",
    "MACB-174": "NO_CONFLICT",
    "MACB-175": "NO_CONFLICT",
    "MACB-176": "NO_CONFLICT",  # 过敏反应：标准分层治疗（非冲突）
    "MACB-178": "NO_CONFLICT",
    "MACB-179": "NO_CONFLICT",
    "MACB-180": "NO_CONFLICT",  # NSTEMI PCI 后：标准抗血小板管理
    "MACB-181": "NO_CONFLICT",
    "MACB-182": "NO_CONFLICT",
    "MACB-183": "NO_CONFLICT",
    "MACB-184": "NO_CONFLICT",
    "MACB-186": "NO_CONFLICT",
    "MACB-187": "NO_CONFLICT",
    "MACB-188": "NO_CONFLICT",
    "MACB-190": "NO_CONFLICT",  # 心律失常管理：标准方案
    "MACB-191": "NO_CONFLICT",  # 阿片类中毒：纳洛酮解救药识别
    "MACB-192": "NO_CONFLICT",  # 药物毒性识别题（齐多夫定中性粒细胞减少）
    "MACB-194": "NO_CONFLICT",
    "MACB-205": "NO_CONFLICT",
    "MACB-214": "NO_CONFLICT",  # MRSA 皮肤脓肿：万古霉素 vs 二氯西林为首选-替代
    "MACB-222": "NO_CONFLICT",
    "MACB-224": "NO_CONFLICT",
    "MACB-236": "NO_CONFLICT",  # 风湿热：长期青霉素预防为标准管理
}

# ── P2b：特殊样本 gold_conflict_type 升级 ─────────────────────────────────────
CONFLICT_TYPE_UPGRADES: Dict[str, str] = {
    "MACB-231": "SC_ABSOLUTE",  # WPW + verapamil：WPW 是患者特异性条件，verapamil 绝对禁忌
}

# ── 需要人工复审（暂保留现状，在报告中标注） ──────────────────────────────────
MANUAL_REVIEW_NEEDED = ["MACB-210", "MACB-223", "MACB-235"]


# ── diff log 工具 ─────────────────────────────────────────────────────────────

def _log(
    diff_log: List[Dict[str, Any]],
    sample_id: str,
    priority: str,
    field: str,
    old_value: Any,
    new_value: Any,
) -> None:
    """将单个字段变更追加到 diff_log。"""
    diff_log.append({
        "sample_id": sample_id,
        "priority": priority,
        "field": field,
        "old": old_value,
        "new": new_value,
    })


# ── P0-P5 各阶段函数 ──────────────────────────────────────────────────────────

def apply_p0_fix_macb085(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    P0：修复 MACB-085 缺少 C 选项的 schema 硬错误。

    C = "Low dose acyclovir"，亚治疗剂量不足以预防肾移植后 HSV，
    且 gold_admissible_set 中不含 C，故状态为 INADMISSIBLE_ABS。

    返回：
        (sample, 修改次数)
    """
    if sample["sample_id"] != "MACB-085":
        return sample, 0

    pas = sample["gold_per_action_status"]
    if "C" not in pas:
        _log(diff_log, "MACB-085", "P0", "gold_per_action_status.C", None, "INADMISSIBLE_ABS")
        sample["gold_per_action_status"]["C"] = "INADMISSIBLE_ABS"
        return sample, 1

    return sample, 0


def apply_p1_rename_inadmissible_rel(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    P1：将 gold_per_action_status 中所有 INADMISSIBLE_REL 改为 CONDITIONALLY_ADMISSIBLE。

    语义修正：SC_RELATIVE 类型的 action 仍在 A(q) 中（需剂量/时机调整），
    标为 INADMISSIBLE 会与理论中"action transformation"的定义矛盾。

    返回：
        (sample, 修改次数)
    """
    count = 0
    pas = sample["gold_per_action_status"]
    for option, status in list(pas.items()):
        if status == "INADMISSIBLE_REL":
            _log(diff_log, sample["sample_id"], "P1",
                 f"gold_per_action_status.{option}",
                 "INADMISSIBLE_REL", "CONDITIONALLY_ADMISSIBLE")
            pas[option] = "CONDITIONALLY_ADMISSIBLE"
            count += 1
    return sample, count


def apply_p2a_treatment_only_not_indicated(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    P2a：对 TREATMENT_ONLY + NO_CONFLICT 样本，将 INADMISSIBLE_* 改为 NOT_INDICATED。

    核心原则：INADMISSIBLE_ABS 只能用于"患者特异性约束导致 action 不可用"。
    TREATMENT_ONLY 样本是普通选择题，错误选项只是医学上不适应，
    没有患者条件约束，不应标为 INADMISSIBLE。

    返回：
        (sample, 修改次数)
    """
    if (
        sample.get("candidate_tag") != "TREATMENT_ONLY"
        or sample.get("gold_conflict_type") != "NO_CONFLICT"
    ):
        return sample, 0

    count = 0
    pas = sample["gold_per_action_status"]
    for option, status in list(pas.items()):
        if status in {"INADMISSIBLE_ABS", "CONDITIONALLY_ADMISSIBLE"}:
            _log(diff_log, sample["sample_id"], "P2a",
                 f"gold_per_action_status.{option}",
                 status, "NOT_INDICATED")
            pas[option] = "NOT_INDICATED"
            count += 1
    return sample, count


def apply_p2b_conflict_type_upgrade(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    P2b：将特定样本的 gold_conflict_type 从 NO_CONFLICT 升级为正确类型。

    MACB-231（WPW + verapamil）：WPW 是患者特异性条件，verapamil 在 WPW 中
    可导致心室颤动，属于患者条件导致的绝对禁忌，应标为 SC_ABSOLUTE。

    返回：
        (sample, 修改次数)
    """
    sid = sample["sample_id"]
    if sid not in CONFLICT_TYPE_UPGRADES:
        return sample, 0

    new_type = CONFLICT_TYPE_UPGRADES[sid]
    old_type = sample.get("gold_conflict_type")
    if old_type != new_type:
        _log(diff_log, sid, "P2b", "gold_conflict_type", old_type, new_type)
        sample["gold_conflict_type"] = new_type
        return sample, 1

    return sample, 0


def apply_p3_fc_reclassification(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    P3：FC 40条逐条重分类（医学判断 + 硬编码决策表）。

    对改为 NO_CONFLICT 的样本，同时将 INADMISSIBLE_ABS/CONDITIONALLY_ADMISSIBLE
    改为 NOT_INDICATED（无患者特异性约束，仅是医学上不适应）。

    对改为 SC_ABSOLUTE 的样本，INADMISSIBLE_ABS 保持不变（语义正确）。

    返回：
        (sample, 修改次数)
    """
    sid = sample["sample_id"]
    if sid not in FC_RECLASSIFICATION:
        return sample, 0

    count = 0
    new_conflict_type = FC_RECLASSIFICATION[sid]
    old_conflict_type = sample.get("gold_conflict_type")

    if old_conflict_type != new_conflict_type:
        _log(diff_log, sid, "P3", "gold_conflict_type", old_conflict_type, new_conflict_type)
        sample["gold_conflict_type"] = new_conflict_type
        count += 1

    # 降级为 NO_CONFLICT 的样本：错误选项不是患者禁忌，改为 NOT_INDICATED
    if new_conflict_type == "NO_CONFLICT":
        pas = sample["gold_per_action_status"]
        for option, status in list(pas.items()):
            if status in {"INADMISSIBLE_ABS", "CONDITIONALLY_ADMISSIBLE"}:
                _log(diff_log, sid, "P3",
                     f"gold_per_action_status.{option}",
                     status, "NOT_INDICATED")
                pas[option] = "NOT_INDICATED"
                count += 1

    return sample, count


def apply_p4_rename_memory_conflict_label(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    P4：将 gold_memory_conflict_label 重命名为 parametric_prior_conflict_label。

    "memory" 语义模糊；"parametric_prior" 准确表达该字段含义：
    LLM 参数记忆（训练集分布）是否与患者特异性条件产生先验冲突。

    返回：
        (sample, 修改次数)
    """
    if "gold_memory_conflict_label" not in sample:
        return sample, 0

    old_value = sample.pop("gold_memory_conflict_label")
    sample["parametric_prior_conflict_label"] = old_value
    _log(diff_log, sample["sample_id"], "P4",
         "gold_memory_conflict_label→parametric_prior_conflict_label",
         old_value, old_value)
    return sample, 1


def apply_p5_clean_options_text(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    P5：清洗 options_text 末尾残留的换行符和多余引号。

    这些字符是数据导出时的格式残留，会干扰 action normalization。

    返回：
        (sample, 修改次数)
    """
    original = sample.get("options_text", "")
    cleaned = original.strip().strip('"').strip()
    if cleaned != original:
        _log(diff_log, sample["sample_id"], "P5", "options_text", original, cleaned)
        sample["options_text"] = cleaned
        return sample, 1
    return sample, 0


# ── 验证与统计 ────────────────────────────────────────────────────────────────

def validate_sample(sample: Dict[str, Any]) -> List[str]:
    """
    验证单个样本的 schema 完整性。

    检查项：
      - gold_per_action_status 含 A-E 五个键
      - 所有 status 值在合法枚举内
      - gold_conflict_type 在合法枚举内
      - parametric_prior_conflict_label 存在（P4 已重命名）

    返回：
        错误列表（空列表表示通过）
    """
    errors: List[str] = []
    sid = sample.get("sample_id", "UNKNOWN")

    pas = sample.get("gold_per_action_status", {})
    missing = VALID_OPTIONS - set(pas.keys())
    if missing:
        errors.append(f"{sid}: gold_per_action_status 缺少选项 {sorted(missing)}")

    for opt, status in pas.items():
        if status not in VALID_STATUSES:
            errors.append(f"{sid}.{opt}: 非法 status '{status}'")

    conflict_type = sample.get("gold_conflict_type", "")
    if conflict_type not in VALID_CONFLICT_TYPES:
        errors.append(f"{sid}: 非法 gold_conflict_type '{conflict_type}'")

    if "gold_memory_conflict_label" in sample:
        errors.append(f"{sid}: 旧字段 gold_memory_conflict_label 未删除（P4 未执行）")

    if "parametric_prior_conflict_label" not in sample:
        errors.append(f"{sid}: 缺少 parametric_prior_conflict_label 字段")

    return errors


def compute_distribution(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算清洗后数据分布，用于 diff 报告和论文统计。

    返回：
        包含各字段分布的字典
    """
    conflict_counter: Counter = Counter()
    status_counter: Counter = Counter()
    prior_label_counter: Counter = Counter()
    candidate_tag_counter: Counter = Counter()

    for s in samples:
        conflict_counter[s.get("gold_conflict_type", "UNKNOWN")] += 1
        candidate_tag_counter[s.get("candidate_tag", "UNKNOWN")] += 1
        prior_label_counter[s.get("parametric_prior_conflict_label", "UNKNOWN")] += 1
        for status in s.get("gold_per_action_status", {}).values():
            status_counter[status] += 1

    return {
        "gold_conflict_type": dict(conflict_counter),
        "status_counts": dict(status_counter),
        "parametric_prior_conflict_label": dict(prior_label_counter),
        "candidate_tag": dict(candidate_tag_counter),
    }


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    主入口：顺序执行 P0-P5 清洗，输出 v3 JSONL 和 diff 报告。

    核心原则：
      - 每个 P 阶段独立函数，可单独测试和 review
      - diff_log 记录所有变更，完整可追踪
      - 验证阶段发现错误时显式抛出，不静默兜底
    """
    # 第一阶段：加载 v2 数据
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

    # 第二阶段：P0 - 修复 MACB-085
    # 第三阶段：P1 - 全量 INADMISSIBLE_REL 重命名
    # 第四阶段：P2a - TREATMENT_ONLY+NO_CONFLICT 改 NOT_INDICATED
    # 第五阶段：P2b - 特殊样本 conflict_type 升级
    # 第六阶段：P3 - FC 样本重分类
    # 第七阶段：P4 - 字段重命名
    # 第八阶段：P5 - options_text 清洗
    diff_log: List[Dict[str, Any]] = []
    p_counts: Dict[str, int] = defaultdict(int)

    for sample in samples:
        sample, n = apply_p0_fix_macb085(sample, diff_log)
        p_counts["P0"] += n

        sample, n = apply_p1_rename_inadmissible_rel(sample, diff_log)
        p_counts["P1"] += n

        # P2a 在 P1 之后运行，确保 CONDITIONALLY_ADMISSIBLE 也被替换
        sample, n = apply_p2a_treatment_only_not_indicated(sample, diff_log)
        p_counts["P2a"] += n

        sample, n = apply_p2b_conflict_type_upgrade(sample, diff_log)
        p_counts["P2b"] += n

        # P3 在 P1 之后运行，FC 降级逻辑需处理 CONDITIONALLY_ADMISSIBLE
        sample, n = apply_p3_fc_reclassification(sample, diff_log)
        p_counts["P3"] += n

        sample, n = apply_p4_rename_memory_conflict_label(sample, diff_log)
        p_counts["P4"] += n

        sample, n = apply_p5_clean_options_text(sample, diff_log)
        p_counts["P5"] += n

    print(f"[第二~八阶段] 修复完成，共 {sum(p_counts.values())} 处变更")

    # 第九阶段：一致性验证
    all_errors: List[str] = []
    for sample in samples:
        all_errors.extend(validate_sample(sample))

    if all_errors:
        print("\n[验证失败] 发现以下错误：", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        raise ValueError(f"验证失败，共 {len(all_errors)} 个错误，v3 文件未写出。")

    print("[第九阶段] 验证通过：所有样本 schema 合法")

    # 第十阶段：输出 v3 JSONL 和 diff 报告
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    distribution = compute_distribution(samples)
    total_changes = sum(p_counts.values())
    report = {
        "summary": {
            "input_file": str(INPUT_PATH),
            "output_file": str(OUTPUT_PATH),
            "total_samples": len(samples),
            "total_changes": total_changes,
            "changes_by_priority": dict(p_counts),
        },
        "changes": diff_log,
        "manual_review_needed": MANUAL_REVIEW_NEEDED,
        "new_distribution": distribution,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 输出摘要
    print(f"\n[第十阶段] 输出完成")
    print(f"\n变更统计：")
    for priority in ["P0", "P1", "P2a", "P2b", "P3", "P4", "P5"]:
        count = p_counts.get(priority, 0)
        print(f"  {priority}: {count} 处")
    print(f"  合计: {total_changes} 处")

    print(f"\n新 gold_conflict_type 分布：")
    for k, v in sorted(distribution["gold_conflict_type"].items()):
        print(f"  {k}: {v}")

    print(f"\n新 status 分布：")
    for k, v in sorted(distribution["status_counts"].items()):
        print(f"  {k}: {v}")

    print(f"\n输出文件：{OUTPUT_PATH}")
    print(f"Diff 报告：{REPORT_PATH}")
    print(f"\n需人工复审（保留现状）：{MANUAL_REVIEW_NEEDED}")


if __name__ == "__main__":
    main()
