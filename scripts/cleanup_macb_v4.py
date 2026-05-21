#!/usr/bin/env python3
"""
MACB Benchmark v3 → v4 targeted cleanup 脚本

修复 GPT 第二轮审查发现的语义一致性问题，生成 macb_treatment_v4.jsonl。

A: gold_admissible_set 扩展（包含 CONDITIONALLY_ADMISSIBLE）+ gold_preferred_set
B: 16条 SC_ABSOLUTE 无 INADMISSIBLE_ABS → per_action_status 修复 or conflict_type 降级
C: 28条 SC_RELATIVE 无 CONDITIONALLY_ADMISSIBLE → conflict_type 升/降级
D: 9条 notes-status 矛盾修复（per_action_status + conflict_type）
E: 21条 gold_scsr_needed=false 但 query 非空 → 清空 scsr_query
F: FC 3条降级为 NO_CONFLICT（FC 从主实验移除）

执行顺序：B→C→D→E→F→A（A 依赖最终 per_action_status，必须最后执行）
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── 路径配置 ──────────────────────────────────────────────────────────────────

INPUT_PATH = Path("data/macb_treatment_v3.jsonl")
OUTPUT_PATH = Path("data/macb_treatment_v4.jsonl")
REPORT_PATH = Path("data/macb_v4_cleanup_report.json")

# ── 合法枚举值 ────────────────────────────────────────────────────────────────

VALID_STATUSES = {
    "ADMISSIBLE",
    "CONDITIONALLY_ADMISSIBLE",
    "INADMISSIBLE_ABS",
    "NOT_INDICATED",
    "NOT_APPLICABLE",
}
VALID_CONFLICT_TYPES = {"SC_ABSOLUTE", "SC_RELATIVE", "NO_CONFLICT"}
VALID_OPTIONS = {"A", "B", "C", "D", "E"}

# ── B 组：SC_ABSOLUTE per_action_status 修复（添加 INADMISSIBLE_ABS）──────────
#
# 这些样本被标为 SC_ABSOLUTE 但无 INADMISSIBLE_ABS 选项。
# 通过添加 INADMISSIBLE_ABS 保持语义一致性。
#
SC_ABS_STATUS_FIXES: Dict[str, Dict[str, str]] = {
    "MACB-025": {"E": "INADMISSIBLE_ABS"},
    # Parkinson's 患者 + 甲氧氯普胺（多巴胺拮抗剂）= 绝对禁忌，会加重症状
    "MACB-040": {"A": "INADMISSIBLE_ABS"},
    # 可卡因胸痛 + 普萘洛尔（β阻滞剂）= 绝对禁忌（unopposed alpha vasospasm）
    "MACB-044": {"C": "INADMISSIBLE_ABS"},
    # 慢性丙型肝炎 + 甲氨蝶呤 = 肝毒性绝对禁忌
    "MACB-231": {"C": "INADMISSIBLE_ABS"},
    # WPW 综合征 + 维拉帕米 = 绝对禁忌（可加速旁路传导致 VF）
}

# ── B 组：SC_ABSOLUTE conflict_type 降级 ──────────────────────────────────────
#
# 这些样本无患者特异性约束创造真正的禁忌，降级到 SC_RELATIVE 或 NO_CONFLICT。
#
SC_ABS_DOWNGRADE: Dict[str, str] = {
    # 降级为 NO_CONFLICT（其他选项 NOT_APPLICABLE，无患者约束排除任何选项）
    "MACB-003": "NO_CONFLICT",   # 孕期滴虫病，错误选项均为错误药类（NOT_APPLICABLE）
    "MACB-023": "NO_CONFLICT",   # 新生儿低氧，PPV 为标准操作，无 SC 冲突
    "MACB-045": "NO_CONFLICT",   # 宫颈环扎 vs 孕酮，双重选项 ADMISSIBLE，无排除约束
    "MACB-050": "NO_CONFLICT",   # 新生儿败血症标准治疗，无冲突
    "MACB-055": "NO_CONFLICT",   # 孕期抗癫痫药 + 叶酸，无 SC 冲突
    "MACB-068": "NO_CONFLICT",   # 无症状菌尿：无患者特异性约束，纯临床知识题
    # 降级为 SC_RELATIVE（已有 CONDITIONALLY_ADMISSIBLE 选项，属于相对约束）
    "MACB-015": "SC_RELATIVE",   # 孕早期口服氟康唑相对禁忌，已有 COND_ADMISSIBLE
    "MACB-018": "SC_RELATIVE",   # 青霉素过敏梅毒，替代药物需监测，已有 COND_ADMISSIBLE
    "MACB-038": "SC_RELATIVE",   # 高剂量吸入激素 + 系统性唑类相互作用，已有 COND_ADMISSIBLE
    "MACB-060": "SC_RELATIVE",   # HIV 暴露新生儿，母体病毒载量阈值（相对约束）
    "MACB-061": "SC_RELATIVE",   # 子痫，不完整方案为 COND_ADMISSIBLE（需调整/完善）
    "MACB-063": "SC_RELATIVE",   # Rh 阴性孕妇，其他补充剂 COND_ADMISSIBLE（非优先）
}

# ── C 组：SC_RELATIVE 升级为 SC_ABSOLUTE ──────────────────────────────────────
#
# 患者条件已创造绝对禁忌（INADMISSIBLE_ABS 已存在），conflict_type 应为 SC_ABSOLUTE。
#
SC_REL_UPGRADE: Set[str] = {
    "MACB-077",   # CPVT 患儿，普鲁卡因胺/起搏器 INADMISSIBLE_ABS（已存在）
    "MACB-082",   # 1期低危高血压，立即用药 INADMISSIBLE_ABS（已存在）
    "MACB-085",   # 肾移植 + 低剂量阿昔洛韦 INADMISSIBLE_ABS（已存在），无相对约束
    "MACB-089",   # 开颅术后+ESRD，抗凝药 INADMISSIBLE_ABS（已存在）
    "MACB-093",   # 肾功能损害 + Metformin INADMISSIBLE_ABS（已存在）
    "MACB-101",   # CKD + 布洛芬 INADMISSIBLE_ABS（已存在）
    "MACB-106",   # 多药物滥用，手术/肿瘤治疗 INADMISSIBLE_ABS（已存在）
    "MACB-109",   # 侵袭性曲霉病+中性粒细胞减少，其他药类 INADMISSIBLE_ABS（已存在）
    "MACB-117",   # 室速，AV 结阻断药 INADMISSIBLE_ABS（已存在）
    "MACB-122",   # 链球菌肾小球肾炎+低血压，利尿药/ACEi INADMISSIBLE_ABS（已存在）
    "MACB-133",   # 心脏停搏非可除颤节律，除颤 INADMISSIBLE_ABS（已存在）
    "MACB-138",   # AML/TLS，促尿酸排泄药 INADMISSIBLE_ABS（已存在）
    "MACB-141",   # 狼疮肾炎，其他免疫抑制剂 INADMISSIBLE_ABS（已存在）
}

# ── C 组：SC_RELATIVE 降级为 NO_CONFLICT ──────────────────────────────────────
#
# 无患者特异性约束，均为普通临床知识题或偏好题。
#
SC_REL_DOWNGRADE: Set[str] = {
    "MACB-081",   # notes 明确"无矛盾或禁忌"
    "MACB-086",   # 播散性球孢子菌病，标准治疗，无患者约束
    "MACB-090",   # ACE 抑制剂咳嗽，临床偏好（换 ARB 或噻嗪）
    "MACB-091",   # 利福平引起的间质性肾炎，停药，无 SC 冲突
    "MACB-096",   # 高钾血症临床分期，无患者特异性禁忌药物
    "MACB-100",   # CKD+蛋白尿，全部 ADMISSIBLE，无约束
    "MACB-103",   # 急性肌张力障碍，苯扎托品标准治疗
    "MACB-104",   # 社区获得性肺炎门诊，临床偏好（多西环素/左氧氟沙星）
    "MACB-116",   # 重度低血糖，纯诊断/治疗知识题
    "MACB-123",   # 乳糜泻，无麸质饮食
    "MACB-127",   # 纤维肌痛，阿米替林
    "MACB-134",   # Whipple 病，静脉头孢曲松标准治疗
    "MACB-143",   # 扁桃体周围脓肿，切开引流+抗生素
    "MACB-147",   # II 期肺腺癌，肺叶切除
    "MACB-158",   # 肺出血肾炎综合征，血浆置换
}

# ── D 组：notes-status 矛盾修复 ──────────────────────────────────────────────
#
# 这些样本在 notes 中明确描述了禁忌/约束，但 per_action_status 未体现。
# 不在 B/C 组中的新增修复。
#
NOTES_STATUS_FIXES: Dict[str, Dict[str, Any]] = {
    # MACB-210 排除：v3 实际数据无 INADMISSIBLE_ABS（A-D 均为 NOT_INDICATED/NOT_APPLICABLE），
    # 会厌炎纯临床知识题，SC_ABSOLUTE 升级前提不成立，保持 NO_CONFLICT。
    "MACB-212": {
        # 哮喘患者 + SVT：腺苷相对禁忌（哮喘），普萘洛尔绝对禁忌（哮喘）
        "conflict_type": "SC_ABSOLUTE",
        "status": {
            "A": "CONDITIONALLY_ADMISSIBLE",  # 腺苷：哮喘相对禁忌 → 可用但需谨慎
            "D": "INADMISSIBLE_ABS",           # 普萘洛尔：哮喘绝对禁忌（非选择性β阻滞剂）
        },
    },
    "MACB-221": {
        # COPD 患者 + 新发房颤：普萘洛尔相对禁忌（COPD 气道痉挛风险）
        "conflict_type": "SC_RELATIVE",
        "status": {
            "E": "CONDITIONALLY_ADMISSIBLE",  # 普萘洛尔：COPD 患者需谨慎使用
        },
    },
    "MACB-223": {
        # 酒精戒断 + 阿片类药物使用中：纳洛酮绝对禁忌（拮抗镇痛效果，诱发戒断）
        "conflict_type": "SC_ABSOLUTE",
        "status": {
            "C": "INADMISSIBLE_ABS",  # 纳洛酮：正在用吗啡镇痛的患者绝对禁忌
        },
    },
    "MACB-229": {
        # 莱姆病 6岁患儿：多西环素相对禁忌（<8岁牙齿染色）
        "conflict_type": "SC_RELATIVE",
        "status": {
            "D": "CONDITIONALLY_ADMISSIBLE",  # 多西环素：<8岁儿童相对禁忌
        },
    },
    "MACB-235": {
        # CABG 后双联抗血小板 + 坏死性筋膜炎：单纯清创出血风险升高
        "conflict_type": "SC_RELATIVE",
        "status": {
            "B": "CONDITIONALLY_ADMISSIBLE",  # 立即清创：双抗血小板增加出血风险，需谨慎
        },
    },
    "MACB-236": {
        # 风湿热伴心脏炎：泼尼松绝对禁忌，短疗程青霉素相对禁忌（防护不足）
        "conflict_type": "SC_RELATIVE",
        "status": {
            "B": "INADMISSIBLE_ABS",          # 泼尼松：风湿热绝对禁忌
            "C": "CONDITIONALLY_ADMISSIBLE",  # 青霉素至40岁：疗程不足（相对禁忌）
            "D": "CONDITIONALLY_ADMISSIBLE",  # 青霉素至21岁：疗程不足
            "E": "CONDITIONALLY_ADMISSIBLE",  # 青霉素5年：疗程不足
        },
    },
}

# ── E 组：SCSR query 清洗 ─────────────────────────────────────────────────────
#
# gold_scsr_needed=false 但 gold_scsr_query 非空，容易误导评估脚本。
#
SCSR_CLEAR: Set[str] = {
    "MACB-040", "MACB-059", "MACB-122", "MACB-134", "MACB-157",
    "MACB-160", "MACB-162", "MACB-175", "MACB-176", "MACB-181",
    "MACB-182", "MACB-184", "MACB-185", "MACB-189", "MACB-190",
    "MACB-202", "MACB-207", "MACB-208", "MACB-212", "MACB-215",
}

# ── F 组：FC 降级为 NO_CONFLICT ───────────────────────────────────────────────
#
# MedQA 本质是单一答案设计，真实多指南冲突无法从中可靠构建。
# 这 3 条 FC 均为首选-替代差异或治疗优先级不同，非值域冲突。
# 降级后 INADMISSIBLE/CONDITIONALLY_ADMISSIBLE → NOT_INDICATED。
#
FC_DOWNGRADE: Set[str] = {
    "MACB-167",   # β地中海贫血特征：叶酸/安慰为首选-替代关系，非指南冲突
    "MACB-185",   # 大规模 PE：全部 ADMISSIBLE，治疗优先级不同，非值域矛盾
    "MACB-193",   # 狗咬伤：阿莫西林-克拉维酸 vs 克林/多西为首选-替代，非冲突
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

def apply_group_b(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    B 组：修复 16 条 SC_ABSOLUTE 无 INADMISSIBLE_ABS 的语义不一致。

    两种策略：
      1. 为应保留 SC_ABSOLUTE 的样本添加 INADMISSIBLE_ABS（SC_ABS_STATUS_FIXES）
      2. 将实际无患者约束的样本降级（SC_ABS_DOWNGRADE）

    返回：(sample, 修改次数)
    """
    sid = sample["sample_id"]
    count = 0

    # 策略1：添加 INADMISSIBLE_ABS
    if sid in SC_ABS_STATUS_FIXES:
        for option, new_status in SC_ABS_STATUS_FIXES[sid].items():
            old_status = sample["gold_per_action_status"].get(option)
            if old_status != new_status:
                _log(diff_log, sid, "B:status_fix",
                     f"gold_per_action_status.{option}", old_status, new_status)
                sample["gold_per_action_status"][option] = new_status
                count += 1

    # 策略2：conflict_type 降级（互斥：一个样本不会同时在两个表中）
    if sid in SC_ABS_DOWNGRADE:
        new_type = SC_ABS_DOWNGRADE[sid]
        old_type = sample.get("gold_conflict_type")
        if old_type != new_type:
            _log(diff_log, sid, "B:downgrade", "gold_conflict_type", old_type, new_type)
            sample["gold_conflict_type"] = new_type
            count += 1
        # 降级为 NO_CONFLICT 时，清理 INADMISSIBLE_ABS/CONDITIONALLY_ADMISSIBLE → NOT_INDICATED
        if new_type == "NO_CONFLICT":
            pas = sample["gold_per_action_status"]
            for option, status in list(pas.items()):
                if status in {"INADMISSIBLE_ABS", "CONDITIONALLY_ADMISSIBLE"}:
                    _log(diff_log, sid, "B:status_clean",
                         f"gold_per_action_status.{option}", status, "NOT_INDICATED")
                    pas[option] = "NOT_INDICATED"
                    count += 1

    return sample, count


def apply_group_c(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    C 组：修复 28 条 SC_RELATIVE 无 CONDITIONALLY_ADMISSIBLE 的语义不一致。

    升级（SC_RELATIVE → SC_ABSOLUTE）：患者条件已创造绝对禁忌。
    降级（SC_RELATIVE → NO_CONFLICT）：无患者特异性约束，普通临床知识题。

    返回：(sample, 修改次数)
    """
    sid = sample["sample_id"]
    count = 0

    if sid in SC_REL_UPGRADE:
        new_type = "SC_ABSOLUTE"
        old_type = sample.get("gold_conflict_type")
        if old_type != new_type:
            _log(diff_log, sid, "C:upgrade", "gold_conflict_type", old_type, new_type)
            sample["gold_conflict_type"] = new_type
            count += 1

    elif sid in SC_REL_DOWNGRADE:
        new_type = "NO_CONFLICT"
        old_type = sample.get("gold_conflict_type")
        if old_type != new_type:
            _log(diff_log, sid, "C:downgrade", "gold_conflict_type", old_type, new_type)
            sample["gold_conflict_type"] = new_type
            count += 1
        # 降级为 NO_CONFLICT 时，清理残留的 scope-inadmissible status
        pas = sample["gold_per_action_status"]
        for option, status in list(pas.items()):
            if status in {"INADMISSIBLE_ABS", "CONDITIONALLY_ADMISSIBLE"}:
                _log(diff_log, sid, "C:status_clean",
                     f"gold_per_action_status.{option}", status, "NOT_INDICATED")
                pas[option] = "NOT_INDICATED"
                count += 1

    return sample, count


def apply_group_d(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    D 组：修复 notes 中描述了禁忌/约束但 per_action_status 未体现的矛盾。

    包含：conflict_type 升级 + per_action_status 精确修复。

    返回：(sample, 修改次数)
    """
    sid = sample["sample_id"]
    if sid not in NOTES_STATUS_FIXES:
        return sample, 0

    count = 0
    fix = NOTES_STATUS_FIXES[sid]

    if "conflict_type" in fix:
        new_type = fix["conflict_type"]
        old_type = sample.get("gold_conflict_type")
        if old_type != new_type:
            _log(diff_log, sid, "D:conflict_type", "gold_conflict_type", old_type, new_type)
            sample["gold_conflict_type"] = new_type
            count += 1

    if "status" in fix:
        pas = sample["gold_per_action_status"]
        for option, new_status in fix["status"].items():
            old_status = pas.get(option)
            if old_status != new_status:
                _log(diff_log, sid, "D:status",
                     f"gold_per_action_status.{option}", old_status, new_status)
                pas[option] = new_status
                count += 1

    return sample, count


def apply_group_e(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    E 组：清空 gold_scsr_needed=false 样本的 gold_scsr_query。

    gold_scsr_needed=false 时存在非空 scsr_query 会误导评估脚本以为需要触发 SCSR。

    返回：(sample, 修改次数)
    """
    sid = sample["sample_id"]
    if sid not in SCSR_CLEAR:
        return sample, 0

    old_query = sample.get("gold_scsr_query", "")
    if old_query:
        _log(diff_log, sid, "E:scsr_clear", "gold_scsr_query", old_query, "")
        sample["gold_scsr_query"] = ""
        return sample, 1

    return sample, 0


def apply_group_f(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    F 组：将 3 条 FC 样本降级为 NO_CONFLICT。

    MedQA 本质是单一答案设计，无法可靠构建真实多指南冲突样本。
    降级时同时清理 INADMISSIBLE_ABS/CONDITIONALLY_ADMISSIBLE → NOT_INDICATED。

    返回：(sample, 修改次数)
    """
    sid = sample["sample_id"]
    if sid not in FC_DOWNGRADE:
        return sample, 0

    count = 0
    old_type = sample.get("gold_conflict_type")
    if old_type != "NO_CONFLICT":
        _log(diff_log, sid, "F:fc_downgrade", "gold_conflict_type", old_type, "NO_CONFLICT")
        sample["gold_conflict_type"] = "NO_CONFLICT"
        count += 1

    # 清理残留 scope-inadmissible status
    pas = sample["gold_per_action_status"]
    for option, status in list(pas.items()):
        if status in {"INADMISSIBLE_ABS", "CONDITIONALLY_ADMISSIBLE"}:
            _log(diff_log, sid, "F:status_clean",
                 f"gold_per_action_status.{option}", status, "NOT_INDICATED")
            pas[option] = "NOT_INDICATED"
            count += 1

    return sample, count


def apply_group_a(
    sample: Dict[str, Any], diff_log: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    """
    A 组：重定义 gold_admissible_set + 新增 gold_preferred_set。

    必须在 B-F 所有 per_action_status 修复完成后执行。

    理论依据：
      gold_admissible_set = {option | status ∈ {ADMISSIBLE, CONDITIONALLY_ADMISSIBLE}}
        = 完整 A(q) 支撑集（CONDITIONALLY_ADMISSIBLE 仍在 A(q) 内，只是需要调整）
      gold_preferred_set  = {option | status == ADMISSIBLE}
        = 无需调整即可使用的最优 action 集合

    返回：(sample, 修改次数)
    """
    sid = sample["sample_id"]
    pas = sample["gold_per_action_status"]

    new_admissible = sorted(
        opt for opt, status in pas.items()
        if status in {"ADMISSIBLE", "CONDITIONALLY_ADMISSIBLE"}
    )
    new_preferred = sorted(
        opt for opt, status in pas.items()
        if status == "ADMISSIBLE"
    )

    old_admissible = sample.get("gold_admissible_set", [])
    old_preferred = sample.get("gold_preferred_set")

    count = 0
    if sorted(old_admissible) != new_admissible:
        _log(diff_log, sid, "A:admissible_set",
             "gold_admissible_set", old_admissible, new_admissible)
        sample["gold_admissible_set"] = new_admissible
        count += 1

    if old_preferred != new_preferred:
        _log(diff_log, sid, "A:preferred_set",
             "gold_preferred_set", old_preferred, new_preferred)
        sample["gold_preferred_set"] = new_preferred
        count += 1

    return sample, count


# ── 验证 ──────────────────────────────────────────────────────────────────────

def validate_sample(sample: Dict[str, Any]) -> List[str]:
    """
    验证单个样本的 schema 完整性与语义一致性。

    一致性规则：
      SC_ABSOLUTE  → ∃ option with INADMISSIBLE_ABS
      SC_RELATIVE  → ∃ option with CONDITIONALLY_ADMISSIBLE
      NO_CONFLICT  → ∀ option: status ∈ {ADMISSIBLE, NOT_INDICATED, NOT_APPLICABLE}
      FC           → 不应存在（已全部移除）
      gold_admissible_set = ADMISSIBLE + CONDITIONALLY_ADMISSIBLE 选项
      gold_preferred_set  = ADMISSIBLE 选项

    返回：
        错误列表（空列表表示通过）
    """
    errors: List[str] = []
    sid = sample.get("sample_id", "UNKNOWN")

    pas = sample.get("gold_per_action_status", {})

    # 五键完整性
    missing = VALID_OPTIONS - set(pas.keys())
    if missing:
        errors.append(f"{sid}: gold_per_action_status 缺少选项 {sorted(missing)}")

    # status 合法值
    for opt, status in pas.items():
        if status not in VALID_STATUSES:
            errors.append(f"{sid}.{opt}: 非法 status '{status}'")

    # conflict_type 合法值（FC 不应存在）
    conflict_type = sample.get("gold_conflict_type", "")
    if conflict_type not in VALID_CONFLICT_TYPES:
        errors.append(f"{sid}: 非法 gold_conflict_type '{conflict_type}'（FC 应已移除）")

    # 语义一致性规则
    statuses = set(pas.values())

    if conflict_type == "SC_ABSOLUTE" and "INADMISSIBLE_ABS" not in statuses:
        errors.append(
            f"{sid}: SC_ABSOLUTE 样本无 INADMISSIBLE_ABS 选项（语义不一致）"
        )

    if conflict_type == "SC_RELATIVE" and "CONDITIONALLY_ADMISSIBLE" not in statuses:
        errors.append(
            f"{sid}: SC_RELATIVE 样本无 CONDITIONALLY_ADMISSIBLE 选项（语义不一致）"
        )

    if conflict_type == "NO_CONFLICT":
        forbidden = {"INADMISSIBLE_ABS", "CONDITIONALLY_ADMISSIBLE"}
        present = statuses & forbidden
        if present:
            errors.append(
                f"{sid}: NO_CONFLICT 样本含禁用 status {present}"
            )

    # gold_admissible_set 与 gold_preferred_set 的内容
    expected_admissible = sorted(
        opt for opt, s in pas.items()
        if s in {"ADMISSIBLE", "CONDITIONALLY_ADMISSIBLE"}
    )
    expected_preferred = sorted(
        opt for opt, s in pas.items()
        if s == "ADMISSIBLE"
    )
    actual_admissible = sorted(sample.get("gold_admissible_set", []))
    actual_preferred = sorted(sample.get("gold_preferred_set", []))

    if actual_admissible != expected_admissible:
        errors.append(
            f"{sid}: gold_admissible_set={actual_admissible} 应为 {expected_admissible}"
        )
    if actual_preferred != expected_preferred:
        errors.append(
            f"{sid}: gold_preferred_set={actual_preferred} 应为 {expected_preferred}"
        )

    return errors


def compute_distribution(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算清洗后数据分布，用于 diff 报告和论文统计。

    返回：
        包含各关键字段分布的字典
    """
    conflict_counter: Counter = Counter()
    status_counter: Counter = Counter()
    candidate_tag_counter: Counter = Counter()

    for s in samples:
        conflict_counter[s.get("gold_conflict_type", "UNKNOWN")] += 1
        candidate_tag_counter[s.get("candidate_tag", "UNKNOWN")] += 1
        for status in s.get("gold_per_action_status", {}).values():
            status_counter[status] += 1

    sc_abs_samples = [
        s for s in samples if s.get("gold_conflict_type") == "SC_ABSOLUTE"
    ]
    sc_rel_samples = [
        s for s in samples if s.get("gold_conflict_type") == "SC_RELATIVE"
    ]
    avg_inadmissible_per_sc_abs = (
        sum(
            sum(1 for v in s["gold_per_action_status"].values() if v == "INADMISSIBLE_ABS")
            for s in sc_abs_samples
        ) / len(sc_abs_samples) if sc_abs_samples else 0
    )
    avg_cond_per_sc_rel = (
        sum(
            sum(1 for v in s["gold_per_action_status"].values() if v == "CONDITIONALLY_ADMISSIBLE")
            for s in sc_rel_samples
        ) / len(sc_rel_samples) if sc_rel_samples else 0
    )

    return {
        "gold_conflict_type": dict(conflict_counter),
        "status_counts": dict(status_counter),
        "candidate_tag": dict(candidate_tag_counter),
        "sc_absolute_count": len(sc_abs_samples),
        "sc_relative_count": len(sc_rel_samples),
        "avg_inadmissible_per_sc_abs": round(avg_inadmissible_per_sc_abs, 2),
        "avg_cond_admissible_per_sc_rel": round(avg_cond_per_sc_rel, 2),
    }


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    主入口：按 B→C→D→E→F→A 顺序执行清洗，输出 v4 JSONL 和 diff 报告。

    执行顺序设计原则：
      - B/C/D 修改 per_action_status 和 conflict_type
      - E 清理 SCSR query（独立，不影响其他字段）
      - F 降级 FC 并清理 status（依赖 per_action_status 的最终状态）
      - A 在所有 status 修改完成后，重新派生 gold_admissible_set 和 gold_preferred_set
      - 验证在 A 之后执行，确保所有字段一致
    """
    # 第一阶段：加载 v3 数据
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

    # 第二～七阶段：逐样本应用 B→C→D→E→F→A
    diff_log: List[Dict[str, Any]] = []
    group_counts: Dict[str, int] = defaultdict(int)

    for sample in samples:
        sample, n = apply_group_b(sample, diff_log)
        group_counts["B"] += n

        sample, n = apply_group_c(sample, diff_log)
        group_counts["C"] += n

        sample, n = apply_group_d(sample, diff_log)
        group_counts["D"] += n

        sample, n = apply_group_e(sample, diff_log)
        group_counts["E"] += n

        sample, n = apply_group_f(sample, diff_log)
        group_counts["F"] += n

    print(f"[第二~六阶段] B-F 修复完成")

    # 第七阶段：A - 重派生 gold_admissible_set 和 gold_preferred_set
    for sample in samples:
        sample, n = apply_group_a(sample, diff_log)
        group_counts["A"] += n

    print(f"[第七阶段] A - gold_admissible_set/preferred_set 重派生完成")

    # 第八阶段：一致性验证
    all_errors: List[str] = []
    for sample in samples:
        all_errors.extend(validate_sample(sample))

    if all_errors:
        print(f"\n[验证失败] 发现 {len(all_errors)} 个错误：", file=sys.stderr)
        for err in all_errors[:30]:
            print(f"  - {err}", file=sys.stderr)
        if len(all_errors) > 30:
            print(f"  ... 以及另外 {len(all_errors) - 30} 个错误", file=sys.stderr)
        raise ValueError(f"验证失败，v4 文件未写出。")

    print("[第八阶段] 验证通过：所有样本 schema 和语义一致性合法")

    # 第九阶段：输出 v4 JSONL 和 diff 报告
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
    print(f"\n[第九阶段] 输出完成，总变更 {total_changes} 处")
    print(f"\n变更统计：")
    for group in ["B", "C", "D", "E", "F", "A"]:
        print(f"  {group}: {group_counts.get(group, 0)} 处")

    print(f"\n新 gold_conflict_type 分布：")
    for k, v in sorted(distribution["gold_conflict_type"].items()):
        print(f"  {k}: {v}")

    print(f"\n新 status 分布：")
    for k, v in sorted(distribution["status_counts"].items()):
        print(f"  {k}: {v}")

    print(f"\n语义一致性检验：")
    print(f"  SC_ABSOLUTE 平均 INADMISSIBLE_ABS 数/样本: {distribution['avg_inadmissible_per_sc_abs']}")
    print(f"  SC_RELATIVE 平均 CONDITIONALLY_ADMISSIBLE 数/样本: {distribution['avg_cond_admissible_per_sc_rel']}")

    print(f"\n输出文件：{OUTPUT_PATH}")
    print(f"Diff 报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
