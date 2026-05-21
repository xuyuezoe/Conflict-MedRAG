#!/usr/bin/env python3
"""
MACB Benchmark 严格 Schema 验证器。

输出所有违规条目，退出码：0=通过，1=有错误。
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

VALID_LETTERS = {"A", "B", "C", "D", "E"}
VALID_STATUSES = {"ADMISSIBLE", "INADMISSIBLE_ABS", "INADMISSIBLE_REL",
                  "NOT_APPLICABLE", "CONDITIONALLY_ADMISSIBLE"}
VALID_CONFLICT_LABELS = {"CONFLICT", "NO_CONFLICT", "NOT_APPLICABLE"}
VALID_TAGS = {"SC_ABSOLUTE_CAND", "SC_RELATIVE_CAND", "FC_CAND", "TREATMENT_ONLY"}


def check(
    benchmark_path: str = "data/macb_treatment_v2.jsonl",
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    验证 benchmark 文件，返回 (错误列表, 有效样本列表)。
    """
    path = Path(benchmark_path)
    samples: List[Dict[str, Any]] = []
    errors: List[str] = []

    # ── 第一阶段：可解析性 ──────────────────────────────────────────
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"[L{i}] JSON 解析失败: {e}")

    print(f"总行数: {len(samples)} 条可解析")
    if errors:
        for e in errors:
            print(f"  {e}")
        return errors, samples

    # ── 第二阶段：逐样本检查 ────────────────────────────────────────
    seen_ids: set = set()
    for s in samples:
        sid = s.get("sample_id", "UNKNOWN")

        # 1. sample_id 重复
        if sid in seen_ids:
            errors.append(f"[{sid}] sample_id 重复")
        seen_ids.add(sid)

        label = s.get("gold_memory_conflict_label", "")
        tag = s.get("candidate_tag", "")
        ans = s.get("answer_idx", "").strip()

        # 2. candidate_tag 合法性
        if tag not in VALID_TAGS:
            errors.append(f"[{sid}] candidate_tag 非法: {repr(tag)}")

        # 3. gold_memory_conflict_label 合法性
        if label not in VALID_CONFLICT_LABELS:
            errors.append(f"[{sid}] gold_memory_conflict_label 非法: {repr(label)}")

        # NOT_APPLICABLE 样本不做以下检查（它们应被移出主集）
        if label == "NOT_APPLICABLE":
            errors.append(f"[{sid}] NOT_APPLICABLE 样本仍在主集，应移至 excluded.jsonl")
            continue

        # 4. answer_idx 缺失
        if not ans:
            errors.append(f"[{sid}] answer_idx 为空")

        # 5. gold_per_action_status 结构检查
        pas = s.get("gold_per_action_status")
        if not isinstance(pas, dict):
            errors.append(f"[{sid}] gold_per_action_status 不是 dict: {type(pas).__name__}")
            pas = {}

        # 5a. 键必须是 A-E
        bad_keys = [k for k in pas.keys() if k not in VALID_LETTERS]
        if bad_keys:
            errors.append(f"[{sid}] per_action_status 非法键: {bad_keys[:5]}")

        # 5b. 值必须是合法 status 字符串（非嵌套 dict）
        for k, v in pas.items():
            if isinstance(v, dict):
                errors.append(f"[{sid}] per_action_status[{k}] 是嵌套 dict，应为字符串")
            elif not isinstance(v, str):
                errors.append(f"[{sid}] per_action_status[{k}] 类型错误: {type(v).__name__}")
            elif v not in VALID_STATUSES:
                errors.append(f"[{sid}] per_action_status[{k}] 非法值: {repr(v)}")

        # 6. gold_admissible_set 类型和一致性
        admissible = s.get("gold_admissible_set")
        if not isinstance(admissible, list):
            errors.append(f"[{sid}] gold_admissible_set 不是 list: {type(admissible).__name__}")
            admissible = []

        # 6a. admissible_set 的元素应都在 per_action_status 的 ADMISSIBLE 键中
        if isinstance(pas, dict) and bad_keys == []:
            expected_admissible = {
                k for k, v in pas.items()
                if isinstance(v, str) and v in {"ADMISSIBLE", "CONDITIONALLY_ADMISSIBLE"}
            }
            actual_admissible = set(admissible)
            if expected_admissible != actual_admissible:
                errors.append(
                    f"[{sid}] gold_admissible_set 与 per_action_status 不一致: "
                    f"expected={sorted(expected_admissible)}, got={sorted(actual_admissible)}"
                )

        # 6b. answer_idx 必须在 admissible_set 中（格式正常时）
        if ans and admissible and bad_keys == [] and ans not in admissible:
            errors.append(f"[{sid}] answer_idx={ans} 不在 gold_admissible_set={admissible}")

        # 7. gold_scsr_needed 逻辑一致性
        scsr_needed = s.get("gold_scsr_needed", False)
        has_inadmissible = isinstance(pas, dict) and any(
            isinstance(v, str) and v in {"INADMISSIBLE_ABS", "INADMISSIBLE_REL"}
            for v in pas.values()
        )
        if scsr_needed and not has_inadmissible and label == "NO_CONFLICT":
            errors.append(
                f"[{sid}] gold_scsr_needed=True 但无 inadmissible 选项且 label=NO_CONFLICT"
            )
        scsr_query = s.get("gold_scsr_query", "").strip()
        if scsr_needed and not scsr_query:
            errors.append(f"[{sid}] gold_scsr_needed=True 但 gold_scsr_query 为空")

        # 8. parametric_prior_disease_query 缺失
        prior = s.get("parametric_prior_disease_query", "").strip()
        if not prior:
            errors.append(f"[{sid}] parametric_prior_disease_query 为空")

        # 9. FC_CAND 不应有 INADMISSIBLE_REL（SC_RELATIVE 约束）
        # （注：FC_CAND 可以有 INADMISSIBLE_ABS，但不应有 SC_RELATIVE 类约束）
        # 此规则暂为警告级别
        if tag == "FC_CAND" and isinstance(pas, dict):
            rel = [k for k, v in pas.items() if isinstance(v, str) and v == "INADMISSIBLE_REL"]
            if rel:
                errors.append(f"[{sid}] FC_CAND 含 INADMISSIBLE_REL（应无患者约束）: {rel}")

    return errors, samples


def print_summary(errors: List[str], samples: List[Dict[str, Any]]) -> None:
    """打印分类汇总。"""
    by_type: Dict[str, List[str]] = defaultdict(list)
    for e in errors:
        # 提取 sample_id
        if e.startswith("[MACB-") or e.startswith("[L"):
            sid = e[1:e.index("]")]
        else:
            sid = "?"
        # 提取错误类型（首词）
        body = e[e.index("]") + 2:]
        etype = body.split(" ")[0] if body else "other"
        by_type[etype].append(sid)

    print(f"\n{'='*60}")
    print(f"验证结果: {len(errors)} 个错误，{len(samples)} 个样本")
    print(f"{'='*60}")
    for etype, sids in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  [{len(sids):3d}] {etype}: {', '.join(sorted(set(sids))[:8])}{'...' if len(set(sids))>8 else ''}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="data/macb_treatment_v2.jsonl")
    args = p.parse_args()

    errors, samples = check(args.benchmark)
    print_summary(errors, samples)
    sys.exit(0 if not errors else 1)
