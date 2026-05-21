#!/usr/bin/env python3
"""
MACB 标注表导出器

将候选池 jsonl 转换为人工标注 CSV 表格。
每行对应一个候选样本，标注者需填写核心字段。

新增列（相对于 GPT v1）：
  - no_keyword_flag        : 来自候选池，提醒标注者此题没有明显关键词信号
  - parametric_prior_disease_query : LLM 生成前的疾病主干查询（由 generate_parametric_prior.py 填充）
  - gold_memory_conflict_label     : 标注者填写，是否存在 context-memory 冲突

标注者需填写的字段（留空）：
  patient_profile_json、gold_admissible_set_json、gold_per_action_status_json、
  gold_scsr_needed、gold_scsr_query、gold_memory_conflict_label、reviewer、notes
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Iterator


def load_jsonl(path: Path) -> Iterator[dict]:
    """逐行加载 jsonl 文件，遇到格式错误立即抛出。"""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[JSON解析失败] {path}:{i} → {e}")


def options_to_text(options: dict) -> str:
    """将选项 dict 序列化为可读文本，用于 CSV 展示。"""
    if not isinstance(options, dict):
        return ""
    return " | ".join(f"{k}: {v}" for k, v in sorted(options.items()))


HEADER = [
    # ── 只读字段（来自候选池） ──────────────────────────────
    "sample_id",
    "candidate_tag",
    "no_keyword_flag",           # 新增：提醒标注者无明显关键词
    "question",
    "options_text",
    "answer_idx",
    "answer",
    "meta_info",
    "source_file",
    "parametric_prior_disease_query",  # 新增：由 generate_parametric_prior.py 填充
    # ── 标注者填写字段 ──────────────────────────────────────
    "patient_profile_json",            # 患者 profile JSON
    "gold_admissible_set_json",        # 可行 action 集合 JSON 数组
    "gold_per_action_status_json",     # per-action 状态 JSON 对象
    "gold_scsr_needed",                # 是否需要 SCSR（True/False）
    "gold_scsr_query",                 # SCSR 查询文本（手工构造，不依赖 LLM）
    "gold_memory_conflict_label",      # 新增：context-memory 冲突标注（yes/no/uncertain）
    "reviewer",
    "notes",
]


def main() -> None:
    p = argparse.ArgumentParser(description="MACB 标注表导出器")
    p.add_argument("--input",  required=True, help="候选池 jsonl 路径")
    p.add_argument("--output", required=True, help="输出 CSV 路径")
    p.add_argument(
        "--tag-filter",
        nargs="*",
        help="只导出指定 tag 的样本（默认全部）",
    )
    args = p.parse_args()

    rows = list(load_jsonl(Path(args.input)))

    if args.tag_filter:
        rows = [r for r in rows if r.get("candidate_tag") in args.tag_filter]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for i, r in enumerate(rows, start=1):
            # parametric_prior_stub 中的 disease_only_query
            pp_stub = r.get("parametric_prior_stub") or {}
            pp_query = pp_stub.get("disease_only_query", "")

            writer.writerow([
                # ── 只读字段 ──
                f"MACB-{i:03d}",
                r.get("candidate_tag", ""),
                str(r.get("no_keyword_flag", False)).lower(),
                r.get("question", ""),
                options_to_text(r.get("options", {})),
                r.get("answer_idx", ""),
                r.get("answer", ""),
                r.get("meta_info", ""),
                r.get("source_file", ""),
                pp_query,
                # ── 标注者填写（初始留空） ──
                "{}",   # patient_profile_json
                "[]",   # gold_admissible_set_json
                "{}",   # gold_per_action_status_json
                "",     # gold_scsr_needed
                "",     # gold_scsr_query
                "",     # gold_memory_conflict_label
                "",     # reviewer
                "",     # notes
            ])

    # 输出标注表分布统计
    from collections import Counter
    tag_dist = Counter(r.get("candidate_tag", "UNKNOWN") for r in rows)
    no_kw_count = sum(1 for r in rows if r.get("no_keyword_flag"))
    print(f"标注表写出 → {out}  ({len(rows)} 行)")
    print(f"  tag 分布: {dict(tag_dist)}")
    print(f"  no_keyword_flag=True: {no_kw_count} 条")


if __name__ == "__main__":
    main()
