#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REQUIRED = ["question", "options", "answer_idx"]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} JSON 解析失败: {e}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="MedQA jsonl 文件路径")
    args = p.parse_args()

    path = Path(args.input)
    rows = load_jsonl(path)

    bad = 0
    option_stats = {}
    for ln, obj in rows:
        for k in REQUIRED:
            if k not in obj:
                print(f"[MISSING] line {ln}: 缺少字段 {k}")
                bad += 1
        options = obj.get("options", {})
        if not isinstance(options, dict):
            print(f"[BAD] line {ln}: options 不是 dict")
            bad += 1
            continue
        option_count = len(options)
        option_stats[option_count] = option_stats.get(option_count, 0) + 1
        ans = obj.get("answer_idx")
        if ans not in options:
            print(f"[BAD] line {ln}: answer_idx={ans} 不在 options keys={list(options.keys())}")
            bad += 1

    print("=" * 50)
    print(f"文件: {path}")
    print(f"总样本: {len(rows)}")
    print(f"异常计数: {bad}")
    print(f"选项数分布: {option_stats}")


if __name__ == "__main__":
    main()
