#!/usr/bin/env python3
"""
结果分析与论文表格生成器

读取 results/ 目录下的评估结果，生成：
  1. 各系统指标汇总表（LaTeX 格式）
  2. SC_ABSOLUTE / SC_RELATIVE / FC 的分层分析
  3. 消融实验表格（MARC vs MARC-noDCR vs MARC-noSCSR）
  4. 显著性检验（paired t-test）

使用方法：
  python3 eval/result_analysis.py --results-dir results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_per_sample_results(
    system_dir: Path,
) -> List[Dict]:
    """
    从 per_sample.jsonl 加载单系统的 per-sample 结果。

    参数：
        system_dir: results/{system_name}/ 目录

    返回：
        per-sample 结果字典列表
    """
    per_sample_path = system_dir / "per_sample.jsonl"
    if not per_sample_path.exists():
        raise FileNotFoundError(
            f"[result_analysis] per_sample.jsonl 不存在: {per_sample_path}"
        )
    results = []
    with per_sample_path.open(encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def load_summary(system_dir: Path) -> Dict:
    """加载 summary.json"""
    summary_path = system_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"[result_analysis] summary.json 不存在: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def generate_latex_table(
    summaries: Dict[str, Dict],
    systems_order: Optional[List[str]] = None,
) -> str:
    """
    生成 LaTeX 格式的指标汇总表。

    参数：
        summaries:     {system_name: summary_dict}
        systems_order: 系统排列顺序（None 时按字母序）

    返回：
        LaTeX tabular 字符串
    """
    if systems_order is None:
        systems_order = sorted(summaries.keys())

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"System & CRR$\downarrow$ & SDR$\uparrow$ & AEC$\uparrow$ & FC-AA$\uparrow$ & SLR$\downarrow$ \\",
        r"\hline",
    ]

    for system_name in systems_order:
        if system_name not in summaries:
            continue
        m = summaries[system_name].get("metrics", {})

        def fmt(key: str) -> str:
            val = m.get(key, {}).get("mean", float("nan"))
            if val != val:
                return "N/A"
            ci_lo = m.get(key, {}).get("ci_lower", float("nan"))
            ci_hi = m.get(key, {}).get("ci_upper", float("nan"))
            if ci_lo == ci_lo and ci_hi == ci_hi:
                return f"{val:.3f}$_{{\\pm {(ci_hi - ci_lo)/2:.3f}}}$"
            return f"{val:.3f}"

        display_name = system_name.replace("_", "\\_")
        row = f"{display_name} & {fmt('CRR')} & {fmt('SDR')} & {fmt('AEC')} & {fmt('FC-AA')} & {fmt('SLR')} \\\\"
        lines.append(row)

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{MARC vs. Baseline systems on MACB. Metrics: CRR (contraindicated recommendation rate), "
        r"SDR (SC detection recall), AEC (alternative evidence coverage), "
        r"FC-AA (FC accuracy within $\mathcal{A}(q)$), SLR (source leakage rate). "
        r"Values are mean $\pm$ 95\% CI (bootstrap).}",
        r"\label{tab:main_results}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def layered_analysis(
    results_dir: Path,
    system_names: List[str],
) -> None:
    """
    分层分析：对 SC_ABSOLUTE / SC_RELATIVE / FC 三类样本分别统计。

    参数：
        results_dir:  结果目录
        system_names: 待分析系统列表
    """
    print("\n[分层分析] SC_ABSOLUTE / SC_RELATIVE / FC / MIXED")
    print("="*60)

    for system_name in system_names:
        system_dir = results_dir / system_name
        if not (system_dir / "per_sample.jsonl").exists():
            continue

        per_sample = load_per_sample_results(system_dir)

        # 按 tag 分组统计 CRR
        tag_crr: Dict[str, List[float]] = {}
        for row in per_sample:
            tag = row.get("candidate_tag", "UNKNOWN")
            gold = row.get("gold_per_action_status", {})
            pred = row.get("per_action_status_pred", {})

            inadmissible = [a for a, s in gold.items() if s == "INADMISSIBLE_ABS"]
            if not inadmissible:
                continue

            pred_lower = {k.lower(): v for k, v in pred.items()}
            crr = 0.0
            for action in inadmissible:
                for pk, pv in pred_lower.items():
                    if action.lower() in pk and pv == "RECOMMENDED":
                        crr = 1.0
                        break

            tag_crr.setdefault(tag, []).append(crr)

        print(f"\n{system_name}:")
        for tag, vals in sorted(tag_crr.items()):
            mean = sum(vals) / len(vals)
            print(f"  {tag:30s}: CRR={mean:.3f} (n={len(vals)})")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="结果分析与论文表格生成")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="分析指定系统（默认分析 results/ 下所有系统）",
    )
    p.add_argument(
        "--latex",
        action="store_true",
        help="输出 LaTeX 格式表格",
    )
    args = p.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(
            f"[result_analysis] results 目录不存在: {args.results_dir}。"
            f"请先运行 experiments/run_experiments.py。"
        )

    # 发现所有系统
    system_dirs = [d for d in args.results_dir.iterdir() if d.is_dir()]
    if args.systems:
        system_dirs = [d for d in system_dirs if d.name in args.systems]

    if not system_dirs:
        raise ValueError(f"[result_analysis] 未找到任何系统结果目录: {args.results_dir}")

    # 加载汇总数据
    summaries: Dict[str, Dict] = {}
    for system_dir in system_dirs:
        try:
            summaries[system_dir.name] = load_summary(system_dir)
        except FileNotFoundError as e:
            print(f"跳过（{e}）", file=sys.stderr)

    # 打印汇总表
    print("\n[汇总指标]")
    print(f"{'系统':<25} {'CRR↓':>8} {'SDR↑':>8} {'AEC↑':>8} {'FC-AA↑':>8} {'SLR↓':>8}")
    print("-"*70)
    for name, summary in sorted(summaries.items()):
        m = summary.get("metrics", {})
        def fmt(key: str) -> str:
            v = m.get(key, {}).get("mean", float("nan"))
            return f"{v:.4f}" if v == v else "  N/A"
        print(f"{name:<25} {fmt('CRR'):>8} {fmt('SDR'):>8} {fmt('AEC'):>8} {fmt('FC-AA'):>8} {fmt('SLR'):>8}")

    # LaTeX 表格
    if args.latex:
        latex = generate_latex_table(summaries)
        print("\n[LaTeX 表格]")
        print(latex)

        latex_path = args.results_dir / "main_results.tex"
        latex_path.write_text(latex, encoding="utf-8")
        print(f"\nLaTeX → {latex_path}")

    # 分层分析
    layered_analysis(args.results_dir, list(summaries.keys()))


if __name__ == "__main__":
    main()
