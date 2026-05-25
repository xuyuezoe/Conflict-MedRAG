#!/usr/bin/env python3
"""
评估入口

批量评估单个系统在 MACB 上的性能：
  1. 读取 MACB benchmark（data/macb_v1.jsonl）
  2. 对每个样本调用 system.run(query)
  3. 计算 5 个指标（CRR/SDR/AEC/FC-AA/SLR）
  4. 写出 per-sample 结果和汇总报告

使用方法（由 experiments/run_experiments.py 调用，也可独立运行）：
  python3 eval/evaluate.py \
      --benchmark data/macb_v1.jsonl \
      --system marc \
      --output-dir results \
      --limit 10  # 调试用
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.base import BaseRAGSystem
from eval.metrics import (
    compute_all_metrics,
    compute_crr, compute_sdr, compute_aec, compute_slr_from_result,
    compute_cdr_metric, compute_rsi_metric, compute_caec,
)
from src.types import EvalSample, SampleResult
from src.run_logger import RunLogger


def load_benchmark(benchmark_path: Path) -> List[EvalSample]:
    """
    从 JSONL 文件加载 MACB 样本。

    参数：
        benchmark_path: macb_v1.jsonl 路径

    返回：
        EvalSample 列表

    异常：
        FileNotFoundError: 文件不存在
        ValueError: JSON 格式非法（不兜底）
    """
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"[evaluate] MACB benchmark 不存在: {benchmark_path}。"
            f"请先完成标注并运行 build_macb_final.py。"
        )

    samples: List[EvalSample] = []
    with benchmark_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[evaluate] MACB {benchmark_path}:{i} JSON 解析失败: {e}")

            samples.append(EvalSample(
                sample_id=row["sample_id"],
                query=row["query"],
                options_text=row.get("options_text", ""),
                answer_idx=row.get("answer_idx", ""),
                candidate_tag=row.get("candidate_tag", ""),
                patient_profile=row.get("patient_profile") or {},
                gold_admissible_set=row.get("gold_admissible_set") or [],
                gold_per_action_status=row.get("gold_per_action_status") or {},
                gold_scsr_needed=bool(row.get("gold_scsr_needed", False)),
                gold_scsr_query=row.get("gold_scsr_query"),
                parametric_prior_conflict_label=row.get("parametric_prior_conflict_label") or row.get("gold_memory_conflict_label", ""),
                parametric_prior_disease_query=row.get("parametric_prior_disease_query", ""),
                gold_preferred_set=row.get("gold_preferred_set") or [],
                gold_conflict_types_present=row.get("gold_conflict_types_present") or [],
                task_type=row.get("task_type", "treatment_recommendation"),
            ))

    return samples


def evaluate_system(
    system: BaseRAGSystem,
    benchmark: List[EvalSample],
    output_dir: Path,
    sample_limit: Optional[int] = None,
    log_dir: Optional[Path] = None,
) -> dict:
    """
    对单个系统在完整 MACB 上进行评估。

    参数：
        system:       待评估系统（实现 BaseRAGSystem 接口）
        benchmark:    EvalSample 列表
        output_dir:   结果输出目录（results/{system_name}/）
        sample_limit: 调试用限制（None = 全量评估）
        log_dir:      运行日志目录（run_log/），None 表示不写详细日志

    返回：
        指标汇总字典 {"CRR": {...}, "SDR": {...}, ...}
    """
    samples = benchmark if sample_limit is None else benchmark[:sample_limit]
    system_dir = output_dir / system.system_name
    system_dir.mkdir(parents=True, exist_ok=True)

    # ── 初始化 RunLogger（若 log_dir 非空） ──────────────────────────────────
    logger: Optional[RunLogger] = None
    if log_dir is not None:
        import os
        from src.llm_client import get_model, get_embedding_model
        log_config = {
            "system":          system.system_name,
            "benchmark":       str(output_dir.parent / "data" / "macb_treatment_v5.jsonl"),
            "n_samples":       len(samples),
            "llm_model":       get_model(),
            "embedding_model": get_embedding_model(),
        }
        logger = RunLogger(
            system_name=system.system_name,
            log_dir=log_dir,
            config=log_config,
        )

    results: List[SampleResult] = []
    errors: List[str] = []

    per_sample_path = system_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as out_f:
        for i, sample in enumerate(samples, start=1):
            # 构造输入 query（question + options）
            query = f"{sample.query}\n\nOptions: {sample.options_text}"

            print(f"  [{i}/{len(samples)}] {sample.sample_id} ({sample.candidate_tag})", end="", flush=True)
            t0 = time.time()
            try:
                result = system.run(
                    query=query,
                    sample_id=sample.sample_id,
                    patient_profile=sample.patient_profile,
                )
                latency = time.time() - t0
                print(f" ✓ ({latency:.1f}s)")
                results.append(result)

                # ── RunLogger：计算 per-sample 指标并写详细日志 ────────────────
                if logger is not None:
                    per_m = {
                        "crr":  compute_crr(result, sample),
                        "sdr":  compute_sdr(result, sample),
                        "aec":  compute_aec(result, sample),
                        "slr":  compute_slr_from_result(result),
                        "cdr":  compute_cdr_metric(result),
                        "rsi":  compute_rsi_metric(result),
                        "caec": compute_caec(result, sample),
                    }
                    logger.log_sample(result, sample, per_m, latency)

                # 写出 per-sample 结果
                row = {
                    "sample_id":             result.sample_id,
                    "system_name":           result.system_name,
                    "candidate_tag":         sample.candidate_tag,
                    "predicted_answer":      result.predicted_answer[:200],
                    "per_action_status_pred": result.per_action_status_pred,
                    "gold_per_action_status": sample.gold_per_action_status,
                    "scsr_triggered":        result.scsr_triggered,
                    "srl_violations":        result.srl_violations,
                    "latency_s":             round(latency, 2),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            except Exception as e:
                latency = time.time() - t0
                error_msg = f"{sample.sample_id}: {type(e).__name__}: {e}"
                errors.append(error_msg)
                print(f" ✗ ({latency:.1f}s) {type(e).__name__}: {str(e)[:80]}")

    # 计算指标：仅在 treatment_recommendation 子集上计算（排除 contraindication_recognition）
    # samples[:len(results)] 与 results 一一对应（按顺序截取）
    finished_samples = samples[:len(results)]
    main_pairs = [
        (r, s) for r, s in zip(results, finished_samples)
        if s.task_type == "treatment_recommendation"
    ]
    main_results = [p[0] for p in main_pairs]
    main_samples = [p[1] for p in main_pairs]
    n_excluded = len(results) - len(main_results)

    metric_summary = compute_all_metrics(main_results, main_samples)

    # 写出汇总报告
    report = {
        "system": system.system_name,
        "n_total": len(samples),
        "n_success": len(results),
        "n_excluded_contraindication": n_excluded,
        "n_evaluated": len(main_results),
        "n_errors": len(errors),
        "errors": errors[:10],
        "metrics": metric_summary,
    }
    report_path = system_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── RunLogger：写最终汇总 ─────────────────────────────────────────────────
    if logger is not None:
        logger.finalize(metric_summary)

    print(f"\n  [结果] 写出 → {report_path}")
    print(f"  CRR:  {metric_summary['CRR']['mean']:.4f} ({metric_summary['CRR']['n']} 个样本)")
    print(f"  SDR:  {metric_summary['SDR']['mean']:.4f} ({metric_summary['SDR']['n']} 个样本)")
    print(f"  AEC:  {metric_summary['AEC']['mean']:.4f} ({metric_summary['AEC']['n']} 个样本)")
    print(f"  SLR:  {metric_summary['SLR']['mean']:.4f} ({metric_summary['SLR']['n']} 个样本)")
    # CDR/RSI/CAEC 仅在 ScopeIndex 已加载时有值
    if metric_summary['CDR']['n'] > 0:
        print(f"  CDR:  {metric_summary['CDR']['mean']:.4f} ({metric_summary['CDR']['n']} 个样本) [scope index]")
        print(f"  RSI:  {metric_summary['RSI']['mean']:.4f} ({metric_summary['RSI']['n']} 个样本) [scope index]")
        print(f"  CAEC: {metric_summary['CAEC']['mean']:.4f} ({metric_summary['CAEC']['n']} 个样本) [scope index]")

    if errors:
        print(f"  警告：{len(errors)} 个样本运行失败（见 {report_path}）")

    return metric_summary
