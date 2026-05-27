#!/usr/bin/env python3
"""
统一实验入口

对所有系统执行 MACB 评估，汇总结果表格。

使用方法：
  # 调试（只跑 5 个样本，只跑 2 个系统）
  python3 experiments/run_experiments.py --limit 5 --systems marc standard_rag

  # 完整实验
  python3 experiments/run_experiments.py

  # 自定义配置
  python3 experiments/run_experiments.py \
      --generate-model claude-opus-4-7 \
      --systems marc marc_no_dcr standard_rag no_retrieval picos_rag
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# 清除系统代理环境变量，避免 socks/http 代理导致 API 连接失败
# MiniMax 等国内 API 通常无需代理即可直接访问
for _proxy_key in ["ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"]:
    os.environ.pop(_proxy_key, None)

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluate import evaluate_system, load_benchmark
from experiments.config import ExperimentConfig


def build_systems(config: ExperimentConfig) -> Dict[str, object]:
    """
    根据 config.systems_to_run 构建所有待评估系统。

    LLM 客户端和模型名从 src/llm_client 读取（来源为 .env 文件）。
    不再接受 api_key 参数，统一由 llm_client 管理。

    参数：
        config: 实验配置

    返回：
        {system_name: BaseRAGSystem 实例}
    """
    from baselines.no_retrieval import NoRetrievalBaseline
    from baselines.picos_rag import PICOsRAG
    from baselines.standard_rag import StandardRAG
    from baselines.bm25_only import BM25OnlyRAG
    from baselines.dense_only import DenseOnlyRAG
    from baselines.marc_no_dcr import MARCNoDCR
    from baselines.marc_no_scsr import MARCNoSCSR, build_marc_no_scsr_pipeline
    from src.pipeline import build_marc_pipeline
    from src.query_decomposer import QueryDecomposer
    from src.retriever import HybridRetriever
    from src.llm_client import get_client, get_model, get_embedding_model

    # 所有系统共用同一个 LLM 客户端和模型（来自 .env）
    client = get_client()
    model = get_model()
    embedding_model = get_embedding_model()
    systems: Dict[str, object] = {}

    # 所有需要检索的系统共用一个 Retriever 实例
    # bm25_only / dense_only 各自指定 method，但共用同一个 retriever 实例
    retriever: HybridRetriever | None = None
    if any(s in config.systems_to_run for s in [
        "marc", "marc_no_dcr", "marc_no_scsr",
        "standard_rag", "bm25_only", "dense_only", "picos_rag",
    ]):
        retriever = HybridRetriever(
            index_dir=config.index_dir,
            embedding_model=embedding_model,
            bm25_weight=config.bm25_weight,
        )

    for system_name in config.systems_to_run:
        if system_name == "marc":
            pipeline = build_marc_pipeline(
                index_dir=str(config.index_dir),
                cache_dir=str(config.cache_dir),
                stage1_top_k=config.stage1_top_k,
                stage2_top_k=config.stage2_top_k,
            )
            from experiments._marc_wrapper import MARCSystemWrapper
            systems["marc"] = MARCSystemWrapper(pipeline)

        elif system_name == "marc_with_fc":
            pipeline_fc = build_marc_pipeline(
                index_dir=str(config.index_dir),
                cache_dir=str(config.cache_dir),
                stage1_top_k=config.stage1_top_k,
                stage2_top_k=config.stage2_top_k,
                enable_fc=True,
            )
            from experiments._marc_wrapper import MARCSystemWrapper
            systems["marc_with_fc"] = MARCSystemWrapper(pipeline_fc)

        elif system_name == "marc_no_dcr":
            decomposer = QueryDecomposer(
                client=client,
                model=model,
                cache_dir=config.cache_dir,
            )
            systems["marc_no_dcr"] = MARCNoDCR(
                retriever=retriever,
                decomposer=decomposer,
                client=client,
                model=model,
            )

        elif system_name == "marc_no_scsr":
            # 独立构建管道（注入 _DisabledConstraintRetriever，禁用 Stage 1B）
            pipeline_no_scsr = build_marc_no_scsr_pipeline(
                index_dir=str(config.index_dir),
                cache_dir=str(config.cache_dir),
                stage1_top_k=config.stage1_top_k,
                stage2_top_k=config.stage2_top_k,
            )
            systems["marc_no_scsr"] = MARCNoSCSR(pipeline_no_scsr)

        elif system_name == "standard_rag":
            systems["standard_rag"] = StandardRAG(
                retriever=retriever,
                client=client,
                model=model,
            )

        elif system_name == "bm25_only":
            systems["bm25_only"] = BM25OnlyRAG(
                retriever=retriever,
                client=client,
                model=model,
            )

        elif system_name == "dense_only":
            systems["dense_only"] = DenseOnlyRAG(
                retriever=retriever,
                client=client,
                model=model,
            )

        elif system_name == "picos_rag":
            systems["picos_rag"] = PICOsRAG(
                retriever=retriever,
                client=client,
                model=model,
            )

        elif system_name == "no_retrieval":
            systems["no_retrieval"] = NoRetrievalBaseline(
                client=client,
                model=model,
            )

        else:
            raise ValueError(
                f"[run_experiments] 系统 {repr(system_name)} 尚未在 build_systems() 中实现。"
            )

    return systems


def run_all(config: ExperimentConfig) -> None:
    """
    按 config.systems_to_run 顺序运行所有系统，汇总结果表格。

    LLM 配置（API key、模型名）通过 src/llm_client 从 .env 读取。

    参数：
        config: 实验配置
    """
    from src.llm_client import print_config
    config.validate()
    print_config()   # 实验开始时打印配置摘要，方便复现确认

    # 加载 benchmark
    print(f"[实验] 加载 MACB benchmark: {config.benchmark_path}")
    benchmark = load_benchmark(config.benchmark_path)

    # 按 tag 过滤（如只跑 SC 样本或指定 tag）
    if config.sample_ids:
        benchmark = [s for s in benchmark if s.sample_id in config.sample_ids]
        print(f"  sample_ids={config.sample_ids}，过滤后: {len(benchmark)} 个样本")
    elif config.tag_filter:
        benchmark = [s for s in benchmark if s.candidate_tag in config.tag_filter]
        print(f"  tag_filter={config.tag_filter}，过滤后: {len(benchmark)} 个样本")
    else:
        print(f"  样本数: {len(benchmark)}")

    if config.sample_limit:
        print(f"  限制: 前 {config.sample_limit} 个样本（调试模式）")

    # 构建系统（LLM 配置来自 .env）
    print(f"\n[实验] 初始化系统: {config.systems_to_run}")
    systems = build_systems(config)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, dict] = {}

    for system_name, system in systems.items():
        print(f"\n{'='*60}")
        print(f"[实验] 评估系统: {system_name}")
        print(f"{'='*60}")
        t0 = time.time()
        metrics = evaluate_system(
            system=system,
            benchmark=benchmark,
            output_dir=config.results_dir,
            sample_limit=config.sample_limit,
            log_dir=config.log_dir,
        )
        elapsed = time.time() - t0
        print(f"  系统总耗时: {elapsed:.1f}s")
        all_metrics[system_name] = metrics

    # 打印汇总表格
    _print_summary_table(all_metrics, config.results_dir)


def _print_summary_table(
    all_metrics: Dict[str, dict],
    results_dir: Path,
) -> None:
    """
    打印并保存所有系统的指标汇总表格。

    格式（适合直接复制到论文）：
      System    | CRR↓   | SDR↑   | AEC↑   | SLR↓
    """
    header = (
        f"{'System':<20} | {'CRR↓':>8} | {'SDR↑':>8} | {'AEC↑':>8} | {'SLR↓':>8}"
        f" | {'CDR↑':>8} | {'RSI↑':>8} | {'CAEC↑':>8}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for system_name, metrics in all_metrics.items():
        def fmt(m: dict) -> str:
            v = m.get("mean", float("nan"))
            if v != v:          # nan check
                return "  N/A  "
            return f"{v:.4f}" if m.get("n", 0) > 0 else "   -   "

        row = (
            f"{system_name:<20} | {fmt(metrics['CRR']):>8} | {fmt(metrics['SDR']):>8} | "
            f"{fmt(metrics['AEC']):>8} | {fmt(metrics['SLR']):>8}"
            f" | {fmt(metrics.get('CDR', {'n': 0})):>8}"
            f" | {fmt(metrics.get('RSI', {'n': 0})):>8}"
            f" | {fmt(metrics.get('CAEC', {'n': 0})):>8}"
        )
        rows.append(row)

    table = "\n".join(rows)
    print(f"\n{'='*60}")
    print("[实验汇总] 所有系统指标")
    print(table)
    print(f"{'='*60}\n")

    # 写出汇总文件
    summary_path = results_dir / "all_systems_summary.txt"
    summary_path.write_text(table + "\n", encoding="utf-8")

    json_path = results_dir / "all_systems_metrics.json"
    json_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"汇总表格 → {summary_path}")
    print(f"JSON 指标 → {json_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="MARC 统一实验入口")
    p.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/macb_treatment_v5.jsonl"),
        help="MACB benchmark 路径",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="教材索引目录",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="结果输出目录",
    )
    p.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="待运行系统列表（默认运行全部）",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个系统只评估前 N 个样本（调试用）",
    )
    p.add_argument(
        "--tag-filter",
        nargs="+",
        default=None,
        metavar="TAG",
        help="只评估指定 candidate_tag 的样本，"
             "如 --tag-filter SC_ABSOLUTE_CAND SC_RELATIVE_CAND",
    )
    p.add_argument(
        "--sc-only",
        action="store_true",
        help="只评估 SC 样本（SC_ABSOLUTE_CAND + SC_RELATIVE_CAND），"
             "等价于 --tag-filter SC_ABSOLUTE_CAND SC_RELATIVE_CAND",
    )
    p.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="只评估指定 sample_id 的样本（如重跑失败样本：--sample-ids MACB-030 MACB-053）",
    )
    args = p.parse_args()

    # LLM 配置（API key、模型名）从 .env 文件读取，无需命令行传入
    # 参考 .env.example 设置你的 .env 文件

    tag_filter = None
    if args.sc_only:
        tag_filter = ["SC_ABSOLUTE_CAND", "SC_RELATIVE_CAND"]
    elif args.tag_filter:
        tag_filter = args.tag_filter

    config = ExperimentConfig(
        benchmark_path=args.benchmark,
        index_dir=args.index_dir,
        results_dir=args.results_dir,
        sample_limit=args.limit,
        tag_filter=tag_filter,
        sample_ids=args.sample_ids,
    )
    if args.systems:
        config.systems_to_run = args.systems

    run_all(config)


if __name__ == "__main__":
    main()
