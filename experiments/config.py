#!/usr/bin/env python3
"""
实验配置

所有超参数、路径、系统选择集中在此，不散落在各模块中。
修改实验参数时只需修改本文件。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """
    MARC 实验统一配置。

    参数分组：
      数据路径:   benchmark、索引、结果输出
      模型选择:   分解/提取模型（Haiku）+ 主生成模型（Sonnet）
      检索超参:   top_k、SCSR 触发阈值、BM25 权重
      实验控制:   待运行系统列表、样本数量限制
    """
    # ── 数据路径 ────────────────────────────────────────────────────────────
    benchmark_path: Path = Path("data/macb_treatment_v5.jsonl")
    index_dir: Path = Path("data/index")
    cache_dir: Path = Path("data/cache")
    results_dir: Path = Path("results")
    log_dir: Path = Path("run_log")

    # ── 模型配置 ────────────────────────────────────────────────────────────
    # 模型名不在此处配置，从 .env 文件的 LLM_MODEL 字段统一读取

    # ── 检索超参 ────────────────────────────────────────────────────────────
    stage1_top_k: int = 20
    stage2_top_k: int = 5
    scsr_min_admissible: int = 2
    scsr_top_k: int = 5
    bm25_weight: float = 0.5

    # ── 实验控制 ────────────────────────────────────────────────────────────
    systems_to_run: List[str] = field(default_factory=lambda: [
        "marc",          # 完整 MARC 系统
        "marc_no_dcr",   # 消融：无 DCR（有 SCOPE BIAS WARNING，无 κ 过滤）
        "marc_no_scsr",  # 消融：无 SCSR（有 DCR，无 Stage 3 gap-filling）
        "standard_rag",  # Baseline：标准混合 RAG
        "bm25_only",     # Baseline：纯 BM25 RAG
        "dense_only",    # Baseline：纯 Dense RAG
        "no_retrieval",  # Baseline：无检索（直接 LLM）
        "picos_rag",     # Baseline：PICOs-RAG 风格
    ])
    sample_limit: Optional[int] = None    # None = 全量；设 N 用于调试
    tag_filter: Optional[List[str]] = None  # None = 全量；设列表按 candidate_tag 过滤

    def validate(self) -> None:
        """
        校验配置合法性。

        异常：
            ValueError: 配置项非法
        """
        if not self.index_dir.exists():
            raise ValueError(
                f"[ExperimentConfig] index_dir 不存在: {self.index_dir}。"
                f"请先运行 python3 scripts/index_textbooks.py。"
            )
        valid_systems = {
            "marc", "marc_no_dcr", "marc_no_scsr",
            "standard_rag", "bm25_only", "dense_only",
            "no_retrieval", "picos_rag",
        }
        for s in self.systems_to_run:
            if s not in valid_systems:
                raise ValueError(
                    f"[ExperimentConfig] 未知系统名: {repr(s)}。"
                    f"有效系统: {valid_systems}"
                )
        if self.bm25_weight < 0 or self.bm25_weight > 1:
            raise ValueError(
                f"[ExperimentConfig] bm25_weight 必须在 [0,1] 范围内，"
                f"当前值: {self.bm25_weight}"
            )
