#!/usr/bin/env python3
"""
运行日志记录器

为每次实验运行创建结构化日志目录，记录：
  - 每样本简要结果（per_sample.jsonl，实时写入）
  - 每样本详细链路日志（samples/{sample_id}.json）
    · MARC 系统：7 个模块的完整输入/输出/延迟
    · Baseline 系统：检索文档 + 生成结果
  - 运行配置快照（config.json）
  - 最终指标汇总（summary.json）

目录结构：
  run_log/{system}_{YYYYMMDD_HHMMSS}/
    config.json
    per_sample.jsonl
    summary.json
    samples/
      MACB-001.json
      MACB-002.json
      ...
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.types import EvalSample, MARCOutput, RetrievedDoc, SampleResult


class RunLogger:
    """
    单次实验运行的日志管理器。

    生命周期：
      1. __init__       → 创建目录，写 config.json
      2. log_sample()   → 每个样本完成后调用（实时）
      3. finalize()     → 运行结束后调用，写 summary.json
    """

    def __init__(
        self,
        system_name: str,
        log_dir: Path,
        config: Dict[str, Any],
    ) -> None:
        """
        初始化日志目录。

        参数：
            system_name: 系统名称（如 "marc"/"standard_rag"）
            log_dir:     顶层日志目录（如 run_log/）
            config:      运行配置快照（模型名、超参数、benchmark 路径等）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = log_dir / f"{system_name}_{timestamp}"
        self._samples_dir = self._run_dir / "samples"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._samples_dir.mkdir(exist_ok=True)

        self._per_sample_path = self._run_dir / "per_sample.jsonl"
        self._per_sample_file = self._per_sample_path.open("w", encoding="utf-8")

        config_path = self._run_dir / "config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"  [RunLogger] 日志目录 → {self._run_dir}")

    # ── 公开接口 ────────────────────────────────────────────────────────────────

    def log_sample(
        self,
        result: SampleResult,
        sample: EvalSample,
        per_sample_metrics: Dict[str, Optional[float]],
        latency_s: float,
    ) -> None:
        """
        记录单个样本的运行结果。

        参数：
            result:              系统输出（SampleResult）
            sample:              MACB 金标准（EvalSample）
            per_sample_metrics:  该样本的指标值（{"crr": 0.0, "sdr": 1.0, ...}）
            latency_s:           该样本总耗时（秒）
        """
        # ── 实时写入简要行 ──────────────────────────────────────────────────────
        brief = {
            "sample_id":      result.sample_id,
            "candidate_tag":  sample.candidate_tag,
            "crr":            per_sample_metrics.get("crr"),
            "sdr":            per_sample_metrics.get("sdr"),
            "aec":            per_sample_metrics.get("aec"),
            "slr":            per_sample_metrics.get("slr"),
            "scsr_triggered": result.scsr_triggered,
            "n_pred_actions": len(result.per_action_status_pred),
            "latency_s":      round(latency_s, 2),
        }
        self._per_sample_file.write(json.dumps(brief, ensure_ascii=False) + "\n")
        self._per_sample_file.flush()

        # ── 写详细日志 ──────────────────────────────────────────────────────────
        if result.marc_output is not None:
            detail = self._build_marc_detail(result, sample, per_sample_metrics, latency_s)
        else:
            detail = self._build_baseline_detail(result, sample, per_sample_metrics, latency_s)

        sample_path = self._samples_dir / f"{result.sample_id}.json"
        sample_path.write_text(
            json.dumps(detail, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def finalize(self, summary_metrics: Dict[str, Any]) -> None:
        """
        写入最终指标汇总，关闭文件句柄。

        参数：
            summary_metrics: compute_all_metrics() 的返回值
        """
        self._per_sample_file.close()
        summary_path = self._run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  [RunLogger] 详细日志 → {self._run_dir}")

    # ── 私有方法：MARC 系统详细日志 ────────────────────────────────────────────

    def _build_marc_detail(
        self,
        result: SampleResult,
        sample: EvalSample,
        metrics: Dict[str, Optional[float]],
        latency_s: float,
    ) -> Dict[str, Any]:
        """
        从 MARCOutput 提取 7 个模块的完整信息。
        """
        mo: MARCOutput = result.marc_output  # type: ignore[assignment]
        dec = mo.decomposition

        # ── Module 0 ──────────────────────────────────────────────────────────
        module_0 = {
            "disease_query": dec.disease_query,
            "constraints": [
                {
                    "type":                c.constraint_type,
                    "target_action":       c.target_action,
                    "raw_text":            c.raw_text,
                    "parameter_value":     c.parameter_value,
                    "parameter_threshold": c.parameter_threshold,
                }
                for c in dec.constraints
            ],
            "decompose_model": dec.decompose_model,
            "latency_s":       round(mo.metrics.get("module0_latency_s", 0.0), 3),
        }

        # ── Stage 1 ───────────────────────────────────────────────────────────
        stage1_docs = mo.stage1_docs
        stage1 = {
            "query_used":  dec.disease_query,
            "n_docs":      len(stage1_docs),
            "top5":        [_doc_brief(d, rank=i + 1) for i, d in enumerate(stage1_docs[:5])],
            "latency_s":   round(mo.metrics.get("stage1_latency_s", 0.0), 3),
        }

        # ── Stage 2 DCR ───────────────────────────────────────────────────────
        n_admissible  = mo.metrics.get("stage2_n_admissible", 0)
        n_excluded    = mo.metrics.get("stage2_n_excluded", 0)
        stage2 = {
            "n_evaluated":    len(stage1_docs),
            "n_admissible":   n_admissible,
            "n_excluded_abs": n_excluded,
            "docs": [
                {
                    "chunk_id":       d.chunk.chunk_id,
                    "source_book":    d.chunk.source_book,
                    "sim_score":      round(d.sim_score, 4),
                    "kappa":          round(d.kappa, 4),
                    "dcr_score":      round(d.dcr_score, 4),
                    "scope_status":   d.scope_status,
                    "scope_predicate": _scope_predicate_dict(d),
                    "decision":       "EXCLUDED" if d.kappa == 0.0 else "ADMITTED",
                    "text_snippet":   d.chunk.text[:300],
                }
                for d in stage1_docs
            ],
            "latency_s": round(mo.metrics.get("stage2_latency_s", 0.0), 3),
        }

        # ── Stage 3 SCSR ──────────────────────────────────────────────────────
        scsr_query = mo.metrics.get("stage3_query", "")
        n_stage2_admissible = len([d for d in mo.stage2_docs if d.kappa > 0.0])
        stage3 = {
            "triggered":      mo.scsr_triggered,
            "trigger_reason": f"admissible_docs={n_stage2_admissible} after DCR" if mo.scsr_triggered else "不需要（admissible_docs 足够）",
            "scsr_query":     scsr_query,
            "n_docs_found":   len(mo.scsr_docs),
            "docs":           [_doc_brief(d, rank=i + 1) for i, d in enumerate(mo.scsr_docs)],
            "latency_s":      round(mo.metrics.get("stage3_latency_s", 0.0), 3),
        }

        # ── FC Handler ────────────────────────────────────────────────────────
        fc = {
            "n_conflicts": len(mo.fc_conflicts),
            "conflicts": [
                {
                    "action":             c.action,
                    "doc_a_id":           c.doc_a_id,
                    "doc_a_claim":        c.doc_a_claim,
                    "doc_b_id":           c.doc_b_id,
                    "doc_b_claim":        c.doc_b_claim,
                    "conflict_type":      c.conflict_type,
                    "resolution":         c.resolution,
                    "resolution_reason":  c.resolution_reason,
                }
                for c in mo.fc_conflicts
            ],
            "latency_s": round(mo.metrics.get("fc_latency_s", 0.0), 3),
        }

        # ── Generation ────────────────────────────────────────────────────────
        generation = {
            "inadmissible_excluded": dec.absolute_target_actions,
            "n_admissible_docs_in_context": len(mo.admissible_docs),
            "generated_answer":     mo.generated_answer,
            "per_action_status":    mo.per_action_status,
            "attribution":          mo.attribution,
            "latency_s":            round(mo.metrics.get("gen_latency_s", 0.0), 3),
        }

        # ── Verification ──────────────────────────────────────────────────────
        verification = {
            "slr":                   mo.metrics.get("slr"),
            "srl_violations":        mo.srl_violations,
            "n_inadmissible_chunks": len(mo.inadmissible_chunk_ids),
            "inadmissible_chunk_ids": list(mo.inadmissible_chunk_ids),
        }

        return {
            "sample_id":      result.sample_id,
            "candidate_tag":  sample.candidate_tag,
            "task_type":      sample.task_type,
            "system":         result.system_name,

            "input": {
                "query":        sample.query,
                "options_text": sample.options_text,
            },

            "module_0_decomposition": module_0,
            "stage_1_retrieval":      stage1,
            "stage_2_dcr":            stage2,
            "stage_3_scsr":           stage3,
            "fc_handler":             fc,
            "generation":             generation,
            "verification":           verification,

            "gold_comparison":        _gold_comparison(result, sample, metrics),
            "total_latency_s":        round(latency_s, 2),
        }

    # ── 私有方法：Baseline 系统详细日志 ────────────────────────────────────────

    def _build_baseline_detail(
        self,
        result: SampleResult,
        sample: EvalSample,
        metrics: Dict[str, Optional[float]],
        latency_s: float,
    ) -> Dict[str, Any]:
        """
        从 SampleResult.context_chunks 提取检索信息 + 生成结果。
        """
        retrieval: Dict[str, Any] = {
            "n_docs": len(result.context_chunks),
            "docs":   result.context_chunks,
        }

        generation = {
            "predicted_answer":  result.predicted_answer,
            "per_action_status": result.per_action_status_pred,
        }

        return {
            "sample_id":     result.sample_id,
            "candidate_tag": sample.candidate_tag,
            "task_type":     sample.task_type,
            "system":        result.system_name,

            "input": {
                "query":        sample.query,
                "options_text": sample.options_text,
            },

            "retrieval":       retrieval,
            "generation":      generation,
            "gold_comparison": _gold_comparison(result, sample, metrics),
            "total_latency_s": round(latency_s, 2),
        }


# ── 模块级辅助函数 ────────────────────────────────────────────────────────────

def _doc_brief(doc: RetrievedDoc, rank: int) -> Dict[str, Any]:
    """将 RetrievedDoc 转为日志友好的简要 dict。"""
    return {
        "rank":          rank,
        "chunk_id":      doc.chunk.chunk_id,
        "source_book":   doc.chunk.source_book,
        "sim_score":     round(doc.sim_score, 4),
        "kappa":         round(doc.kappa, 4),
        "scope_status":  doc.scope_status,
        "text_snippet":  doc.chunk.text[:300],
    }


def _scope_predicate_dict(doc: RetrievedDoc) -> Optional[Dict[str, Any]]:
    """将 ScopePredicate 转为 dict（若不存在则返回 None）。"""
    sp = doc.scope_predicate
    if sp is None:
        return None
    return {
        "recommended_action":    sp.recommended_action,
        "population":            sp.population,
        "contraindications":     sp.contraindications,
        "relative_restrictions": sp.relative_restrictions,
    }


def _gold_comparison(
    result: SampleResult,
    sample: EvalSample,
    metrics: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    """构造 gold vs pred 的对比字典。"""
    return {
        "gold_per_action_status": sample.gold_per_action_status,
        "pred_per_action_status": result.per_action_status_pred,
        "gold_admissible_set":    sample.gold_admissible_set,
        "gold_preferred_set":     sample.gold_preferred_set,
        "gold_conflict_types":    sample.gold_conflict_types_present,
        "crr":                    metrics.get("crr"),
        "sdr":                    metrics.get("sdr"),
        "aec":                    metrics.get("aec"),
        "slr":                    metrics.get("slr"),
    }
