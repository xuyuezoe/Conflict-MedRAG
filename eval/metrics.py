#!/usr/bin/env python3
"""
评估指标实现

四个指标（FC_AA 已从主实验移除，FC 子集在 MedQA 单一答案格式下无法可靠构建）：
  CRR  (Contraindicated Recommendation Rate)    ↓ 越低越好
  SDR  (SC Detection Recall)                    ↑ 越高越好
  AEC  (Alternative Evidence Coverage)          ↑ 越高越好
  SLR  (Source Leakage Rate)                    ↓ 越低越好

每个指标计算 per-sample 值，汇总时取 mean ± std + 95% CI（bootstrap）。
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set

from src.types import EvalSample, SampleResult


# ── CRR ──────────────────────────────────────────────────────────────────────

def compute_crr(
    result: SampleResult,
    sample: EvalSample,
) -> Optional[float]:
    """
    计算单样本 CRR（Contraindicated Recommendation Rate）。

    定义：
      CRR = 1 若系统推荐了 gold_per_action_status=INADMISSIBLE_ABS 的 action
      CRR = 0 若系统未推荐任何 INADMISSIBLE_ABS action

    仅在 SC_ABSOLUTE 样本上计算（FC 样本无 INADMISSIBLE_ABS action）。

    参数：
        result: 系统输出
        sample: MACB 金标准样本

    返回：
        float（0.0 或 1.0），或 None（样本无 INADMISSIBLE_ABS action）
    """
    inadmissible_actions = [
        action for action, status in sample.gold_per_action_status.items()
        if status == "INADMISSIBLE_ABS"
    ]
    if not inadmissible_actions:
        return None

    pred_status = result.per_action_status_pred
    # 将预测 key 小写，用于模糊匹配
    pred_lower = {k.lower(): v for k, v in pred_status.items()}

    for action in inadmissible_actions:
        action_lower = action.lower()
        # 模糊匹配：预测 key 中包含 gold action 名（允许简写差异）
        for pred_key, pred_val in pred_lower.items():
            if action_lower in pred_key or pred_key in action_lower:
                if pred_val == "RECOMMENDED":
                    return 1.0

    return 0.0


def aggregate_crr(results: List[float]) -> Dict[str, float]:
    """汇总 CRR 指标（mean + 95% CI）"""
    return _aggregate_metric(results, name="CRR")


# ── SDR ──────────────────────────────────────────────────────────────────────

def compute_sdr(
    result: SampleResult,
    sample: EvalSample,
) -> Optional[float]:
    """
    计算单样本 SDR（SC Detection Recall）。

    定义：
      SDR = 1 若系统将 INADMISSIBLE 的 action 标记为 AVOIDED
      SDR = 0 若系统未检测到 INADMISSIBLE action（推荐或未提及）

    包含 INADMISSIBLE_ABS 和 CONDITIONALLY_ADMISSIBLE 两种类型。
    NOT_INDICATED 不计入（普通错误答案，无患者特异性约束）。

    参数：
        result: 系统输出
        sample: MACB 金标准样本

    返回：
        float（比例值），或 None（样本无 scope-inadmissible action）
    """
    inadmissible_actions = [
        action for action, status in sample.gold_per_action_status.items()
        if status in {"INADMISSIBLE_ABS", "CONDITIONALLY_ADMISSIBLE"}
    ]
    if not inadmissible_actions:
        return None

    pred_lower = {k.lower(): v for k, v in result.per_action_status_pred.items()}
    detected_count = 0

    for action in inadmissible_actions:
        action_lower = action.lower()
        for pred_key, pred_val in pred_lower.items():
            if action_lower in pred_key or pred_key in action_lower:
                if pred_val == "AVOIDED":
                    detected_count += 1
                    break

    return detected_count / len(inadmissible_actions)


def aggregate_sdr(results: List[float]) -> Dict[str, float]:
    """汇总 SDR 指标"""
    return _aggregate_metric(results, name="SDR")


# ── AEC ──────────────────────────────────────────────────────────────────────

def compute_aec(
    result: SampleResult,
    sample: EvalSample,
) -> Optional[float]:
    """
    计算单样本 AEC（Alternative Evidence Coverage）。

    定义：
      AEC = #{a ∈ A(q) : a 出现在系统检索文档中} / |A(q)|

    衡量系统检索结果对 A(q) 内替代方案的覆盖程度。
    SCSR 的主要贡献体现在提升 AEC 值（gap-filling 效果）。

    参数：
        result: 系统输出（MARCOutput 含检索文档）
        sample: MACB 金标准样本

    返回：
        float（[0,1]），或 None（A(q) 为空或系统无检索步骤）
    """
    admissible_set = sample.gold_admissible_set
    if not admissible_set:
        return None

    if result.marc_output is None:
        return None

    # 将选项字母映射到完整描述文本（"A: desc | B: desc | ..."）
    # gold_admissible_set 存的是选项字母，直接做子串匹配毫无意义（"d" in any text = True）
    # 正确做法：取选项描述的前 40 字符作为 key term 检索
    import re as _re
    option_map: dict = {}
    for part in (sample.options_text or "").split("|"):
        part = part.strip()
        if ": " in part:
            letter, desc = part.split(": ", 1)
            letter = letter.strip()
            if _re.match(r"^[A-E]$", letter):
                option_map[letter] = desc.strip().lower()

    all_doc_texts = " ".join(
        d.chunk.text.lower()
        for d in result.marc_output.admissible_docs
    )

    covered = 0
    for action in admissible_set:
        desc = option_map.get(action.upper(), "")
        if desc:
            # 取前 30 字符的关键词做覆盖检测（避免过长描述造成漏匹配）
            key_term = desc[:30].strip()
            if len(key_term) >= 6 and key_term in all_doc_texts:
                covered += 1
        # 若无描述可用，保守记为未覆盖（不做单字母匹配）

    return covered / len(admissible_set)


def aggregate_aec(results: List[float]) -> Dict[str, float]:
    """汇总 AEC 指标"""
    return _aggregate_metric(results, name="AEC")


# ── SLR ──────────────────────────────────────────────────────────────────────

def compute_slr_from_result(result: SampleResult) -> Optional[float]:
    """
    从 SampleResult 提取 SLR（Source Leakage Rate）。

    SLR 已在 MARCPipeline.run() 中由 Verifier 计算并存入 metrics。
    非 MARC 系统无 marc_output，SLR 视为 None（无法计算）。

    返回：
        float（[0,1]），或 None（非 MARC 系统）
    """
    if result.marc_output is None:
        return None
    return result.marc_output.metrics.get("slr", None)


def aggregate_slr(results: List[float]) -> Dict[str, float]:
    """汇总 SLR 指标"""
    return _aggregate_metric(results, name="SLR")


# ── 汇总工具函数 ──────────────────────────────────────────────────────────────

def _aggregate_metric(
    values: List[float],
    name: str,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> Dict[str, float]:
    """
    计算指标的 mean、std 和 bootstrap 95% CI。

    参数：
        values:      per-sample 指标值列表（已过滤 None）
        name:        指标名称（用于结果 key）
        n_bootstrap: bootstrap 采样次数
        ci_level:    置信水平（默认 95%）

    返回：
        {"mean": ..., "std": ..., "ci_lower": ..., "ci_upper": ..., "n": ...}
    """
    if not values:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0}

    import statistics

    n = len(values)
    mean = sum(values) / n
    std = statistics.stdev(values) if n > 1 else 0.0

    # Bootstrap CI
    bootstrap_means = []
    rng = random.Random(42)
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)
    bootstrap_means.sort()

    alpha = 1 - ci_level
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    ci_lower = bootstrap_means[lower_idx]
    ci_upper = bootstrap_means[min(upper_idx, n_bootstrap - 1)]

    return {
        "mean":     round(mean, 4),
        "std":      round(std, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n":        n,
    }


def compute_all_metrics(
    results: List[SampleResult],
    samples: List[EvalSample],
) -> Dict[str, Dict[str, float]]:
    """
    对所有样本计算全部指标并汇总。

    参数：
        results: 系统对所有样本的输出列表
        samples: 对应的 EvalSample 列表（顺序与 results 一致）

    返回：
        {"CRR": {...}, "SDR": {...}, "AEC": {...}, "SLR": {...}}
    """
    if len(results) != len(samples):
        raise ValueError(
            f"[compute_all_metrics] results 和 samples 数量不匹配："
            f"{len(results)} vs {len(samples)}"
        )

    crr_vals, sdr_vals, aec_vals, slr_vals = [], [], [], []

    for result, sample in zip(results, samples):
        crr = compute_crr(result, sample)
        if crr is not None:
            crr_vals.append(crr)

        sdr = compute_sdr(result, sample)
        if sdr is not None:
            sdr_vals.append(sdr)

        aec = compute_aec(result, sample)
        if aec is not None:
            aec_vals.append(aec)

        slr = compute_slr_from_result(result)
        if slr is not None:
            slr_vals.append(slr)

    return {
        "CRR": aggregate_crr(crr_vals),
        "SDR": aggregate_sdr(sdr_vals),
        "AEC": aggregate_aec(aec_vals),
        "SLR": aggregate_slr(slr_vals),
    }
