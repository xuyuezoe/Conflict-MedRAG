#!/usr/bin/env python3
"""
评估指标实现

基础指标（FC_AA 已从主实验移除，FC 子集在 MedQA 单一答案格式下无法可靠构建）：
  CRR  (Contraindicated Recommendation Rate)    ↓ 越低越好
  SDR  (SC Detection Recall)                    ↑ 越高越好
  AEC  (Alternative Evidence Coverage)          ↑ 越高越好
  SLR  (Source Leakage Rate)                    ↓ 越低越好

CVFR 阶段性指标（仅当 ScopeIndex 已加载时有值，否则为 None）：
  CDR  (Conditional Document Relevance)         ↑ 越高越好
       = avg cos(e^scope_d, e_C)，衡量检索文档在约束维度的平均对齐程度
  RSI  (Retrieval Specificity Index)            ↑ 越高越好
       = 高特异性文档比例（cos ≥ 0.6），衡量约束特异性文档的覆盖率
  CAEC (Constrained AEC)                        ↑ 越高越好
       = #{a ∈ A(q) : ∃d ∈ E(q), a ∈ d.text AND scope_cos(d) ≥ θ_low} / |A(q)|
       在 AEC 基础上增加 scope-aware 约束，仅计算来自有 scope 信息文档的覆盖

每个指标计算 per-sample 值，汇总时取 mean ± std + 95% CI（bootstrap）。
"""
from __future__ import annotations

import re
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

    # 构建全量检索文本：
    #   MARC 系统：使用 admissible_docs（Stage 2 scope-filtered 文档，全文）
    #   非 MARC 系统：使用 context_chunks（baseline 填充的检索片段，text_snippet[:300]）
    #   无检索系统（no_retrieval）：context_chunks 为空 → all_doc_texts="" → AEC=0
    import re as _re
    if result.marc_output is not None:
        all_doc_texts = " ".join(
            d.chunk.text.lower()
            for d in result.marc_output.admissible_docs
        )
    else:
        all_doc_texts = " ".join(
            chunk.get("text_snippet", "").lower()
            for chunk in result.context_chunks
        )

    # 将选项字母映射到完整描述文本（"A: desc | B: desc | ..."）
    # gold_admissible_set 存的是选项字母，直接做子串匹配毫无意义（"d" in any text = True）
    # 正确做法：取选项描述的前 40 字符作为 key term 检索
    option_map: dict = {}
    for part in (sample.options_text or "").split("|"):
        part = part.strip()
        if ": " in part:
            letter, desc = part.split(": ", 1)
            letter = letter.strip()
            if _re.match(r"^[A-E]$", letter):
                option_map[letter] = desc.strip().lower()

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


# ── CDR ──────────────────────────────────────────────────────────────────────

def compute_cdr_metric(result: SampleResult) -> Optional[float]:
    """
    从 SampleResult 提取 CDR（Conditional Document Relevance）。

    CDR 由 MARCPipeline.run() 在 Stage 2 后计算（需 ScopeIndex 已加载），
    存于 marc_output.metrics["cdr"]。

    定义（cvfr_theory.md §9.4）：
      CDR = (1/|E(q)|) × Σ_{d ∈ E(q)} cos(e^scope_d, e_C)

    返回：
        float（[0,1]），或 None（ScopeIndex 未加载 / 非 MARC 系统）
    """
    if result.marc_output is None:
        return None
    return result.marc_output.metrics.get("cdr", None)


def aggregate_cdr(results: List[float]) -> Dict[str, float]:
    """汇总 CDR 指标"""
    return _aggregate_metric(results, name="CDR")


# ── RSI ──────────────────────────────────────────────────────────────────────

def compute_rsi_metric(result: SampleResult) -> Optional[float]:
    """
    从 SampleResult 提取 RSI（Retrieval Specificity Index）。

    RSI 由 MARCPipeline.run() 计算，存于 marc_output.metrics["rsi"]。

    定义（cvfr_theory.md §9.4）：
      RSI = #{d ∈ E(q) : cos(e^scope_d, e_C) ≥ θ_high} / |E(q)|，θ_high=0.6

    返回：
        float（[0,1]），或 None（ScopeIndex 未加载 / 非 MARC 系统）
    """
    if result.marc_output is None:
        return None
    return result.marc_output.metrics.get("rsi", None)


def aggregate_rsi(results: List[float]) -> Dict[str, float]:
    """汇总 RSI 指标"""
    return _aggregate_metric(results, name="RSI")


# ── CAEC ─────────────────────────────────────────────────────────────────────

# θ_low: scope-aware 文档的最低 cos 阈值（低于此值视为无 scope 信息）
_CAEC_THETA_LOW: float = 0.3


def _parse_option_map(options_text: str) -> Dict[str, str]:
    """
    解析选项文本为 {字母 → 描述} 的映射。

    选项格式："A: description | B: description | ..."
    返回：{"A": "description", "B": "description", ...}，描述均转小写
    """
    option_map: Dict[str, str] = {}
    for part in (options_text or "").split("|"):
        part = part.strip()
        if ": " in part:
            letter, desc = part.split(": ", 1)
            letter = letter.strip()
            if re.match(r"^[A-E]$", letter):
                option_map[letter] = desc.strip().lower()
    return option_map


def compute_caec(
    result: SampleResult,
    sample: EvalSample,
) -> Optional[float]:
    """
    计算单样本 CAEC（Constrained AEC，约束感知的替代方案覆盖率）。

    定义（cvfr_theory.md §9.4）：
      CAEC = #{a ∈ A(q) : ∃d ∈ E(q), a ∈ d.text AND cos(e^scope_d, e_C) ≥ θ_low}
             ──────────────────────────────────────────────────────────────────────
                                     |A(q)|

    CAEC 与 AEC 的区别：
      AEC：文档覆盖 action 即算（不考虑 scope 信息质量）
      CAEC：只计算来自 scope-aware 文档的覆盖（cos ≥ θ_low），
            要求系统不仅检索了 action 相关文档，还检索了约束特异性证据

    参数：
        result: 系统输出（需 marc_output 含 cdr_per_doc 和 stage2_docs）
        sample: MACB 金标准样本

    返回：
        float（[0,1]），或 None（ScopeIndex 未加载 / 非 MARC 系统 / A(q) 为空）
    """
    if result.marc_output is None:
        return None

    cdr_per_doc = result.marc_output.metrics.get("cdr_per_doc")
    if not cdr_per_doc:
        # ScopeIndex 未加载时无 cdr_per_doc，CAEC 不可计算
        return None

    admissible_set = sample.gold_admissible_set
    if not admissible_set:
        return None

    # 构建 scope-aware 文档集合：cos(e^scope, e_C) ≥ θ_low
    scope_aware_chunk_ids: Set[str] = {
        doc["chunk_id"]
        for doc in cdr_per_doc
        if doc.get("scope_cos", 0.0) >= _CAEC_THETA_LOW
    }

    if not scope_aware_chunk_ids:
        return 0.0

    # 构建 scope-aware 文档文本（仅来自 stage2_docs 且 cos ≥ θ_low 的文档）
    scope_aware_texts = " ".join(
        d.chunk.text.lower()
        for d in result.marc_output.stage2_docs
        if d.chunk.chunk_id in scope_aware_chunk_ids
    )

    # 将选项字母映射到描述
    option_map = _parse_option_map(sample.options_text)

    covered = 0
    for action_letter in admissible_set:
        desc = option_map.get(action_letter.upper(), "")
        if not desc:
            continue
        key_term = desc[:30].strip()
        if len(key_term) >= 6 and key_term in scope_aware_texts:
            covered += 1

    return covered / len(admissible_set)


def aggregate_caec(results: List[float]) -> Dict[str, float]:
    """汇总 CAEC 指标"""
    return _aggregate_metric(results, name="CAEC")


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
        {
          "CRR": {...}, "SDR": {...}, "AEC": {...}, "SLR": {...},
          "CDR": {...},   # 仅 MARC + ScopeIndex 已加载时有值
          "RSI": {...},   # 同上
          "CAEC": {...},  # 同上
        }
    """
    if len(results) != len(samples):
        raise ValueError(
            f"[compute_all_metrics] results 和 samples 数量不匹配："
            f"{len(results)} vs {len(samples)}"
        )

    crr_vals, sdr_vals, aec_vals, slr_vals = [], [], [], []
    cdr_vals, rsi_vals, caec_vals = [], [], []

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

        cdr = compute_cdr_metric(result)
        if cdr is not None:
            cdr_vals.append(cdr)

        rsi = compute_rsi_metric(result)
        if rsi is not None:
            rsi_vals.append(rsi)

        caec = compute_caec(result, sample)
        if caec is not None:
            caec_vals.append(caec)

    return {
        "CRR":  aggregate_crr(crr_vals),
        "SDR":  aggregate_sdr(sdr_vals),
        "AEC":  aggregate_aec(aec_vals),
        "SLR":  aggregate_slr(slr_vals),
        "CDR":  aggregate_cdr(cdr_vals),
        "RSI":  aggregate_rsi(rsi_vals),
        "CAEC": aggregate_caec(caec_vals),
    }
