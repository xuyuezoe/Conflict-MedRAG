#!/usr/bin/env python3
"""
CVFR Phase 0：KKT 条件查询向量构造器

数学基础（docs/cvfr_theory.md §5.1）：
  将疾病查询向量 e_D 修正为条件查询向量 e*(D, C)，使其在最大化疾病相关性的同时
  满足约束方向对齐阈值 τ。这是一个 Hilbert 空间约束优化问题的 KKT 闭合解。

优化问题：
  e* = argmax_{e ∈ S^{d-1}}  ⟨e, e_D⟩
       subject to: ⟨e, ê_C⟩ ≥ τ

闭合解：
  情况1（cos(e_D, ê_C) ≥ τ）：约束已隐含，e* = ê_D
  情况2（cos(e_D, ê_C) < τ）：约束激活，e* = normalize(e_D + λ*(τ, c) · ê_C)

  λ*(τ, c) = -c + τ√(1-c²) / √(1-τ²)
  其中 c = cos(e_D, ê_C)

多约束聚合：
  ê_C = normalize(Σ_i w_i · ê_{C_i})
  ABSOLUTE 约束权重 w=1.0，RELATIVE 约束权重 w=0.5

设计约定：
  - 使用 raw_text 字段编码约束语义（target_action 字段存在 bug，见实验日志 V1 §5.3）
  - 约束向量编码与 HybridRetriever 共享同一 SentenceTransformer 实例（避免重复加载）
  - 所有向量均 L2 归一化（与 FAISS IndexFlatIP 的余弦相似度约定一致）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.types import PatientConstraint


@dataclass
class CVFRResult:
    """
    KKT 条件查询向量的计算结果（含完整调试信息）。

    参数：
        e_star:           条件查询向量（归一化，float32，shape=(d,)）
        e_D:              原始疾病查询向量（归一化，用于对比调试）
        lambda_star:      KKT 乘数（0 表示约束未激活）
        cos_dc:           cos(e_D, ê_C)（约束与疾病方向的余弦相似度）
        angular_shift_deg: e_D 到 e* 的角度偏移（度数）
        case:             "constraint_implicit" / "constraint_active" / "no_constraints"
        n_active:         参与聚合的有效约束数量
        debug:            其他调试信息
    """
    e_star: np.ndarray
    e_D: np.ndarray
    lambda_star: float
    cos_dc: Optional[float]
    angular_shift_deg: float
    case: str
    n_active: int
    debug: Dict[str, Any] = field(default_factory=dict)


class CVFRQueryConstructor:
    """
    CVFR Phase 0：KKT 条件查询向量构造器。

    使用与 HybridRetriever 相同的 SentenceTransformer 实例进行编码，
    保证向量空间一致性（同一 embedding 空间内的几何运算才有意义）。

    参数：
        embedding_model: SentenceTransformer 实例（由 HybridRetriever 提供）
        tau:             约束满足阈值（默认 0.3）
                         当 cos(e_D, ê_C) ≥ τ 时，约束已隐含在疾病方向中，不修正
                         τ 越大，修正越保守；τ 越小，修正越激进
    """

    def __init__(
        self,
        embedding_model,          # SentenceTransformer，类型不直接标注避免循环依赖
        tau: float = 0.5,
    ) -> None:
        """
        参数：
            embedding_model: SentenceTransformer 实例
            tau:             约束激活阈值（0.0~1.0），默认 0.5
                             基于 BGE-M3 空间标定：医学查询-约束对 cos_DC ∈ [0.39, 0.55]
                             τ=0.5 可激活大多数患者约束（6/7 测试对），同时过滤真正隐含的约束
        """
        if embedding_model is None:
            raise ValueError(
                "[CVFRQueryConstructor] embedding_model 不可为 None。"
                "请确保 HybridRetriever 已加载 Dense 索引。"
            )
        self._model = embedding_model
        self._tau = tau

    def compute_conditional_vector(
        self,
        disease_query: str,
        constraints: List[PatientConstraint],
    ) -> CVFRResult:
        """
        计算条件查询向量 e*(D, C)。

        参数：
            disease_query: D_q，纯疾病查询文本（不含患者约束）
            constraints:   C_q，患者约束列表（PatientConstraint）

        返回：
            CVFRResult（含 e_star、λ* 和完整调试信息）

        数学流程：
            第一步：编码 disease_query → e_D（归一化）
            第二步：筛选有效约束（排除 NONE 类型）
            第三步：编码约束文本 → {ê_{C_i}}，加权聚合 → ê_C
            第四步：计算 cos(e_D, ê_C)，判断情况 1 或情况 2
            第五步：情况 2 时计算 λ*，得到 e*(D, C)
        """
        # 第一步：编码疾病查询
        e_D = self._model.encode(
            [disease_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        # 第二步：筛选有效约束（排除 NONE 类型）
        active = [c for c in constraints if c.constraint_type != "NONE"]

        if not active:
            # 无有效约束 → e* = e_D（不修正）
            return CVFRResult(
                e_star=e_D.copy(),
                e_D=e_D,
                lambda_star=0.0,
                cos_dc=None,
                angular_shift_deg=0.0,
                case="no_constraints",
                n_active=0,
            )

        # 第三步：编码约束，加权聚合
        # 使用 raw_text 而非 target_action（后者存在语义 bug，见实验日志 V1 §5.3）
        constraint_texts = [c.raw_text for c in active]
        weights = np.array(
            [1.0 if c.constraint_type == "ABSOLUTE" else 0.5 for c in active],
            dtype=np.float32,
        )

        e_C_list = self._model.encode(
            constraint_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)   # shape=(n_active, d)

        # 加权求和 + 归一化 → ê_C（聚合约束方向）
        e_C_raw = (weights[:, None] * e_C_list).sum(axis=0)   # shape=(d,)
        e_C_norm = float(np.linalg.norm(e_C_raw))

        if e_C_norm < 1e-8:
            # 约束向量退化为零向量（极罕见）
            return CVFRResult(
                e_star=e_D.copy(),
                e_D=e_D,
                lambda_star=0.0,
                cos_dc=None,
                angular_shift_deg=0.0,
                case="no_constraints",
                n_active=len(active),
                debug={"note": "e_C collapsed to zero vector"},
            )

        e_hat_C = (e_C_raw / e_C_norm).astype(np.float32)

        # 第四步：计算 cos(e_D, ê_C)，判断情况
        cos_dc = float(np.dot(e_D, e_hat_C))

        if cos_dc >= self._tau:
            # 情况1：约束方向已被疾病方向自然覆盖，不需修正
            return CVFRResult(
                e_star=e_D.copy(),
                e_D=e_D,
                lambda_star=0.0,
                cos_dc=cos_dc,
                angular_shift_deg=0.0,
                case="constraint_implicit",
                n_active=len(active),
                debug={
                    "tau": self._tau,
                    "cos_dc": cos_dc,
                    "reason": f"cos_dc={cos_dc:.4f} >= tau={self._tau}，约束已隐含",
                },
            )

        # 第五步：情况2——约束激活，计算 λ*(τ, c)
        #
        # 正确闭合解（二次方程 KKT 推导，详见 docs/cvfr_theory.md §5.2-5.3）：
        #
        #   λ*(τ, c) = -c + τ√(1-c²) / √(1-τ²)
        #
        # 推导：令 cos(normalize(e_D + λ·ê_C), ê_C) = τ
        #   (c + λ) / √(1 + 2λc + λ²) = τ
        # 两边平方并整理，得到关于 λ 的二次方程：
        #   (1-τ²)λ² + 2c(1-τ²)λ + (c²-τ²) = 0
        # 判别式 Δ/4 = τ²(1-τ²)(1-c²)，取正根（λ≥0，KKT）：
        #   λ* = [-c + τ√(1-c²)/√(1-τ²)]
        #
        # 注意：理论文档 §5.3 中原始公式 λ*(τ,c)=[τ√(1-τ²+c²)-c(1-τ²)]/(1-τ²)
        # 仅在 c=0 时成立，对一般 c 有误，此处使用正确推导结果。
        tau = self._tau
        c = cos_dc

        if abs(1.0 - tau**2) < 1e-10:
            # τ → 1 的极限情况（理论上不应发生，τ 应远小于 1）
            raise ValueError(
                f"[CVFRQueryConstructor] τ={tau} 过于接近 1.0，λ* 分母 √(1-τ²) 退化。"
                f"请将 τ 设置在 [0.0, 0.95] 范围内。"
            )

        lambda_star = float(-c + tau * np.sqrt(max(0.0, 1.0 - c**2)) / np.sqrt(1.0 - tau**2))

        # 条件查询向量：e* = normalize(e_D + λ* · ê_C)
        e_star_raw = e_D + lambda_star * e_hat_C
        e_star_norm = float(np.linalg.norm(e_star_raw))

        if e_star_norm < 1e-8:
            raise ValueError(
                f"[CVFRQueryConstructor] e* 归一化异常：e_star_raw 接近零向量。"
                f"lambda_star={lambda_star:.4f}, cos_dc={cos_dc:.4f}"
            )

        e_star = (e_star_raw / e_star_norm).astype(np.float32)

        # 计算角度偏移（用于调试，验证修正幅度是否合理）
        cos_shift = float(np.dot(e_D, e_star))
        cos_shift_clipped = float(np.clip(cos_shift, -1.0, 1.0))
        angular_shift_deg = float(np.degrees(np.arccos(cos_shift_clipped)))

        # 验证 cos(e*, ê_C) ≈ τ（数值校验）
        cos_e_star_C = float(np.dot(e_star, e_hat_C))

        return CVFRResult(
            e_star=e_star,
            e_D=e_D,
            lambda_star=lambda_star,
            cos_dc=cos_dc,
            angular_shift_deg=angular_shift_deg,
            case="constraint_active",
            n_active=len(active),
            debug={
                "tau": tau,
                "cos_dc": cos_dc,
                "lambda_star": lambda_star,
                "cos_e_star_C": cos_e_star_C,    # 应 ≈ τ（数值验证）
                "angular_shift_deg": angular_shift_deg,
                "n_active_constraints": len(active),
                "constraint_types": [c.constraint_type for c in active],
            },
        )
