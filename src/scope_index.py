#!/usr/bin/env python3
"""
Scope Embedding 索引管理器（CVFR Phase 1）

功能：
  1. 离线批处理：为语料库所有 chunk 生成结构化 ScopePredicate（v2）+ scope 嵌入向量
  2. 在线查询：根据患者约束向量 e_C 检索最近邻 scope 相关文档
  3. 指标计算：为 CDR（Conditional Document Relevance）和 RSI（Retrieval Specificity Index）
               提供 scope embedding 支持

数学基础（cvfr_theory.md §6）：
  每个文档 chunk 有两个独立嵌入：
    e^content_i：内容嵌入（讲什么疾病/治疗，对应查询侧 e_D）
    e^scope_i：  适用域嵌入（适用于什么患者，对应查询侧 e_C）

  e^scope_i 由以下生成（而非直接对原文做 embedding）：
    ScopePredicate → 结构化适用域 predicate 转换为 scope 描述文本 → embedding

  这样做的原因：
    - 直接对原文做 embedding 会将内容语义（疾病/治疗）和适用域语义（患者特征）纠缠
    - scope 描述文本聚焦于患者特征维度，使得 cos(e_scope, e_C) 更能反映适用域对齐程度

  scope_status="not_specified" 的文档：
    生成一个远离所有具体约束方向的 scope embedding
    （通过 "general population, applicability not specified" 描述文本生成）

设计约定：
  - scope embedding 索引与内容索引（FAISS）分开存储，但共用同一 SentenceTransformer 实例
  - 结构化 ScopePredicate 存入 JSON 元数据库（供 hard gate 和蕴含判断使用）
  - scope embedding 存入 FAISS IndexFlatIP（供相似度召回使用）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.types import RetrievedDoc, ScopePredicate, TextChunk


# ── scope 描述文本生成 ─────────────────────────────────────────────────────────

def scope_predicate_to_text(predicate: ScopePredicate) -> str:
    """
    将结构化 ScopePredicate 转换为适合 embedding 的 scope 描述文本。

    参数：
        predicate: 结构化适用域 predicate（v2，含 polarity/scope_inclusion/scope_exclusion）

    返回：
        str：聚焦于患者特征维度的描述文本，供 embedding 编码

    生成规则：
        not_specified → 中性描述（远离所有具体约束方向）
        contraindicated → 明确排除描述
        其他 → 基于 scope_inclusion + scope_exclusion + cautions 组合描述

    为什么不直接对原文 embedding：
        原文混合了疾病语义和患者特征语义，直接 embedding 后 cos(e_scope, e_C) 会
        被疾病语义污染（疾病相关文本的 embedding 天然靠近疾病 query，而非约束 query）。
        scope 描述文本只保留患者特征维度，使 e_scope 在患者约束语义空间中定位准确。
    """
    if predicate.scope_status == "not_specified":
        # 适用域未声明：生成中性描述，使 e_scope 不偏向任何具体约束方向
        # 避免因"无信息"而在余弦相似度计算中产生虚假高分
        return "general population, applicability scope not explicitly stated in text"

    parts: List[str] = []

    # 治疗动作的极性描述
    action = predicate.recommended_action
    if action and action != "none":
        if predicate.polarity == "contraindicated":
            parts.append(f"contraindicated: {action}")
        elif predicate.polarity == "recommended":
            parts.append(f"recommended for use: {action}")
        elif predicate.polarity in ("caution", "dose_adjustment"):
            parts.append(f"use with caution: {action}")

    # 适用人群（inclusion scope）
    if predicate.scope_inclusion:
        parts.append(f"applicable to: {', '.join(predicate.scope_inclusion)}")

    # 排除人群（exclusion scope）
    if predicate.scope_exclusion:
        parts.append(f"contraindicated for: {', '.join(predicate.scope_exclusion)}")

    # 传统禁忌证（后向兼容 v1 字段）
    if predicate.contraindications and not predicate.scope_exclusion:
        parts.append(f"contraindications: {'; '.join(predicate.contraindications[:3])}")

    if not parts:
        return "general population, no specific applicability criteria stated"

    return "; ".join(parts)


# ── ScopeIndex 主类 ────────────────────────────────────────────────────────────

class ScopeIndex:
    """
    Scope Embedding 索引管理器。

    管理两类数据结构：
      1. scope_embeddings：FAISS IndexFlatIP，shape=(N, d)，每行是一个 chunk 的 scope 向量
      2. scope_predicates：Dict[chunk_id → ScopePredicate]，供 hard gate 和蕴含判断使用

    关键设计决策：
      - scope embedding 与内容 embedding 共用同一 SentenceTransformer（保证向量空间一致）
      - scope_status="not_specified" 文档用中性文本生成 scope embedding，而非赋零向量
        （零向量在 cosine 空间无法归一化，且会导致所有查询的相似度为 NaN）
      - 索引构建是 I/O 密集型操作（LLM scope extraction 批量进行），
        建议离线运行 build() 一次，在线 load() 复用

    典型使用模式：
        # 离线构建（需要 KappaScorer 提取 ScopePredicate）
        index = ScopeIndex(embedding_model=retriever.embedding_model)
        index.build(chunks, kappa_scorer, cache_dir)
        index.save(scope_index_dir)

        # 在线加载
        index = ScopeIndex.load(scope_index_dir, embedding_model)

        # 检索
        cdr = index.compute_cdr(retrieved_docs, e_C)
        rsi = index.compute_rsi(retrieved_docs, e_C, theta_high=0.6)
    """

    def __init__(self, embedding_model) -> None:
        """
        参数：
            embedding_model: SentenceTransformer 实例（与 HybridRetriever 共用）
        """
        if embedding_model is None:
            raise ValueError(
                "[ScopeIndex] embedding_model 不可为 None。"
                "请确保 Dense 索引已加载（HybridRetriever 初始化后提供）。"
            )
        self._model = embedding_model
        self._scope_vectors: Optional[np.ndarray] = None   # shape=(N, d)，float32
        self._chunk_ids: List[str] = []                    # 与 _scope_vectors 行对齐
        self._predicates: Dict[str, ScopePredicate] = {}   # chunk_id → ScopePredicate

    @property
    def is_loaded(self) -> bool:
        """是否已加载 scope embedding 数据"""
        return self._scope_vectors is not None and len(self._chunk_ids) > 0

    def get_scope_vector(self, chunk_id: str) -> Optional[np.ndarray]:
        """
        获取单个 chunk 的 scope embedding 向量。

        参数：
            chunk_id: 文本块 ID

        返回：
            float32 向量（shape=(d,)，L2 归一化）；chunk 不存在时返回 None
        """
        if self._scope_vectors is None:
            return None
        try:
            idx = self._chunk_ids.index(chunk_id)
            return self._scope_vectors[idx]
        except ValueError:
            return None

    def get_scope_predicate(self, chunk_id: str) -> Optional[ScopePredicate]:
        """获取 chunk 的结构化 ScopePredicate（供 hard gate 和蕴含判断使用）"""
        return self._predicates.get(chunk_id)

    # ── 离线构建 ──────────────────────────────────────────────────────────────

    def build(
        self,
        chunks: List[TextChunk],
        kappa_scorer,          # KappaScorer 实例（类型不直接标注避免循环依赖）
        cache_dir: Optional[Path] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> None:
        """
        离线构建 scope embedding 索引。

        流程：
          第一步：对所有 chunk 调用 KappaScorer 提取 ScopePredicate（批量，带缓存）
          第二步：将 ScopePredicate 转换为 scope 描述文本
          第三步：批量编码 scope 描述文本 → e^scope 向量（L2 归一化）
          第四步：存入内存索引（_scope_vectors + _chunk_ids）

        参数：
            chunks:       所有 TextChunk（通常 43K 条）
            kappa_scorer: KappaScorer 实例（用于批量提取 ScopePredicate）
            cache_dir:    缓存目录（ScopePredicate 缓存复用 KappaScorer 的 scope_predicates 目录）
            batch_size:   embedding 编码批大小（默认 32，控制显存占用）
            show_progress: 是否打印进度
        """
        from src.retriever import RetrievalResult

        if show_progress:
            print(f"[ScopeIndex.build] 开始构建 scope index，共 {len(chunks)} 个 chunk")

        # 第一步：批量提取 ScopePredicate（复用 KappaScorer 的缓存机制）
        predicates: Dict[str, ScopePredicate] = {}
        cache_miss_chunks: List[TextChunk] = []

        scope_cache_dir = (
            cache_dir / "scope_predicates" if cache_dir else None
        )

        for chunk in chunks:
            if scope_cache_dir and (scope_cache_dir / f"{chunk.chunk_id}.json").exists():
                data = json.loads(
                    (scope_cache_dir / f"{chunk.chunk_id}.json").read_text(encoding="utf-8")
                )
                predicates[chunk.chunk_id] = kappa_scorer._scope_predicate_from_cache_dict(data)
            else:
                cache_miss_chunks.append(chunk)

        if show_progress and cache_miss_chunks:
            print(f"[ScopeIndex.build] 缓存未命中 {len(cache_miss_chunks)} 个 chunk，批量提取...")

        # 批量 LLM 提取（复用 KappaScorer 的批量提取逻辑）
        if cache_miss_chunks and kappa_scorer._client is not None:
            fake_results = [
                RetrievalResult(chunk=c, score=1.0, retrieval_method="scope_build")
                for c in cache_miss_chunks
            ]
            kappa_scorer._batch_extract_and_cache(fake_results)
            # 写入缓存后重新读取
            for chunk in cache_miss_chunks:
                if scope_cache_dir and (scope_cache_dir / f"{chunk.chunk_id}.json").exists():
                    data = json.loads(
                        (scope_cache_dir / f"{chunk.chunk_id}.json").read_text(encoding="utf-8")
                    )
                    predicates[chunk.chunk_id] = kappa_scorer._scope_predicate_from_cache_dict(data)
                else:
                    # LLM 提取失败：用 not_specified 占位
                    predicates[chunk.chunk_id] = ScopePredicate(
                        chunk_id=chunk.chunk_id,
                        recommended_action="none",
                        population="",
                        contraindications=[],
                        relative_restrictions=[],
                        extraction_model="unavailable",
                        raw_output="",
                        scope_status="not_specified",
                    )

        # 第二步：生成 scope 描述文本
        scope_texts: List[str] = []
        ordered_chunk_ids: List[str] = []
        for chunk in chunks:
            pred = predicates.get(chunk.chunk_id)
            if pred is None:
                pred = ScopePredicate(
                    chunk_id=chunk.chunk_id,
                    recommended_action="none",
                    population="",
                    contraindications=[],
                    relative_restrictions=[],
                    extraction_model="missing",
                    raw_output="",
                    scope_status="not_specified",
                )
            scope_texts.append(scope_predicate_to_text(pred))
            ordered_chunk_ids.append(chunk.chunk_id)

        if show_progress:
            print(f"[ScopeIndex.build] 编码 {len(scope_texts)} 个 scope 描述文本...")

        # 第三步：批量编码（L2 归一化）
        all_vectors: List[np.ndarray] = []
        for i in range(0, len(scope_texts), batch_size):
            batch = scope_texts[i: i + batch_size]
            vecs = self._model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            all_vectors.append(vecs)
            if show_progress and (i // batch_size) % 20 == 0:
                print(f"  已处理 {min(i + batch_size, len(scope_texts))}/{len(scope_texts)}")

        # 第四步：存入内存
        self._scope_vectors = np.vstack(all_vectors)   # shape=(N, d)
        self._chunk_ids = ordered_chunk_ids
        self._predicates = predicates

        if show_progress:
            print(f"[ScopeIndex.build] 完成，共 {len(self._chunk_ids)} 条 scope 向量")

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def save(self, scope_index_dir: Path) -> None:
        """
        将 scope index 持久化到磁盘。

        存储格式：
          scope_vectors.npy: np.ndarray，shape=(N, d)，float32
          chunk_ids.json:    List[str]，与 scope_vectors 行对齐

        ScopePredicate 本身由 KappaScorer 的 scope_predicates 缓存目录管理，
        不重复存储（避免数据重复和格式不一致）。

        参数：
            scope_index_dir: 存储目录（不存在时自动创建）
        """
        if not self.is_loaded:
            raise RuntimeError("[ScopeIndex.save] 索引未构建，请先调用 build()")

        scope_index_dir.mkdir(parents=True, exist_ok=True)
        np.save(scope_index_dir / "scope_vectors.npy", self._scope_vectors)
        (scope_index_dir / "chunk_ids.json").write_text(
            json.dumps(self._chunk_ids, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[ScopeIndex.save] 已保存至 {scope_index_dir}，共 {len(self._chunk_ids)} 条")

    @classmethod
    def load(
        cls,
        scope_index_dir: Path,
        embedding_model,
        scope_predicates_cache_dir: Optional[Path] = None,
        kappa_scorer=None,
    ) -> "ScopeIndex":
        """
        从磁盘加载 scope index。

        参数：
            scope_index_dir:           scope index 目录（含 scope_vectors.npy + chunk_ids.json）
            embedding_model:           SentenceTransformer 实例
            scope_predicates_cache_dir: ScopePredicate 缓存目录（用于加载结构化 predicate）
            kappa_scorer:              KappaScorer 实例（用于 _scope_predicate_from_cache_dict）

        返回：
            已加载的 ScopeIndex 实例
        """
        vectors_path = scope_index_dir / "scope_vectors.npy"
        ids_path = scope_index_dir / "chunk_ids.json"

        if not vectors_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"[ScopeIndex.load] scope index 文件不存在: {scope_index_dir}\n"
                f"请先运行离线构建脚本：python3 scripts/build_scope_index.py"
            )

        index = cls(embedding_model=embedding_model)
        index._scope_vectors = np.load(str(vectors_path)).astype(np.float32)
        index._chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))

        # 加载结构化 ScopePredicate（可选）
        if scope_predicates_cache_dir and kappa_scorer and scope_predicates_cache_dir.exists():
            for chunk_id in index._chunk_ids:
                pred_path = scope_predicates_cache_dir / f"{chunk_id}.json"
                if pred_path.exists():
                    data = json.loads(pred_path.read_text(encoding="utf-8"))
                    index._predicates[chunk_id] = kappa_scorer._scope_predicate_from_cache_dict(
                        data
                    )

        print(f"[ScopeIndex.load] 已加载 {len(index._chunk_ids)} 条 scope 向量")
        return index

    # ── 指标计算 ──────────────────────────────────────────────────────────────

    def compute_cdr(
        self,
        retrieved_docs: List[RetrievedDoc],
        e_C: np.ndarray,
    ) -> Dict[str, Any]:
        """
        计算 CDR（Conditional Document Relevance，条件文档相关性）。

        定义（cvfr_theory.md §9.4）：
          CDR(E(q), C) = (1/|E(q)|) × Σ_{d ∈ E(q)} cos(e^scope_d, e_C)

        含义：
          衡量检索到的文档集合中，scope embedding 与患者约束 e_C 的平均对齐程度。
          CDR 高 → 系统在"适用于当前患者"的维度上检索了文档
          CDR 低 → 系统检索的证据在约束维度上随机（等同于无条件检索）

        参数：
            retrieved_docs: 检索结果列表（RetrievedDoc）
            e_C:            患者约束聚合嵌入向量（float32，L2 归一化，shape=(d,)）

        返回：
            {
              "cdr": float（平均 scope-constraint cosine，[0,1]），
              "per_doc": List[{"chunk_id": str, "scope_cos": float}],
              "n_docs": int,
              "n_scope_available": int（有 scope embedding 的文档数）
            }
        """
        if not self.is_loaded:
            raise RuntimeError("[ScopeIndex.compute_cdr] scope index 未加载")

        e_C_norm = np.linalg.norm(e_C)
        if e_C_norm < 1e-8:
            raise ValueError("[ScopeIndex.compute_cdr] e_C 为零向量，无法计算 CDR")
        e_C_hat = (e_C / e_C_norm).astype(np.float32)

        per_doc: List[Dict[str, Any]] = []
        cosines: List[float] = []

        for doc in retrieved_docs:
            cid = doc.chunk.chunk_id
            e_scope = self.get_scope_vector(cid)
            if e_scope is None:
                continue
            cos_val = float(np.dot(e_scope, e_C_hat))
            cosines.append(cos_val)
            per_doc.append({"chunk_id": cid, "scope_cos": cos_val})

        return {
            "cdr": float(np.mean(cosines)) if cosines else 0.0,
            "per_doc": per_doc,
            "n_docs": len(retrieved_docs),
            "n_scope_available": len(cosines),
        }

    def compute_rsi(
        self,
        retrieved_docs: List[RetrievedDoc],
        e_C: np.ndarray,
        theta_high: float = 0.6,
    ) -> Dict[str, Any]:
        """
        计算 RSI（Retrieval Specificity Index，检索特异性指数）。

        定义（cvfr_theory.md §9.4）：
          RSI = #{d ∈ E(q) : cos(e^scope_d, e_C) >= θ_high} / |E(q)|

        含义：
          衡量检索结果中"约束特异性文档"的比例。
          RSI = 0 → 系统只检索通用疾病证据（无条件检索）
          RSI = 1 → 系统全部检索约束特异性证据（完美条件检索）

        参数：
            retrieved_docs: 检索结果列表
            e_C:            患者约束聚合嵌入（float32，L2 归一化）
            theta_high:     高特异性阈值（默认 0.6，需从验证集标定）

        返回：
            {
              "rsi": float（比例，[0,1]），
              "n_specific": int（高特异性文档数），
              "n_total": int（总文档数），
              "theta_high": float
            }
        """
        if not self.is_loaded:
            raise RuntimeError("[ScopeIndex.compute_rsi] scope index 未加载")

        e_C_norm = np.linalg.norm(e_C)
        if e_C_norm < 1e-8:
            raise ValueError("[ScopeIndex.compute_rsi] e_C 为零向量")
        e_C_hat = (e_C / e_C_norm).astype(np.float32)

        n_specific = 0
        n_evaluated = 0

        for doc in retrieved_docs:
            cid = doc.chunk.chunk_id
            e_scope = self.get_scope_vector(cid)
            if e_scope is None:
                continue
            n_evaluated += 1
            cos_val = float(np.dot(e_scope, e_C_hat))
            if cos_val >= theta_high:
                n_specific += 1

        return {
            "rsi": float(n_specific / n_evaluated) if n_evaluated > 0 else 0.0,
            "n_specific": n_specific,
            "n_total": n_evaluated,
            "theta_high": theta_high,
        }

    def encode_constraint(self, constraint_texts: List[str]) -> np.ndarray:
        """
        将患者约束文本编码为归一化向量（供 CDR/RSI 计算使用）。

        参数：
            constraint_texts: 患者约束的 raw_text 列表

        返回：
            聚合约束向量 e_C（float32，L2 归一化，shape=(d,)）
        """
        if not constraint_texts:
            raise ValueError("[ScopeIndex.encode_constraint] constraint_texts 为空")

        vecs = self._model.encode(
            constraint_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # 均值聚合 + 归一化
        e_C_raw = vecs.mean(axis=0)
        norm = np.linalg.norm(e_C_raw)
        if norm < 1e-8:
            raise ValueError("[ScopeIndex.encode_constraint] 聚合向量退化为零向量")
        return (e_C_raw / norm).astype(np.float32)
