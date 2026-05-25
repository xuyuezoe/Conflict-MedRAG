#!/usr/bin/env python3
"""
Scope Index 离线构建脚本

功能：
  对 data/index/chunks.jsonl 中的全量 43,238 个 chunk 构建 scope embedding 索引，
  保存至 data/index/scope/，供 ScopeIndex.load() 在线加载使用。

三层数据关系（理解背景）：
  Layer 1：data/index/chunks.jsonl         ← 文本切片（已完成）
  Layer 2：data/index/dense/faiss.index    ← 内容 Embedding，对应疾病语义 e^content（已完成）
  Layer 3：data/index/scope/               ← Scope Embedding，对应适用域语义 e^scope（本脚本构建）

Scope Embedding 的来源：
  每个 chunk 的 e^scope 不是直接对原文做 embedding，而是：
    ScopePredicate → scope_predicate_to_text() → scope 描述文本 → embedding
  这样做使 cos(e^scope, e_C) 聚焦于"适用域"维度，不被疾病语义污染。

两阶段策略（成本控制）：
  阶段 A（LLM 提取，昂贵）：
    - 默认 --no-llm 跳过此阶段
    - 运行实验后，KappaScorer 自动缓存检索过的 chunk 的 ScopePredicate
    - 重复运行此脚本时覆盖率自动提升
  阶段 B（Embedding 编码，便宜）：
    - 用缓存 ScopePredicate 生成 scope 描述文本
    - 未缓存的 chunk 用 "not_specified" 中性文本占位（e^scope 为中性方向）
    - 全量 43K 条编码约 10-20 分钟（GPU）

用法：
  # 最常用：只用缓存，不触发 LLM（约 10-20 分钟 GPU 编码）
  python3 scripts/build_scope_index.py --no-llm

  # 允许 LLM 补充提取（实验积累缓存后使用）
  python3 scripts/build_scope_index.py --model claude-haiku-4-5-20251001

  # 指定输出目录（默认 data/index/scope）
  python3 scripts/build_scope_index.py --no-llm --output data/index/scope

  # 仅对已缓存的 chunk 建索引（跳过未缓存的，不做 not_specified 占位）
  python3 scripts/build_scope_index.py --no-llm --cached-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 清除代理环境变量，防止 httpx 因 socks:// 协议崩溃
# 同时强制 HuggingFace 离线模式，避免 etag 网络检查（模型已在本地缓存）
for _k in ["ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"]:
    os.environ.pop(_k, None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.kappa_scorer import KappaScorer
from src.llm_client import get_client, get_embedding_model
from src.scope_index import ScopeIndex, scope_predicate_to_text
from src.types import ScopePredicate, TextChunk


# ── 常量 ──────────────────────────────────────────────────────────────────────

DEFAULT_CHUNKS_PATH = _ROOT / "data" / "index" / "chunks.jsonl"
DEFAULT_CACHE_DIR   = _ROOT / "data" / "cache"
DEFAULT_OUTPUT_DIR  = _ROOT / "data" / "index" / "scope"
DEFAULT_BATCH_SIZE  = 256   # embedding 编码批大小（无 LLM，可用大 batch 充分利用 GPU）


# ── Phase 1：加载 chunk 列表 ──────────────────────────────────────────────────

def load_chunks(chunks_path: Path) -> List[TextChunk]:
    """
    从 JSONL 文件加载全量 TextChunk 列表。

    参数：
        chunks_path: chunks.jsonl 路径

    返回：
        List[TextChunk]，顺序与文件行序一致

    异常：
        FileNotFoundError: 文件不存在
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"[build_scope_index] chunks.jsonl 不存在: {chunks_path}\n"
            f"请先运行 python3 scripts/index_textbooks.py"
        )

    chunks: List[TextChunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            chunks.append(TextChunk(
                chunk_id=row["chunk_id"],
                source_book=row["source_book"],
                text=row["text"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                token_count=row.get("token_count", 0),
            ))

    return chunks


# ── Phase 2：加载 ScopePredicate 缓存 ─────────────────────────────────────────

def load_cached_predicates(
    chunks: List[TextChunk],
    cache_dir: Path,
    kappa_scorer: KappaScorer,
) -> Tuple[Dict[str, ScopePredicate], List[TextChunk]]:
    """
    从磁盘缓存加载已有 ScopePredicate，返回缓存命中和未命中的 chunk。

    参数：
        chunks:       全量 TextChunk 列表
        cache_dir:    缓存基础目录（内含 scope_predicates/ 子目录）
        kappa_scorer: KappaScorer 实例（用于 v1/v2 兼容解析）

    返回：
        (predicates, miss_chunks)
        predicates:  chunk_id → ScopePredicate 的缓存命中字典
        miss_chunks: 缓存未命中的 chunk 列表（需 LLM 提取或占位）
    """
    scope_cache_dir = cache_dir / "scope_predicates"
    predicates: Dict[str, ScopePredicate] = {}
    miss_chunks: List[TextChunk] = []

    for chunk in chunks:
        pred_path = scope_cache_dir / f"{chunk.chunk_id}.json"
        if pred_path.exists():
            data = json.loads(pred_path.read_text(encoding="utf-8"))
            predicates[chunk.chunk_id] = kappa_scorer._scope_predicate_from_cache_dict(data)
        else:
            miss_chunks.append(chunk)

    return predicates, miss_chunks


# ── Phase 3（可选）：LLM 补充提取 ────────────────────────────────────────────

def llm_extract_missing(
    miss_chunks: List[TextChunk],
    predicates: Dict[str, ScopePredicate],
    cache_dir: Path,
    kappa_scorer: KappaScorer,
    batch_size_llm: int = 10,
) -> None:
    """
    对缓存未命中的 chunk 批量调用 LLM 提取 ScopePredicate，写入缓存。

    操作是原地修改 predicates dict（而非返回新 dict），
    以便后续 Phase 4 能统一读取。

    参数：
        miss_chunks:    需要 LLM 提取的 chunk 列表
        predicates:     当前已缓存的 predicate 字典（原地更新）
        cache_dir:      缓存基础目录
        kappa_scorer:   含有效 LLM client 的 KappaScorer 实例
        batch_size_llm: 每批次处理的 chunk 数（默认 10）
    """
    from src.retriever import RetrievalResult

    print(f"[build_scope_index] Phase 3：LLM 提取 {len(miss_chunks)} 个缓存未命中的 chunk...")
    scope_cache_dir = cache_dir / "scope_predicates"

    total_batches = (len(miss_chunks) + batch_size_llm - 1) // batch_size_llm
    for batch_idx in range(0, len(miss_chunks), batch_size_llm):
        batch = miss_chunks[batch_idx: batch_idx + batch_size_llm]
        current_batch = batch_idx // batch_size_llm + 1
        if current_batch % 10 == 1 or current_batch == total_batches:
            print(f"  批次 {current_batch}/{total_batches}（{batch_idx + len(batch)}/{len(miss_chunks)} 条）...")

        # 构造 RetrievalResult 供 _batch_extract_and_cache 使用
        fake_results = [
            RetrievalResult(chunk=c, score=1.0, retrieval_method="scope_build")
            for c in batch
        ]
        kappa_scorer._batch_extract_and_cache(fake_results)

        # 读取刚写入的缓存，更新 predicates dict
        for chunk in batch:
            pred_path = scope_cache_dir / f"{chunk.chunk_id}.json"
            if pred_path.exists():
                data = json.loads(pred_path.read_text(encoding="utf-8"))
                predicates[chunk.chunk_id] = kappa_scorer._scope_predicate_from_cache_dict(data)


# ── Phase 4：生成占位 predicate 并统计 ────────────────────────────────────────

def fill_placeholders(
    chunks: List[TextChunk],
    predicates: Dict[str, ScopePredicate],
    cached_only: bool = False,
) -> Tuple[List[TextChunk], List[str], Dict[str, int]]:
    """
    为缓存未命中的 chunk 生成 not_specified 占位 ScopePredicate，
    或在 --cached-only 模式下过滤掉这些 chunk。

    参数：
        chunks:      全量 TextChunk 列表
        predicates:  当前已有 predicate 字典（含缓存命中 + LLM 提取结果）
        cached_only: True → 仅保留有真实 predicate 的 chunk；False → 未命中用占位

    返回：
        (filtered_chunks, scope_texts, stats)
        filtered_chunks: 最终参与 embedding 编码的 chunk 列表
        scope_texts:     对应的 scope 描述文本列表
        stats:           统计信息 dict
    """
    filtered_chunks: List[TextChunk] = []
    scope_texts: List[str] = []
    n_explicit = 0
    n_placeholder = 0
    n_skipped = 0

    for chunk in chunks:
        pred = predicates.get(chunk.chunk_id)

        if pred is None:
            if cached_only:
                n_skipped += 1
                continue
            # 占位：not_specified 中性文本
            pred = ScopePredicate(
                chunk_id=chunk.chunk_id,
                recommended_action="none",
                population="",
                contraindications=[],
                relative_restrictions=[],
                extraction_model="placeholder",
                raw_output="",
                scope_status="not_specified",
            )
            n_placeholder += 1
        else:
            n_explicit += 1

        filtered_chunks.append(chunk)
        scope_texts.append(scope_predicate_to_text(pred))

    stats = {
        "n_explicit":    n_explicit,
        "n_placeholder": n_placeholder,
        "n_skipped":     n_skipped,
        "n_total":       len(filtered_chunks),
    }
    return filtered_chunks, scope_texts, stats


# ── Phase 5：批量 embedding 编码 ──────────────────────────────────────────────

def batch_encode_scope_texts(
    scope_texts: List[str],
    embedding_model,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    对 scope 描述文本批量编码，返回 L2 归一化向量矩阵。

    参数：
        scope_texts:     scope 描述文本列表
        embedding_model: SentenceTransformer 实例
        batch_size:      编码批大小

    返回：
        np.ndarray，shape=(N, d)，float32，每行已 L2 归一化
    """
    all_vectors: List[np.ndarray] = []
    n_total = len(scope_texts)

    for i in range(0, n_total, batch_size):
        batch = scope_texts[i: i + batch_size]
        vecs = embedding_model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        all_vectors.append(vecs)

        # 每 20 批打印一次进度（约每 5120 条）
        if (i // batch_size) % 20 == 0 or i + batch_size >= n_total:
            print(f"  已编码 {min(i + batch_size, n_total)}/{n_total} 条")

    return np.vstack(all_vectors)  # shape=(N, d)


# ── Phase 6：保存并验证 ───────────────────────────────────────────────────────

def save_and_verify(
    output_dir: Path,
    scope_vectors: np.ndarray,
    chunk_ids: List[str],
) -> None:
    """
    保存 scope index 文件并做基础完整性校验。

    保存格式：
        scope_vectors.npy: float32，shape=(N, d)，L2 归一化
        chunk_ids.json:    List[str]，与 scope_vectors 行对齐

    参数：
        output_dir:    输出目录（不存在时自动创建）
        scope_vectors: 已归一化的 scope 向量矩阵
        chunk_ids:     chunk ID 列表（与矩阵行对齐）

    异常：
        AssertionError: 完整性校验失败
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = output_dir / "scope_vectors.npy"
    ids_path     = output_dir / "chunk_ids.json"

    np.save(str(vectors_path), scope_vectors)
    ids_path.write_text(
        json.dumps(chunk_ids, ensure_ascii=False),
        encoding="utf-8",
    )

    # 完整性校验：向量数与 ID 数应一致
    assert scope_vectors.shape[0] == len(chunk_ids), (
        f"向量行数 {scope_vectors.shape[0]} ≠ chunk_id 数 {len(chunk_ids)}"
    )

    # 归一化校验：所有向量的 L2 范数应 ≈ 1.0
    norms = np.linalg.norm(scope_vectors, axis=1)
    norm_min, norm_max = float(norms.min()), float(norms.max())
    if not (0.99 < norm_min and norm_max < 1.01):
        raise ValueError(
            f"[build_scope_index] L2 归一化校验失败：norm 范围 [{norm_min:.4f}, {norm_max:.4f}]，"
            f"期望 [0.99, 1.01]。检查 embedding 模型的 normalize_embeddings 参数。"
        )

    print(f"[build_scope_index] 已保存至 {output_dir}")
    print(f"  scope_vectors.npy: shape={scope_vectors.shape}, dtype={scope_vectors.dtype}")
    print(f"  chunk_ids.json:    {len(chunk_ids)} 条")
    print(f"  L2 norm 范围: [{norm_min:.4f}, {norm_max:.4f}] ✓")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="构建 Scope Embedding 索引（CVFR Phase 1 离线准备）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="禁用 LLM 提取（只使用现有缓存 + not_specified 占位）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM 提取模型 ID（仅 --no-llm 未设时有效，默认读 LLM_MODEL 环境变量）",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--chunks",
        default=str(DEFAULT_CHUNKS_PATH),
        help=f"chunks.jsonl 路径（默认 {DEFAULT_CHUNKS_PATH}）",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"缓存基础目录（默认 {DEFAULT_CACHE_DIR}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"embedding 编码批大小（默认 {DEFAULT_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="仅对已有缓存的 chunk 建索引（跳过未缓存 chunk，不做占位）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    output_dir = Path(args.output)
    cache_dir  = Path(args.cache_dir)
    chunks_path = Path(args.chunks)

    print("=" * 60)
    print("[build_scope_index] Scope Index 构建启动")
    print(f"  输出目录:   {output_dir}")
    print(f"  缓存目录:   {cache_dir}")
    print(f"  LLM 模式:   {'禁用（--no-llm）' if args.no_llm else '启用'}")
    print(f"  仅缓存模式: {'是（--cached-only）' if args.cached_only else '否（含 not_specified 占位）'}")
    print(f"  编码批大小: {args.batch_size}")
    print("=" * 60)

    # ── Phase 1：加载 chunk 列表 ──────────────────────────────────────────────
    print("\n[Phase 1] 加载 chunk 列表...")
    t0 = time.time()
    chunks = load_chunks(chunks_path)
    print(f"  加载完成：{len(chunks)} 个 chunk（耗时 {time.time() - t0:.1f}s）")

    # ── 初始化 KappaScorer（client=None 时不触发 LLM）────────────────────────
    llm_client = None
    if not args.no_llm:
        print("\n[build_scope_index] 初始化 LLM 客户端...")
        llm_client = get_client()

    kappa_scorer = KappaScorer(
        client=llm_client,
        model=args.model or "claude-haiku-4-5-20251001",
        cache_dir=cache_dir,
    )

    # ── Phase 2：加载磁盘缓存 ─────────────────────────────────────────────────
    print("\n[Phase 2] 加载 ScopePredicate 缓存...")
    t0 = time.time()
    predicates, miss_chunks = load_cached_predicates(chunks, cache_dir, kappa_scorer)
    hit_rate = len(predicates) / len(chunks) * 100
    print(f"  缓存命中：{len(predicates)} 条（{hit_rate:.1f}%）")
    print(f"  缓存未命中：{len(miss_chunks)} 条（耗时 {time.time() - t0:.1f}s）")

    # ── Phase 3（可选）：LLM 补充提取 ─────────────────────────────────────────
    if miss_chunks and not args.no_llm:
        t0 = time.time()
        llm_extract_missing(miss_chunks, predicates, cache_dir, kappa_scorer)
        newly_extracted = sum(
            1 for c in miss_chunks if c.chunk_id in predicates
        )
        print(f"  LLM 提取完成：{newly_extracted}/{len(miss_chunks)} 条成功（耗时 {time.time() - t0:.1f}s）")
    elif miss_chunks and args.no_llm:
        mode = "跳过（--cached-only）" if args.cached_only else f"占位 not_specified（{len(miss_chunks)} 条）"
        print(f"\n[Phase 3] LLM 提取已禁用，缓存未命中处理：{mode}")

    # ── Phase 4：生成 scope 描述文本 + 填充占位 ───────────────────────────────
    print("\n[Phase 4] 生成 scope 描述文本...")
    t0 = time.time()
    filtered_chunks, scope_texts, stats = fill_placeholders(chunks, predicates, args.cached_only)
    print(f"  真实 predicate：{stats['n_explicit']} 条")
    print(f"  占位 not_specified：{stats['n_placeholder']} 条")
    if stats['n_skipped']:
        print(f"  已跳过（--cached-only）：{stats['n_skipped']} 条")
    print(f"  参与编码总量：{stats['n_total']} 条（耗时 {time.time() - t0:.1f}s）")

    if stats['n_total'] == 0:
        raise RuntimeError(
            "[build_scope_index] 没有可编码的 chunk。"
            "请先运行实验以积累 ScopePredicate 缓存，或去掉 --cached-only 参数。"
        )

    # ── Phase 5：加载 embedding 模型并批量编码 ────────────────────────────────
    print(f"\n[Phase 5] 加载 embedding 模型并批量编码 {stats['n_total']} 条 scope 文本...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model_name = get_embedding_model()
    print(f"  模型：{model_name}")
    embedding_model = SentenceTransformer(model_name)

    scope_vectors = batch_encode_scope_texts(scope_texts, embedding_model, args.batch_size)
    chunk_ids = [c.chunk_id for c in filtered_chunks]
    print(f"  编码完成：shape={scope_vectors.shape}（耗时 {time.time() - t0:.1f}s）")

    # ── Phase 6：保存并验证 ───────────────────────────────────────────────────
    print(f"\n[Phase 6] 保存 scope index 至 {output_dir}...")
    save_and_verify(output_dir, scope_vectors, chunk_ids)

    # ── 最终统计 ──────────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("[build_scope_index] 构建完成")
    print(f"  总耗时：{t_total:.1f}s（{t_total / 60:.1f}min）")
    print(f"  Scope Index 覆盖率：{stats['n_explicit']}/{stats['n_total']} 条有真实 predicate "
          f"（{stats['n_explicit'] / max(stats['n_total'], 1) * 100:.1f}%）")
    print(f"  向量维度：{scope_vectors.shape[1]}")
    print(f"\n下一步：")
    print(f"  1. 运行实验以积累更多 ScopePredicate 缓存")
    print(f"  2. 重新运行此脚本提升覆盖率")
    print(f"  3. 在 eval/evaluate.py 中调用 ScopeIndex.load() 计算 CDR/RSI 指标")
    print("=" * 60)


if __name__ == "__main__":
    main()
