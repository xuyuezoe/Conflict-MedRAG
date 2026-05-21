#!/usr/bin/env python3
"""
教材索引构建器

将 18 本英文医学教材切分为文本块，构建 BM25 索引和 FAISS 向量索引。

执行三阶段流程：
  第一阶段：文本切分 → data/index/chunks.jsonl
  第二阶段：BM25 索引 → data/index/bm25/（rank_bm25 + pickle）
  第三阶段：向量索引 → data/index/dense/（FAISS IndexFlatIP）

使用方法：
  python3 scripts/index_textbooks.py \\
      --textbook-dir data_clean/textbooks/en \\
      --output-dir   data/index \\
      --chunk-size   400 \\
      --chunk-overlap 50 \\
      --embedding-model sentence-transformers/all-MiniLM-L6-v2

块 ID 格式：{book_slug}_{idx:05d}，如 InternalMed_Harrison_00042

设计决策：
  切分单位优先段落（\\n\\n），避免跨段落的语义碎片。
  FAISS 使用 IndexFlatIP + L2 归一化，等价于余弦相似度，便于后续 RRF 融合。
  块 ID 保序映射持久化到 chunk_ids.json，与 FAISS 索引一一对应。
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.types import TextChunk


# ── 文本切分 ──────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    粗略估计 token 数（按空格分词，实际 BPE token 数约为此值的 0.75）。

    用于切分决策，不需要精确计数。
    """
    return len(text.split())


def split_text_into_chunks(
    text: str,
    book_slug: str,
    chunk_size: int,
    overlap_tokens: int,
    min_chunk_tokens: int = 50,
) -> List[TextChunk]:
    """
    将单本教材文本切分为重叠文本块。

    参数：
        text:             原始教材文本
        book_slug:        教材标识符（用于生成 chunk_id）
        chunk_size:       目标块大小（token 数）
        overlap_tokens:   相邻块重叠 token 数
        min_chunk_tokens: 最小块大小，过小的块直接丢弃

    返回：
        TextChunk 列表（按文档顺序）

    核心逻辑：
      第一步：按双换行（\\n\\n）分割为段落
      第二步：段落累积，达到 chunk_size 时截断，保留 overlap 的 token 到下一块
      第三步：过长单段落按句（'. '）进一步切分
    """
    # 第一步：段落切分
    raw_paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    chunks: List[TextChunk] = []
    chunk_idx = 0
    current_tokens: List[str] = []    # 当前块的 token 列表（按空格分词）
    current_start_char = 0
    char_pos = 0

    for para in paragraphs:
        para_tokens = para.split()

        # 单段落超过 1.5×chunk_size 时拆分为句子
        if len(para_tokens) > int(1.5 * chunk_size):
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub_units = sentences
        else:
            sub_units = [para]

        for unit in sub_units:
            unit_tokens = unit.split()

            # 累积 token，未超限时直接追加
            if len(current_tokens) + len(unit_tokens) <= chunk_size:
                current_tokens.extend(unit_tokens)
            else:
                # 当前块已满，先落盘当前块
                if len(current_tokens) >= min_chunk_tokens:
                    chunk_text = " ".join(current_tokens)
                    end_char = current_start_char + len(chunk_text)
                    chunks.append(TextChunk(
                        chunk_id=f"{book_slug}_{chunk_idx:05d}",
                        source_book=book_slug,
                        text=chunk_text,
                        start_char=current_start_char,
                        end_char=end_char,
                        token_count=len(current_tokens),
                    ))
                    chunk_idx += 1

                    # 保留 overlap 用于下一块
                    overlap_start = max(0, len(current_tokens) - overlap_tokens)
                    current_tokens = current_tokens[overlap_start:]
                    current_start_char = end_char - len(" ".join(current_tokens))

                # 将当前 unit 加入新块
                current_tokens.extend(unit_tokens)

                # 若 unit 本身超过 chunk_size，强制切断
                while len(current_tokens) > chunk_size:
                    chunk_text = " ".join(current_tokens[:chunk_size])
                    end_char = current_start_char + len(chunk_text)
                    chunks.append(TextChunk(
                        chunk_id=f"{book_slug}_{chunk_idx:05d}",
                        source_book=book_slug,
                        text=chunk_text,
                        start_char=current_start_char,
                        end_char=end_char,
                        token_count=chunk_size,
                    ))
                    chunk_idx += 1
                    overlap_start = chunk_size - overlap_tokens
                    current_tokens = current_tokens[overlap_start:]
                    current_start_char = end_char - len(" ".join(current_tokens))

        char_pos += len(para) + 2   # +2 for "\n\n"

    # 落盘最后一块（可能不满 chunk_size）
    if len(current_tokens) >= min_chunk_tokens:
        chunk_text = " ".join(current_tokens)
        chunks.append(TextChunk(
            chunk_id=f"{book_slug}_{chunk_idx:05d}",
            source_book=book_slug,
            text=chunk_text,
            start_char=current_start_char,
            end_char=current_start_char + len(chunk_text),
            token_count=len(current_tokens),
        ))

    return chunks


def process_all_textbooks(
    textbook_dir: Path,
    chunk_size: int,
    overlap_tokens: int,
) -> List[TextChunk]:
    """
    处理目录下所有 .txt 教材文件，返回全部文本块。

    参数：
        textbook_dir:   教材目录（data_clean/textbooks/en）
        chunk_size:     每块目标 token 数
        overlap_tokens: 相邻块重叠 token 数

    返回：
        全部教材的 TextChunk 列表（按教材顺序拼接）
    """
    txt_files = sorted(textbook_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"[索引构建] 在 {textbook_dir} 中未找到 .txt 文件")

    all_chunks: List[TextChunk] = []
    for txt_path in txt_files:
        book_slug = txt_path.stem
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        chunks = split_text_into_chunks(
            text=text,
            book_slug=book_slug,
            chunk_size=chunk_size,
            overlap_tokens=overlap_tokens,
        )
        print(f"  {book_slug}: {len(chunks)} 块（{len(text):,} 字符）")
        all_chunks.extend(chunks)

    return all_chunks


# ── BM25 索引 ─────────────────────────────────────────────────────────────────

def build_bm25_index(
    chunks: List[TextChunk],
    output_dir: Path,
) -> None:
    """
    构建 BM25 索引并序列化到磁盘。

    使用 rank_bm25.BM25Okapi（业界标准参数 k1=1.5, b=0.75）。
    词条化：简单空格分词 + 小写（医学缩写对大小写敏感，故不去停用词）。
    序列化：pickle 存储（BM25 对象 + 保序 chunk_id 列表）。

    参数：
        chunks:     全部文本块
        output_dir: BM25 索引输出目录（data/index/bm25）
    """
    from rank_bm25 import BM25Okapi

    print("\n[第二阶段] 构建 BM25 索引...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 词条化：小写，空格分词（保留医学缩写的大小写区分依赖于查询时同样小写）
    tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    # 持久化
    index_path = output_dir / "bm25_index.pkl"
    with index_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    print(f"  BM25 索引写出 → {index_path}（{len(chunks):,} 块）")


# ── Dense 向量索引 ────────────────────────────────────────────────────────────

def build_dense_index(
    chunks: List[TextChunk],
    output_dir: Path,
    embedding_model: str,
    batch_size: int = 64,
) -> None:
    """
    构建 FAISS 向量索引。

    使用 sentence-transformers 计算 embedding。
    FAISS 类型：IndexFlatIP（内积）+ L2 归一化 → 等价于余弦相似度。
    保序映射：chunk_ids.json 与 FAISS 行号严格对应。

    参数：
        chunks:          全部文本块
        output_dir:      Dense 索引输出目录（data/index/dense）
        embedding_model: SentenceTransformers 模型名或路径
        batch_size:      批处理大小（GPU 可加大）
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"\n[第三阶段] 构建 Dense 索引（模型: {embedding_model}）...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(embedding_model)
    texts = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    print(f"  计算 {len(texts):,} 个文本块的 embedding（batch_size={batch_size}）...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,    # L2 归一化，与 IndexFlatIP 配合使用余弦相似度
    )

    # 构建 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # 持久化
    faiss.write_index(index, str(output_dir / "faiss.index"))
    with (output_dir / "chunk_ids.json").open("w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)

    print(f"  Dense 索引写出 → {output_dir}/faiss.index（dim={dim}, n={len(chunks):,}）")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="教材索引构建器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--textbook-dir",
        type=Path,
        default=Path("data_clean/textbooks/en"),
        help="英文教材目录（默认 data_clean/textbooks/en）",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/index"),
        help="索引输出目录（默认 data/index）",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="每块目标 token 数（默认 400）",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="相邻块重叠 token 数（默认 50）",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers 模型（默认 all-MiniLM-L6-v2）",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding 批处理大小（默认 64）",
    )
    p.add_argument(
        "--skip-dense",
        action="store_true",
        help="跳过 Dense 索引构建（仅用于快速测试 BM25）",
    )
    args = p.parse_args()

    if not args.textbook_dir.exists():
        raise FileNotFoundError(f"[教材目录不存在] {args.textbook_dir}")

    # 第一阶段：文本切分
    print(f"[第一阶段] 文本切分（chunk_size={args.chunk_size}, overlap={args.chunk_overlap}）")
    all_chunks = process_all_textbooks(
        textbook_dir=args.textbook_dir,
        chunk_size=args.chunk_size,
        overlap_tokens=args.chunk_overlap,
    )
    print(f"\n  总文本块数: {len(all_chunks):,}")

    # 写出 chunks.jsonl（元数据持久化）
    chunks_path = args.output_dir / "chunks.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            row = {
                "chunk_id":    chunk.chunk_id,
                "source_book": chunk.source_book,
                "text":        chunk.text,
                "start_char":  chunk.start_char,
                "end_char":    chunk.end_char,
                "token_count": chunk.token_count,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  文本块元数据 → {chunks_path}")

    # 第二阶段：BM25 索引
    build_bm25_index(
        chunks=all_chunks,
        output_dir=args.output_dir / "bm25",
    )

    # 第三阶段：Dense 索引（可选）
    if not args.skip_dense:
        build_dense_index(
            chunks=all_chunks,
            output_dir=args.output_dir / "dense",
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
        )
    else:
        print("\n[第三阶段] 已跳过 Dense 索引构建（--skip-dense）")

    print(f"\n索引构建完成 → {args.output_dir}")
    print(f"  总块数: {len(all_chunks):,}")
    from collections import Counter
    book_dist = Counter(c.source_book for c in all_chunks)
    print("  各教材块数:")
    for book, cnt in sorted(book_dist.items(), key=lambda x: -x[1]):
        print(f"    {book}: {cnt}")


if __name__ == "__main__":
    main()
