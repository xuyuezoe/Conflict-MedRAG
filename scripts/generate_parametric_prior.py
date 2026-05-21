#!/usr/bin/env python3
"""
PARAMETRIC_PRIOR 生成器

为每个 SC 候选样本生成 parametric_prior 字段：
  - disease_only_query:    去除患者约束的纯疾病查询（LLM 精化版）
  - disease_only_response: LLM 在无上下文情况下的推荐（近似 P_LLM(a|D)）
  - marginal_bias_confirmed: 该推荐是否落在 INADMISSIBLE action 上（人工标注）

理论依据（research.md §2.3）：
  P_LLM(a|D) ≈ Σ_q P(a|D,q)·P(q|D) 是边缘分布，被一般人群主导。
  当患者有绝对禁忌时，P_LLM(a_INADMISSIBLE|D) 仍为高概率，
  与 P(a|D,q)=0 形成显著差距 → Experiment 2 的直接测量。

使用方法：
  LLM 配置从 .env 文件读取（参考 .env.example）。

  python3 scripts/generate_parametric_prior.py \
      --candidates data/interim/macb_candidates_v2.jsonl \
      --output     data/interim/macb_candidates_v2_with_prior.jsonl

  # 先用 dry-run 检查生成质量（不消耗 API）
  python3 scripts/generate_parametric_prior.py \
      --candidates data/interim/macb_candidates_v2.jsonl \
      --output     data/interim/macb_candidates_v2_with_prior.jsonl \
      --dry-run --limit 3

成本估算：
  150 × SC 样本 × ~200 tokens/样本 × 2 次调用 ≈ 60K tokens → 约 $0.01
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import LLMClient, get_client, get_model


SC_TAGS = {"SC_ABSOLUTE_CAND", "SC_RELATIVE_CAND"}

# ── 疾病主干查询提取的 Prompt ─────────────────────────────────────────────────
DISEASE_QUERY_PROMPT = """\
You are helping construct a medical RAG benchmark.

Given this USMLE question, extract a "disease-only query" — a simplified question \
that asks about the standard treatment for the condition, WITHOUT mentioning:
- The patient's allergies or contraindications
- The patient's lab values (eGFR, creatinine, etc.)
- Any specific patient constraints

The disease-only query should represent what a general-population patient with \
this condition would receive as standard treatment, i.e., it approximates P(a|D) \
rather than P(a|D,q_specific).

Format: Return ONLY the query text (1-2 sentences). No explanation.

USMLE Question:
{question}

Disease-only query:"""

# ── 边缘分布推荐生成的 Prompt ────────────────────────────────────────────────
MARGINAL_RESPONSE_PROMPT = """\
Answer the following medical question concisely. \
Give only the most appropriate treatment recommendation in 1-2 sentences. \
Do not add caveats or ask for more information.

Question: {disease_only_query}

Answer:"""


def generate_prior_for_sample(
    client: LLMClient,
    sample: dict,
    dry_run: bool,
) -> dict:
    """
    为单个 SC 样本生成 parametric_prior 字段。

    参数：
        client:  LLMClient 实例
        sample:  候选样本 dict
        dry_run: True 时不调用 API，只打印将要发送的内容

    返回：
        更新后的 parametric_prior dict
    """
    question = sample.get("question", "")

    # 第一步：生成 disease_only_query
    dq_prompt = DISEASE_QUERY_PROMPT.format(question=question)
    if dry_run:
        print(f"\n[DRY-RUN] {sample['candidate_id']}")
        print(f"  prompt 前 200 字:\n  {dq_prompt[:200]}...")
        return {
            "disease_only_query": "[DRY-RUN]",
            "disease_only_response": None,
            "marginal_bias_confirmed": None,
        }

    disease_only_query = client.chat(
        messages=[{"role": "user", "content": dq_prompt}],
        max_tokens=2000,  # 推理模型的 think 链可达 1500+ token，需留足空间给最终答案
    )

    # 第二步：获取边缘分布推荐（无上下文，纯参数记忆）
    mr_prompt = MARGINAL_RESPONSE_PROMPT.format(disease_only_query=disease_only_query)
    disease_only_response = client.chat(
        messages=[{"role": "user", "content": mr_prompt}],
        max_tokens=1500,
    )

    # 第三步：marginal_bias_confirmed 留给人工核对（此处置 null）
    return {
        "disease_only_query": disease_only_query,
        "disease_only_response": disease_only_response,
        "marginal_bias_confirmed": None,
        "model_used": client.model,
    }


def update_annotation_csv(
    candidates_path: Path,
    csv_path: Path,
) -> None:
    """
    将生成好的 disease_only_query 同步回标注 CSV 的
    parametric_prior_disease_query 列。

    参数：
        candidates_path: 含 parametric_prior_stub 的 candidates jsonl
        csv_path:        annotations/macb_v2_sheet.csv
    """
    if not csv_path.exists():
        print(f"[跳过 CSV 同步] 未找到 {csv_path}", file=sys.stderr)
        return

    # 建立 candidate_id → disease_only_query 映射
    prior_map: dict = {}
    with candidates_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stub = row.get("parametric_prior_stub") or {}
            dq = stub.get("disease_only_query", "")
            if dq and not dq.startswith("[STUB]") and dq != "[DRY-RUN]":
                prior_map[row["candidate_id"]] = dq

    # CSV 的 sample_id 与 candidate_id 同格式（MACB-001 vs CAND-SC_ABS-0001 不同）
    # 需要通过 question 内容匹配，或通过 source_line 对应
    # 实际上 macb_v2_sheet.csv 的 sample_id 与 candidates 的 candidate_id 是一一对应的
    # 按顺序写入（两者均按相同顺序从同一源生成）
    with candidates_path.open("r", encoding="utf-8") as f:
        candidates_list = [json.loads(l) for l in f if l.strip()]

    # 建立 question → disease_only_query 的映射（question 是唯一键）
    question_to_dq: dict = {}
    for row in candidates_list:
        stub = row.get("parametric_prior_stub") or {}
        dq = stub.get("disease_only_query", "")
        if dq and not dq.startswith("[STUB]") and dq != "[DRY-RUN]":
            question_to_dq[row["question"]] = dq

    # 读取并更新 CSV
    csv_rows = []
    updated = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            q = row.get("question", "")
            if q in question_to_dq:
                row["parametric_prior_disease_query"] = question_to_dq[q]
                updated += 1
            csv_rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"CSV 同步完成 → {csv_path}（更新了 {updated} 行）")


def main() -> None:
    p = argparse.ArgumentParser(
        description="为 SC 候选样本生成 parametric_prior 字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--candidates",
        default="data/interim/macb_candidates_v2.jsonl",
        help="候选池 jsonl 路径（默认 data/interim/macb_candidates_v2.jsonl）",
    )
    p.add_argument(
        "--output",
        default="data/interim/macb_candidates_v2_with_prior.jsonl",
        help="写出路径（默认 data/interim/macb_candidates_v2_with_prior.jsonl）",
    )
    p.add_argument(
        "--annotation-csv",
        default="annotations/macb_v2_sheet.csv",
        help="标注 CSV 路径，生成后自动同步 parametric_prior_disease_query 列",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="不调用 API，仅打印将要发送的 prompt（用于检查质量）",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="每次 API 调用后的休眠秒数（避免速率限制），默认 0.3",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 个 SC 样本（用于测试）",
    )
    args = p.parse_args()

    # 读取候选池
    candidates_path = Path(args.candidates)
    candidates = []
    with candidates_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    # 初始化 LLM 客户端（dry-run 时跳过）
    client: LLMClient | None = None
    if not args.dry_run:
        client = get_client()

    # 筛选 SC 样本
    sc_samples = [s for s in candidates if s.get("candidate_tag") in SC_TAGS]
    if args.limit:
        sc_samples = sc_samples[: args.limit]

    print(f"候选池总数: {len(candidates)}")
    print(f"SC 样本数（待处理）: {len(sc_samples)}")
    if not args.dry_run:
        print(f"模型: {client.model}")
        print(f"预计 API 调用次数: {len(sc_samples) * 2}（每样本 2 次）")
    print()

    # 为每个 SC 样本生成 prior
    sc_id_set = {s["candidate_id"] for s in sc_samples}
    errors: list = []
    processed = 0

    # 在 candidates 列表上原地更新（非 SC 样本保持不变）
    for sample in candidates:
        if sample["candidate_id"] not in sc_id_set:
            continue
        cid = sample["candidate_id"]
        try:
            prior = generate_prior_for_sample(
                client=client,
                sample=sample,
                dry_run=args.dry_run,
            )
            sample["parametric_prior_stub"] = prior
            processed += 1
            if not args.dry_run:
                print(f"  [OK] {cid}: {prior['disease_only_query'][:60]}...")
                time.sleep(args.sleep)
        except Exception as e:
            errors.append(f"{cid}: {e}")
            print(f"  [ERROR] {cid}: {e}", file=sys.stderr)

    if errors:
        print(f"\n[警告] {len(errors)} 个样本处理失败：", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)

    # 写出更新后的 candidates jsonl
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in candidates:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if not args.dry_run:
        print(f"\nParametric prior 写出 → {out_path}")
        print(f"  成功: {processed}，失败: {len(errors)}")

        # 同步回标注 CSV
        csv_path = Path(args.annotation_csv)
        update_annotation_csv(out_path, csv_path)

        print()
        print("注意：marginal_bias_confirmed 字段需要人工核对后在 CSV 中填写。")
        print("  判断标准：disease_only_response 是否推荐了 INADMISSIBLE 的 action？")


if __name__ == "__main__":
    main()
