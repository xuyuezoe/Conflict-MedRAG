#!/usr/bin/env python3
"""
MACB Benchmark 一次性修复脚本。

操作：
  1. 移出 8 个不合格样本 → data/macb_excluded.jsonl
  2. 自动修复：拼写、嵌套 dict、键格式、admissible_set 一致性、scsr_needed 逻辑
  3. 重标注 7 个格式完全错误的样本（MACB-004/058/059/085/122/124/222）
  4. 补全 5 个 answer_idx
  5. 生成缺失 parametric_prior_disease_query
  6. 新增 gold_conflict_type 派生字段
  7. 写出 data/macb_treatment_v2.jsonl（覆盖）
"""
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 配置 ───────────────────────────────────────────────────────────────────────
BENCHMARK_PATH = Path("data/macb_treatment_v2.jsonl")
EXCLUDED_PATH  = Path("data/macb_excluded.jsonl")
CSV_PATH       = Path("annotations/macb_v3_sheet.csv")

MODEL    = os.environ["LLM_MODEL"]
API_KEY  = os.environ["LLM_API_KEY"]
BASE_URL = os.environ["LLM_BASE_URL"]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

VALID_LETTERS   = {"A", "B", "C", "D", "E"}
VALID_STATUSES  = {"ADMISSIBLE", "INADMISSIBLE_ABS", "INADMISSIBLE_REL", "NOT_APPLICABLE"}

# 移出主集的样本（诊断题 / 药物警戒 / 逻辑自相矛盾）
EXCLUDED_IDS = {
    "MACB-002",  # 药物警戒诊断题
    "MACB-016", "MACB-078", "MACB-083", "MACB-092",
    "MACB-118", "MACB-144", "MACB-154",  # NOT_APPLICABLE 诊断题
}

# answer_idx 从 answer 字段读取（单字母的情况）
MISSING_ANSWER_IDX = {
    "MACB-013": "D",
    "MACB-014": "D",
    "MACB-018": "D",
    "MACB-069": "C",
    "MACB-150": "D",
}


# ── LLM 辅助函数 ───────────────────────────────────────────────────────────────

def _llm(prompt: str) -> str:
    """调用 MiniMax，剥离 think chain，返回文本。"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        except Exception as exc:
            if attempt < 2:
                time.sleep(2)
            else:
                raise


def _extract_json(text: str) -> Dict[str, Any]:
    """从文本中提取 JSON 对象。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"未找到 JSON: {text[:200]}")
    return json.loads(m.group())


# ── 格式修复函数 ───────────────────────────────────────────────────────────────

def _fix_spelling(v: str) -> str:
    """修正 INADMISIBLE / INADMISIBILE 拼写错误。"""
    return re.sub(r"INADMISI+BLE?_ABS", "INADMISSIBLE_ABS", v)


def _extract_letter_from_key(key: str) -> Optional[str]:
    """
    从非标准键中提取选项字母。
    支持: "A: Drug", "A (Drug)", "A_Drug", "A Drug text", "A"
    """
    m = re.match(r"^([A-Ea-e])[^A-Za-z]", key)
    if m:
        return m.group(1).upper()
    if key.upper() in VALID_LETTERS:
        return key.upper()
    return None


def _normalize_per_action(pas_raw: Any, options_text: str) -> Optional[Dict[str, str]]:
    """
    将各种格式的 per_action_status 规范化为 {A: status, B: status, ...}。
    返回 None 表示无法自动修复，需要重标注。
    """
    if isinstance(pas_raw, str):
        return None  # 整个字段是字符串，无法自动修复

    if not isinstance(pas_raw, dict):
        return None

    result: Dict[str, str] = {}

    # 检查键格式
    all_standard = all(k in VALID_LETTERS for k in pas_raw.keys())

    if all_standard:
        # 键已标准，只修复值
        for k, v in pas_raw.items():
            if isinstance(v, dict):
                # 嵌套 dict：提取 .status 字段
                status = v.get("status") or v.get("Status") or v.get("admissibility")
                if status and isinstance(status, str):
                    result[k] = _fix_spelling(status)
                else:
                    return None  # 嵌套 dict 中找不到 status，需要重标注
            elif isinstance(v, str):
                result[k] = _fix_spelling(v)
            else:
                return None
    else:
        # 键非标准，尝试提取字母
        for k, v in pas_raw.items():
            letter = _extract_letter_from_key(k)
            if letter is None:
                # 尝试从 options_text 匹配药品名
                letter = _match_drug_to_letter(k, options_text)
            if letter is None:
                return None  # 无法映射，需要重标注
            status: Any = v
            if isinstance(status, dict):
                status = status.get("status") or status.get("admissibility")
            if not isinstance(status, str):
                return None
            result[letter] = _fix_spelling(status)

    # 验证所有值合法
    for v in result.values():
        if v not in VALID_STATUSES:
            return None

    return result if result else None


def _match_drug_to_letter(drug_name: str, options_text: str) -> Optional[str]:
    """
    将药品名映射到选项字母（用于 MACB-085/222 类型）。
    options_text 格式: "A: Drug1 | B: Drug2 | ..."
    """
    for part in options_text.split("|"):
        part = part.strip()
        m = re.match(r"^([A-E]):\s*(.+)$", part)
        if m:
            letter = m.group(1)
            opt_drug = m.group(2).strip().lower()
            query = drug_name.strip().lower()
            if query in opt_drug or opt_drug in query or opt_drug.startswith(query[:10]):
                return letter
    return None


def _compute_admissible_set(pas: Dict[str, str]) -> List[str]:
    """从 per_action_status 自动计算 admissible_set。"""
    return sorted(
        k for k, v in pas.items()
        if v in {"ADMISSIBLE", "CONDITIONALLY_ADMISSIBLE"}
    )


def _derive_conflict_type(tag: str, pas: Dict[str, str], label: str) -> str:
    """根据 candidate_tag 和 per_action_status 派生 gold_conflict_type。"""
    if label == "NOT_APPLICABLE":
        return "NOT_APPLICABLE"
    if tag == "SC_ABSOLUTE_CAND":
        return "SC_ABSOLUTE"
    if tag == "SC_RELATIVE_CAND":
        return "SC_RELATIVE"
    if tag == "FC_CAND":
        return "FC"
    return "NO_CONFLICT"


# ── LLM 重标注（针对格式完全损坏的样本） ──────────────────────────────────────

REANNOTATE_PROMPT = """\
You are a medical annotation expert. Re-annotate this USMLE treatment question for the MACB benchmark.

Question: {question}
Options: {options}
Correct USMLE Answer: {answer_idx}

Candidate Tag: {candidate_tag}
Patient Profile: {patient_profile}

For each option A-E, determine:
- ADMISSIBLE: safe and appropriate for this patient
- INADMISSIBLE_ABS: absolutely contraindicated (allergy, absolute CI, wrong drug class for disease)
- INADMISSIBLE_REL: relatively contraindicated (dose adjustment needed)
- NOT_APPLICABLE: not a treatment option (diagnostic test, irrelevant)

RULE: The correct USMLE answer ({answer_idx}) MUST be ADMISSIBLE.

Return ONLY valid JSON (no markdown):
{{
  "per_action_status": {{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}},
  "gold_admissible_set": ["..."],
  "gold_scsr_needed": true or false,
  "gold_memory_conflict_label": "CONFLICT" or "NO_CONFLICT",
  "notes": "one sentence"
}}"""


def _reannotate(s: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """LLM 重标注单个样本，返回修复后的字段（失败返回 None）。"""
    prompt = REANNOTATE_PROMPT.format(
        question=s.get("query", "")[:500],
        options=s.get("options_text", ""),
        answer_idx=s.get("answer_idx", "?"),
        candidate_tag=s.get("candidate_tag", ""),
        patient_profile=json.dumps(s.get("patient_profile", {}))[:300],
    )
    raw = _llm(prompt)
    result = _extract_json(raw)

    # 安全保障：answer_idx 必须 ADMISSIBLE
    ans = s.get("answer_idx", "").strip()
    pas = result.get("per_action_status", {})
    if ans:
        if pas.get(ans) != "ADMISSIBLE":
            pas[ans] = "ADMISSIBLE"
            result["per_action_status"] = pas
        admissible = result.get("gold_admissible_set", [])
        if ans not in admissible:
            admissible.append(ans)
            result["gold_admissible_set"] = admissible

    return result


# ── 生成 parametric_prior ─────────────────────────────────────────────────────

PRIOR_PROMPT = """\
Generate a concise medical query (one sentence) for retrieval of evidence about the \
standard first-line treatment for the disease/condition described.
Remove any patient-specific constraints (allergies, organ impairment, etc.).

Question: {question}
Options: {options}

Return ONLY the query string."""


def _generate_prior(s: Dict[str, Any]) -> str:
    """生成 parametric_prior_disease_query。"""
    raw = _llm(PRIOR_PROMPT.format(
        question=s.get("query", "")[:300],
        options=s.get("options_text", ""),
    ))
    lines = [l.strip() for l in raw.split("\n") if l.strip() and not l.strip().startswith("<")]
    return lines[0][:200] if lines else ""


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"使用模型: {MODEL}")

    # 读取 benchmark
    with BENCHMARK_PATH.open(encoding="utf-8") as f:
        samples: List[Dict[str, Any]] = [json.loads(l) for l in f if l.strip()]
    print(f"读入 {len(samples)} 个样本")

    excluded: List[Dict[str, Any]] = []
    main_samples: List[Dict[str, Any]] = []

    # ── 第一步：移出不合格样本 ───────────────────────────────────────
    for s in samples:
        if s["sample_id"] in EXCLUDED_IDS or s.get("gold_memory_conflict_label") == "NOT_APPLICABLE":
            excluded.append(s)
        else:
            main_samples.append(s)

    print(f"移出 {len(excluded)} 个样本 → {EXCLUDED_PATH}")
    with EXCLUDED_PATH.open("w", encoding="utf-8") as f:
        for s in excluded:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ── 第二步：逐样本修复 ───────────────────────────────────────────
    needs_reannotate: List[Dict[str, Any]] = []
    needs_prior: List[Dict[str, Any]] = []

    for s in main_samples:
        sid = s["sample_id"]
        changed = False

        # 2a. 补全 answer_idx
        if not s.get("answer_idx", "").strip():
            if sid in MISSING_ANSWER_IDX:
                s["answer_idx"] = MISSING_ANSWER_IDX[sid]
                print(f"  [{sid}] answer_idx 补全 → {s['answer_idx']}")
                changed = True

        # 2b. 规范化 per_action_status
        pas_raw = s.get("gold_per_action_status")
        fixed_pas = _normalize_per_action(pas_raw, s.get("options_text", ""))

        if fixed_pas is None:
            needs_reannotate.append(s)
            continue  # 跳过后续，等重标注

        if fixed_pas != pas_raw:
            s["gold_per_action_status"] = fixed_pas
            print(f"  [{sid}] per_action_status 规范化")
            changed = True

        # 2c. 修复 gold_admissible_set 一致性
        expected_admissible = _compute_admissible_set(fixed_pas)
        if sorted(s.get("gold_admissible_set") or []) != expected_admissible:
            s["gold_admissible_set"] = expected_admissible
            print(f"  [{sid}] admissible_set 修正 → {expected_admissible}")
            changed = True

        # 2d. 修复 gold_scsr_needed 逻辑
        has_inadmissible = any(
            v in {"INADMISSIBLE_ABS", "INADMISSIBLE_REL"}
            for v in fixed_pas.values()
        )
        label = s.get("gold_memory_conflict_label", "")
        if s.get("gold_scsr_needed") and not has_inadmissible and label == "NO_CONFLICT":
            s["gold_scsr_needed"] = False
            print(f"  [{sid}] gold_scsr_needed → False（无 inadmissible 选项）")
            changed = True

        # 2e. 标记缺失 prior
        if not s.get("parametric_prior_disease_query", "").strip():
            needs_prior.append(s)

    # ── 第三步：LLM 重标注 ──────────────────────────────────────────
    print(f"\n需重标注: {len(needs_reannotate)} 个")
    reannotate_failures: List[str] = []
    for s in needs_reannotate:
        sid = s["sample_id"]
        print(f"  [{sid}] 重标注...", end=" ", flush=True)
        try:
            result = _reannotate(s)
            if result:
                s["gold_per_action_status"] = result["per_action_status"]
                # 重新计算 admissible_set（保证一致性）
                s["gold_admissible_set"] = _compute_admissible_set(result["per_action_status"])
                s["gold_scsr_needed"] = result.get("gold_scsr_needed", False)
                s["gold_memory_conflict_label"] = result.get("gold_memory_conflict_label", s.get("gold_memory_conflict_label", "NO_CONFLICT"))
                if result.get("notes"):
                    s["notes"] = result["notes"]
                print(f"OK ({s['gold_memory_conflict_label']})")
            else:
                print("FAILED")
                reannotate_failures.append(sid)
            # 标记缺失 prior
            if not s.get("parametric_prior_disease_query", "").strip():
                needs_prior.append(s)
        except Exception as e:
            print(f"FAILED: {e}")
            reannotate_failures.append(sid)
        time.sleep(0.8)

    main_samples.extend(needs_reannotate)  # 将重标注样本重新加入（含失败的）

    # ── 第四步：生成 parametric_prior ───────────────────────────────
    print(f"\n需生成 prior: {len(needs_prior)} 个")
    for s in needs_prior:
        if s["sample_id"] in reannotate_failures:
            continue
        sid = s["sample_id"]
        print(f"  [{sid}] 生成 prior...", end=" ", flush=True)
        try:
            prior = _generate_prior(s)
            if prior:
                s["parametric_prior_disease_query"] = prior
                print(f"OK: {prior[:80]}")
            else:
                print("空结果")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(0.5)

    # ── 第五步：新增 gold_conflict_type 字段 ───────────────────────
    for s in main_samples:
        pas = s.get("gold_per_action_status", {})
        if isinstance(pas, dict):
            s["gold_conflict_type"] = _derive_conflict_type(
                s.get("candidate_tag", ""),
                pas,
                s.get("gold_memory_conflict_label", ""),
            )

    # ── 第六步：按 sample_id 排序写出 ───────────────────────────────
    def sort_key(s: Dict[str, Any]) -> int:
        try:
            return int(s["sample_id"].replace("MACB-", ""))
        except (ValueError, KeyError):
            return 9999

    main_samples.sort(key=sort_key)

    with BENCHMARK_PATH.open("w", encoding="utf-8") as f:
        for s in main_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    labels = Counter(s["gold_memory_conflict_label"] for s in main_samples)
    tags = Counter(s["candidate_tag"] for s in main_samples)
    print(f"\n写出 {len(main_samples)} 个样本 → {BENCHMARK_PATH}")
    print(f"  conflict_label: {dict(labels)}")
    print(f"  candidate_tag: {dict(tags)}")
    if reannotate_failures:
        print(f"  重标注失败（需人工处理）: {reannotate_failures}")


if __name__ == "__main__":
    main()
