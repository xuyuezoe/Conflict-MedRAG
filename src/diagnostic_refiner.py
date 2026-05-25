#!/usr/bin/env python3
"""
Module 0A：DiagnosticRefiner — 临床诊断精化器

将完整的临床叙述（含症状、体征、实验室检查）精化为高特异性的诊断描述，
生成用于精准检索的查询字符串。

设计原理：
  LLM 的核心优势在于临床推理（symptom complex → specific diagnosis），
  而非结构化信息提取（structure extraction）。

  DiagnosticRefiner 将"诊断推理"这一 LLM 擅长的任务前置于检索，
  使 Stage 1A 的检索 query 从泛化的疾病名（如 "neonatal conjunctivitis treatment"）
  精化为特异性的诊断描述（如 "treatment for chlamydial neonatal conjunctivitis with pneumonia"）。

  这对应 *Diagnose First, Then Retrieve* 的设计原则（research.md §3.5.2 扩展）。

与 QueryDecomposer 的分工：
  QueryDecomposer：尝试同时提取 D_q 和 C_q（两件事，容易引入误差）
  DiagnosticRefiner：仅做"诊断推理"（一件事，LLM 专注度更高，输出质量更好）
  C_q 提取：完全由 ConstraintExpander（规则，确定性）完成

无答案泄露保证：
  DiagnosticRefiner 的输入中不包含答案选项（Options 部分被剥离），
  保证检索 query 仅来自临床推理，而非反向推断答案。

缓存策略（与 QueryDecomposer 一致）：
  cache key = MD5(clinical_text)，其中 clinical_text = 剥离选项后的 query。
  同一临床描述只调用 LLM 一次，结果磁盘持久化。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm_client import LLMClient


# ── 诊断精化 Prompt ───────────────────────────────────────────────────────────

DIAGNOSTIC_REFINER_SYSTEM = """\
You are an expert clinical diagnostician. Your ONLY task is to identify the \
most specific clinical diagnosis from the presented case.

CRITICAL RULES:
1. Do NOT suggest any treatments or medications
2. There are no answer options in the input — do not reference them
3. Do NOT extract patient constraints (allergies, pregnancy, lab thresholds)
4. ONLY analyze clinical findings: symptoms, signs, lab values, history, imaging
5. Output ONLY valid JSON — no markdown fences, no explanation text"""

DIAGNOSTIC_REFINER_PROMPT = """\
Analyze this clinical presentation and identify the MOST SPECIFIC diagnosis.

Reason through discriminating features:
- What symptoms, signs, or lab findings are present?
- What pathogens, subtypes, or clinical variants do these suggest?
- What single most specific diagnosis do they point to?

Clinical presentation:
{clinical_text}

Output ONLY valid JSON (no markdown):
{{
  "refined_diagnosis": "<specific diagnosis, e.g., 'chlamydial neonatal conjunctivitis with interstitial pneumonia'>",
  "discriminating_features": ["<feature 1>", "<feature 2>", "<feature 3>"],
  "retrieval_query": "<treatment for [specific diagnosis]>"
}}"""


# ── 诊断精化结果 ──────────────────────────────────────────────────────────────

@dataclass
class DiagnosticRefinement:
    """
    DiagnosticRefiner 的输出。

    参数：
        refined_diagnosis:       最特异性的诊断描述（如 "chlamydial neonatal conjunctivitis"）
        discriminating_features: 关键鉴别特征列表（用于调试和论文分析）
        retrieval_query:         用于 Stage 1A 的精化检索 query（如 "treatment for chlamydial..."）
        original_query:          原始临床描述（剥离选项后的输入，用于调试）
        debug:                   LLM 原始输出和中间状态
    """
    refined_diagnosis: str
    discriminating_features: List[str]
    retrieval_query: str
    original_query: str
    debug: Dict[str, Any] = field(default_factory=dict)


# ── DiagnosticRefiner 主类 ────────────────────────────────────────────────────

class DiagnosticRefiner:
    """
    Module 0A：临床诊断精化器。

    核心操作：
      query（含症状叙述）→ [剥离选项] → LLM 临床推理 → DiagnosticRefinement

    缓存：
      cache key = MD5(clinical_text)（与 QueryDecomposer 一致）
      同一临床描述只计算一次，磁盘持久化。

    无答案泄露：
      clinical_text = query.split("\\n\\nOptions: ")[0]（选项不参与 LLM 推理）
    """

    def __init__(
        self,
        client: LLMClient,
        model: str,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        参数：
            client:    LLM 客户端（统一 Anthropic + OpenAI 兼容接口）
            model:     诊断推理模型（建议使用推理能力强的模型，如 claude-sonnet-4-6）
            cache_dir: 缓存根目录（None 表示不缓存）
                       实际缓存写入 cache_dir/diagnostic_refiner/{cache_key}.json
        """
        self._client = client
        self._model = model
        self._cache_dir = cache_dir
        if cache_dir:
            (cache_dir / "diagnostic_refiner").mkdir(parents=True, exist_ok=True)

    def refine(self, query: str) -> DiagnosticRefinement:
        """
        精化临床描述，生成诊断特异性检索 query。

        参数：
            query: 原始输入（可能含 "\\n\\nOptions: ..." 后缀，会自动剥离）

        返回：
            DiagnosticRefinement（含精化诊断描述和检索 query）

        异常：
            ValueError: LLM 返回空响应或 JSON 格式非法（不兜底，让上层处理）
        """
        # 第一步：剥离选项部分（防止答案泄露进入诊断推理）
        clinical_text = query
        if "\n\nOptions: " in query:
            clinical_text = query.split("\n\nOptions: ", 1)[0]

        # 第二步：检查磁盘缓存
        cache_key = hashlib.md5(clinical_text.encode("utf-8")).hexdigest()
        if self._cache_dir:
            cache_path = self._cache_dir / "diagnostic_refiner" / f"{cache_key}.json"
            if cache_path.exists():
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return DiagnosticRefinement(
                    refined_diagnosis=data["refined_diagnosis"],
                    discriminating_features=data["discriminating_features"],
                    retrieval_query=data["retrieval_query"],
                    original_query=clinical_text,
                    debug={"source": "cache", "cache_key": cache_key},
                )

        # 第三步：调用 LLM 做临床推理
        prompt = DIAGNOSTIC_REFINER_PROMPT.format(clinical_text=clinical_text)
        raw_output = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            system=DIAGNOSTIC_REFINER_SYSTEM,
        )

        if not raw_output:
            raise ValueError(
                f"[DiagnosticRefiner] LLM 返回空响应（推理模型 token 耗尽）。\n"
                f"  查询（前 80 字）: {clinical_text[:80]}"
            )

        # 第四步：解析 JSON（失败直接抛出，不兜底）
        json_str = raw_output.strip()
        if "```" in json_str:
            start = json_str.index("```") + 3
            if json_str[start : start + 4].lower().startswith("json"):
                start += 4
            end = json_str.rindex("```")
            json_str = json_str[start:end].strip()

        # 兼容部分模型无 fence 格式（MiniMax 风格）
        if not json_str.startswith("{"):
            match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if match:
                json_str = match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[DiagnosticRefiner] JSON 解析失败。\n"
                f"  查询（前 80 字）: {clinical_text[:80]}\n"
                f"  原始输出（前 300 字）: {raw_output[:300]}\n"
                f"  解析错误: {e}"
            )

        refined_diagnosis: str = data.get("refined_diagnosis", "")
        discriminating_features: List[str] = data.get("discriminating_features", [])
        retrieval_query: str = data.get("retrieval_query", "") or clinical_text

        # 第五步：写入磁盘缓存
        if self._cache_dir:
            cache_data = {
                "refined_diagnosis":       refined_diagnosis,
                "discriminating_features": discriminating_features,
                "retrieval_query":         retrieval_query,
                "raw_output":              raw_output,
            }
            (self._cache_dir / "diagnostic_refiner" / f"{cache_key}.json").write_text(
                json.dumps(cache_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return DiagnosticRefinement(
            refined_diagnosis=refined_diagnosis,
            discriminating_features=discriminating_features,
            retrieval_query=retrieval_query,
            original_query=clinical_text,
            debug={"raw_output": raw_output[:500], "cache_key": cache_key},
        )
