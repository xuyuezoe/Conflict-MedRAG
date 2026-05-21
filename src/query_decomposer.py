#!/usr/bin/env python3
"""
Module 0：Query Decomposer

将自然语言 query 分解为 (D_q, C_q) 两个正交组件。

数学目标（research.md §3.5.1）：
  DCR 评分函数需要 D_q 和 C_q 独立计算：
    score(q, d) = sim(D_q, d) · κ(C_q, π_d)
  若将 C_q（患者约束）混入 D_q 的 embedding，约束信息在高维空间中被疾病语义稀释。
  因此必须将 query 分解为正交的两部分。

实现：
  使用 LLM（Haiku）进行结构化信息提取，返回 JSON 格式。
  缓存：同一 query 的分解结果用 MD5(query) 作为 key 缓存到磁盘，避免重复 API 调用。
  错误处理：JSON 解析失败直接抛出 ValueError（禁止兜底逻辑）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm_client import LLMClient

from src.types import PatientConstraint, QueryDecomposition


# ── Prompt 模板 ───────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM_PROMPT = """\
You are a medical information extractor. Your task is to decompose a medical \
clinical query into two orthogonal components:

1. disease_query: A simplified query describing only the disease/condition \
   and seeking standard treatment, WITHOUT any patient-specific constraints \
   (no allergies, no lab values, no contraindications).

2. constraints: A list of patient-specific constraints that may affect which \
   treatments are admissible for this particular patient.

Return ONLY valid JSON in this exact format:
{
  "disease_query": "standard treatment for <condition>",
  "constraints": [
    {
      "type": "ABSOLUTE",
      "target_action": "<drug or treatment name>",
      "text": "<exact text describing the constraint>",
      "parameter_value": null,
      "parameter_threshold": null
    }
  ]
}

Constraint types:
- ABSOLUTE: Hard contraindication (allergy, pregnancy contraindication, absolute ban)
  target_action: the specific drug/treatment that is contraindicated
  parameter_value and parameter_threshold: null

- RELATIVE: Soft constraint requiring dose adjustment (renal/hepatic impairment)
  target_action: the drug class or treatment requiring adjustment
  parameter_value: actual patient value (e.g., eGFR=28 → 28.0)
  parameter_threshold: the safety threshold (e.g., eGFR threshold=60 → 60.0)

- NONE: No constraint (patient characteristics mentioned but do not restrict treatment)
  Use NONE sparingly; only when a characteristic is explicitly non-constraining.

If no constraints exist, return "constraints": [].
Do not add constraints not mentioned in the query.
"""

DECOMPOSE_USER_TEMPLATE = """\
Medical query:
{query}

Decompose this into disease_query and constraints as specified."""


# ── 主类 ─────────────────────────────────────────────────────────────────────

class QueryDecomposer:
    """
    Module 0：自然语言医学 query 分解器。

    将含有疾病描述和患者约束的混合 query 分解为：
      D_q (disease_query): 纯疾病查询，用于 sim(D_q, d) 计算
      C_q (constraints):   患者约束结构化列表，用于 κ(C_q, π_d) 计算

    设计决策：
      - LLM 调用而非规则：医学 query 结构复杂，规则覆盖率有限
      - 缓存磁盘结果：避免评估阶段重复调用（同一 query 评估多个系统时）
      - 严格 JSON 验证：不接受格式错误输出（禁止兜底降级）
    """

    def __init__(
        self,
        client: LLMClient,
        model: str = "claude-haiku-4-5-20251001",
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        参数：
            client:    Anthropic API 客户端
            model:     分解模型 ID（默认 Haiku，成本低）
            cache_dir: 缓存目录（None 表示不缓存）
        """
        self._client = client
        self._model = model
        self._cache_dir = cache_dir
        if cache_dir:
            (cache_dir / "decompositions").mkdir(parents=True, exist_ok=True)

    def decompose(self, query: str) -> QueryDecomposition:
        """
        将自然语言 query 分解为 (D_q, C_q)。

        参数：
            query: 原始自然语言医学查询（含疾病描述+患者约束）

        返回：
            QueryDecomposition（含 disease_query 和 constraints 列表）

        异常：
            ValueError:  LLM 输出不符合预期 JSON 格式（不兜底，直接抛出）
            anthropic.APIError: API 调用失败
        """
        # 第一步：检查缓存
        cached = self._load_cache(query)
        if cached is not None:
            return cached

        # 第二步：调用 LLM 分解
        raw_output = self._call_llm(query)

        # 第三步：解析 JSON（失败则抛出，不兜底）
        decomposition = self._parse_output(query, raw_output)

        # 第四步：写入缓存
        self._save_cache(query, decomposition)

        return decomposition

    def _call_llm(self, query: str) -> str:
        """
        调用 LLM 执行分解，返回原始文本响应。
        """
        return self._client.chat(
            messages=[{
                "role": "user",
                "content": DECOMPOSE_USER_TEMPLATE.format(query=query),
            }],
            max_tokens=32000,
            system=DECOMPOSE_SYSTEM_PROMPT,
        )

    def _parse_output(self, original_query: str, raw_output: str) -> QueryDecomposition:
        """
        解析 LLM 输出的 JSON，构造 QueryDecomposition。

        参数：
            original_query: 原始查询（用于错误信息）
            raw_output:     LLM 原始文本输出

        异常：
            ValueError: JSON 格式非法或缺少必要字段
        """
        # 提取 JSON 块（LLM 有时会在 JSON 前后添加说明文字或代码块标记）
        if not raw_output:
            raise ValueError(
                f"[QueryDecomposer] LLM 返回空响应（推理模型 token 耗尽）。\n"
                f"  查询（前 100 字）: {original_query[:100]}"
            )

        json_str = raw_output
        if "```json" in raw_output:
            start = raw_output.index("```json") + 7
            end = raw_output.rindex("```")
            json_str = raw_output[start:end].strip()
        elif "```" in raw_output:
            start = raw_output.index("```") + 3
            end = raw_output.rindex("```")
            json_str = raw_output[start:end].strip()
        else:
            # 无代码块时，使用正则提取最外层 {} 对象（兼容 MiniMax 无 fence 格式）
            import re as _re
            m = _re.search(r"\{.*\}", raw_output, _re.DOTALL)
            if m:
                json_str = m.group()

        try:
            data: Dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[QueryDecomposer] LLM 输出 JSON 解析失败。\n"
                f"  查询（前 100 字）: {original_query[:100]}\n"
                f"  原始输出（前 200 字）: {raw_output[:200]}\n"
                f"  解析错误: {e}"
            )

        # 校验必要字段
        if "disease_query" not in data:
            raise ValueError(
                f"[QueryDecomposer] LLM 输出缺少 'disease_query' 字段。"
                f"  原始输出: {raw_output[:200]}"
            )
        if "constraints" not in data or not isinstance(data["constraints"], list):
            raise ValueError(
                f"[QueryDecomposer] LLM 输出缺少 'constraints' 字段或类型非数组。"
                f"  原始输出: {raw_output[:200]}"
            )

        # 解析约束列表
        constraints: List[PatientConstraint] = []
        for i, c in enumerate(data["constraints"]):
            ctype = c.get("type", "NONE").upper()
            if ctype not in {"ABSOLUTE", "RELATIVE", "NONE"}:
                raise ValueError(
                    f"[QueryDecomposer] 约束[{i}] type 值非法: {repr(ctype)}。"
                    f"仅接受 ABSOLUTE/RELATIVE/NONE。"
                )
            # NONE 约束不参与 κ 计算，直接跳过
            if ctype == "NONE":
                continue
            target = c.get("target_action", "")
            if not target:
                raise ValueError(
                    f"[QueryDecomposer] 约束[{i}] 缺少 target_action 字段。"
                    f"  约束数据: {c}"
                )

            param_val: Optional[float] = None
            param_thr: Optional[float] = None
            if ctype == "RELATIVE":
                raw_val = c.get("parameter_value")
                raw_thr = c.get("parameter_threshold")
                if raw_val is None or raw_thr is None:
                    raise ValueError(
                        f"[QueryDecomposer] RELATIVE 约束[{i}] 缺少 parameter_value "
                        f"或 parameter_threshold。约束: {c}"
                    )
                try:
                    param_val = float(raw_val)
                    param_thr = float(raw_thr)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"[QueryDecomposer] RELATIVE 约束[{i}] 参数值类型错误: {e}"
                    )

            constraints.append(PatientConstraint(
                constraint_type=ctype,  # type: ignore[arg-type]
                target_action=target.strip().lower(),
                raw_text=c.get("text", "").strip(),
                parameter_value=param_val,
                parameter_threshold=param_thr,
            ))

        return QueryDecomposition(
            original_query=original_query,
            disease_query=data["disease_query"].strip(),
            constraints=constraints,
            decompose_model=self._model,
            debug={"raw_llm_output": raw_output},
        )

    def _cache_key(self, query: str) -> str:
        """生成 query 的 MD5 缓存键"""
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def _load_cache(self, query: str) -> Optional[QueryDecomposition]:
        """
        从磁盘加载缓存的分解结果。
        缓存未命中时返回 None（不抛出异常）。
        """
        if self._cache_dir is None:
            return None
        cache_file = self._cache_dir / "decompositions" / f"{self._cache_key(query)}.json"
        if not cache_file.exists():
            return None

        data = json.loads(cache_file.read_text(encoding="utf-8"))
        constraints = [
            PatientConstraint(
                constraint_type=c["constraint_type"],
                target_action=c["target_action"],
                raw_text=c["raw_text"],
                parameter_value=c.get("parameter_value"),
                parameter_threshold=c.get("parameter_threshold"),
            )
            for c in data["constraints"]
        ]
        return QueryDecomposition(
            original_query=data["original_query"],
            disease_query=data["disease_query"],
            constraints=constraints,
            decompose_model=data["decompose_model"],
            debug=data.get("debug", {}),
        )

    def _save_cache(self, query: str, decomp: QueryDecomposition) -> None:
        """将分解结果写入磁盘缓存"""
        if self._cache_dir is None:
            return
        cache_file = self._cache_dir / "decompositions" / f"{self._cache_key(query)}.json"
        data = {
            "original_query": decomp.original_query,
            "disease_query": decomp.disease_query,
            "constraints": [
                {
                    "constraint_type": c.constraint_type,
                    "target_action": c.target_action,
                    "raw_text": c.raw_text,
                    "parameter_value": c.parameter_value,
                    "parameter_threshold": c.parameter_threshold,
                }
                for c in decomp.constraints
            ],
            "decompose_model": decomp.decompose_model,
            "debug": decomp.debug,
        }
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
