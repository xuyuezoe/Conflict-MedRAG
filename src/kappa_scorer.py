#!/usr/bin/env python3
"""
κ-Scorer：适用范围相容性计算器

计算 κ(C_q, π_d) = ∏_i κ_single(c_i, π_d)

数学基础（research.md §3.5.2）：
  单约束：
    κ_single(c_i, π_d) = 0          若 c_i 是针对 π_d 推荐 action 的绝对禁忌
    κ_single(c_i, π_d) = f(δ)       若 c_i 是相对禁忌（f(δ) = 1 - max(0,thr-val)/thr）
    κ_single(c_i, π_d) = 1          否则（不冲突）
  多约束组合：乘积形式（任意 κ_single=0 → 整体 κ=0）

实现分两层：
  层 1（规则层，优先）：基于药物-禁忌对的确定性规则，O(1)，无 API 成本
  层 2（LLM 层，fallback）：从文档文本提取 π_d，仅规则层无法确定时触发

π_d 提取结果缓存到磁盘，同一 chunk 不重复提取。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.llm_client import LLMClient

from src.types import PatientConstraint, QueryDecomposition, RetrievedDoc, ScopePredicate, TextChunk
from src.retriever import RetrievalResult


# ── 规则库：绝对禁忌对 ────────────────────────────────────────────────────────

# 格式：(患者约束 target_action 关键词, 文档推荐 action 关键词)
# target_action 字符串需与 PatientConstraint.target_action（小写）匹配
# 文档 action 关键词用于匹配 ScopePredicate.recommended_action 或 TextChunk.text

ABSOLUTE_CONTRAINDICATION_RULES: List[Tuple[str, str]] = [
    # 青霉素类过敏 → 青霉素族所有药物
    ("penicillin",     "amoxicillin"),
    ("penicillin",     "amoxicillin-clavulanate"),
    ("penicillin",     "ampicillin"),
    ("penicillin",     "ampicillin-sulbactam"),
    ("penicillin",     "piperacillin"),
    ("penicillin",     "piperacillin-tazobactam"),
    ("penicillin",     "nafcillin"),
    ("penicillin",     "oxacillin"),
    ("penicillin",     "dicloxacillin"),
    # 磺胺类过敏
    ("sulfa",          "trimethoprim-sulfamethoxazole"),
    ("sulfa",          "sulfamethoxazole"),
    ("sulfonamide",    "trimethoprim-sulfamethoxazole"),
    # 阿司匹林/NSAIDs 过敏
    ("aspirin",        "aspirin"),
    ("aspirin",        "ibuprofen"),
    # 妊娠禁忌
    ("pregnancy",      "warfarin"),
    ("pregnant",       "warfarin"),
    ("pregnancy",      "isotretinoin"),
    ("pregnant",       "isotretinoin"),
    ("pregnancy",      "thalidomide"),
    ("pregnant",       "thalidomide"),
    ("pregnancy",      "methotrexate"),
    ("pregnant",       "methotrexate"),
    ("pregnancy",      "tetracycline"),
    ("pregnant",       "tetracycline"),
    ("pregnancy",      "doxycycline"),
    ("pregnant",       "doxycycline"),
    ("pregnancy",      "fluoroquinolone"),
    ("pregnant",       "fluoroquinolone"),
    ("pregnancy",      "ciprofloxacin"),
    ("pregnant",       "ciprofloxacin"),
    ("pregnancy",      "levofloxacin"),
    ("pregnant",       "levofloxacin"),
    # HIT（肝素诱导血小板减少）
    ("heparin-induced thrombocytopenia", "heparin"),
    ("hit",            "heparin"),
    # 哺乳期
    ("breastfeeding",  "tetracycline"),
    ("lactation",      "tetracycline"),
    ("breastfeeding",  "doxycycline"),
    ("lactation",      "doxycycline"),
    # 儿童阿司匹林（Reye 综合征）
    ("reye",           "aspirin"),
    ("child",          "aspirin"),      # 仅在有 Reye 综合征上下文时（粗略匹配）
]

# 格式：(患者约束 target_action 关键词, 文档推荐 action 关键词)
# RELATIVE 规则：存在时不返回 κ=0，而是触发 f(δ) 计算
RELATIVE_RESTRICTION_RULES: List[Tuple[str, str]] = [
    ("renal", "metformin"),
    ("kidney", "metformin"),
    ("egfr",   "metformin"),
    ("renal",  "methotrexate"),
    ("renal",  "gabapentin"),
    ("renal",  "pregabalin"),
    ("renal",  "allopurinol"),
    ("liver",  "acetaminophen"),
    ("hepatic","acetaminophen"),
]


# ── π_d 提取 Prompt ───────────────────────────────────────────────────────────

SCOPE_EXTRACTION_PROMPT = """\
Extract the scope predicate from this medical text passage.
Identify what treatment is being recommended and who it is recommended for.

Return ONLY valid JSON:
{{
  "recommended_action": "<primary drug or treatment mentioned>",
  "population": "<who this treatment is for, e.g., 'adults with ABRS'>",
  "contraindications": ["<explicit contraindication 1>", "<explicit contraindication 2>"],
  "relative_restrictions": [
    {{
      "condition": "<e.g., 'renal impairment'>",
      "threshold": "<e.g., 'eGFR < 30 mL/min'>"
    }}
  ]
}}

If no clear treatment is recommended, use recommended_action: "none".
If no explicit contraindications are mentioned, use contraindications: [].

Medical text:
{text}
"""


# ── κ-Scorer 主类 ─────────────────────────────────────────────────────────────

class KappaScorer:
    """
    适用范围相容性计算器。

    计算 κ(C_q, π_d) = ∏_i κ_single(c_i, π_d)

    关键数学性质（research.md §3.5.2）：
      - 任意 κ_single=0 → 整体 κ=0（乘积形式，不可补偿）
      - κ 值域 [0,1]；κ=0 是绝对排除的硬边界
      - κ 的计算与 sim(D_q,d) 完全独立（两因子正交）

    实现优先级：
      1. 规则层（O(1)，无 API 成本）
      2. LLM π_d 提取（仅规则层无法确定时）
      3. 缓存 π_d（同一 chunk_id 只提取一次）
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        model: str = "claude-haiku-4-5-20251001",
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        参数：
            client:    Anthropic 客户端（None 表示只用规则层，不调用 LLM）
            model:     π_d 提取模型
            cache_dir: π_d 缓存目录（None 表示不缓存）
        """
        self._client = client
        self._model = model
        self._cache_dir = cache_dir
        if cache_dir:
            (cache_dir / "scope_predicates").mkdir(parents=True, exist_ok=True)

        # 构建规则查找集合（加速匹配）
        self._abs_rule_set: Set[Tuple[str, str]] = set(ABSOLUTE_CONTRAINDICATION_RULES)
        self._rel_rule_set: Set[Tuple[str, str]] = set(RELATIVE_RESTRICTION_RULES)

    def score_documents(
        self,
        retrieval_results: List[RetrievalResult],
        decomposition: QueryDecomposition,
    ) -> List[RetrievedDoc]:
        """
        对 Stage 1 检索结果批量计算 κ，生成 RetrievedDoc 列表。

        优化：对缓存未命中的文档执行批量 π_d 提取（N 个文档一次 LLM 调用，
        而非 N 次），大幅降低推理模型首次运行的 API 调用次数。

        参数：
            retrieval_results: Stage 1 返回的检索结果（含 sim_score）
            decomposition:     Module 0 的分解结果（含 C_q 约束列表）

        返回：
            RetrievedDoc 列表（含 sim_score、kappa、dcr_score、scope_status）
            按 dcr_score 降序排列
        """
        from src.types import RetrievedDoc

        # 第一步：规则层快速筛选 + 收集 cache-miss 文档
        needs_llm: List[RetrievalResult] = []
        if decomposition.constraints:
            for rr in retrieval_results:
                rule_result = self._rule_layer_check(rr.chunk.text, decomposition.constraints)
                if rule_result is None:
                    # 规则层未能判断，需要 LLM
                    cache_path = (
                        self._cache_dir / "scope_predicates" / f"{rr.chunk.chunk_id}.json"
                        if self._cache_dir else None
                    )
                    if cache_path is None or not cache_path.exists():
                        needs_llm.append(rr)

        # 第二步：批量提取所有 cache-miss 文档的 π_d（一次 LLM 调用）
        if needs_llm and self._client is not None:
            self._batch_extract_and_cache(needs_llm)

        # 第三步：逐文档计算 κ（此时 cache 已预热，LLM 层读缓存）
        results: List[RetrievedDoc] = []
        for rr in retrieval_results:
            kappa, scope_status, scope_pred = self.compute_kappa(
                chunk=rr.chunk,
                constraints=decomposition.constraints,
            )
            dcr_score = rr.score * kappa
            results.append(RetrievedDoc(
                chunk=rr.chunk,
                sim_score=rr.score,
                kappa=kappa,
                dcr_score=dcr_score,
                scope_predicate=scope_pred,
                scope_status=scope_status,
            ))

        results.sort(key=lambda d: -d.dcr_score)
        return results

    _BATCH_SIZE: int = 10   # 每次 LLM 调用处理的文档数（推理模型下 10 个约 4000 tokens）

    def _batch_extract_and_cache(self, retrieval_results: List[RetrievalResult]) -> None:
        """
        对多个 cache-miss 文档批量提取 π_d，写入磁盘缓存。

        策略：将文档切分为 _BATCH_SIZE 大小的批次，每批一次 LLM 调用，
        相比逐文档提取将调用次数从 N 降至 ceil(N/_BATCH_SIZE)。

        参数：
            retrieval_results: 需要 LLM 提取的文档列表（均为 cache-miss）
        """
        if not self._cache_dir:
            return
        cache_dir = self._cache_dir / "scope_predicates"
        cache_dir.mkdir(parents=True, exist_ok=True)

        batch_prompt_template = """\
Extract scope predicates from the following {n} medical text passages.
For each passage, identify the treatment recommended and any contraindications.

{passages}

Return a JSON array with exactly {n} objects in order:
[
  {{
    "chunk_id": "<id from above>",
    "recommended_action": "<primary drug or treatment, or 'none'>",
    "population": "<who this treatment is for>",
    "contraindications": ["<contraindication 1>", "..."],
    "relative_restrictions": [{{"condition": "...", "threshold": "..."}}]
  }},
  ...
]
Return ONLY the JSON array, no other text."""

        for i in range(0, len(retrieval_results), self._BATCH_SIZE):
            batch = retrieval_results[i: i + self._BATCH_SIZE]

            # 构造批量 prompt（每个文档截取前 400 字符，防止超出上下文）
            passages_parts = []
            for j, rr in enumerate(batch):
                passages_parts.append(
                    f"Passage {j+1} (chunk_id: {rr.chunk.chunk_id}):\n{rr.chunk.text[:400]}"
                )
            passages_text = "\n\n".join(passages_parts)

            prompt = batch_prompt_template.format(
                n=len(batch),
                passages=passages_text,
            )

            raw = self._client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32000,
            )

            # 解析并缓存每个 π_d
            try:
                json_str = raw
                if "```" in raw:
                    start = raw.index("```") + 3
                    if "json" in raw[start: start + 4]:
                        start += 4
                    end = raw.rindex("```")
                    json_str = raw[start:end].strip()
                items: List[dict] = json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                # 批量解析失败：退回逐文档模式
                for rr in batch:
                    try:
                        pred = self._extract_scope_predicate(rr.chunk)
                        data = {
                            "chunk_id": pred.chunk_id,
                            "recommended_action": pred.recommended_action,
                            "population": pred.population,
                            "contraindications": pred.contraindications,
                            "relative_restrictions": pred.relative_restrictions,
                            "extraction_model": pred.extraction_model,
                            "raw_output": pred.raw_output,
                        }
                        (cache_dir / f"{pred.chunk_id}.json").write_text(
                            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                    except Exception:
                        pass
                continue

            # 将批量结果写入缓存
            for item in items:
                cid = item.get("chunk_id", "")
                if not cid:
                    continue
                data = {
                    "chunk_id": cid,
                    "recommended_action": item.get("recommended_action", "none"),
                    "population": item.get("population", ""),
                    "contraindications": item.get("contraindications", []),
                    "relative_restrictions": item.get("relative_restrictions", []),
                    "extraction_model": self._model,
                    "raw_output": raw,
                }
                (cache_dir / f"{cid}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    def compute_kappa(
        self,
        chunk: TextChunk,
        constraints: List[PatientConstraint],
    ) -> Tuple[float, str, Optional[ScopePredicate]]:
        """
        计算单文档的 κ 值。

        参数：
            chunk:       文本块
            constraints: 患者约束列表 C_q

        返回：
            (kappa, scope_status, scope_predicate)
            kappa:            [0,1] 综合值
            scope_status:     ADMISSIBLE / INADMISSIBLE_ABS / INADMISSIBLE_REL / UNKNOWN
            scope_predicate:  π_d（LLM 提取的结果，可为 None）
        """
        if not constraints:
            return 1.0, "ADMISSIBLE", None

        # 第一步：规则层快速判断
        rule_result = self._rule_layer_check(chunk.text, constraints)
        if rule_result is not None:
            kappa, scope_status = rule_result
            return kappa, scope_status, None

        # 第二步：LLM 层提取 π_d（仅当客户端可用）
        if self._client is None:
            return 1.0, "UNKNOWN", None

        scope_pred = self._get_or_extract_scope_predicate(chunk)
        kappa, scope_status = self._predicate_layer_check(scope_pred, constraints)
        return kappa, scope_status, scope_pred

    # 文档文本中"负面上下文"的模式（出现时表示文档是在说某药物不适用，而非推荐）
    _NEGATIVE_CONTEXT_PATTERNS: List[str] = [
        r"cannot\s+tolerate",
        r"alternative\s+(?:to|for)",
        r"instead\s+of",
        r"contraindicated",
        r"avoid(?:ance)?\s+of",
        r"not\s+(?:be\s+)?(?:used|given|recommended)",
        r"allergic\s+to",
        r"allerg[yi]",
        r"hypersensitiv",
        r"in\s+patients\s+who\s+(?:cannot|are\s+unable)",
        r"penicillin.{0,20}allergic",
    ]

    def _is_positive_recommendation(self, doc_text_lower: str, drug_name: str) -> bool:
        """
        判断文档是否在正面推荐 drug_name（而非仅提及或声明其禁忌）。

        策略：
          1. 定位 drug_name 在文本中的所有出现位置
          2. 检查每处出现的上下文窗口（±150 字符）
          3. 若窗口内出现负面上下文模式，视为"提及而非推荐"
          4. 若所有出现位置均有负面上下文，返回 False（不推荐）
          5. 至少一处无负面上下文 → 返回 True（可能在推荐）
        """
        positions = [m.start() for m in re.finditer(r"\b" + re.escape(drug_name) + r"\b", doc_text_lower)]
        if not positions:
            return False

        for pos in positions:
            window_start = max(0, pos - 150)
            window_end = min(len(doc_text_lower), pos + len(drug_name) + 150)
            window = doc_text_lower[window_start:window_end]

            # 检查窗口内是否有负面上下文
            has_negative = any(
                re.search(pat, window)
                for pat in self._NEGATIVE_CONTEXT_PATTERNS
            )
            if not has_negative:
                # 此处出现 drug_name 没有负面上下文，视为正面推荐
                return True

        # 所有出现位置均有负面上下文，视为"提及但不推荐"
        return False

    def _rule_layer_check(
        self,
        doc_text: str,
        constraints: List[PatientConstraint],
    ) -> Optional[Tuple[float, str]]:
        """
        规则层：基于 ABSOLUTE_CONTRAINDICATION_RULES 快速判断。

        规则触发条件（两个条件同时满足）：
          1. 患者约束 target 与规则约束匹配
          2. 文档在正面推荐（非仅提及）规则中的 action

        引入 _is_positive_recommendation() 避免误判：
          "doxycycline is an alternative for patients who cannot tolerate amoxicillin"
          → amoxicillin 出现，但上下文为负面 → 不触发规则 → κ 不为 0
        """
        doc_text_lower = doc_text.lower()

        for constraint in constraints:
            c_target = constraint.target_action.lower()

            if constraint.constraint_type == "ABSOLUTE":
                # 检查是否命中任何绝对禁忌规则
                for (rule_constraint, rule_action) in self._abs_rule_set:
                    if rule_constraint in c_target or c_target in rule_constraint:
                        # 仅在文档正面推荐该 action 时才触发（避免负面上下文误判）
                        if self._is_positive_recommendation(doc_text_lower, rule_action):
                            return 0.0, "INADMISSIBLE_ABS"

            elif constraint.constraint_type == "RELATIVE":
                # 相对禁忌：规则层只能判断"是否可能相关"，f(δ) 计算需要具体参数
                for (rule_constraint, rule_action) in self._rel_rule_set:
                    if rule_constraint in c_target or c_target in rule_constraint:
                        if re.search(r"\b" + re.escape(rule_action) + r"\b", doc_text_lower):
                            # 命中相对禁忌规则 → 需要 f(δ) 计算，不能在规则层解决
                            if constraint.parameter_value is not None and constraint.parameter_threshold is not None:
                                kappa = constraint.compute_kappa_single()
                                status = "INADMISSIBLE_REL" if kappa < 1.0 else "ADMISSIBLE"
                                return kappa, status
                            return None  # 参数缺失，交给 LLM 层

        # 无规则命中 → 交给 LLM 层（或返回 ADMISSIBLE 若 LLM 不可用）
        return None

    def _predicate_layer_check(
        self,
        scope_pred: ScopePredicate,
        constraints: List[PatientConstraint],
    ) -> Tuple[float, str]:
        """
        LLM 层：基于 π_d 的 contraindications 列表计算 κ。

        产品规则：∏_i κ_single(c_i, π_d)
        """
        kappa_product = 1.0
        worst_status = "ADMISSIBLE"

        for constraint in constraints:
            c_target = constraint.target_action.lower()

            if constraint.constraint_type == "ABSOLUTE":
                # 检查患者约束 target 是否在文档禁忌证列表中的 action 所属药物族
                for ci in scope_pred.contraindications:
                    ci_lower = ci.lower()
                    # 宽松匹配：患者约束目标在禁忌证描述中出现
                    if c_target in ci_lower:
                        kappa_product = 0.0
                        worst_status = "INADMISSIBLE_ABS"
                        break
                # 反向：文档推荐的 action 属于患者绝对禁忌的药物族
                action_lower = scope_pred.recommended_action.lower()
                for (rule_c, rule_a) in self._abs_rule_set:
                    if rule_c in c_target and rule_a in action_lower:
                        kappa_product = 0.0
                        worst_status = "INADMISSIBLE_ABS"
                        break

            elif constraint.constraint_type == "RELATIVE":
                # 相对禁忌：直接用 PatientConstraint.compute_kappa_single()
                k_single = constraint.compute_kappa_single()
                if k_single < kappa_product:
                    kappa_product = k_single
                    worst_status = "INADMISSIBLE_REL"

        return kappa_product, worst_status

    def _get_or_extract_scope_predicate(self, chunk: TextChunk) -> ScopePredicate:
        """
        获取或提取文本块的 π_d。

        优先从缓存读取；缓存未命中时调用 LLM 提取并写入缓存。

        异常：
            ValueError: LLM 输出格式非法（不兜底）
        """
        # 第一步：检查缓存
        if self._cache_dir:
            cache_path = self._cache_dir / "scope_predicates" / f"{chunk.chunk_id}.json"
            if cache_path.exists():
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return ScopePredicate(
                    chunk_id=data["chunk_id"],
                    recommended_action=data["recommended_action"],
                    population=data["population"],
                    contraindications=data["contraindications"],
                    relative_restrictions=data["relative_restrictions"],
                    extraction_model=data["extraction_model"],
                    raw_output=data["raw_output"],
                )

        # 第二步：LLM 提取
        if self._client is None:
            raise RuntimeError(
                f"[KappaScorer] 需要 LLM 提取 π_d 但 client=None。"
                f"chunk_id: {chunk.chunk_id}"
            )
        scope_pred = self._extract_scope_predicate(chunk)

        # 第三步：写入缓存
        if self._cache_dir:
            data = {
                "chunk_id":             scope_pred.chunk_id,
                "recommended_action":   scope_pred.recommended_action,
                "population":           scope_pred.population,
                "contraindications":    scope_pred.contraindications,
                "relative_restrictions": scope_pred.relative_restrictions,
                "extraction_model":     scope_pred.extraction_model,
                "raw_output":           scope_pred.raw_output,
            }
            cache_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return scope_pred

    def _extract_scope_predicate(self, chunk: TextChunk) -> ScopePredicate:
        """
        调用 LLM 从文本块中提取 π_d。

        异常：
            ValueError: JSON 格式非法（不兜底）
        """
        prompt = SCOPE_EXTRACTION_PROMPT.format(text=chunk.text[:2000])
        raw_output = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32000,
        )

        # 解析 JSON（失败直接抛出，不兜底）
        if not raw_output:
            raise ValueError(
                f"[KappaScorer] π_d 提取 LLM 返回空响应（推理模型 token 耗尽）。"
                f"  chunk_id: {chunk.chunk_id}"
            )

        json_str = raw_output
        if "```" in raw_output:
            start = raw_output.index("```") + 3
            if "json" in raw_output[start:start+4]:
                start += 4
            end = raw_output.rindex("```")
            json_str = raw_output[start:end].strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[KappaScorer] π_d 提取 JSON 解析失败。"
                f"  chunk_id: {chunk.chunk_id}\n"
                f"  原始输出（前 200 字）: {raw_output[:200]}\n"
                f"  解析错误: {e}"
            )

        return ScopePredicate(
            chunk_id=chunk.chunk_id,
            recommended_action=data.get("recommended_action", "unknown"),
            population=data.get("population", ""),
            contraindications=data.get("contraindications", []),
            relative_restrictions=data.get("relative_restrictions", []),
            extraction_model=self._model,
            raw_output=raw_output,
        )
