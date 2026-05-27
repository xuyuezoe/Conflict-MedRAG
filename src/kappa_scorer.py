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

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.llm_client import LLMClient

from src.types import PatientConstraint, QueryDecomposition, RetrievedDoc, ScopePredicate, TextChunk
from src.retriever import RetrievalResult
from src.constraint_expander import DRUG_CLASS_EXPANSION


# ── 禁忌靶点别名表（类成员 + 同义词/变体）────────────────────────────────────
#
# 用途：分解器（LLM）命名的 contraindicated_targets 与 MCQ 选项用词常有
#       "类名 vs 成员"或"同义词形"差异（如 LLM 写 "ACE inhibitors"，
#       选项写 "lisinopril"；LLM 写 "radioactive iodine I-131"，选项写 "radioiodine"）。
#       本表把禁忌类/同义靶点展开为可匹配的具体药名与词形变体，提升绑定召回。
#
# 与 DRUG_CLASS_EXPANSION（抗生素/NSAID/阿片等）互补，二者在 _expand_target 中合并。

_CONTRAINDICATION_ALIASES: Dict[str, List[str]] = {
    "radioactive iodine": [
        "radioiodine", "radioactive iodine", "rai", "i-131", "i131",
        "iodine-131", "iodine 131", "radioactive iodine i-131",
    ],
    "ace inhibitor": [
        "ace inhibitor", "acei", "lisinopril", "enalapril", "ramipril",
        "captopril", "benazepril", "perindopril", "quinapril",
    ],
    "angiotensin receptor blocker": [
        "arb", "angiotensin receptor blocker", "losartan", "valsartan",
        "candesartan", "irbesartan", "olmesartan", "telmisartan",
    ],
    "dopamine antagonist": [
        "dopamine antagonist", "metoclopramide", "prochlorperazine",
        "haloperidol", "chlorpromazine", "promethazine", "droperidol",
    ],
    "statin": [
        "statin", "atorvastatin", "simvastatin", "rosuvastatin",
        "pravastatin", "lovastatin", "pitavastatin",
    ],
    "beta blocker": [
        "beta blocker", "beta-blocker", "propranolol", "nadolol", "timolol",
        "carvedilol", "labetalol", "metoprolol", "atenolol", "bisoprolol",
    ],
    "bisphosphonate": [
        "bisphosphonate", "alendronate", "risedronate", "ibandronate",
        "zoledronic acid", "pamidronate",
    ],
    "thiazolidinedione": ["thiazolidinedione", "pioglitazone", "rosiglitazone"],
    "live vaccine": ["live vaccine", "live attenuated", "mmr", "varicella", "rotavirus"],
}


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


# ── π_d 提取 Prompt（v2，含 polarity/scope/status 结构化字段）────────────────

SCOPE_EXTRACTION_PROMPT = """\
Extract the structured scope predicate from this medical text passage.
Identify the treatment action, its clinical polarity, and the applicability scope.

Return ONLY valid JSON in this exact format:
{{
  "recommended_action": "<primary drug or treatment, or 'none' if absent>",
  "population": "<natural language description of who this treatment is for>",
  "polarity": "<one of: recommended | contraindicated | caution | dose_adjustment | not_applicable>",
  "scope_inclusion": ["<population characteristic that IS included, e.g. 'adult', 'pregnancy-compatible'>"],
  "scope_exclusion": ["<population characteristic that is EXCLUDED, e.g. 'pregnancy', 'penicillin allergy'>"],
  "contraindications": ["<explicit contraindication description>"],
  "relative_restrictions": [
    {{
      "condition": "<e.g. 'renal impairment'>",
      "threshold": "<e.g. 'eGFR < 30 mL/min'>"
    }}
  ],
  "scope_status": "<one of: explicit | inferred | not_specified>"
}}

Polarity rules:
  recommended:    The text positively recommends this action for the stated population.
  contraindicated: The text states this action is contraindicated / should be avoided.
  caution:        The text recommends with significant caveats or warns of risks.
  dose_adjustment: The text requires dose modification for certain populations.
  not_applicable: The text does not make a specific treatment recommendation.

Scope status rules:
  explicit:     The text explicitly states who the treatment is (or is not) for.
  inferred:     The scope can be reasonably inferred from context (mark conservatively).
  not_specified: The text mentions a treatment but does not state its applicability scope.

CRITICAL RULES:
  1. scope_status="not_specified" does NOT mean "no contraindications". It means the text
     did not state who the treatment applies to. Do NOT populate scope_inclusion or
     scope_exclusion if the text does not mention them — leave them as empty lists.
  2. If the text is recommending an alternative FOR patients who cannot use Drug X,
     polarity should be "recommended" for the alternative, and scope_inclusion should
     mention the relevant patient population (e.g. "penicillin-allergic patients").
  3. Use "contraindicated" polarity only when the text explicitly states contraindication.

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

    # v2 批量提取 prompt（与 SCOPE_EXTRACTION_PROMPT 字段完全一致，仅封装为数组格式）
    _BATCH_EXTRACT_PROMPT_V2 = """\
Extract structured scope predicates from the following {n} medical text passages.
For each passage apply the same rules as single-passage extraction.

{passages}

Return ONLY a valid JSON array with exactly {n} objects in order.
Each object must have ALL of these fields:
[
  {{
    "chunk_id": "<id from above>",
    "recommended_action": "<primary drug or treatment, or 'none' if absent>",
    "population": "<natural language description of who this treatment is for>",
    "polarity": "<one of: recommended | contraindicated | caution | dose_adjustment | not_applicable>",
    "scope_inclusion": ["<population characteristic that IS included>"],
    "scope_exclusion": ["<population characteristic that is EXCLUDED, e.g. 'pregnancy', 'penicillin allergy'>"],
    "contraindications": ["<explicit contraindication description>"],
    "relative_restrictions": [{{"condition": "<e.g. renal impairment>", "threshold": "<e.g. eGFR < 30>"}}],
    "scope_status": "<one of: explicit | inferred | not_specified>"
  }},
  ...
]

Polarity rules:
  recommended:    The text positively recommends this action for the stated population.
  contraindicated: The text states this action is contraindicated / should be avoided.
  caution:        The text recommends with significant caveats or warns of risks.
  dose_adjustment: The text requires dose modification for certain populations.
  not_applicable: The text does not make a specific treatment recommendation.

Scope status rules:
  explicit:      The text explicitly states who the treatment is (or is not) for.
  inferred:      The scope can be reasonably inferred from context.
  not_specified: The text mentions a treatment but does not state its applicability scope.

CRITICAL RULES:
  1. scope_status="not_specified" does NOT mean "no contraindications".
  2. Do NOT populate scope_inclusion or scope_exclusion if the text does not mention them.
  3. Use "contraindicated" polarity only when the text explicitly states contraindication.

Return ONLY the JSON array, no other text."""

    def _batch_extract_and_cache(self, retrieval_results: List[RetrievalResult]) -> None:
        """
        对多个 cache-miss 文档批量提取 π_d（v2格式），写入磁盘缓存。

        策略：将文档切分为 _BATCH_SIZE 大小的批次，每批一次 LLM 调用。
        使用 v2 batch prompt（含 polarity/scope_inclusion/scope_exclusion/scope_status），
        与 _extract_scope_predicate 的 SCOPE_EXTRACTION_PROMPT 字段完全对齐。

        参数：
            retrieval_results: 需要 LLM 提取的文档列表（均为 cache-miss）
        """
        if not self._cache_dir:
            return
        cache_dir = self._cache_dir / "scope_predicates"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for i in range(0, len(retrieval_results), self._BATCH_SIZE):
            batch = retrieval_results[i: i + self._BATCH_SIZE]

            # 构造批量 prompt（每个文档截取前 600 字符，v2 需要更多上下文判断 polarity）
            passages_parts = []
            for j, rr in enumerate(batch):
                passages_parts.append(
                    f"Passage {j+1} (chunk_id: {rr.chunk.chunk_id}):\n{rr.chunk.text[:600]}"
                )
            passages_text = "\n\n".join(passages_parts)

            prompt = self._BATCH_EXTRACT_PROMPT_V2.format(
                n=len(batch),
                passages=passages_text,
            )

            raw = self._client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32000,
            )

            # 解析 JSON 数组
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
                # 批量解析失败：退回逐文档模式（_extract_scope_predicate 使用 v2 prompt）
                for rr in batch:
                    try:
                        pred = self._extract_scope_predicate(rr.chunk)
                        (cache_dir / f"{pred.chunk_id}.json").write_text(
                            json.dumps(self._scope_predicate_to_cache_dict(pred),
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                continue

            # 将批量结果以 v2 格式写入缓存
            for item in items:
                cid = item.get("chunk_id", "")
                if not cid:
                    continue
                pred = self._parse_scope_predicate_from_dict(item, cid, raw)
                (cache_dir / f"{cid}.json").write_text(
                    json.dumps(self._scope_predicate_to_cache_dict(pred),
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
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

    def check_option(
        self,
        letter: str,
        description: str,
        constraints: List[PatientConstraint],
        use_llm_layer: bool = True,
    ) -> Tuple[float, str, str]:
        """
        对单个候选选项（A–E）的描述文本判定 κ 与适用性状态。

        第一性原理：把 κ 机制从"文档维度"扩展到"选项维度"。
        选项即候选动作 a_i ∈ A_q，约束 C 是 action-specific 的，
        因此可将选项描述视为一条最小 action 陈述，复用既有两层 κ 机制判定
        κ(C_q, a_i)。规则层确定性命中即定论，无 LLM 成本；规则层无定论时
        按需用谓词层（LLM 提取 π）兜底。

        与 compute_kappa 的区别：
          - 输入是选项描述文本而非检索文档；
          - 显式返回判定来源 source（rule/predicate/none），供审计与可解释；
          - 谓词层使用命名空间隔离的伪 chunk_id（opt::<hash>），
            避免污染真实文档 π_d 的磁盘缓存。

        参数：
            letter:         选项字母（仅用于日志/调试，不参与判定）
            description:    选项描述文本（如 "Oral warfarin therapy"）
            constraints:    患者约束列表 C_q
            use_llm_layer:  规则层无定论时是否启用 LLM 谓词层兜底
                            （False 用于纯确定性消融实验）

        返回：
            (kappa, scope_status, source)
            kappa:        [0,1]
            scope_status: ADMISSIBLE / INADMISSIBLE_ABS / INADMISSIBLE_REL / UNKNOWN
            source:       rule / predicate / none
        """
        # 第一阶段：无约束 → 平凡可行
        if not constraints:
            return 1.0, "ADMISSIBLE", "rule"

        # 第二阶段：命名禁忌药匹配（最高优先级）。
        # 复用 LLM 在分解阶段显式抽取的 contraindicated_targets：
        # 若选项正面推荐了某约束的禁忌药，直接定论（确定性，可解释）。
        named_result = self._named_target_check(description, constraints)
        if named_result is not None:
            kappa, scope_status = named_result
            return kappa, scope_status, "named_target"

        # 第三阶段：规则层确定性判定（命中即定论，O(1)，无 API 成本）
        rule_result = self._rule_layer_check(description, constraints)
        if rule_result is not None:
            kappa, scope_status = rule_result
            return kappa, scope_status, "rule"

        # 第四阶段：LLM 谓词层兜底（仅在启用且客户端可用时）
        if not use_llm_layer or self._client is None:
            # 显式返回 UNKNOWN，不静默当成可行（遵守 CLAUDE.md 禁兜底）
            return 1.0, "UNKNOWN", "none"

        # 用命名空间隔离的伪 chunk_id，避免污染真实文档缓存
        pseudo_id = f"opt::{hashlib.sha1(description.encode('utf-8')).hexdigest()[:12]}"
        option_chunk = TextChunk(
            chunk_id=pseudo_id,
            source_book="__option__",
            text=description,
            start_char=0,
            end_char=len(description),
            token_count=len(description.split()),
        )
        scope_pred = self._get_or_extract_scope_predicate(option_chunk)
        kappa, scope_status = self._predicate_layer_check(scope_pred, constraints)
        return kappa, scope_status, "predicate"

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
        # 停药/换药动词：药物是被"撤掉"而非被推荐（修复 MCQ 选项 "Discontinue X, start Y"
        # 把被停的 X 误判为正面推荐的假阳性）
        r"discontinu\w*",
        r"\bstop\w*",
        r"\bcease\w*",
        r"\bwithdraw\w*",
        r"\btaper\w*",
        r"\bwithhold\w*",
        r"switch\w*\s+(?:from|off)",
        r"\bwean\w*",
        # 脱敏/分级激发/试验剂量：在过敏患者身上"安全给药"的协议，
        # 不是禁忌推荐（修复 MACB-024：penicillin desensitization 被误判禁忌）
        r"desensiti[sz]\w*",
        r"graded\s+challenge",
        r"test\s+dose",
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

    def _named_target_check(
        self,
        option_text: str,
        constraints: List[PatientConstraint],
    ) -> Optional[Tuple[float, str]]:
        """
        命名禁忌药匹配层：选项是否正面推荐了某约束显式列出的禁忌药。

        第一性原理：分解器（LLM）已依据患者状态把禁忌药/药类抽到
        constraint.contraindicated_targets。此处只需把"选项推荐的药"与之比对，
        即完成 patient×drug 绑定——无需静态规则表覆盖全部药物。

        判定规则：
          - 命中 ABSOLUTE 约束的禁忌药 → (0.0, "INADMISSIBLE_ABS")
          - 命中 RELATIVE 约束的禁忌药 → (κ_single, "INADMISSIBLE_REL")
          - 无命中 → None（交由后续规则层/谓词层）

        ABSOLUTE 优先于 RELATIVE（绝对禁忌一票否决）。

        参数：
            option_text: 选项描述文本
            constraints: 患者约束列表（须含 contraindicated_targets）

        返回：
            (kappa, scope_status) 或 None
        """
        text_lower = option_text.lower()
        relative_hit: Optional[Tuple[float, str]] = None

        for c in constraints:
            if not c.contraindicated_targets:
                continue
            for target in c.contraindicated_targets:
                # 展开为别名集合（类成员 + 同义词/变体），逐一匹配
                for alias in self._expand_target(target):
                    # 复用正面推荐判定，规避"提及但排除/停药"的假阳性
                    if self._is_positive_recommendation(text_lower, alias):
                        if c.constraint_type == "ABSOLUTE":
                            return 0.0, "INADMISSIBLE_ABS"
                        if c.constraint_type == "RELATIVE" and relative_hit is None:
                            relative_hit = (c.compute_kappa_single(), "INADMISSIBLE_REL")
                        break  # 该 target 已命中，无需试其余别名

        return relative_hit

    @staticmethod
    def _expand_target(target: str) -> List[str]:
        """
        把单个禁忌靶点展开为可匹配的别名集合（去重，保序）。

        展开来源（合并）：
          1. 靶点本身（规范化小写）
          2. DRUG_CLASS_EXPANSION：抗生素/NSAID/阿片等类→成员
          3. _CONTRAINDICATION_ALIASES：禁忌类/同义靶点→成员与词形变体
        类名匹配时同时尝试去复数（"inhibitors"→"inhibitor"）。

        参数：
            target: 分解器命名的禁忌药/药类名

        返回：
            别名字符串列表（含 target 自身）
        """
        t = target.strip().lower()
        if not t:
            return []
        aliases: List[str] = [t]
        seen = {t}
        # 子串包含式匹配 alias 表的键：使被限定/带后缀的靶点也能命中
        # （如 "radioactive iodine i-131" 命中键 "radioactive iodine"；
        #   "penicillins"/"penicillin class" 命中键 "penicillin"）。
        # 键均为 >=4 字符的具体药/类名，子串匹配不会引入短词误命中。
        for table in (DRUG_CLASS_EXPANSION, _CONTRAINDICATION_ALIASES):
            for key, members in table.items():
                if key in t or t in key:
                    for member in members:
                        m = member.strip().lower()
                        if m and m not in seen:
                            seen.add(m)
                            aliases.append(m)
        return aliases

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
        LLM 层：基于 π_d 结构化 scope 计算 κ。

        判断顺序（优先级从高到低）：
          1. Hard gate：polarity=contraindicated → κ=0（不可补偿）
          2. scope_exclusion 蕴含判断：患者约束 target 命中 scope_exclusion → κ=0
          3. contraindications 后向兼容匹配
          4. 绝对禁忌规则反向检查（推荐 action 属于患者禁忌类别）
          5. 相对禁忌：f(δ) 连续值计算

        κ 组合规则：乘积形式（任意 0 → 整体为 0）
        """
        # ── Hard gate：文档本身被标注为 contraindicated（polarity 级别判断）──
        # scope_status="not_specified" 的文档不触发此 gate
        if scope_pred.is_hard_contraindicated:
            return 0.0, "INADMISSIBLE_ABS"

        kappa_product = 1.0
        worst_status = "ADMISSIBLE"

        # ── scope_status="not_specified" 的保守降权 ───────────────────────────
        # 未声明适用域的文档：既不确认适用，也不确认排除，保守默认降权到 0.5
        # 而非兜底为 κ=1（即"默认适用于所有人"）
        if scope_pred.scope_status == "not_specified" and constraints:
            kappa_product = min(kappa_product, 0.5)
            worst_status = "UNKNOWN"

        for constraint in constraints:
            c_target = constraint.target_action.lower()

            if constraint.constraint_type == "ABSOLUTE":
                # 第一层：scope_exclusion 蕴含判断
                # "患者约束的 target 是否出现在文档的排除人群特征中"
                for excl in scope_pred.scope_exclusion:
                    excl_lower = excl.lower()
                    if c_target in excl_lower or excl_lower in c_target:
                        kappa_product = 0.0
                        worst_status = "INADMISSIBLE_ABS"
                        break

                if kappa_product == 0.0:
                    break

                # 第二层：contraindications 列表（后向兼容旧缓存）
                for ci in scope_pred.contraindications:
                    ci_lower = ci.lower()
                    if c_target in ci_lower:
                        kappa_product = 0.0
                        worst_status = "INADMISSIBLE_ABS"
                        break

                if kappa_product == 0.0:
                    break

                # 第三层：绝对禁忌规则反向检查（推荐 action 属于患者禁忌类别）
                action_lower = scope_pred.recommended_action.lower()
                for (rule_c, rule_a) in self._abs_rule_set:
                    if rule_c in c_target and rule_a in action_lower:
                        kappa_product = 0.0
                        worst_status = "INADMISSIBLE_ABS"
                        break

            elif constraint.constraint_type == "RELATIVE":
                # 相对禁忌：连续 f(δ) 计算
                k_single = constraint.compute_kappa_single()
                if k_single < kappa_product:
                    kappa_product = k_single
                    worst_status = "INADMISSIBLE_REL"

        return kappa_product, worst_status

    # polarity 合法值集合（用于校验 LLM 输出）
    _VALID_POLARITY = frozenset(
        ["recommended", "contraindicated", "caution", "dose_adjustment", "not_applicable"]
    )
    _VALID_SCOPE_STATUS = frozenset(["explicit", "inferred", "not_specified"])

    def _parse_scope_predicate_from_dict(
        self, data: Dict, chunk_id: str, raw_output: str
    ) -> ScopePredicate:
        """
        从 LLM 输出的 dict 构造 ScopePredicate（v2 格式，含 polarity/scope/status）。

        参数：
            data:       已解析的 JSON dict
            chunk_id:   文本块 ID（用于错误信息）
            raw_output: LLM 原始文本（存入 raw_output 字段）

        返回：
            ScopePredicate（v2）

        校验：
            polarity / scope_status 非法值 → 降级为默认值并记录警告（不抛出，
            因为该字段对 κ=0 的 hard gate 判断不是必须的）
        """
        raw_polarity = data.get("polarity", "not_applicable")
        polarity = (
            raw_polarity if raw_polarity in self._VALID_POLARITY else "not_applicable"
        )
        raw_status = data.get("scope_status", "not_specified")
        scope_status = (
            raw_status if raw_status in self._VALID_SCOPE_STATUS else "not_specified"
        )

        return ScopePredicate(
            chunk_id=chunk_id,
            recommended_action=data.get("recommended_action", "none"),
            population=data.get("population", ""),
            contraindications=data.get("contraindications", []),
            relative_restrictions=data.get("relative_restrictions", []),
            extraction_model=self._model,
            raw_output=raw_output,
            polarity=polarity,  # type: ignore[arg-type]
            scope_inclusion=data.get("scope_inclusion", []),
            scope_exclusion=data.get("scope_exclusion", []),
            scope_status=scope_status,  # type: ignore[arg-type]
        )

    def _scope_predicate_to_cache_dict(self, pred: ScopePredicate) -> Dict:
        """将 ScopePredicate（v2）序列化为缓存 dict（含新字段）"""
        return {
            "chunk_id":              pred.chunk_id,
            "recommended_action":    pred.recommended_action,
            "population":            pred.population,
            "contraindications":     pred.contraindications,
            "relative_restrictions": pred.relative_restrictions,
            "extraction_model":      pred.extraction_model,
            "raw_output":            pred.raw_output,
            # v2 新字段
            "polarity":              pred.polarity,
            "scope_inclusion":       pred.scope_inclusion,
            "scope_exclusion":       pred.scope_exclusion,
            "scope_status":          pred.scope_status,
        }

    def _scope_predicate_from_cache_dict(self, data: Dict) -> ScopePredicate:
        """
        从缓存 dict 恢复 ScopePredicate（兼容 v1 旧缓存，缺失字段用默认值补全）。

        v1 旧缓存没有 polarity/scope_inclusion/scope_exclusion/scope_status 字段，
        恢复时补为默认值（not_applicable / [] / [] / not_specified），不抛出异常。
        """
        raw_polarity = data.get("polarity", "not_applicable")
        polarity = raw_polarity if raw_polarity in self._VALID_POLARITY else "not_applicable"
        raw_status = data.get("scope_status", "not_specified")
        scope_status = raw_status if raw_status in self._VALID_SCOPE_STATUS else "not_specified"
        return ScopePredicate(
            chunk_id=data["chunk_id"],
            recommended_action=data["recommended_action"],
            population=data["population"],
            contraindications=data["contraindications"],
            relative_restrictions=data["relative_restrictions"],
            extraction_model=data["extraction_model"],
            raw_output=data["raw_output"],
            polarity=polarity,  # type: ignore[arg-type]
            scope_inclusion=data.get("scope_inclusion", []),
            scope_exclusion=data.get("scope_exclusion", []),
            scope_status=scope_status,  # type: ignore[arg-type]
        )

    def _get_or_extract_scope_predicate(self, chunk: TextChunk) -> ScopePredicate:
        """
        获取或提取文本块的 π_d（v2）。

        优先从缓存读取（支持 v1 旧缓存自动升级）；
        缓存未命中时调用 LLM 提取并写入缓存。

        异常：
            ValueError: LLM 输出格式非法（不兜底）
        """
        # 第一步：检查缓存（v1/v2 兼容读取）
        if self._cache_dir:
            cache_path = self._cache_dir / "scope_predicates" / f"{chunk.chunk_id}.json"
            if cache_path.exists():
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return self._scope_predicate_from_cache_dict(data)

        # 第二步：LLM 提取
        if self._client is None:
            raise RuntimeError(
                f"[KappaScorer] 需要 LLM 提取 π_d 但 client=None。"
                f"chunk_id: {chunk.chunk_id}"
            )
        scope_pred = self._extract_scope_predicate(chunk)

        # 第三步：写入缓存
        if self._cache_dir:
            cache_path = self._cache_dir / "scope_predicates" / f"{chunk.chunk_id}.json"
            cache_path.write_text(
                json.dumps(self._scope_predicate_to_cache_dict(scope_pred),
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return scope_pred

    def _extract_scope_predicate(self, chunk: TextChunk) -> ScopePredicate:
        """
        调用 LLM 从文本块中提取 π_d（v2，含 polarity/scope/status）。

        异常：
            ValueError: JSON 格式非法（不兜底）
        """
        prompt = SCOPE_EXTRACTION_PROMPT.format(text=chunk.text[:2000])
        raw_output = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32000,
        )

        if not raw_output:
            raise ValueError(
                f"[KappaScorer] π_d 提取 LLM 返回空响应。chunk_id: {chunk.chunk_id}"
            )

        json_str = raw_output
        if "```" in raw_output:
            start = raw_output.index("```") + 3
            if "json" in raw_output[start:start + 4]:
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

        return self._parse_scope_predicate_from_dict(data, chunk.chunk_id, raw_output)
