#!/usr/bin/env python3
"""
MACB 候选样本池构建器 v3

核心改动（相比 v2）：
  1. 增加"治疗性问题主干过滤器"——question stem 必须询问治疗方案/药物
  2. 增加"选项类型过滤器"——全部/绝大多数选项必须是药物或干预措施，
     非诊断名、检查名或疾病名
  3. 由此根本解决 v2 中 ~80% 候选为诊断题的问题
  4. 目标：从 ~27K US USMLE 题中筛选 100+ 高质量 TREATMENT 候选

使用方法：
  python3 scripts/build_macb_candidates.py \
      --input data_clean/questions/US/test.jsonl \
               data_clean/questions/US/dev.jsonl \
               data_clean/questions/US/train.jsonl \
               data_clean/questions/US/US_qbank.jsonl \
      --output data/macb_candidates_v3.jsonl \
      --target-sc-abs 60 --target-sc-rel 30 --pool-factor 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ── 治疗性主干模式 ────────────────────────────────────────────────────────────
# 问题主干必须至少命中其中一条，方可进入候选池
TREATMENT_STEM_PATTERNS: List[str] = [
    # 最常见的 USMLE 治疗提问框架
    r"\bmost\s+appropriate\s+(?:treatment|therapy|management|medication|drug|next\s+step\s+in\s+(?:treatment|management|therapy))\b",
    r"\bmost\s+appropriate\s+(?:pharmacologic(?:al)?\s+)?(?:treatment|therapy|management)\b",
    r"\bbest\s+(?:treatment|therapy|management|medication|drug|pharmacotherapy)\b",
    r"\bfirst[\s-]line\s+(?:treatment|therapy|medication|drug|management|agent|antibiotic|antifungal|antiviral)\b",
    r"\btreatment\s+of\s+choice\b",
    r"\bdrug\s+of\s+choice\b",
    r"\bwhich\s+(?:of\s+the\s+following\s+)?(?:medication|drug|antibiotic|agent|therapy|treatment|intervention)\s+(?:is|should|would|will)\b",
    r"\bwhich\s+(?:of\s+the\s+following\s+)?(?:should|is\s+the\s+most\s+appropriate)\s+(?:treatment|therapy|medication|drug|management)\b",
    r"\bshould\s+(?:be\s+)?(?:given|prescribed|administered|started|initiated|added|used|treated)\b",
    r"\bwould\s+(?:be\s+)?(?:given|prescribed|administered|started|initiated|most\s+appropriate)\b",
    r"\bmost\s+likely\s+to\s+(?:benefit|improve|treat|resolve|alleviate)\b",
    r"\bmost\s+effective\s+(?:treatment|therapy|management|medication|drug)\b",
    r"\bmanagement\s+(?:of|for|involves?|includes?|should)\b",
    r"\bhow\s+(?:should|would)\s+(?:this|the)\s+(?:patient|condition|disease|infection|pain)\s+(?:be\s+)?(?:treated|managed|handled)\b",
    r"\bnext\s+(?:best|most\s+appropriate)\s+(?:step|management|treatment|therapy)\b",
    r"\bpharmacological(?:ly)?\s+(?:treat|manage|intervention)\b",
    r"\bwhich\s+(?:medication|drug|agent|antibiotic)\s+(?:is|would\s+be)\s+(?:most\s+)?(?:appropriate|indicated|recommended)\b",
    r"\bprescrib(?:e|ing)\b.{0,50}\b(?:medication|drug|antibiotic|agent)\b",
    r"\badminister(?:ing)?\b.{0,50}\b(?:medication|drug|dose)\b",
    r"\bstart(?:ing)?\b.{0,50}\b(?:medication|drug|therapy|treatment)\b",
    r"\binitiat(?:e|ing)\b.{0,50}\b(?:medication|drug|therapy|treatment)\b",
]

# ── 药物/干预措施识别模式（阳性）────────────────────────────────────────────────
# 选项文本命中此列表 → 视为 TREATMENT 选项
DRUG_POSITIVE_PATTERNS: List[str] = [
    # 常见药物后缀（INN命名规范）
    r"\b\w+(?:cillin|mycin|cycline|azole|mab|tinib|oxacin|statin|sartan|pril|olol|"
    r"dipine|vir|vudine|navir|lukast|limus|fenac|profen|formin|gliptin|gliflozin|"
    r"zepam|pam|barbital|bital|caine|dine|tidine|razole|prazole|sone|lone|olone|"
    r"sterone|onide|terol|phylline|tropine|cromone|fentanil|morphine|codeine|"
    r"warfarin|heparin|xaban|gatran|parin)\b",
    # 常见独立药物名
    r"\b(?:aspirin|ibuprofen|acetaminophen|paracetamol|metformin|insulin|"
    r"prednisone|prednisolone|dexamethasone|hydrocortisone|methylprednisolone|"
    r"amoxicillin|penicillin|ampicillin|vancomycin|clindamycin|metronidazole|"
    r"ciprofloxacin|levofloxacin|azithromycin|erythromycin|doxycycline|"
    r"trimethoprim|sulfamethoxazole|nitrofurantoin|rifampin|isoniazid|ethambutol|"
    r"pyrazinamide|fluconazole|itraconazole|voriconazole|amphotericin|"
    r"acyclovir|valacyclovir|ganciclovir|oseltamivir|remdesivir|"
    r"lisinopril|enalapril|captopril|ramipril|losartan|valsartan|irbesartan|"
    r"amlodipine|nifedipine|diltiazem|verapamil|metoprolol|atenolol|carvedilol|"
    r"propranolol|furosemide|hydrochlorothiazide|spironolactone|"
    r"atorvastatin|simvastatin|rosuvastatin|pravastatin|lovastatin|"
    r"warfarin|heparin|enoxaparin|rivaroxaban|apixaban|dabigatran|clopidogrel|"
    r"digoxin|amiodarone|lidocaine|adenosine|atropine|epinephrine|norepinephrine|"
    r"dopamine|dobutamine|phenylephrine|vasopressin|"
    r"morphine|oxycodone|hydrocodone|fentanyl|tramadol|codeine|naloxone|"
    r"ondansetron|metoclopramide|promethazine|prochlorperazine|"
    r"haloperidol|risperidone|olanzapine|quetiapine|clozapine|aripiprazole|"
    r"sertraline|fluoxetine|paroxetine|citalopram|escitalopram|venlafaxine|"
    r"duloxetine|bupropion|mirtazapine|amitriptyline|nortriptyline|"
    r"alprazolam|diazepam|lorazepam|clonazepam|midazolam|"
    r"phenytoin|valproate|carbamazepine|levetiracetam|lamotrigine|"
    r"albuterol|salbutamol|salmeterol|formoterol|tiotropium|ipratropium|"
    r"montelukast|fluticasone|budesonide|beclomethasone|"
    r"levothyroxine|methimazole|propylthiouracil|"
    r"methotrexate|azathioprine|mycophenolate|cyclophosphamide|"
    r"rituximab|infliximab|adalimumab|etanercept|tocilizumab|"
    r"cisplatin|carboplatin|paclitaxel|docetaxel|doxorubicin|"
    r"tamoxifen|letrozole|anastrozole|exemestane|"
    r"imatinib|erlotinib|gefitinib|sunitinib|sorafenib|"
    r"omeprazole|pantoprazole|lansoprazole|esomeprazole|"
    r"ranitidine|cimetidine|famotidine|"
    r"loperamide|bismuth|sucralfate|misoprostol|"
    r"colchicine|allopurinol|febuxostat|probenecid|"
    r"cyclosporine|tacrolimus|sirolimus|everolimus|"
    r"iron|ferrous|folic\s+acid|vitamin\s+[bBdDkK]|thiamine|pyridoxine|"
    r"calcium|magnesium|potassium|sodium\s+bicarbonate|"
    r"albumin|fresh\s+frozen\s+plasma|packed\s+red\s+blood\s+cells|"
    r"desmopressin|octreotide|vasopressin|terlipressin|"
    r"mannitol|acetazolamide|"
    r"nitroglycerine|nitroglycerin|isosorbide|"
    r"sildenafil|tadalafil|"
    r"methadone|buprenorphine|naltrexone)\b",
    # 手术 / 侵入性干预
    r"\b(?:surgery|surgical|resection|excision|appendectomy|cholecystectomy|"
    r"colectomy|gastrectomy|thyroidectomy|mastectomy|hysterectomy|"
    r"nephrectomy|splenectomy|orchiectomy|prostatectomy|"
    r"coronary\s+artery\s+bypass|bypass\s+graft(?:ing)?|cabg|"
    r"angioplasty|stent(?:ing)?|catheterization|ablation|"
    r"transfusion|plasmapheresis|hemodialysis|dialysis|"
    r"intubation|mechanical\s+ventilation|"
    r"thoracentesis|paracentesis|lumbar\s+puncture|"
    r"incision\s+and\s+drainage|debridement|"
    r"radiation\s+therapy|chemotherapy|immunotherapy|"
    r"physical\s+therapy|occupational\s+therapy|"
    r"electroconvulsive\s+therapy|ect)\b",
    # 给药/输液
    r"\b(?:intravenous(?:ly)?|iv\s+fluid|oral\s+rehydration|"
    r"supplementation|replacement\s+therapy)\b",
]

# ── 诊断/检查/疾病名识别模式（阴性）─────────────────────────────────────────────
# 选项文本命中此列表 → 视为非 TREATMENT 选项
DIAGNOSIS_NEGATIVE_PATTERNS: List[str] = [
    # 影像学检查
    r"\b(?:mri|ct\s+scan|computed\s+tomography|x[\s-]ray|ultrasound|"
    r"echocardiogram|echo|pet\s+scan|bone\s+scan|scintigraphy|"
    r"angiography|fluoroscopy|mammography|colonoscopy|endoscopy|"
    r"bronchoscopy|cystoscopy|laparoscopy|arthroscopy)\b",
    # 实验室检查 / 活检
    r"\b(?:biopsy|culture|sensitivity|gram\s+stain|blood\s+culture|"
    r"urinalysis|urine\s+culture|throat\s+culture|spinal\s+tap|"
    r"lumbar\s+puncture|bone\s+marrow\s+biopsy|liver\s+biopsy|"
    r"fine\s+needle\s+aspiration|fna|"
    r"complete\s+blood\s+count|cbc|basic\s+metabolic\s+panel|bmp|"
    r"comprehensive\s+metabolic\s+panel|cmp|"
    r"prothrombin\s+time|partial\s+thromboplastin|"
    r"arterial\s+blood\s+gas|abg|"
    r"ecg|electrocardiogram|eeg|electromyography|emg|"
    r"pulmonary\s+function\s+test|pft|spirometry|"
    r"skin\s+test|mantoux|tuberculin|ppd|"
    r"serology|titer|antigen\s+test|antibody\s+test|"
    r"glucose\s+tolerance\s+test|schilling\s+test)\b",
    # 疾病 / 综合征名称（常见用作 distractor）
    r"\b(?:syndrome|disease|disorder|deficiency|malignancy|cancer|"
    r"carcinoma|lymphoma|leukemia|sarcoma|adenoma|"
    r"infection|inflammation|torsion|rupture|perforation|obstruction|"
    r"stenosis|thrombosis|embolism|infarction|ischemia|"
    r"hypertension|hypotension|arrhythmia|fibrillation|"
    r"appendicitis|cholecystitis|pancreatitis|hepatitis|"
    r"pneumonia|meningitis|encephalitis|pyelonephritis|"
    r"gastroenteritis|colitis|diverticulitis|peritonitis)\b",
    # 观察/等待类
    r"\b(?:observation|watchful\s+waiting|reassurance|"
    r"supportive\s+care\s+only|discharge\s+home|"
    r"referral\s+to|counseling\s+only|no\s+treatment)\b",
]

# ── SC_ABSOLUTE 触发模式 ─────────────────────────────────────────────────────
SC_ABS_PATTERNS: List[str] = [
    r"\ballerg(?:y|ic)\s+to\b",
    r"\bknown\s+allerg\w+\b",
    r"\bpenicillin\b.{0,50}\ballerg\w+\b",
    r"\bsulfa\b.{0,50}\ballerg\w+\b",
    r"\baspirin\b.{0,50}\ballerg\w+\b",
    r"\bdrug\s+allerg\w+\b",
    r"\bmedication\s+allerg\w+\b",
    r"\banaphylax(?:is|tic)\b",
    r"\bcontraindica(?:ted|tion|tions?)\b",
    r"\bcannot\s+(?:be\s+)?(?:given|used|administered|taken|receive)\b",
    r"\bmust\s+not\s+(?:be\s+)?(?:given|used|administered)\b",
    r"\bnot\s+safe\s+(?:to\s+use\s+in|for|in|during)\b",
    r"\babsolute\s+contraindic\w+",
    r"\bpregnant\b",
    r"\bpregnancy\b",
    r"\bgestat(?:ion|ional)\b",
    r"\bgravida\b",
    r"\btrimester\b",
    r"\bbreastfeed(?:ing)?\b",
    r"\blactat(?:ing|ion)\b",
    r"\bnursing\s+(?:mother|infant|baby|child)\b",
    r"\bheparin[-\s]induced\s+thrombocytopenia\b",
    r"\bwarfarin\b.{0,80}\bpregnant\b",
    r"\bthalidomide\b",
    r"\bisotretinoin\b",
    r"\bmethotrexate\b.{0,80}\bpregnant\b",
    r"\blive\s+(?:attenuated\s+)?vaccine\b.{0,80}\bimmunosuppress\w+\b",
    r"\baspirin\b.{0,40}\breye\b",
]

# ── SC_RELATIVE 触发模式 ─────────────────────────────────────────────────────
SC_REL_PATTERNS: List[str] = [
    r"\begfr\b",
    r"\bcreatinine\b",
    r"\brenal\s+(?:impairment|failure|insufficiency|disease|function|dosing)\b",
    r"\bchronic\s+kidney\s+disease\b",
    r"\b(?:ckd|aki)\b",
    r"\bhepatic\s+(?:impairment|failure|insufficiency|disease)\b",
    r"\bliver\s+(?:failure|disease|impairment|cirrhosis)\b",
    r"\bchild[-\s]pugh\b",
    r"\bdose\s+(?:reduction|adjustment|modification)\b",
    r"\bdose[-\s]adjust\w+\b",
    r"\badjust(?:ed|ment)?\s+(?:the\s+)?dose\b",
    r"\brenal(?:ly)?\s+(?:dosed|cleared|excreted|adjusted)\b",
    r"\bglomerular\s+filtration\s+rate\b",
    r"\bmoderate\s+(?:renal|hepatic|kidney|liver)\b",
    r"\bsevere\s+(?:renal|hepatic|kidney|liver)\b",
]


def _any_pattern(patterns: List[str], text: str) -> bool:
    """词边界安全的多模式匹配；任一命中返回 True。"""
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _count_drug_options(options: Dict[str, str]) -> Tuple[int, int]:
    """
    统计选项中有多少是药物/干预（阳性），多少是诊断/检查（阴性）。

    返回：
        (n_positive, n_negative) — 阳性计数，阴性计数
    """
    n_pos = 0
    n_neg = 0
    for opt_text in options.values():
        is_pos = _any_pattern(DRUG_POSITIVE_PATTERNS, opt_text)
        is_neg = _any_pattern(DIAGNOSIS_NEGATIVE_PATTERNS, opt_text)
        if is_pos and not is_neg:
            n_pos += 1
        elif is_neg and not is_pos:
            n_neg += 1
        # 两者均命中或均未命中：不计入
    return n_pos, n_neg


def is_treatment_question(
    question: str,
    options: Dict[str, str],
    min_drug_ratio: float = 0.6,
) -> bool:
    """
    判断一道题是否为"治疗性问题"。

    必须同时满足：
      1. 问题主干匹配至少一条 TREATMENT_STEM_PATTERNS
      2. 选项中药物/干预占比 ≥ min_drug_ratio（默认 60%，即 5 项中 3+ 项）
         且诊断/检查选项数不超过 1 个

    参数：
        question:      问题主干文本
        options:       选项字典 {A: text, ...}
        min_drug_ratio: 药物选项最低占比阈值

    返回：
        True 表示是治疗性题目
    """
    # 第一阶段：主干必须询问治疗
    if not _any_pattern(TREATMENT_STEM_PATTERNS, question):
        return False

    # 第二阶段：选项类型过滤
    if not options:
        return False

    n_pos, n_neg = _count_drug_options(options)
    n_total = len(options)

    # 诊断型选项超过 1 个 → 排除（混合题，难以判定冲突类型）
    if n_neg > 1:
        return False

    # 药物/干预比例不足 → 排除
    ratio = n_pos / n_total
    return ratio >= min_drug_ratio


def heuristic_tag(
    question: str,
    options: Dict[str, str],
) -> Tuple[str, bool]:
    """
    v3 启发式分类器，在确认是治疗性问题后进行冲突类型分类。

    分类优先级（高→低）：SC_ABSOLUTE > SC_RELATIVE > TREATMENT_ONLY

    参数：
        question: 问题主干文本（已通过 is_treatment_question 过滤）
        options:  选项字典

    返回：
        (candidate_tag, no_keyword_flag)
        no_keyword_flag = True 表示 SC 触发词仅出现于选项，主干无明显关键词
    """
    options_text = " ".join(options.values()) if isinstance(options, dict) else ""
    full_text = question + " " + options_text

    # SC_ABSOLUTE：过敏 / 妊娠 / 绝对禁忌
    sc_abs_in_stem = _any_pattern(SC_ABS_PATTERNS, question)
    if _any_pattern(SC_ABS_PATTERNS, full_text):
        return "SC_ABSOLUTE_CAND", not sc_abs_in_stem

    # SC_RELATIVE：肾功能 / 肝功能 / 剂量调整
    sc_rel_in_stem = _any_pattern(SC_REL_PATTERNS, question)
    if _any_pattern(SC_REL_PATTERNS, full_text):
        return "SC_RELATIVE_CAND", not sc_rel_in_stem

    # 无特定冲突信号 → 纯治疗题（无约束冲突）
    return "TREATMENT_ONLY", False


def _question_hash(question: str) -> str:
    """基于问题文本的 MD5，用于跨文件去重。"""
    return hashlib.md5(question.strip().lower().encode("utf-8")).hexdigest()


def _extract_disease_query_stub(question: str) -> str:
    """
    从 USMLE 问题中提取疾病主干查询占位符。
    后续由 generate_parametric_prior.py 通过 LLM 精化。
    """
    sentences = re.split(r'(?<=[.!?])\s+', question.strip())
    stub = " ".join(sentences[:2])[:200]
    return f"[STUB] {stub.strip()}"


def load_jsonl(path: Path) -> List[Tuple[int, dict]]:
    """
    加载 jsonl 文件，返回 (行号, 对象) 列表。

    异常：
        ValueError: JSON 解析失败时，附带文件名和行号
    """
    rows: List[Tuple[int, dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                raise ValueError(f"[JSON解析失败] {path}:{i} → {e}") from e
    return rows


def build_candidates(
    input_paths: List[Path],
    pool_factor: int,
    seed: int,
    target_map: Dict[str, int],
    min_drug_ratio: float = 0.6,
) -> Tuple[List[dict], Dict[str, int], Dict[str, int]]:
    """
    构建候选池主逻辑。

    第一阶段：加载所有题目，执行 is_treatment_question() 过滤，
              再执行 heuristic_tag() 分类，MD5 去重。
    第二阶段：按 pool_factor 从每类随机采样。

    参数：
        input_paths:    输入 jsonl 文件列表
        pool_factor:    候选池倍增因子（目标数 × pool_factor）
        seed:           随机种子
        target_map:     各类别目标样本数 {tag: n}
        min_drug_ratio: 药物选项最低占比阈值

    返回：
        (sampled_candidates, full_distribution_counts, filtered_counts)
        filtered_counts: 被 is_treatment_question() 过滤掉的统计
    """
    seen_hashes: set = set()
    all_tagged: Dict[str, List[dict]] = defaultdict(list)
    total_loaded = 0
    dup_count = 0
    non_treatment_count = 0

    for path in input_paths:
        rows = load_jsonl(path)
        total_loaded += len(rows)
        for line_no, obj in rows:
            q = obj.get("question", "")
            options = obj.get("options", {})
            if not isinstance(options, dict) or not q:
                continue

            # 跨文件 MD5 去重
            q_hash = _question_hash(q)
            if q_hash in seen_hashes:
                dup_count += 1
                continue
            seen_hashes.add(q_hash)

            # v3 核心过滤：必须是治疗性问题
            if not is_treatment_question(q, options, min_drug_ratio):
                non_treatment_count += 1
                continue

            tag, no_kw_flag = heuristic_tag(q, options)
            candidate = {
                "candidate_id": f"CAND-{path.stem}-L{line_no:05d}",
                "question": q,
                "options": options,
                "answer_idx": obj.get("answer_idx", ""),
                "answer": obj.get("answer", ""),
                "meta_info": obj.get("meta_info", ""),
                "candidate_tag": tag,
                "no_keyword_flag": no_kw_flag,
                "source_file": str(path),
                "source_split": path.stem,
                "source_line": line_no,
                "parametric_prior_stub": {
                    "disease_only_query": _extract_disease_query_stub(q),
                    "disease_only_response": None,
                    "marginal_bias_confirmed": None,
                    "note": "通过 generate_parametric_prior.py 填充",
                },
            }
            all_tagged[tag].append(candidate)

    # 全量分布统计
    full_dist = {tag: len(items) for tag, items in all_tagged.items()}

    # 第二阶段：按 pool_factor 采样
    random.seed(seed)
    sampled: List[dict] = []
    for tag, n_target in target_map.items():
        pool = all_tagged.get(tag, [])
        n_pool = min(n_target * pool_factor, len(pool))
        if n_pool < n_target:
            print(
                f"[警告] {tag}: 仅找到 {len(pool)} 个治疗候选，"
                f"目标 {n_target}，pool_factor={pool_factor}",
                file=sys.stderr,
            )
        chosen = random.sample(pool, n_pool)
        for i, c in enumerate(chosen, start=1):
            c["candidate_id"] = f"CAND-{tag[:8]}-{i:04d}"
        sampled.extend(chosen)

    # 控制台输出统计
    print("\n" + "=" * 65)
    print(f"加载总行数:              {total_loaded}")
    print(f"跨文件重复剔除:          {dup_count}")
    print(f"非治疗题过滤:            {non_treatment_count}")
    n_passed = sum(full_dist.values())
    print(f"通过治疗过滤的候选数:    {n_passed}")
    print(f"\n治疗候选分布（heuristic_tag 分类后）:")
    for tag in ["SC_ABSOLUTE_CAND", "SC_RELATIVE_CAND", "TREATMENT_ONLY"]:
        n = full_dist.get(tag, 0)
        pct = n / n_passed * 100 if n_passed else 0
        print(f"  {tag:<25}: {n:>5}  ({pct:.1f}%)")
    print(f"\n输出候选池（{pool_factor}× 目标数）:")
    out_dist: Dict[str, int] = defaultdict(int)
    for c in sampled:
        out_dist[c["candidate_tag"]] += 1
    for tag in ["SC_ABSOLUTE_CAND", "SC_RELATIVE_CAND", "TREATMENT_ONLY"]:
        n_no_kw = sum(
            1 for c in sampled
            if c["candidate_tag"] == tag and c["no_keyword_flag"]
        )
        print(
            f"  {tag:<25}: {out_dist.get(tag, 0):>5}"
            f"  (其中 no_keyword_flag=True: {n_no_kw})"
        )
    print(f"\n  总计输出候选:            {len(sampled)}")
    print("=" * 65 + "\n")

    filtered_counts = {"non_treatment": non_treatment_count, "duplicates": dup_count}
    return sampled, full_dist, filtered_counts


def main() -> None:
    p = argparse.ArgumentParser(
        description="MACB 候选样本池构建器 v3（治疗性题目过滤）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        nargs="+",
        metavar="FILE",
        help="一个或多个 MedQA jsonl 文件（test/dev/train/US_qbank）",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="输出候选池 jsonl 路径",
    )
    p.add_argument(
        "--pool-factor",
        type=int,
        default=4,
        metavar="N",
        help="候选池倍增因子（目标数 × N），默认 4",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    p.add_argument(
        "--target-sc-abs",
        type=int,
        default=60,
        help="SC_ABSOLUTE 目标数，默认 60",
    )
    p.add_argument(
        "--target-sc-rel",
        type=int,
        default=30,
        help="SC_RELATIVE 目标数，默认 30",
    )
    p.add_argument(
        "--target-treatment-only",
        type=int,
        default=20,
        help="TREATMENT_ONLY（无冲突对照组）目标数，默认 20",
    )
    p.add_argument(
        "--min-drug-ratio",
        type=float,
        default=0.6,
        help="选项中药物/干预占比下限，默认 0.6（5 选项中至少 3 个）",
    )
    args = p.parse_args()

    input_paths = [Path(f) for f in args.input]
    for ip in input_paths:
        if not ip.exists():
            raise FileNotFoundError(f"[输入文件不存在] {ip}")

    target_map: Dict[str, int] = {
        "SC_ABSOLUTE_CAND":  args.target_sc_abs,
        "SC_RELATIVE_CAND":  args.target_sc_rel,
        "TREATMENT_ONLY":    args.target_treatment_only,
    }

    candidates, _, _ = build_candidates(
        input_paths=input_paths,
        pool_factor=args.pool_factor,
        seed=args.seed,
        target_map=target_map,
        min_drug_ratio=args.min_drug_ratio,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"候选池写出完成 → {out_path}  ({len(candidates)} 条)")


if __name__ == "__main__":
    main()
