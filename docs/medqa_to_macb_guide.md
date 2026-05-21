# MedQA -> MACB 数据集构建指南（对齐 research.md）

本指南严格对齐 `research.md` 第 4 节（MACB v1，80 样本）。

## 1) 目标产物

你最终需要 4 份文件：

- `data/raw/medqa_usmle.jsonl`：原始 MedQA-USMLE
- `data/interim/macb_candidates_v1.jsonl`：候选样本池（自动粗筛）
- `annotations/macb_v1_sheet.csv`：人工标注表
- `data/processed/macb_v1.jsonl`：最终 gold benchmark

## 2) 目录规范

已创建目录：

- `data/raw`
- `data/interim`
- `data/processed`
- `annotations`
- `scripts`

## 3) 先准备 MedQA 原始数据

把 MedQA-USMLE 的 `jsonl` 文件放到：

- `data/raw/medqa_usmle.jsonl`

脚本假设字段至少有：

- `question`
- `options`（字典，键通常为 A/B/C/D）
- `answer_idx`（正确选项键）

## 4) 运行数据校验

```bash
python3 scripts/validate_medqa.py --input data/raw/medqa_usmle.jsonl
```

通过标准：

- `异常计数 = 0`
- `answer_idx` 都在 `options` keys 内

## 5) 生成 MACB 候选池（按 v1 配比）

```bash
python3 scripts/build_macb_candidates.py \
  --input data/raw/medqa_usmle.jsonl \
  --output data/interim/macb_candidates_v1.jsonl
```

当前自动配比（与你方案一致）：

- `SC_ABSOLUTE_CAND`: 35
- `SC_RELATIVE_CAND`: 15
- `FC_CAND`: 20
- `MIXED_OR_OTHER`: 10

注意：这一步是关键词启发式粗筛，不是最终标注。

## 6) 导出人工标注表

```bash
python3 scripts/export_annotation_sheet.py \
  --input data/interim/macb_candidates_v1.jsonl \
  --output annotations/macb_v1_sheet.csv
```

标注者需要填的核心列：

- `patient_profile_json`
- `gold_admissible_set_json`
- `gold_per_action_status_json`
- `gold_scsr_needed`
- `gold_scsr_query`

## 7) 标注规则（最小执行版）

每个 action 只能是三类之一：

- `INADMISSIBLE`（SC_ABSOLUTE）
- `CONDITIONALLY_ADMISSIBLE`（SC_RELATIVE）
- `ADMISSIBLE`（NO_CONFLICT 或 FC）

并且每个非 ADMISSIBLE action 必须有 `scope_basis`（可追溯证据依据）。

## 8) 双标一致性（Kappa）

你可以让两位标注者各交一份 csv（至少 30 条重叠样本），并提供列：

- `sample_id`
- `sc_fc_label`（例如：SC 或 FC）

计算：

```bash
python3 scripts/compute_kappa.py \
  --ann1 annotations/reviewer1.csv \
  --ann2 annotations/reviewer2.csv \
  --sample-col sample_id \
  --label-col sc_fc_label
```

目标：`kappa > 0.75`。

## 9) 你和团队的推荐分工

- 你：完成第 3-6 步，整理候选池和标注表
- 医学生/住院医生：重点审核 `SC_ABSOLUTE` 子集（35 条优先）
- 第二标注者：完成 30 条重叠标注用于 Kappa
- 你：汇总成 `data/processed/macb_v1.jsonl`

## 10) 常见坑

- 只标注“正确答案”，不做 per-action 标注：会导致后续无法算 CRR/SDR
- `gold_scsr_query` 用模型自动生成：会引入 oracle 污染
- 把 FC/SC 混成一个标签：会丢掉你论文最核心的不对称失效验证

