# MARC: Medical Admissibility Conflict-aware RAG

> **论文**：*Decomposed Conditional Retrieval for Scope-Conflict-Aware Medical RAG*  
> **会议**：BIBM 2026（投稿中）

---

## 概述

现有 RAG 知识冲突框架存在一个未被指出的结构性盲点：它们将所有冲突建模为**同一条件概率空间内的值域竞争**（即"谁的权重更高"）。但医学场景中最危险的冲突类型是**条件空间本身的不相容**——证据所适用的患者群体与当前患者的约束条件不一致。

MARC（Medical Admissibility Conflict-aware RAG）基于以下核心区分构建：

| 操作层次 | 冲突类型 | 数学含义 | 处理方式 |
|---|---|---|---|
| **支撑集操作** | SC（Scope Conflict） | 确定哪些 action 对患者 $q$ 可行 | κ=0 物理排除（不可补偿） |
| **值域操作** | FC（Factual Conflict） | 在可行 action 集内估计效用 | 证据加权仲裁 |

核心算法 **DCR（Decomposed Conditional Retrieval）** 将查询分解为疾病查询 $D_q$ 和患者约束 $C_q$，通过乘法结构保证 SC_ABSOLUTE 文档在物理上不进入生成上下文：

$$\text{score}_{\text{MARC}}(q, d) = \text{sim}(D_q, d) \cdot \kappa(C_q, \pi_d)$$

---

## 项目结构

```
Conflict-RAG/
├── src/                    # MARC 核心系统
│   ├── types.py            # 共享数据类型（PatientConstraint, MARCOutput 等）
│   ├── pipeline.py         # 端到端 Pipeline + build_marc_pipeline() 工厂函数
│   ├── retriever.py        # HybridRetriever（BM25 + FAISS，RRF 融合）
│   ├── query_decomposer.py # Module 0：q → (D_q, C_q)
│   ├── kappa_scorer.py     # κ(C_q, π_d) 计算（规则层优先 + LLM fallback）
│   ├── dcr.py              # Stage 2 DCR 重排序（κ=0 物理排除）
│   ├── scsr.py             # Stage 3 SCSR 补充检索（gap-filling）
│   ├── fc_handler.py       # FC 值域冲突仲裁
│   ├── generator.py        # Scope-anchored 生成（含 SCOPE BIAS WARNING）
│   ├── verifier.py         # SLR 归因校验
│   └── llm_client.py       # 统一 LLM 客户端（Anthropic + OpenAI 兼容）
├── baselines/              # 对比基线系统
│   ├── standard_rag.py     # 标准混合 RAG（无 scope filtering）
│   ├── bm25_only.py        # 纯 BM25 RAG
│   ├── dense_only.py       # 纯 Dense RAG
│   ├── no_retrieval.py     # 无检索（参数记忆基准）
│   ├── picos_rag.py        # PICOs-RAG 风格
│   ├── marc_no_dcr.py      # 消融：无 DCR
│   └── marc_no_scsr.py     # 消融：无 SCSR
├── eval/                   # 评估框架
│   ├── metrics.py          # CRR / SDR / AEC / SLR 四指标 + bootstrap CI
│   ├── evaluate.py         # 批量评估入口
│   └── result_analysis.py  # LaTeX 表格生成 + 分层分析
├── experiments/
│   ├── config.py           # ExperimentConfig 统一配置
│   ├── run_experiments.py  # 统一实验入口
│   └── _marc_wrapper.py    # MARC pipeline 适配器
├── scripts/
│   ├── index_textbooks.py  # 教材切分 + BM25 + FAISS 索引构建
│   ├── build_macb_final.py # MACB benchmark 构建
│   └── compute_kappa.py    # 标注一致性（Cohen's κ）计算
├── data/
│   ├── macb_treatment_v5.jsonl  # MACB benchmark（最新版，231 个样本）
│   └── index/              # 构建后的索引目录（见下方步骤）
├── annotations/            # 人工标注 CSV
├── docs/                   # 设计文档
└── .env.example            # LLM 配置模板
```

---

## 环境要求

- Python 3.10+
- 至少 8GB 内存（FAISS 向量索引约 64MB，BM25 索引约 200MB）
- LLM API 访问（支持 Anthropic / OpenAI 兼容接口）

**安装依赖**：

```bash
pip install anthropic openai rank-bm25 faiss-cpu sentence-transformers \
            numpy scipy tqdm python-dotenv
```

> 若使用 GPU 加速 FAISS：将 `faiss-cpu` 替换为 `faiss-gpu`。

---

## 快速开始

### 第一步：配置 LLM

```bash
cp .env.example .env
# 编辑 .env，填入 API key 和模型名
```

`.env` 支持以下配置方案：

```bash
# 方案 A：Anthropic（默认）
ANTHROPIC_API_KEY=sk-ant-xxxx
LLM_MODEL=claude-sonnet-4-6

# 方案 B：任意 OpenAI 兼容接口（GPT / DeepSeek / GLM / 本地 Ollama）
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

# Embedding 模型（本地运行，无需 API）
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 第二步：准备教材数据

将英文医学教材（PDF 或 TXT 格式）放入 `data_clean/textbooks/en/`，然后构建索引：

```bash
python3 scripts/index_textbooks.py \
    --textbook-dir data_clean/textbooks/en \
    --output-dir   data/index \
    --chunk-size   400 \
    --chunk-overlap 50
```

构建完成后目录结构如下：

```
data/index/
├── chunks.jsonl           # 所有文本块（含元数据）
├── chunk_ids.json         # FAISS 索引 ID 映射表
├── bm25/
│   └── bm25_index.pkl     # BM25 索引
└── dense/
    └── faiss.index        # FAISS IndexFlatIP（余弦相似度）
```

> **注**：论文实验使用 18 本英文医学教材（Harrison's、Robbins、Nelson 等），共 43,238 个文本块。受版权限制，本仓库不分发教材文件，请自行准备。若需复现 MACB 构建流程，原始 MedQA 问题数据可从 [jind11/MedQA](https://github.com/jind11/MedQA) 下载。

### 第三步：运行实验

```bash
# 快速调试（5 个样本，2 个系统）
python3 experiments/run_experiments.py \
    --limit 5 --systems marc standard_rag

# 完整实验（所有系统，MACB benchmark v5，231 个样本）
python3 experiments/run_experiments.py

# 仅跑 SC_ABSOLUTE 子集
python3 experiments/run_experiments.py --tag SC_ABSOLUTE
```

结果保存至 `results/`：

```
results/
├── all_systems_metrics.json   # 所有系统指标（含 bootstrap CI）
└── all_systems_summary.txt    # 可读摘要表格
```

### 第四步：分析结果

```bash
# 生成 LaTeX 表格
python3 eval/result_analysis.py \
    --results results/all_systems_metrics.json \
    --output  results/table.tex

# 分层分析（按冲突类型）
python3 eval/result_analysis.py \
    --results results/all_systems_metrics.json \
    --stratify conflict_type
```

---

## MACB Benchmark

**MACB（Medical Admissibility Conflict Benchmark）** 是本文构建的第一个以 *per-action admissibility*（逐 action 可行性）为金标准的医学治疗推荐评测集。

| 子集 | 样本数 | 说明 |
|---|---|---|
| SC_ABSOLUTE | 75 | 绝对禁忌（κ=0，physical exclusion） |
| SC_RELATIVE | 78 | 相对禁忌（κ∈(0,1)，降权 + action 变换） |
| FC | 3 | 真实多指南事实性冲突 |
| NO_CONFLICT | 75 | 无冲突对照组 |
| **总计** | **231** | |

**gold_per_action_status 枚举值**：

```
ADMISSIBLE              # 对当前患者可行
CONDITIONALLY_ADMISSIBLE # 可行但需剂量/方案调整（SC_RELATIVE）
INADMISSIBLE_ABS        # 绝对禁忌（κ=0）
NOT_INDICATED           # 非适应证（非禁忌，但不推荐）
NOT_APPLICABLE          # 选项与问题无关
```

数据集文件：`data/macb_treatment_v5.jsonl`（每行一个 JSON 样本）。

---

## 评估指标

| 指标 | 方向 | 定义 |
|---|---|---|
| **CRR**（Contraindicated Recommendation Rate） | ↓ 越低越好 | 系统推荐了 INADMISSIBLE_ABS action 的比例（仅 SC_ABSOLUTE 子集） |
| **SDR**（SC Detection Recall） | ↑ 越高越好 | Scope Conflict 被正确识别的召回率 |
| **AEC**（Alternative Evidence Coverage） | ↑ 越高越好 | 替代治疗方案有对应检索证据的覆盖率 |
| **SLR**（Source Leakage Rate） | ↓ 越低越好 | 生成答案中引用了 κ=0 来源的比例 |

所有指标均报告均值 ± 标准差，以及 95% bootstrap 置信区间。

---

## 系统列表

实验对比以下 8 个系统：

| 系统名 | 说明 |
|---|---|
| `marc` | 完整 MARC（DCR + SCSR） |
| `marc_no_dcr` | 消融：去掉 DCR，保留其余模块 |
| `marc_no_scsr` | 消融：去掉 SCSR，保留 DCR |
| `standard_rag` | 标准混合 RAG（BM25+Dense，无 scope filtering） |
| `bm25_only` | 纯 BM25 检索 |
| `dense_only` | 纯 Dense 检索 |
| `no_retrieval` | 无检索，直接 LLM 参数记忆回答 |
| `picos_rag` | PICO 框架改写查询的 RAG |

---

## 单样本推理（API 调用示例）

```python
from src.pipeline import build_marc_pipeline

pipeline = build_marc_pipeline(
    index_dir="data/index",
    cache_dir="data/cache",
)

result = pipeline.run(
    query="A 45-year-old patient with ABRS and penicillin anaphylaxis. "
          "Which antibiotic is recommended?\n\nOptions: "
          "A. Amoxicillin  B. Doxycycline  C. Azithromycin  D. Levofloxacin"
)

print(result.generated_answer)
print(result.per_action_status)   # {'Amoxicillin': 'INADMISSIBLE_ABS', ...}
print(result.metrics)             # 延迟、token 数、κ=0 排除数等
```

---

## 注意事项

**代理设置**：部分国内 API（DeepSeek、MiniMax 等）无需代理，`run_experiments.py` 已自动清除代理环境变量。若 Anthropic API 需要代理，请在 Shell 中显式设置，而非通过 socks 代理（httpx 不兼容）。

**Think chain**：使用 MiniMax 等带 `<think>` 的模型时，`llm_client.py` 会自动剥离 think chain。`max_tokens` 已设为 8000 以防截断。

**缓存**：`QueryDecomposer` 和 `KappaScorer` 均有本地磁盘缓存（`data/cache/`），同一 query+model 组合只调用 LLM 一次。

---

## 引用

```bibtex
@inproceedings{marc2026bibm,
  title     = {Decomposed Conditional Retrieval for Scope-Conflict-Aware Medical RAG},
  booktitle = {Proceedings of the 2026 IEEE International Conference on
               Bioinformatics and Biomedicine (BIBM)},
  year      = {2026},
  note      = {Under review}
}
```

---

## 许可证

MIT License。MACB benchmark 数据（`data/macb_*.jsonl`）基于公开医学指南构建，仅限学术研究使用。
