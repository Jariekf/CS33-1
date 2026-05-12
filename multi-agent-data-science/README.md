# Multi-Agent Data Science Pipeline

A LangGraph-powered multi-agent pipeline that automates the full machine learning workflow — from raw CSV to trained and evaluated models — using LLM agents for every reasoning step.

---

## Pipeline Overview

```
init_run            (timestamped workspace under artifacts/workspace/)
    |
    +--[single CSV]--> split_agent  (infer_task + split_dataset tool → train/test paths on disk)
    |
    +--[--train + --test already set]--> (skip split_agent)
    |
agent_manager       (profile_dataset on TRAIN only; infer_task if task_type/target not set;
    |                create_cleaning_plan for train + for test)
    |
set_clean_train -> data_clean_train -> set_clean_test -> data_clean_test
    |
eda_agent           (EDA, correlation, charts, select 3 candidate models)
    |
feature_agent       (encodes & scales — one loop iteration per candidate model)
    |
model_agent         (trains & evaluates — one loop iteration per candidate model)
    |
business_translator (optional: KB RAG + arXiv → Markdown business report, if best_model exists)
```

CLI data source (see `main.py --help`):

- Single file: `--data path.csv` (split_agent runs inside the graph).
- Pre-split files: `--train train.csv --test test.csv` (skips split_agent).
- Optional explicit supervision metadata: `--task-type classification|regression --target <column>` (skips infer_task inside agent_manager).
- Dataset description as text: `--description "..."` **or** `--description-file path/to/file.txt` (UTF-8 `.txt` only).

Pipeline diagrams: enable `--show-graph` → writes `artifacts/graphs/full_pipeline.mmd` and `full_pipeline.png`.

All intermediate files and reports are written to the run's workspace directory (`artifacts/workspace/YYYY-MM-DD-HH-MM/`), so parallel or back-to-back runs never overwrite each other.

---

## Requirements

| Component | Version |
|---|---|
| Python | 3.12+ |
| langgraph | 1.1.10 |
| langchain-core | 1.3.2 |
| langchain-openai | 1.2.1 |
| langsmith | 0.8.0 |
| openai | 2.33.0 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| python-dotenv | 1.2.2 |
| pytest | 9.0.3 |

---

## Installation

```bash
git clone https://github.com/your-org/multi-agent-data-science.git
cd multi-agent-data-science

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

Copy the example env file and fill in at least one LLM provider key:

```bash
cp .env.example .env
```

`.env` reference — only the provider(s) you plan to use need to be set:

```dotenv
# ── DeepSeek ──────────────────────────────────────────────
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# ── OpenAI ────────────────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# ── Zhipu (GLM) ───────────────────────────────────────────
ZHIPU_API_KEY=...
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL=glm-4-plus

# ── Google Gemini ─────────────────────────────────────────
GEMINI_API_KEY=...
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_MODEL=gemini-2.5-pro

GEMINI_FLASH_API_KEY=...
GEMINI_FLASH_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_FLASH_MODEL=gemini-2.0-flash

# ── Qwen (Alibaba) ────────────────────────────────────────
QWEN_API_KEY=...
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

# ── MiniMax ───────────────────────────────────────────────
MINIMAX_API_KEY=...
MINIMAX_BASE_URL=https://api.minimax.chat/v1
MINIMAX_MODEL=MiniMax-Text-01

# ── Kimi (Moonshot) ───────────────────────────────────────
KIMI_API_KEY=...
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=moonshot-v1-8k
```

The active LLM is selected in `core/my_llm.py`. By default the pipeline uses `llm_deepseek`; swap the import in each agent file to use a different provider.

### Business Action Translator (BAT) — McKinsey RAG

After `model_agent`, the **business_translator** subgraph generates `reports/business_report.md` using:

- Retrieval from Chroma collection **`mckinsey_kb`** (three buckets: `mckinsey`, `equity_research`, `textbook`)
- **arXiv** fallback when bucket hits fall below `MCKINSEY_MIN_HITS`

**First-time setup (one-shot ingest):**

```bash
# From multi-agent-data-science/ — requires embedding API (see RAG_* in .env.example)
python scripts/build_mckinsey_kb.py
```

This reads `data/mckinsey_kb/manifest.lock.yaml`, embeds `data/mckinsey_kb/raw/*.md`, and writes vectors under `artifacts/chroma_rag/` (override with `RAG_CHROMA_DIR`). Copy variables from `.env.example` (`RAG_EMBEDDING_*`, `MCKINSEY_*`, `ARXIV_MAX_RESULTS`, etc.). Details: [`rag/README.md`](rag/README.md).

---

## Quick Start

```bash
python main.py \
  --data datasets/bank-full.csv \
  --description "Binary classification: predict whether a client subscribes to a term deposit (target column: y)" \
  --request "Please run the full pipeline — clean the data, split it, perform EDA, engineer features, train models, and select the best classifier"

# Or load the same context from a UTF-8 .txt file:
python main.py \
  --data datasets/bank-full.csv \
  --description-file datasets/dataset_description.txt \
  --request "Please run the full pipeline and select the best classifier"
```

**Required arguments**

| Argument | Description |
|---|---|
| `--data` *or* `--train` + `--test` | CSV path(s) — see `--help` |
| `--description` *or* `--description-file` | Dataset context (inline text or `.txt` UTF-8 path) |
| `--request` | Natural-language instruction |
| *(optional)* `--target` + `--task-type` | Skips `infer_task` in Agent Manager when both set |
| *(optional)* `--show-graph` | Save `artifacts/graphs/full_pipeline.{mmd,png}` |
| *(optional)* `--recursion-limit` | LangGraph max steps (default `200`) |

### Use as a library

```python
from main import run_pipeline

result = run_pipeline(
    data_path="datasets/bank-full.csv",
    description="Binary classification: predict term deposit subscription (target: y)",
    user_request="Run the full pipeline and select the best classifier",
)

print(result["best_model"])
# {'model_name': 'XGBoost', 'metrics': {'accuracy': 0.91, 'f1_weighted': 0.90}, ...}
```

---

## Modular Agent Execution

Every agent can be run **standalone** without invoking the full pipeline. This is useful when you are iterating on a single stage and don't want to re-run everything upstream.

Each agent file contains a `__main__` block and each test file contains a matching `_run_xxx()` helper. Run them directly and interact with the agent in the terminal:

```bash
# Run only the Agent Manager (dataset profiling + cleaning plan)
python agent/agent_manager.py

# Run only the EDA Agent (given a pre-split training CSV)
python tests/test_eda_agent.py

# Run only the Feature Agent (given a training CSV + plan)
python tests/test_feature_agent.py

# Run only the Model Agent (given feature-engineered CSVs)
python tests/test_model_agent.py

# Run only the full pipeline interactively (streaming output)
python tests/test_full_pipeline.py
```

These entry points stream every agent message to the terminal and print `[stage]` transitions so you can see exactly what the LLM decided at each step.

---

## Testing and Debugging

The test suite is split into two layers so you can iterate quickly without an API key:

### Pure-function tests (no API key needed)

These cover graph structure, routing logic, and tool behaviour using synthetic fixtures in `tests/testenv/`. They run in seconds.

```bash
# Run everything
python -m pytest tests/ -v

# Focus on one agent
python -m pytest tests/test_feature_agent.py -v

# Run a single test case
python -m pytest tests/test_model_agent.py::test_train_model_regression -v
```

**What each test file covers:**

| File | Covers |
|---|---|
| `test_profile_dataset.py` | `profile_dataset` tool — shape, missing values, type detection |
| `test_agent_manager.py` | Graph structure, routing, tool outputs |
| `test_data_clean_agent.py` | Graph structure, routing, all cleaning tools |
| `test_full_pipeline.py` | Full graph nodes & edges, split node, pipeline routing |
| `test_eda_agent.py` | Graph structure, model selection validation, EDA tools |
| `test_feature_agent.py` | Graph structure, encode/scale tools, state updates |
| `test_model_agent.py` | Graph structure, train/evaluate tools, model routing |

### LLM invoke tests (API key required)

Each test file contains one end-to-end `test_xxx_invoke` test that calls the real LLM. These are skipped automatically when no key is present, and can be run explicitly once your key is configured:

```bash
# Example: run the EDA agent end-to-end against the test dataset
python -m pytest tests/test_eda_agent.py::test_eda_agent_invoke -v -s
```

### Typical debugging workflow

If you modify an agent and want to verify it still behaves correctly:

1. Run its test file with `-v` to check all pure-function assertions pass.
2. Run `python tests/test_xxx.py` to invoke the agent interactively with the testenv CSV — inspect the streamed messages and `[stage]` output.
3. Check `artifacts/workspace/<timestamp>/reports/` for the agent's saved report to see exactly what the LLM produced.

---

## Run Artifacts

Every pipeline run writes all its output to a dedicated timestamped directory so nothing is ever overwritten:

```
artifacts/workspace/2026-05-09-14-30/
|
|-- reports/                          # Markdown report saved after each agent finishes
|   |-- agent_manager_report.md       #   dataset profile, task inference, cleaning plan
|   |-- data_clean_agent_report.md    #   cleaning actions taken, rows dropped/fixed
|   |-- eda_agent_report.md           #   EDA findings, model selection rationale
|   |-- feature_agent_report.md       #   encoding/scaling decisions per model
|   |-- model_agent_report.md         #   training scores and evaluation metrics
|   `-- business_report.md           #   BAT: McKinsey-style business insight report
|
|-- features/                         # Feature-engineered datasets (one per model)
|   |-- encoders/                     #   fitted LabelEncoders / OneHotEncoders (.pkl)
|   |-- LogisticRegression_train_featured.csv
|   |-- RandomForest_train_featured.csv
|   `-- XGBoost_train_featured.csv
|
|-- models/                           # Trained model files
|   |-- LogisticRegression_model.pkl
|   |-- RandomForest_model.pkl
|   `-- XGBoost_model.pkl
|
|-- bank-full_cleaned_train.csv       # Post-cleaning train split
`-- bank-full_cleaned_test.csv        # Post-cleaning test split
```

You can inspect any intermediate result without re-running the pipeline — open the reports for the LLM's reasoning, load the featured CSVs to inspect what encoding was applied, or load a `.pkl` model directly for inference.

---

## Directory Structure

```
|-- main.py                        # CLI entry point
|-- requirements.txt
|-- .env                           # API keys (not committed)
|-- scripts/
|   `-- build_mckinsey_kb.py       # One-shot ingest raw/*.md → Chroma mckinsey_kb
|-- data/
|   `-- mckinsey_kb/               # McKinsey corpus manifest + raw markdown (BAT RAG)
|-- rag/                           # Chroma RAG pipeline (embeddings, ingest, retrieve)
|-- agent/
|   |-- agent_manager.py           # Dataset profiling, task inference, cleaning plan
|   |-- data_clean_agent.py        # Data cleaning execution
|   |-- eda_agent.py               # EDA, feature engineering plan, model selection
|   |-- feature_agent.py           # Feature engineering (one pass per candidate model)
|   |-- model_agent.py             # Train + evaluate (one pass per candidate model)
|   `-- business_translator_agent.py  # BAT: RAG + business Markdown report
|-- core/
|   |-- global_state.py            # GlobalState TypedDict (shared across all nodes)
|   |-- graph.py                   # Pipeline assembly (build_full_pipeline)
|   |-- dataset_split.py           # Split node (pure function)
|   |-- dataset_store.py           # In-memory dataset cache (dataset_id -> DataFrame)
|   |-- report_writer.py           # Saves per-agent reports to run_dir
|   |-- my_llm.py                  # LLM instances (one per provider)
|   `-- env_utils.py               # PROJECT_ROOT, env var loading
|-- tools/
|   |-- agentmanager_tools/        # profile_dataset, infer_task, create_cleaning_plan
|   |-- dataclean_tools/           # handle_missing, remove_duplicates, fix_types, ...
|   |-- eda_tools/                 # profile_training_data, correlation_matrix, select_candidate_models
|   |-- feature_tools/             # encode_categorical, scale_features, save_featured_dataset
|   |-- model_tools/               # train_model, evaluate_model
|   `-- business_tools/            # arxiv_tool, file_reader (BAT)
|-- tests/
|   |-- conftest.py                # Shared path constants
|   |-- testenv/                   # Synthetic CSV fixtures (no external data needed)
|   |   |-- raw/sample.csv
|   |   |-- clean/sample_cleaned.csv
|   |   |-- split/sample_train.csv, sample_test.csv
|   |   `-- feature/{Model}_train/test_featured.csv
|   `-- test_*.py
`-- artifacts/
    `-- workspace/                 # Per-run output directories (git-ignored)
```

---

## Candidate Model Pools

The EDA Agent selects exactly **3 models** from the relevant pool based on dataset characteristics.

**Classification:** `LogisticRegression`, `RandomForest`, `XGBoost`, `SVM`, `KNN`

**Regression:** `LinearRegression`, `Ridge`, `RandomForestRegressor`, `XGBoostRegressor`, `SVR`
