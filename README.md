# Multi-Agent Data Science Pipeline

A LangGraph-powered multi-agent pipeline that automates the full machine learning workflow — from raw CSV to trained and evaluated models — using LLM agents for every reasoning step.

---

## Pipeline Overview

```
init_run            (creates timestamped workspace under artifacts/workspace/)
    |
agent_manager       (profiles dataset, infers task type & target column, produces cleaning plan)
    |
data_clean_agent    (executes cleaning: missing values, outliers, type fixes)
    |
split               (train / test split, stratified for classification)
    |
eda_agent           (correlation analysis, distribution profiling, selects 3 candidate models)
    |
feature_agent       (encodes & scales features — one loop iteration per candidate model)
    |
model_agent         (trains & evaluates — one loop iteration per candidate model)
```

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

---

## Quick Start

**Linux / macOS (bash / zsh):**

```bash
python main.py \
  --data datasets/bank-full.csv \
  --description "Binary classification: predict whether a client subscribes to a term deposit (target column: y)" \
  --request "Please run the full pipeline — clean the data, split it, perform EDA, engineer features, train models, and select the best classifier"
```

**Windows PowerShell** — use a backtick `` ` `` for line continuation (no trailing spaces), or put everything on one line. The bash `\` continuation does **not** work in PowerShell:

```powershell
python main.py `
  --data datasets/bank-full.csv `
  --description "Binary classification: predict whether a client subscribes to a term deposit (target column: y)" `
  --request "Please run the full pipeline — clean the data, split it, perform EDA, engineer features, train models, and select the best classifier"
```

**Windows CMD** — use `^` for line continuation, or run as a single line:

```bat
python main.py ^
  --data datasets/bank-full.csv ^
  --description "Binary classification: predict whether a client subscribes to a term deposit (target column: y)" ^
  --request "Please run the full pipeline — clean the data, split it, perform EDA, engineer features, train models, and select the best classifier"
```

**All arguments are required:**

| Argument | Description |
|---|---|
| `--data` | Path to the raw CSV dataset |
| `--description` | Dataset context — task type, target column, domain background |
| `--request` | Natural-language instruction given to the first agent |
| `--show-graph` | *(optional)* Save pipeline graph as `artifacts/full_pipeline.png` |
| `--recursion-limit` | *(optional)* LangGraph max recursion steps (default `200`) |

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
|   `-- model_agent_report.md         #   training scores and evaluation metrics
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
|-- agent/
|   |-- agent_manager.py           # Dataset profiling, task inference, cleaning plan
|   |-- data_clean_agent.py        # Data cleaning execution
|   |-- eda_agent.py               # EDA, feature engineering plan, model selection
|   |-- feature_agent.py           # Feature engineering (one pass per candidate model)
|   `-- model_agent.py             # Train + evaluate (one pass per candidate model)
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
|   `-- model_tools/               # train_model, evaluate_model
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
