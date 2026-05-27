# Multi-Agent Data Science Pipeline

A LangGraph-powered multi-agent pipeline that automates the full machine learning workflow — from raw CSV to trained and evaluated models — using LLM agents for every reasoning step.

---

## Pipeline Overview

```
init_run
    |
    +--[single CSV]-----------> Agent_Manager_split  (infer_task + split_dataset → train/test CSVs on disk)
    |
    +--[--train + --test set]--> (skip Agent_Manager_split)
    |
agent_manager       (profile dataset; build RAP context; infer task if needed; create cleaning plans)
    |
set_clean_train -> data_clean_train -> set_clean_test -> data_clean_test
    |
eda_agent           (EDA, correlation, charts; select 3 candidate models and evaluation metrics)
    |
feature_agent       (encode & scale — one loop iteration per candidate model)
    |
model_agent         (HPO, train, evaluate — one loop iteration per candidate model)
    |
business_translator (McKinsey-style business report via KB RAG + arXiv, if best_model exists)
```

CLI data source (see `main.py --help`):

- **Single file**: `--data path.csv` — `Agent_Manager_split` infers the task and splits internally.
- **Pre-split files**: `--train train.csv --test test.csv` — skips `Agent_Manager_split`.
- **Explicit supervision**: `--task-type classification|regression --target <column>` — skips `infer_task` inside Agent Manager when both are set.
- **Dataset description**: `--description "..."` **or** `--description-file path/to/file.txt` (UTF-8 `.txt` only).

Pipeline diagrams: enable `--show-graph` → writes `artifacts/graphs/full_pipeline.mmd` and `full_pipeline.png`.

All intermediate files and reports are written to a timestamped run directory (`artifacts/workspace/YYYY-MM-DD-HH-MM/`), so parallel or back-to-back runs never overwrite each other.

---

## Agent Responsibilities & Report Boundaries

Each agent owns a **specific deliverable** and may not encroach on others.
**Business recommendations are produced exclusively by the Business Action Translator (BAT)**; every other agent's report is filtered before disk write to strip any business-advice sections that leak through.

| Agent | Responsibility | Report file | May write business advice? |
|---|---|---|---|
| `supervisor` | Gating checks between stages | `supervisor_report.md` | No |
| `agent_manager` | Dataset profile + Manager RAP (requirement parse & retrieval) + cleaning plan | `agent_manager_report.md` | No |
| `Agent_Manager_split` | Task inference + train/test split | `Agent_Manager_split_report.md` | No |
| `data_clean_agent` | Type / missing / outlier / duplicate cleaning | `data_clean_agent_report.md` | No |
| `eda_agent` | Profiling, correlation, charts, candidate model & metric selection | `eda_agent_report.md` | No |
| `feature_agent` | Per-model feature engineering (encode + scale) | `feature_agent_report.md` | No |
| `model_agent` | HPO, training, evaluation (technical conclusions only) | `model_agent_report.md` | No |
| `business_translator` (BAT) | McKinsey-style business strategy report | `business_insight_report.md` / `.html` / `.pdf` | **Yes — sole owner** |

### How the boundary is enforced

Three layers, all in [`multi-agent-data-science/`](.):

1. **Context isolation at the source.** EDA and Model agents trim the shared `state["messages"]` down to their own tool-call chain before invoking the LLM (`_trim_messages_to_current_round` in `agent/eda_agent.py`, `_trim_messages_to_modeling_round` in `agent/model_agent.py`). This blocks the original `user_request` HumanMessage and `agent_manager`'s RAP business-strategy AIMessage from polluting their context.
2. **Prompt-level prohibitions.** The EDA / Model agent system prompts list forbidden heading titles (e.g. `Business Recommendation(s)`, `Recommended Actions`, `Next Step(s)`) and state that business advice belongs to BAT.
3. **Save-time filter.** `core/report_writer.save_agent_report` calls `_strip_non_bat_recommendations` on the final AI message for every non-BAT agent. Any matching heading is removed along with its section (until the next heading of equal or higher level).

Run the safety net's unit tests:

```bash
pytest tests/test_report_writer_filter.py tests/test_agent_message_trim.py -q
```

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

# Or load the dataset description from a UTF-8 .txt file:
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
| `test_data_clean_tools.py` | Individual cleaning tool correctness |
| `test_data_clean_plan_normalize.py` | Cleaning plan normalisation logic |
| `test_agentmanager_dataclean_pipeline.py` | Agent Manager → data clean pipeline integration |
| `test_dataset_split.py` | Train/test split tool |
| `test_full_pipeline.py` | Full graph nodes & edges, split node, pipeline routing |
| `test_eda_agent.py` | Graph structure, model selection validation, EDA tools |
| `test_feature_agent.py` | Graph structure, encode/scale tools, state updates |
| `test_model_agent.py` | Graph structure, train/evaluate tools, model routing |
| `test_supervisor_cleaning_guard.py` | Supervisor gating logic for cleaning stages |
| `test_report_writer_filter.py` | Business-advice strip filter for non-BAT agents |
| `test_agent_message_trim.py` | Message trimming for EDA/Model agents |
| `test_bat.py` | Business Action Translator graph and tools |

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
|-- train_{dataset}.csv               # Raw train split (output of Agent_Manager_split)
|-- test_{dataset}.csv                # Raw test split
|
|-- clean/                            # Post-cleaning datasets
|   |-- cleaned_train_{dataset}.csv
|   `-- cleaned_test_{dataset}.csv
|
|-- plots/                            # EDA visualisations
|   |-- corr_heatmap.png
|   |-- rf_importance.png
|   |-- numeric_distributions.png
|   |-- categorical_distributions.png
|   |-- numeric_by_target_density.png
|   |-- categorical_by_target_rate.png
|   |-- class_balance.png
|   `-- outlier_boxplots.png
|
|-- features/                         # Feature-engineered datasets (one set per candidate model)
|   |-- encoders/                     #   fitted LabelEncoders / OneHotEncoders (.pkl)
|   |-- scalers/                      #   fitted StandardScalers / MinMaxScalers (.pkl)
|   |-- {Model}_train_featured.csv
|   `-- {Model}_test_featured.csv
|
`-- reports/                          # Markdown report saved after each agent finishes
    |-- agent_manager_report.md       #   dataset profile, task inference, cleaning plan
    |-- Agent_Manager_split_report.md #   inferred task type and split summary
    |-- data_clean_agent_report.md    #   cleaning actions taken, rows dropped / fixed
    |-- eda_agent_report.md           #   EDA findings, model selection rationale
    |-- feature_agent_report.md       #   encoding / scaling decisions per model
    |-- model_agent_report.md         #   training scores and evaluation metrics
    |-- supervisor_report.md          #   supervisor gating decisions
    `-- business_report.md            #   BAT: McKinsey-style business insight report
```

You can inspect any intermediate result without re-running the pipeline — open the reports for the LLM's reasoning, load the featured CSVs to inspect what encoding was applied, or load a `.pkl` encoder/scaler directly for inference.

---

## Directory Structure

```
|-- main.py                           # CLI entry point
|-- requirements.txt
|-- .env                              # API keys (not committed)
|
|-- agent/
|   |-- agent_manager.py             # Dataset profiling, task inference, cleaning plan
|   |-- agent_manager_split.py       # Task inference + train/test split (Agent_Manager_split node)
|   |-- supervisor.py                # Supervisor gating node (between pipeline stages)
|   |-- data_clean_agent.py          # Data cleaning execution
|   |-- eda_agent.py                 # EDA, correlation analysis, model & metric selection
|   |-- feature_agent.py             # Feature engineering (one pass per candidate model)
|   |-- model_agent.py               # HPO, train & evaluate (one pass per candidate model)
|   `-- business_translator_agent.py # BAT: RAG + business Markdown report
|
|-- core/
|   |-- global_state.py              # GlobalState TypedDict (shared across all nodes)
|   |-- graph.py                     # Full pipeline assembly (build_full_pipeline)
|   |-- dataset_split.py             # Train/test split logic (pure function)
|   |-- dataset_store.py             # In-memory dataset cache (dataset_id → DataFrame)
|   |-- report_writer.py             # Saves per-agent Markdown reports to run_dir
|   |-- timer_utils.py               # Per-agent wall-clock timing helpers
|   |-- my_llm.py                    # LLM instances (one per provider)
|   `-- env_utils.py                 # PROJECT_ROOT, env var loading
|
|-- tools/
|   |-- agentmanager_tools/
|   |   |-- profile_dataset.py       # Dataset shape, types, missing-value summary
|   |   |-- infer_task.py            # Infer classification vs regression and target column
|   |   |-- create_cleaning_plan.py  # Generate structured cleaning plan from profile
|   |   `-- manager_rap.py           # RAP: summarise requirements + retrieve context (arXiv / web)
|   |
|   |-- split_tools/
|   |   `-- split_dataset_tool.py    # Stratified / random train-test split
|   |
|   |-- dataclean_tools/
|   |   |-- load_dataset_to_store.py # Load CSV into the in-memory dataset store
|   |   |-- save_dataset_from_store.py # Flush cleaned dataset to disk
|   |   |-- clean_missing_values.py  # Impute or drop missing values
|   |   |-- clean_data_types.py      # Fix column dtypes (parse dates, cast numerics, etc.)
|   |   |-- clean_duplicates.py      # Detect and remove duplicate rows
|   |   |-- clean_outliers.py        # Cap or drop outliers (IQR / z-score)
|   |   `-- clean_drop_columns.py    # Drop irrelevant or leaky columns
|   |
|   |-- eda_tools/
|   |   |-- profile_training_data.py          # Full statistical profile of training set
|   |   |-- compute_correlation_matrix.py     # Pearson / Spearman correlation matrix
|   |   |-- generate_eda_charts.py            # Heatmap, importance, distribution charts
|   |   |-- plot_distributions.py             # Per-column distribution plots
|   |   `-- select_candidate_models_and_metrics.py  # Validate and commit model + metric choices
|   |
|   |-- feature_tools/
|   |   |-- encode_categorical.py    # Label / one-hot encoding of categorical features
|   |   |-- encode_target_column.py  # Target label encoding (XGBoost / XGBoostRegressor)
|   |   |-- scale_features.py        # StandardScaler / MinMaxScaler
|   |   `-- save_featured_dataset.py # Persist featured CSV and fitted preprocessors to disk
|   |
|   |-- model_tools/
|   |   |-- tune_and_select_model.py # Cross-validated HPO (GridSearch / RandomSearch)
|   |   |-- train_model.py           # Final training on full train set; save .pkl
|   |   |-- evaluate_model.py        # Test-set evaluation across all configured metrics
|   |   `-- explain_model.py         # Feature importance / SHAP explanation
|   |
|   |-- business_tools/
|   |   |-- bat_tools.py             # BAT-specific retrieval and formatting tools
|   |   |-- arxiv_tool.py            # arXiv paper search for BAT
|   |   `-- file_reader.py           # Read local Markdown files for BAT RAG
|   |
|   `-- tool_trace.py                # Unified tool-call logging
|
|-- scripts/
|   |-- build_mckinsey_kb.py         # One-shot ingest raw/*.md → Chroma mckinsey_kb
|   |-- benchmark_autokaggle.py      # Run AutoKaggle benchmark suite
|   |-- benchmark_eval_system.py     # Evaluate a completed pipeline run against benchmark
|   `-- aggregate_results.py         # Aggregate multi-run benchmark results into summary CSV
|
|-- rag/                             # Chroma RAG pipeline (embeddings, ingest, retrieve)
|-- data/
|   `-- mckinsey_kb/                 # McKinsey corpus manifest + raw Markdown (BAT RAG)
|-- datasets/                        # Sample input datasets
|-- tests/
|   |-- conftest.py                  # Shared path constants
|   |-- testenv/                     # Synthetic CSV fixtures (no external data needed)
|   |   |-- raw/sample.csv
|   |   |-- clean/sample_cleaned.csv
|   |   |-- split/sample_train.csv, sample_test.csv
|   |   `-- feature/{Model}_train/test_featured.csv
|   `-- test_*.py
`-- artifacts/
    `-- workspace/                   # Per-run output directories (git-ignored)
```

---

## Candidate Model Pools

The EDA Agent selects exactly **3 models** from the relevant pool based on dataset characteristics. All 3 are trained and evaluated; the best is reported as `best_model`.

### Classification (11 options)

| Model | Notes |
|---|---|
| `LogisticRegression` | Linear baseline, highly interpretable |
| `RandomForest` | Ensemble tree, stable and noise-resistant |
| `XGBoost` | Gradient boosting, competition-grade performance |
| `LightGBM` | Faster gradient boosting, preferred for large datasets |
| `GradientBoosting` | sklearn-native gradient boosting, no extra dependencies |
| `ExtraTrees` | Extremely randomised trees, fast with low variance |
| `AdaBoost` | Adaptive boosting, effective on imbalanced data |
| `SVM` | Kernel method, strong on small-to-medium datasets |
| `KNN` | Instance-based learning, no parametric assumptions |
| `GaussianNB` | Naive Bayes, very fast probabilistic baseline |
| `MLPClassifier` | Multi-layer perceptron, good for dense / large datasets |

### Regression (12 options)

| Model | Notes |
|---|---|
| `LinearRegression` | Linear baseline |
| `Ridge` | L2 regularisation, handles multicollinearity |
| `Lasso` | L1 regularisation, built-in feature selection |
| `ElasticNet` | L1+L2, balances sparsity and stability |
| `HuberRegressor` | Robust regression, insensitive to outliers |
| `RandomForestRegressor` | Ensemble tree |
| `XGBoostRegressor` | Gradient boosting |
| `LightGBMRegressor` | Faster gradient boosting |
| `GradientBoostingRegressor` | sklearn-native gradient boosting |
| `ExtraTreesRegressor` | Extremely randomised trees |
| `SVR` | Support vector regression |
| `MLPRegressor` | Multi-layer perceptron, optional for complex non-linear relationships |

---

## Evaluation Metrics

The EDA Agent selects one **primary metric** (used for CV hyperparameter tuning) and a list of **evaluation metrics** (reported on the final test set). All names are case-sensitive.

### Classification

| Metric | When to use |
|---|---|
| `f1_weighted` | Class-balanced; overall accuracy is the main concern |
| `f1_macro` | Minority classes exist; weights each class equally |
| `roc_auc` | Severely imbalanced; focus on minority-class recall |
| `accuracy` | Supplementary; good when classes are balanced |
| `precision` | When false positives are costly |
| `recall` | When false negatives are costly |
| `log_loss` | Probability calibration tasks (CTR, risk scoring) — requires `predict_proba` support |

> **Note:** For multi-class tasks, `roc_auc` is automatically upgraded to `roc_auc_ovr_weighted`. If `SVM` is a candidate (no `predict_proba` by default), avoid `log_loss` and `roc_auc` as the primary metric.

### Regression

| Metric | When to use |
|---|---|
| `r2` | Default; global goodness-of-fit, intuitive |
| `mae` | Significant outliers present (more robust than RMSE) |
| `rmse` | Standard squared-error loss |
| `rmsle` | Skewed targets (e.g. house prices, sales volume); requires y ≥ 0 |
| `mape` | Business cares about percentage deviation; requires y ≠ 0 |
| `max_error` | Worst-case precision requirements (e.g. part-tolerance prediction) |
