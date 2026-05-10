# Demo2 Development Context

This document summarizes the current development context for a new AI assistant or developer taking over the `demo2` project.

## 1. Project Purpose

`demo2` is a redesigned version of the previous Capstone multi-agent data science workflow.

The main design change is:

```text
Old approach:
LLM generates full Python code for each data science stage.

New approach:
LLM makes high-level decisions, while deterministic project-owned tools execute data processing, validation, modelling, and reporting.
```

The goal is to improve:

- Stability
- Reproducibility
- Debuggability
- Agent quality control
- Adaptability to different datasets and user goals

## 2. Core Architecture

Current workflow:

```text
Manager
-> Data Cleaning
-> Supervisor
-> Manager Replanning Checkpoint
-> Feature Decision
-> Feature Engineering
-> Supervisor
-> Manager Replanning Checkpoint
-> Model Selection
-> Supervisor
-> Business Translator
-> Supervisor
```

Current runner:

```text
main.py
```

The project does not yet use LangGraph. The current implementation is a minimal sequential runner so the redesigned agent/tool logic can be developed and tested first.

## 3. Main Design Principles

- Manager should define task intent, target/task inference, stage contracts, and downstream-only replanning patches.
- Worker agents should not write full scripts from scratch.
- Worker agents should call deterministic tools from `tools/`.
- Supervisor should validate artifacts deterministically before any LLM-based judgement is added.
- Business Translator can use an LLM because it mainly generates business narrative from structured evidence.
- LLM failure should not break the core data science pipeline when a deterministic fallback is possible.

## 4. Directory Structure

```text
demo2/
  agents/
    manager.py
    data_cleaning.py
    feature_decision.py
    feature_engineering.py
    model_selection.py
    business_translator.py
    supervisor.py

  core/
    state_schema.py
    llm.py

  tools/
    data_cleaning_tools.py
    target_inference_tools.py
    feature_engineering_tools.py
    model_selection_tools.py
    validation_tools.py
    registry.py

  docs/
    architecture_design.md

  workspace/
    runtime artifacts

  data/
    sample datasets

  main.py
  .env.example
```

## 5. Agent Responsibilities

### Manager Agent

File:

```text
agents/manager.py
```

Responsibilities:

- Read user goal and dataset preview.
- Normalize column names and parse numeric-like columns for planning.
- Infer target column and task type using LLM-assisted inference.
- Build initial `global_plan`.
- Define stage contracts.
- Run checkpoint-based replanning after selected stages.

Important functions:

- `manager_node(state)`
- `manager_replanning_checkpoint(state, completed_stage)`

Current replanning rules:

- After Data Cleaning:
  - If classification target appears imbalanced, patch Feature Engineering and Model Selection contracts.
  - If high-missingness columns exist, patch Feature Engineering requirements.
- After Feature Engineering:
  - If feature table is high-dimensional, patch Model Selection requirements.
  - If FE was skipped, require Model Selection to confirm model-ready numeric features.

Patch records are written to:

```text
workspace/plan_patches.json
```

### Data Cleaning Agent

File:

```text
agents/data_cleaning.py
```

Responsibilities:

- Load raw CSV.
- Apply deterministic cleaning tools.
- Write cleaned data.
- Split train/test.
- Write structured cleaning report.

Artifacts:

```text
workspace/cleaned_data.csv
workspace/train_data.csv
workspace/test_data.csv
workspace/cleaning_report.json
```

### Feature Decision Node

File:

```text
agents/feature_decision.py
```

Responsibilities:

- Read task summary and cleaned data profile.
- Decide final feature engineering policy.
- Choose one of the supported FE modes: `skip`, `light`, `standard`, `task_specific`, `model_specific`.

Artifact:

```text
workspace/feature_policy_report.json
```

### Feature Engineering Agent

File:

```text
agents/feature_engineering.py
```

Responsibilities:

- Read train/test data.
- Apply selected FE policy.
- Preserve target column.
- Write model-ready train/test artifacts.
- Write structured feature report.

Artifacts:

```text
workspace/fe_train.csv
workspace/fe_test.csv
workspace/feature_report.json
```

### Model Selection Agent

File:

```text
agents/model_selection.py
```

Responsibilities:

- Read feature-engineered train/test data.
- Train candidate models.
- Select best model by cross-validation.
- Evaluate on test set.
- Write model report and feature importance.

Artifacts:

```text
workspace/model_report.json
workspace/feature_importance.json
```

Current candidate models:

- Classification: Logistic Regression, Random Forest Classifier
- Regression: Ridge, Random Forest Regressor

### Business Translator Agent

File:

```text
agents/business_translator.py
```

Responsibilities:

- Read structured workflow evidence.
- Generate business-facing markdown report.
- Use LLM if configured.
- Use deterministic fallback report if LLM fails.

Artifact:

```text
workspace/business_report.md
```

### Supervisor Agent

File:

```text
agents/supervisor.py
```

Responsibilities:

- Validate required artifacts for each stage.
- Check CSV readability and train/test column consistency.
- Check JSON artifacts.
- Check business report existence, length, and required sections.

Current Supervisor is deterministic. It does not yet perform LLM semantic review.

## 6. Tool Layer

### Data Cleaning Tools

File:

```text
tools/data_cleaning_tools.py
```

Key functions:

- `normalize_column_names(df)`
- `infer_target_column(columns, user_goal)`
- `profile_dataframe(df, target_col)`
- `parse_numeric_like_columns(df, target_col)`
- `basic_clean_dataframe(df, target_col)`
- `split_cleaned_data(df, target_col, task_type)`

Note:

`infer_target_column` is now only deterministic fallback logic. Manager target inference should use `tools/target_inference_tools.py`.

### Target Inference Tools

File:

```text
tools/target_inference_tools.py
```

Key function:

```text
infer_target_and_task(df, user_goal, use_llm=True)
```

Current behavior:

- Uses LLM to infer `target_column`, `task_type`, `confidence`, and `reason`.
- Validates that returned target exists in dataset columns.
- Falls back to deterministic rules if LLM fails.
- Adds `source: "llm"` or `source: "rules"` to the result.

Example result:

```json
{
  "target_column": "profit",
  "task_type": "regression",
  "confidence": 0.9,
  "reason": "profit is a measurable financial outcome aligned with the user goal",
  "source": "llm"
}
```

### Feature Engineering Tools

File:

```text
tools/feature_engineering_tools.py
```

Key functions:

- `recommend_feature_engineering_policy(task_type, model_family, data_profile)`
- `apply_feature_policy(train_df, test_df, policy, target_col)`
- `create_missing_indicators(train_df, test_df)`
- `create_datetime_features(train_df, test_df, target_col)`
- `encode_categorical_features(train_df, test_df, target_col)`

### Model Selection Tools

File:

```text
tools/model_selection_tools.py
```

Key functions:

- `infer_task_type(user_goal, target_series)`
- `select_metric(task_type)`
- `candidate_models(task_type)`
- `train_and_select_model(train_df, test_df, target_col, task_type)`

### Validation Tools

File:

```text
tools/validation_tools.py
```

Key functions:

- `ensure_workspace(path)`
- `write_json(path, payload)`
- `write_json(path, payload)`
- `read_json(path)`
- `validate_csv(path, min_rows)`
- `validate_json_object(path)`
- `validate_matching_columns(train_path, test_path)`

## 7. LLM Integration

LLM client:

```text
core/llm.py
```

Configuration file:

```text
.env
```

Template:

```text
.env.example
```

Required:

```text
OPENAI_API_KEY=your_api_key_here
```

Optional:

```text
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=optional_openai_compatible_base_url
LLM_TEMPERATURE=0.2
```

Current LLM usage:

- Target/task inference in Manager.
- Business report generation in Business Translator.

The core data processing, feature engineering, model selection, and artifact validation remain deterministic.

## 8. How To Run

From `demo2/`:

```bash
python main.py data/Financials.csv "Predict business-relevant financial outcomes and generate actionable recommendations"
```

Expected output:

```text
Workflow completed.
Plan version: 1
Business report: workspace/business_report.md
Artifacts written to ./workspace
```

If LLM is configured, `business_report.md` should be LLM-generated.

If LLM is unavailable, Business Translator should write a fallback report instead.

## 9. Current Verified Behavior

The workflow has been tested on:

```text
data/Financials.csv
```

Observed behavior:

- Target inference selected `profit`.
- Task type selected `regression`.
- Data cleaning normalized column names and parsed currency-like fields.
- Feature Decision selected light feature engineering.
- Model Selection selected Ridge for the Financials sample.
- Business Translator generated `workspace/business_report.md`.

## 10. Known Issues and Risks

### Target leakage risk

For `Financials.csv`, `profit` may be directly derivable from other columns such as sales, COGS, discounts, or gross sales. This can produce unrealistically high model performance.

Recommended next step:

```text
Implement leakage_detection_tool and make Supervisor or Model Selection report possible leakage.
```

### Limited task support

Current Model Selection mainly supports supervised classification and regression.

Not fully supported yet:

- Clustering
- Time-series forecasting
- Text modelling
- Causal analysis
- Recommendation systems

### Feature Engineering is still basic

Current FE supports:

- Missing indicators
- Datetime feature extraction
- One-hot encoding
- Pass-through behavior

Missing:

- Scaling for non-tree models
- High-cardinality categorical handling
- Time-series lag/rolling features
- Text vectorization
- Leakage-aware feature filtering

### Supervisor is artifact-focused, not semantic enough

Current Supervisor validates files and basic structure.

Missing:

- Contract-by-contract semantic validation
- Target leakage checks
- Metric suitability checks
- Dataset drift checks
- Business report evidence consistency checks

### No LangGraph integration yet

The project currently uses a sequential Python runner. LangGraph can be added later after the agent/tool architecture stabilizes.

## 11. Recommended Next Development Steps

1. Add leakage detection.
2. Improve Supervisor contract validation.
3. Add structured tool execution logs.
4. Add unit tests for tools.
5. Add support for descriptive and clustering tasks.
6. Add time-series split and time-series feature tools.
7. Add model persistence.
8. Add richer business report evidence checks.
9. Add LangGraph orchestration after the deterministic pipeline is stable.

## 12. Important Design Decision

Do not return to the old pattern where worker agents ask the LLM to generate full executable scripts.

The intended direction is:

```text
LLM = planning, target inference, report writing, limited tool selection
Deterministic tools = data transformation, modelling, validation, artifact writing
Supervisor = deterministic checks first, optional LLM semantic review later
```

This design is the core reason `demo2` exists.
