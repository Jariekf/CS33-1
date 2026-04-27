"""
Feature Engineering Agent (特征工程智能体)
职责：
  根据 Manager 的全局计划中的 feature_engineering_plan，
  生成 Python 特征工程代码并在沙盒中执行（含内部自救机制）。

设计参照 AutoKaggle Developer Agent 的核心模式：
  - 结构化 Prompt (CONTEXT + PLAN + CONSTRAINTS + TASK)
  - 上一阶段摘要注入上下文 (previous_summary)
  - 代码提取 (regex ```python...```)

The column names in the output CSV should be readable by the downstream "Feature Importance / Business Report": retain the business column names, and use meaningful names for new features;

It is forbidden to change them to anonymous column names such as x0, x1, feature_0, etc. without justification (enforced by the agent's Prompt constraints).
"""
from __future__ import annotations

import json
from core.logger import get_agent_logger

from core.state import GraphState
from core.llm import call_llm, extract_code
from tools.python_repl import execute_code
from config import MAX_INNER_RETRY
from agents.data_cleaning import TRAIN_DATA_PATH

logger = get_agent_logger("feature_engineering")

# Fixed per-phase output path. FE always reads the cleaned CSV produced by
# the data_cleaning node and writes its output here. Using a fixed path
# (rather than ``state["current_data_path"]``) prevents Supervisor retries
# from re-feeding FE's own rejected output back into itself.
FE_OUTPUT_PATH = "./workspace/fe_output.csv"
FE_TEST_OUTPUT_PATH = "./workspace/fe_test_output.csv"


def feature_engineering_node(state: GraphState) -> dict:
    """Feature Engineering 的 LangGraph 节点函数。"""

    logger.info("Feature Engineering node started.")
    # 【边界控制】只抽取属于自己的子计划
    plan = state["global_plan"].get("feature_engineering_plan", {})
    # Always read from the train split produced by data_cleaning, NOT from
    # ``state["current_data_path"]`` — on a Supervisor retry the latter may
    # point at FE's own previous rejected output.
    data_path = TRAIN_DATA_PATH
    feedback = state.get("current_error", "")
    dataset_description = (state.get("dataset_description") or "").strip()
    # 只读取最近的 Memory 摘要 (上一阶段清洗成果)
    previous_summary = state["memory"][-1] if state.get("memory") else ""

    # ------------------------------------------------------------------
    # Step 1: 生成特征工程代码
    # ------------------------------------------------------------------
    from agents.data_cleaning import TEST_DATA_PATH
    code = _generate_fe_code(plan, data_path, TEST_DATA_PATH, previous_summary, feedback, dataset_description)
    logger.info("Feature engineering code generation completed.")

    # ------------------------------------------------------------------
    # Step 2: 内部语法自救循环
    # ------------------------------------------------------------------
    stdout, stderr = "", ""
    for attempt in range(MAX_INNER_RETRY):
        stdout, stderr = execute_code(code)
        if not stderr:
            logger.info("Feature engineering code executed successfully.")
            break
        # Show head (traceback origin) + tail (exception type & message) for diagnosis.
        _snippet = stderr if len(stderr) <= 2000 else f"{stderr[:1000]}\n...[truncated {len(stderr)} chars total]...\n{stderr[-1000:]}"
        logger.info("内部自救第 %d 次: %s", attempt + 1, _snippet)
        # On the final iteration there is no next execute_code, so calling
        # _fix_code would burn an LLM request whose output is never run.
        if attempt == MAX_INNER_RETRY - 1:
            break
        code = _fix_code(code, stderr, plan, previous_summary, dataset_description)

    # ------------------------------------------------------------------
    # Step 3: 返回状态更新
    # ------------------------------------------------------------------
    if stderr:
        logger.error(f"Feature engineering failed after {MAX_INNER_RETRY} attempts: {stderr}")
        return {
            "current_phase": "feature_engineering",
            "current_code": code,
            "current_execution_log": stdout,
            "current_error": f"[InnerRetryExhausted] {stderr}",
        }

    # Note: we intentionally do NOT set ``current_data_path`` here.
    # The downstream model_selection node reads from a fixed ``FE_OUTPUT_PATH``,
    # so on a Supervisor-driven retry we also avoid re-feeding FE's own
    # previous rejected output back into itself.
    return {
        "current_phase": "feature_engineering",
        "current_code": code,
        "current_execution_log": stdout,
        "current_error": "",
    }


# ======================================================================
# 私有辅助函数
# ======================================================================

# ---------- Prompt 模板 (参照 AutoKaggle PROMPT_DEVELOPER 结构) ----------

_MODERN_SKLEARN_API_GUIDE = """\
# TARGET LIBRARY VERSIONS (IMPORTANT) #
Assume scikit-learn >= 1.4, pandas >= 2.0, numpy >= 1.24. \
Do NOT use deprecated or removed APIs. In particular:
  - `OneHotEncoder`: use `sparse_output=False` (NEVER `sparse=`; that kwarg is REMOVED in sklearn 1.4).
  - `OrdinalEncoder`: use `handle_unknown="use_encoded_value"` with `unknown_value=...` when needed.
  - `SGDClassifier` / `Ridge` / `Lasso` / `LinearRegression`: do NOT pass `normalize=`; it is REMOVED. Use a `StandardScaler` in a Pipeline instead.
  - `LogisticRegression`: do NOT pass `multi_class=`; it is deprecated. Rely on defaults.
  - `GridSearchCV` / `RandomizedSearchCV`: do NOT pass `iid=`; it is REMOVED.
  - `mean_squared_error(..., squared=False)` is deprecated; use `root_mean_squared_error` (sklearn >= 1.4) or `np.sqrt(mean_squared_error(...))`.
  - `make_scorer(..., needs_proba=True)` is deprecated; use `make_scorer(..., response_method="predict_proba")`.
  - pandas `DataFrame.append` is REMOVED; use `pd.concat([...])`.
  - pandas `df.iteritems()` is REMOVED; use `df.items()`.
  - numpy `np.int`/`np.float`/`np.bool` are REMOVED; use built-in `int`/`float`/`bool` or `np.int64`/`np.float64`.
If you are unsure whether a kwarg still exists, omit it and rely on defaults rather than guessing."""


_SYSTEM_PROMPT_GENERATE = """\
You are an expert Feature Engineering developer. \
You write Python code according to the provided plan and constraints. \
Your code must be complete, self-contained, and directly runnable. \
You must only use columns that actually exist in the loaded DataFrame. Never invent schema fields such as `product_id`, `country`, `period`, or target-like columns unless they are present in the data. \
If a planned grouping key, datetime column, or source column is missing, degrade gracefully: skip that advanced feature family and continue with simpler valid features instead of failing. \
Column names in the saved CSV must stay human-interpretable for downstream model reports: \
keep original names where columns are retained; name new features descriptively (snake_case). \
Never bulk-rename features to generic x0, x1, x2, or feature_0 unless the prompt explicitly requires it.

""" + _MODERN_SKLEARN_API_GUIDE

_USER_PROMPT_GENERATE = """\
# CONTEXT #
Current phase: Feature Engineering

# DATASET DESCRIPTION (authoritative) #
{dataset_description_block}

# PREVIOUS PHASE SUMMARY #
{previous_summary}

# PLAN #
{plan}

# CONSTRAINTS #
1. Load training data from: `{data_path}` (80% train split) using `pd.read_csv`.
   Also load the held-out test set from: `{test_data_path}` (20% test split).
2. Save feature-engineered training data to: `./workspace/fe_output.csv` and feature-engineered test data to: `./workspace/fe_test_output.csv` using `df.to_csv(..., index=False)`.
   Both files MUST have exactly the same columns. Fit all encoders/scalers on the training set only, then transform both train and test sets with the fitted objects.
3. Always work on a copy of the DataFrame, not the original.
4. Apply consistent modifications (e.g., scaling, encoding) to ensure reproducibility.
5. Use `print()` to output a summary (feature count before/after, new features created, etc.).
6. Import all necessary libraries at the top of the code.
7. Common feature engineering operations to consider:
   - Derived features (interaction, polynomial, aggregation)
   - Category encoding (Label Encoding, One-Hot, Target Encoding)
   - Feature scaling (StandardScaler, MinMaxScaler)
   - Handling of datetime features (extraction of year/month/day/hour etc.)
8. Column names for `./workspace/fe_output.csv` must remain meaningful for business-facing reports:
   - Keep original input column names when those columns still exist (do not rename to x0/x1/...).
   - New or transformed columns: use descriptive snake_case names, e.g. `units_sold_log`, `gross_sales_sqrt`, `country_Germany` (one-hot suffix), `order_date_year`.
   - Do NOT replace all column names with sequential generics (`x0`, `x1`, `feature_0`, `col_0`, etc.).
9. If you drop columns, that is fine; if you one-hot or encode, use clear prefixes/suffixes so names trace back to the source column.
10. Before using any column in transformations, grouping, sorting, lagging, rolling, or encoding, validate that the column exists in `df.columns`.
11. If the dataset does not contain reliable datetime or grouping columns, do not force time-series or hierarchical feature engineering. Fall back to safe row-level or column-level features that are supported by the actual schema.
12. Do not raise an exception only because an expected column is missing. Skip unsupported feature families and still produce a valid `./workspace/fe_output.csv`.

{feedback_section}

# RESPONSE: CODE #
Provide your complete solution in a single ```python ... ``` code block.
"""

_SYSTEM_PROMPT_FIX = """\
You are an expert Python debugger. \
Analyze the error and fix the code. \
Output the complete corrected code. \
Only use columns that actually exist in the loaded DataFrame, and if expected columns are missing, degrade gracefully instead of failing. \
Preserve the same column-naming rules as feature engineering: descriptive names, no generic x0/x1 renames.

""" + _MODERN_SKLEARN_API_GUIDE

_USER_PROMPT_FIX = """\
# CONTEXT #
Current phase: Feature Engineering

# DATASET DESCRIPTION (authoritative) #
{dataset_description_block}

# PREVIOUS PHASE SUMMARY #
{previous_summary}

# PLAN #
{plan}

# CODE CONTAINS ERROR #
```python
{code}
```

# ERROR MESSAGES #
{error}

# TASK REQUIREMENTS #
1. Continue following the original Feature Engineering plan and constraints.
2. Save the feature-engineered output to `./workspace/fe_output.csv` (train) and `./workspace/fe_test_output.csv` (test). Both must have the same columns. Fit encoders/scalers on train only, then transform both.
3. Preserve meaningful, human-interpretable column names.
4. If the previous attempt was rejected, fix the implementation gap rather than only fixing syntax.
5. Do not use or reference columns that are not present in the loaded dataframe.
6. If sort/group/time columns are missing, skip those advanced features and continue with simpler valid transformations.

# RESPONSE: CORRECTED CODE #
Provide the complete corrected code in a single ```python ... ``` code block.
"""


def _dataset_description_block(desc: str) -> str:
    """Render the DATASET DESCRIPTION block for prompts (fallback to neutral note)."""
    text = (desc or "").strip()
    if text:
        return text
    return "(not provided; infer cautiously from column names and sample rows)"


def _compress_supervisor_feedback(feedback: str, max_chars: int = 1200) -> str:
    """
    Trim overly long supervisor feedback before injecting it into the generate
    prompt. See ``agents.model_selection._compress_supervisor_feedback`` for
    the rationale — the two helpers intentionally mirror each other so that
    both worker agents behave the same way under a verbose supervisor.
    """
    text = (feedback or "").strip()
    if len(text) <= max_chars:
        return text
    head = (max_chars * 2) // 3
    tail = max_chars - head
    return (
        text[:head].rstrip()
        + "\n\n[...supervisor feedback truncated by pipeline: focus on the TOP 2-3 issues above; do NOT try to implement every sub-bullet in a single pass...]\n\n"
        + text[-tail:].lstrip()
    )


def _generate_fe_code(
    plan: dict,
    data_path: str,
    test_data_path: str,
    previous_summary: str,
    feedback: str,
    dataset_description: str = "",
) -> str:
    """
    调用 LLM 生成特征工程代码。
    参照 AutoKaggle Developer._execute 的 Prompt 构建逻辑，注入上一阶段摘要。
    若 LLM 调用失败则降级返回桩代码模板。
    """
    feedback_section = ""
    if feedback:
        compressed = _compress_supervisor_feedback(feedback)
        feedback_section = (
            "# SUPERVISOR FEEDBACK (compressed) #\n"
            "Previous attempt was rejected. Focus on the TOP 2-3 issues; do NOT\n"
            "try to implement every sub-bullet in a single pass:\n"
            f"{compressed}"
        )

    user_prompt = _USER_PROMPT_GENERATE.format(
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        data_path=data_path,
        test_data_path=test_data_path,
        previous_summary=previous_summary if previous_summary else "无上一阶段摘要",
        feedback_section=feedback_section,
        dataset_description_block=_dataset_description_block(dataset_description),
    )

    reply = call_llm(_SYSTEM_PROMPT_GENERATE, user_prompt, agent_name="feature_engineering")

    if reply:
        code = extract_code(reply)
        if code:
            return code

    # 降级：返回桩代码模板
    logger.warning("LLM 调用失败或未返回代码，使用桩代码模板。")
    return _fallback_fe_code(plan, data_path, previous_summary)


def _fix_code(
    code: str,
    error: str,
    plan: dict,
    previous_summary: str,
    dataset_description: str = "",
) -> str:
    """
    调用 LLM 根据报错修复代码。
    参照 AutoKaggle PROMPT_DEVELOPER_DEBUG_FIX 的简化版。
    若 LLM 不可用则返回原代码。
    """
    user_prompt = _USER_PROMPT_FIX.format(
        code=code,
        error=error,
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        previous_summary=previous_summary if previous_summary else "无上一阶段摘要",
        dataset_description_block=_dataset_description_block(dataset_description),
    )
    reply = call_llm(_SYSTEM_PROMPT_FIX, user_prompt, agent_name="feature_engineering")

    if reply:
        fixed_code = extract_code(reply)
        if fixed_code:
            return fixed_code

    logger.warning("LLM 修复调用失败，返回原代码。")
    return code


def _fallback_fe_code(plan: dict, data_path: str, previous_summary: str) -> str:
    """LLM 不可用时的降级桩代码模板。"""
    path_lit = json.dumps(data_path)
    return f"""\
import pandas as pd
import numpy as np

df = pd.read_csv({path_lit})
# 特征工程计划: {plan.get('objectives', [])}
# 上一阶段摘要: {previous_summary}

# Work on a copy to keep transformations explicit.
df = df.copy()
original_cols = list(df.columns)
created_features = []

# 1. Try to parse likely datetime columns and derive safe date parts when parsing succeeds.
datetime_candidates = []
for col in df.columns:
    lc = str(col).lower()
    if any(token in lc for token in ("date", "time", "timestamp", "period")):
        datetime_candidates.append(col)

for col in datetime_candidates:
    parsed = pd.to_datetime(df[col], errors="coerce")
    if parsed.notna().sum() > 0:
        df[col] = parsed
        df[f"{{col}}_year"] = parsed.dt.year
        df[f"{{col}}_month"] = parsed.dt.month
        df[f"{{col}}_day"] = parsed.dt.day
        created_features.extend([f"{{col}}_year", f"{{col}}_month", f"{{col}}_day"])

# 2. Safe numeric features: fill missing values and add log1p / sqrt transforms for non-negative columns.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

for col in numeric_cols:
    series = pd.to_numeric(df[col], errors="coerce")
    if series.notna().sum() == 0:
        continue
    min_val = series.min()
    if pd.notna(min_val) and min_val >= 0:
        log_col = f"{{col}}_log1p"
        sqrt_col = f"{{col}}_sqrt"
        df[log_col] = np.log1p(series)
        df[sqrt_col] = np.sqrt(series)
        created_features.extend([log_col, sqrt_col])

# 3. Simple interaction features for the first few numeric columns.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
base_numeric = [c for c in numeric_cols[:4] if c in df.columns]
for i in range(len(base_numeric)):
    for j in range(i + 1, min(len(base_numeric), i + 3)):
        c1, c2 = base_numeric[i], base_numeric[j]
        inter_col = f"{{c1}}_x_{{c2}}"
        df[inter_col] = pd.to_numeric(df[c1], errors="coerce").fillna(0) * pd.to_numeric(df[c2], errors="coerce").fillna(0)
        created_features.append(inter_col)

# 4. Safe categorical encoding: one-hot only for low-cardinality object/category columns.
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
low_card_cols = []
for col in categorical_cols:
    nunique = df[col].nunique(dropna=True)
    if 1 < nunique <= 10:
        low_card_cols.append(col)

if low_card_cols:
    df = pd.get_dummies(df, columns=low_card_cols, dummy_na=False)
    created_features.extend([c for c in df.columns if c not in original_cols and c not in created_features])

# 5. Final cleanup: drop all-empty columns and save.
df = df.dropna(axis=1, how="all")
output_path = "./workspace/fe_output.csv"
df.to_csv(output_path, index=False)
print(f"Feature engineering complete. Original cols: {{len(original_cols)}}; final cols: {{df.shape[1]}}; created features: {{len(created_features)}}; saved to {{output_path}}")
"""
