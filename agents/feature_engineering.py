"""
Feature Engineering agent.

Responsibilities:
  Read ``feature_engineering_plan`` from the Manager's global plan, generate Python
  feature code, and execute it in the sandbox (with inner self-heal).

  EDA charts and narrative are produced by ``agents.eda_analysis`` after the data_cleaning
  supervisor passes; this stage does not own EDA.

Design follows the AutoKaggle Developer pattern:
  - Structured prompt (CONTEXT + PLAN + CONSTRAINTS + TASK)
  - Prior-stage summary in context (``previous_summary``)
  - Code extraction via ```python``` regex

Output CSV column names should stay interpretable for feature importance / business reports:
keep business-meaningful names; avoid anonymous ``x0`` / ``feature_0`` renames without justification
(enforced via prompt constraints).
"""
from __future__ import annotations

import json
import pandas as pd
from core.logger import get_agent_logger

from core.state import GraphState
from core.llm import call_llm, extract_code
from tools.python_repl import execute_code
from config import MAX_INNER_RETRY
from agents.data_cleaning import TRAIN_DATA_PATH, TEST_DATA_PATH

logger = get_agent_logger("feature_engineering")

# Fixed per-phase output path. FE always reads the cleaned CSV produced by
# the data_cleaning node and writes its output here. Using a fixed path
# (rather than ``state["current_data_path"]``) prevents Supervisor retries
# from re-feeding FE's own rejected output back into itself.
FE_OUTPUT_PATH = "./workspace/fe_output.csv"
FE_TEST_OUTPUT_PATH = "./workspace/fe_test_output.csv"
EDA_SUMMARY_START = "===EDA_SUMMARY_START==="
EDA_SUMMARY_END = "===EDA_SUMMARY_END==="


def feature_engineering_node(state: GraphState) -> dict:
    """LangGraph node for feature engineering."""

    logger.info("Feature Engineering node started.")
    # Scope: only this phase's sub-plan
    plan = state["global_plan"].get("feature_engineering_plan", {})
    # Always read from the train split produced by data_cleaning, NOT from
    # ``state["current_data_path"]`` — on a Supervisor retry the latter may
    # point at FE's own previous rejected output.
    data_path = TRAIN_DATA_PATH
    feedback = state.get("current_error", "")
    dataset_description = (state.get("dataset_description") or "").strip()
    # Latest memory entry (prior stage, e.g. cleaning)
    previous_summary = state["memory"][-1] if state.get("memory") else ""
    # Authoritative target column from state (Manager's three-tier resolver:
    # CLI --target > Manager LLM > last-column fallback). Empty string only
    # for genuinely unsupervised tasks or when the input header was unreadable.
    state_target_col = (state.get("target_column") or "").strip()
    target_hint = _build_target_hint(data_path, target_col_override=state_target_col)

    # ------------------------------------------------------------------
    # Step 1: generate feature code
    # ------------------------------------------------------------------
    code = _generate_fe_code(
        plan,
        data_path,
        TEST_DATA_PATH,
        previous_summary,
        feedback,
        dataset_description,
        target_hint,
    )
    logger.info("Feature engineering code generation completed.")

    # ------------------------------------------------------------------
    # Step 2: inner self-heal loop
    # ------------------------------------------------------------------
    stdout, stderr = "", ""
    error_history: list[str] = []
    for attempt in range(MAX_INNER_RETRY):
        previous_signatures = {
            FE_OUTPUT_PATH: _get_file_signature(FE_OUTPUT_PATH),
            FE_TEST_OUTPUT_PATH: _get_file_signature(FE_TEST_OUTPUT_PATH),
        }
        stdout, stderr = execute_code(code)
        if not stderr:
            validation_error = _validate_fe_artifacts(previous_signatures)
            if not validation_error:
                logger.info("Feature engineering code executed successfully.")
                break
            stderr = f"[PostValidationFailed] {validation_error}"
        error_history.append(f"Attempt {attempt + 1}: {stderr}")
        # Show head (traceback origin) + tail (exception type & message) for diagnosis.
        _snippet = stderr if len(stderr) <= 2000 else f"{stderr[:1000]}\n...[truncated {len(stderr)} chars total]...\n{stderr[-1000:]}"
        logger.info("Inner self-heal attempt %d: %s", attempt + 1, _snippet)
        if _is_target_related_error(stderr):
            logger.warning(
                "Detected target-column-related FE failure. Switching to deterministic fallback template."
            )
            code = _deterministic_fallback_fe_code(plan, data_path, TEST_DATA_PATH, previous_summary)
            continue
        # On the final iteration there is no next execute_code, so calling
        # _fix_code would burn an LLM request whose output is never run.
        if attempt == MAX_INNER_RETRY - 1:
            break
        code = _fix_code(
            code,
            "\n\n---\n\n".join(error_history),
            plan,
            previous_summary,
            dataset_description,
            target_hint,
        )

    # ------------------------------------------------------------------
    # Step 3: return state updates
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
# Private helpers
# ======================================================================

# ---------- Prompt templates (AutoKaggle PROMPT_DEVELOPER style) ----------

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

# COMMON RUNTIME FOOT-GUNS (must be avoided) #
- `get_feature_names_out` REQUIRES a fitted estimator. NEVER call it on a freshly-constructed `OneHotEncoder` / `OrdinalEncoder` / `ColumnTransformer` / `Pipeline` — that raises `NotFittedError`. The correct order is always: build the encoder/pipeline → `fit` (or `fit_transform`) it on the training data → THEN extract feature names. For a `Pipeline` whose preprocessing step is named (e.g. `"prep"`), use `pipeline.named_steps["prep"].get_feature_names_out()` AFTER `pipeline.fit(...)`. If you only need a column-name list at that exact moment, call `fit_transform` first and ignore the matrix.
- numpy scalars (`np.int32`, `np.int64`, `np.float32`, `np.float64`, `np.bool_`) are NOT JSON-serializable by stdlib `json`. Before `json.dump` / `json.dumps`, convert them. The robust pattern is `json.dump(obj, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x)`. Equivalently, build the dict from native Python types up front: `{str(k): float(v) for k, v in importances.items()}` for floats, or `{str(k): int(v) for k, v in counts.items()}` for ints. Do NOT just write `json.dump(sorted_feature_importances, f)` when the values came out of pandas/numpy/sklearn — that fails with `TypeError: Object of type int32 is not JSON serializable`.

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

# TARGET HINT FROM RUNTIME DATA INSPECTION #
{target_hint}

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
5. Use `print()` to output a brief summary (feature count before/after, new features created, etc.).
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
13. If obvious class imbalance is detected in the training target column, include at least one mitigation strategy in produced features or metadata (e.g., class_weight hint, resampling note) and mention it in the printed summary.
14. For supervised tasks, preserve the target column in both output CSVs. Do not drop, scale, one-hot encode, or otherwise transform the target column; keep its original column name and values unchanged.
15. NEVER hardcode `train_df['y']` or any target reference before checking `df.columns`. If no reliable target is found at runtime, skip target-specific logic and continue with generic feature engineering.

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

# TARGET HINT FROM RUNTIME DATA INSPECTION #
{target_hint}

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
7. Never raise an exception only because a target column cannot be confidently identified. If no reliable target is found, skip target-specific transformations and preserve generic features only.

# RESPONSE: CORRECTED CODE #
Provide the complete corrected code in a single ```python ... ``` code block.
"""


def _dataset_description_block(desc: str) -> str:
    """Render the DATASET DESCRIPTION block for prompts (fallback to neutral note)."""
    text = (desc or "").strip()
    if text:
        return text
    return "(not provided; infer cautiously from column names and sample rows)"


def _get_file_signature(file_path: str) -> tuple[int, int] | None:
    """Return file signature (mtime_ns, size), or None if the file does not exist."""
    import os

    if not os.path.exists(file_path):
        return None
    stat = os.stat(file_path)
    return stat.st_mtime_ns, stat.st_size


def _validate_one_fe_csv(
    output_path: str,
    previous_signature: tuple[int, int] | None,
) -> tuple[str, pd.DataFrame | None]:
    """Validate one FE CSV artifact and return the loaded DataFrame when valid."""
    import os

    if not os.path.exists(output_path):
        return f"feature-engineered output not found: {output_path}", None
    if not os.path.isfile(output_path):
        return f"feature-engineered output is not a file: {output_path}", None

    current_signature = _get_file_signature(output_path)
    if current_signature is None:
        return f"failed to stat feature-engineered output: {output_path}", None
    if current_signature[1] <= 0:
        return f"feature-engineered output file is empty: {output_path}", None
    if previous_signature is not None and current_signature == previous_signature:
        return f"feature-engineered output file was not updated in current run: {output_path}", None

    try:
        df = pd.read_csv(output_path)
    except Exception as e:
        return f"feature-engineered CSV could not be read: {output_path}: {e}", None
    if df.empty:
        return f"feature-engineered CSV has no rows: {output_path}", None
    if df.shape[1] == 0:
        return f"feature-engineered CSV has no columns: {output_path}", None
    return "", df


def _validate_fe_artifacts(
    previous_signatures: dict[str, tuple[int, int] | None],
) -> str:
    """Validate train/test FE artifacts before passing the phase to Supervisor."""
    train_error, train_df = _validate_one_fe_csv(
        FE_OUTPUT_PATH,
        previous_signatures.get(FE_OUTPUT_PATH),
    )
    if train_error:
        return train_error

    test_error, test_df = _validate_one_fe_csv(
        FE_TEST_OUTPUT_PATH,
        previous_signatures.get(FE_TEST_OUTPUT_PATH),
    )
    if test_error:
        return test_error

    if train_df is None or test_df is None:
        return "feature-engineered train/test artifacts could not be loaded"
    if list(train_df.columns) != list(test_df.columns):
        return "feature-engineered train/test outputs must have exactly the same columns"
    return ""


def _build_target_hint(data_path: str, target_col_override: str = "") -> str:
    """
    Render a hard-edged target-passthrough hint for the FE LLM.

    Resolution order for the column name:
      1. ``target_col_override`` from ``state["target_column"]`` (Manager's
         three-tier resolver — CLI / Manager LLM / last-column fallback).
         This is the authoritative answer for the whole pipeline.
      2. Local semantic-name match (kept as emergency fallback for callers
         that do not pass the override, e.g. legacy unit tests).
      3. Last-column heuristic if the override is missing AND no semantic hit.

    The hint deliberately separates two ideas the previous wording conflated:
      (a) FE must NOT add a target-derived **feature column** to the output
          CSV (e.g. ``{target}_log`` as an extra column next to the real
          ``{target}`` column) — this is the leakage source we keep tripping on.
      (b) Whether to apply ``log(y)`` / Box-Cox / etc. to the **target** for
          training is a Model Selection decision, not a FE concern. FE just
          passes the original target values through unchanged so MS sees the
          true labels.
    """
    try:
        df = pd.read_csv(data_path, nrows=50)
    except Exception as exc:
        return f"(runtime target hint unavailable: {exc})"
    cols = list(df.columns)
    if not cols:
        return (
            "No columns could be read from the training CSV. Skip target-specific "
            "logic if absent and still produce FE outputs."
        )

    candidate: str | None = None
    source: str = ""
    override = (target_col_override or "").strip()
    if override and override in cols:
        candidate = override
        source = "Manager-resolved (state['target_column'])"

    if candidate is None:
        semantic_exact = {"target", "label", "labels", "outcome", "response", "dependent", "y", "class"}
        semantic_contains = ("target", "label", "outcome", "response", "dependent")
        lowered = {c: str(c).strip().lower().replace(" ", "_") for c in cols}
        exact_hits = [c for c, lc in lowered.items() if lc in semantic_exact]
        contains_hits = [c for c, lc in lowered.items() if any(tok in lc for tok in semantic_contains)]
        if exact_hits:
            candidate = exact_hits[0]
            source = "semantic name match (exact)"
        elif contains_hits:
            candidate = contains_hits[0]
            source = "semantic name match (contains)"

    if candidate is None:
        candidate = cols[-1]
        source = "last-column fallback"

    uniques = df[candidate].dropna().astype(str).str.strip().str.lower().unique().tolist()[:10]
    return (
        f"Supervised target column for this run is `{candidate}` "
        f"(resolved via {source}). Observed sample values: {uniques}.\n"
        "\n"
        f"You MUST handle `{candidate}` carefully:\n"
        f"  - DO copy `{candidate}` straight from the input into "
        "`./workspace/fe_output.csv` with its original values intact. The "
        "downstream Model Selection node reads this column as `y`.\n"
        f"  - DO NOT include `{candidate}` in any list of feature columns "
        "passed to a transformer (e.g. exclude it from `numeric_cols`, "
        "`categorical_cols`, `ColumnTransformer` selectors, or any feature-side "
        "scaling/encoding/imputation step).\n"
        f"  - DO NOT add NEW columns to the output CSV that are derived from "
        f"`{candidate}` (e.g. `{candidate}_log`, `log_{candidate}`, "
        f"`{candidate}_squared`, `{candidate}_bin`, "
        f"`scaled_{candidate}`, etc.). Such columns become target-leakage "
        "features at Model Selection — the model learns to predict the target "
        "from a near-copy of itself, producing fake high scores that "
        "Supervisor will reject.\n"
        "\n"
        f"Note: applying log / Box-Cox / Yeo-Johnson to the target series "
        "for training (e.g. for log-RMSE on a skewed regression target) is "
        "perfectly fine — but that is a Model Selection responsibility. MS "
        "reads the same authoritative target column name from state and can "
        "transform `y_train` internally before model training, then invert "
        "before reporting. FE's job here is just to keep the target column "
        f"values pristine and out of the feature transformer pipeline.\n"
        "\n"
        f"In the test set: if `{candidate}` is missing or all-NaN (held-out "
        "predict-only mode), do NOT synthesise values — preserve NaN as-is."
    )


def _is_target_related_error(stderr: str) -> bool:
    text = (stderr or "").lower()
    return any(
        token in text
        for token in (
            "keyerror: 'y'",
            'keyerror: "y"',
            "target column",
            "detect_target_column",
            "could not detect binary target column",
            "no reliable target",
        )
    )


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
    target_hint: str = "",
) -> str:
    """
    Ask the LLM to generate feature engineering code.
    Mirrors AutoKaggle Developer-style prompt construction with the prior-stage summary injected.
    On LLM failure, falls back to a stub template.
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
        previous_summary=previous_summary if previous_summary else "No previous-stage summary.",
        feedback_section=feedback_section,
        dataset_description_block=_dataset_description_block(dataset_description),
        target_hint=target_hint or "(no runtime target hint available)",
    )

    reply = call_llm(_SYSTEM_PROMPT_GENERATE, user_prompt, agent_name="feature_engineering")

    if reply:
        code = extract_code(reply)
        if code:
            return code

    # Fallback: return template code.
    logger.warning("LLM call failed or returned no code; using fallback code template.")
    return _deterministic_fallback_fe_code(plan, data_path, test_data_path, previous_summary)


def _fix_code(
    code: str,
    error: str,
    plan: dict,
    previous_summary: str,
    dataset_description: str = "",
    target_hint: str = "",
) -> str:
    """
    Ask the LLM to fix code given the traceback.
    Simplified analogue of AutoKaggle PROMPT_DEVELOPER_DEBUG_FIX.
    If the LLM is unavailable, returns the original code unchanged.
    """
    user_prompt = _USER_PROMPT_FIX.format(
        code=code,
        error=error,
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        previous_summary=previous_summary if previous_summary else "No previous-stage summary.",
        dataset_description_block=_dataset_description_block(dataset_description),
        target_hint=target_hint or "(no runtime target hint available)",
    )
    reply = call_llm(_SYSTEM_PROMPT_FIX, user_prompt, agent_name="feature_engineering")

    if reply:
        fixed_code = extract_code(reply)
        if fixed_code:
            return fixed_code

    logger.warning("LLM fix call failed; returning original code.")
    return code


def _fallback_fe_code(plan: dict, data_path: str, previous_summary: str) -> str:
    """Fallback stub template when the LLM is unavailable."""
    path_lit = json.dumps(data_path)
    return f"""\
import pandas as pd
import numpy as np
import json

df = pd.read_csv({path_lit})
# Feature engineering plan: {plan.get('objectives', [])}
# Previous-stage summary: {previous_summary}

# Work on a copy to keep transformations explicit.
df = df.copy()
original_cols = list(df.columns)
created_features = []

# Preserve a likely supervised target column unchanged if one is present.
semantic_exact = {{"target", "label", "labels", "outcome", "response", "dependent", "y", "class"}}
semantic_contains = ("target", "label", "outcome", "response", "dependent")
lowered = {{c: str(c).strip().lower().replace(" ", "_") for c in df.columns}}
exact_hits = [c for c, lc in lowered.items() if lc in semantic_exact]
contains_hits = [c for c, lc in lowered.items() if any(tok in lc for tok in semantic_contains)]
target_col = (exact_hits or contains_hits or [None])[0]

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
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target_col]
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
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target_col]
base_numeric = [c for c in numeric_cols[:4] if c in df.columns]
for i in range(len(base_numeric)):
    for j in range(i + 1, min(len(base_numeric), i + 3)):
        c1, c2 = base_numeric[i], base_numeric[j]
        inter_col = f"{{c1}}_x_{{c2}}"
        df[inter_col] = pd.to_numeric(df[c1], errors="coerce").fillna(0) * pd.to_numeric(df[c2], errors="coerce").fillna(0)
        created_features.append(inter_col)

# 4. Safe categorical encoding: one-hot only for low-cardinality object/category columns.
categorical_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns.tolist() if c != target_col]
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


def _deterministic_fallback_fe_code(
    plan: dict,
    data_path: str,
    test_data_path: str,
    previous_summary: str,
) -> str:
    """Deterministic fallback template for robust train/test feature engineering."""
    train_path_lit = json.dumps(data_path)
    test_path_lit = json.dumps(test_data_path)
    return f"""\
import pandas as pd
import numpy as np

train_df = pd.read_csv({train_path_lit})
test_df = pd.read_csv({test_path_lit})
# Feature engineering plan: {plan.get('objectives', [])}
# Previous phase summary: {previous_summary}

def infer_target_col(df):
    semantic_exact = {{"target", "label", "labels", "outcome", "response", "dependent", "y", "class"}}
    semantic_contains = ("target", "label", "outcome", "response", "dependent")
    lowered = {{c: str(c).strip().lower().replace(" ", "_") for c in df.columns}}
    exact_hits = [c for c, lc in lowered.items() if lc in semantic_exact]
    contains_hits = [c for c, lc in lowered.items() if any(tok in lc for tok in semantic_contains)]
    if exact_hits or contains_hits:
        return (exact_hits or contains_hits)[0]
    return df.columns[-1] if len(df.columns) > 0 else None

target_col = infer_target_col(train_df)
if target_col and target_col not in test_df.columns:
    target_col = None

target_train = train_df[target_col].copy() if target_col else None
target_test = test_df[target_col].copy() if target_col else None
train_features = train_df.drop(columns=[target_col]) if target_col else train_df.copy()
test_features = test_df.drop(columns=[target_col]) if target_col else test_df.copy()

original_cols = list(train_features.columns)
created_features = []

datetime_candidates = []
for col in train_features.columns:
    lc = str(col).lower()
    if any(token in lc for token in ("date", "time", "timestamp", "period")):
        datetime_candidates.append(col)

for col in datetime_candidates:
    parsed_train = pd.to_datetime(train_features[col], errors="coerce")
    parsed_test = pd.to_datetime(test_features[col], errors="coerce")
    if parsed_train.notna().sum() > 0:
        train_features[col] = parsed_train
        test_features[col] = parsed_test
        for suffix, accessor in (("year", "year"), ("month", "month"), ("day", "day")):
            new_col = f"{{col}}_{{suffix}}"
            train_features[new_col] = getattr(parsed_train.dt, accessor)
            test_features[new_col] = getattr(parsed_test.dt, accessor)
            created_features.append(new_col)

numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    median = pd.to_numeric(train_features[col], errors="coerce").median()
    train_features[col] = pd.to_numeric(train_features[col], errors="coerce").fillna(median)
    test_features[col] = pd.to_numeric(test_features[col], errors="coerce").fillna(median)

for col in numeric_cols:
    series = pd.to_numeric(train_features[col], errors="coerce")
    if series.notna().sum() == 0:
        continue
    min_val = series.min()
    if pd.notna(min_val) and min_val >= 0:
        log_col = f"{{col}}_log1p"
        sqrt_col = f"{{col}}_sqrt"
        train_features[log_col] = np.log1p(series)
        train_features[sqrt_col] = np.sqrt(series)
        test_nonneg = pd.to_numeric(test_features[col], errors="coerce").fillna(0).clip(lower=0)
        test_features[log_col] = np.log1p(test_nonneg)
        test_features[sqrt_col] = np.sqrt(test_nonneg)
        created_features.extend([log_col, sqrt_col])

base_numeric = [c for c in numeric_cols[:4] if c in train_features.columns]
for i in range(len(base_numeric)):
    for j in range(i + 1, min(len(base_numeric), i + 3)):
        c1, c2 = base_numeric[i], base_numeric[j]
        inter_col = f"{{c1}}_x_{{c2}}"
        train_features[inter_col] = train_features[c1].fillna(0) * train_features[c2].fillna(0)
        test_features[inter_col] = test_features[c1].fillna(0) * test_features[c2].fillna(0)
        created_features.append(inter_col)

categorical_cols = train_features.select_dtypes(include=["object", "category"]).columns.tolist()
low_card_cols = []
for col in categorical_cols:
    nunique = train_features[col].nunique(dropna=True)
    if 1 < nunique <= 10:
        low_card_cols.append(col)

if low_card_cols:
    train_features = pd.get_dummies(train_features, columns=low_card_cols, dummy_na=False)
    test_features = pd.get_dummies(test_features, columns=low_card_cols, dummy_na=False)
    created_features.extend([c for c in train_features.columns if c not in original_cols and c not in created_features])

test_features = test_features.reindex(columns=train_features.columns, fill_value=0)
train_features = train_features.dropna(axis=1, how="all")
test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

if target_col:
    train_features[target_col] = target_train.values
    test_features[target_col] = target_test.values

train_output_path = "./workspace/fe_output.csv"
test_output_path = "./workspace/fe_test_output.csv"
train_features.to_csv(train_output_path, index=False)
test_features.to_csv(test_output_path, index=False)
print(f"Feature engineering complete. Original cols: {{len(original_cols)}}; final train cols: {{train_features.shape[1]}}; created features: {{len(created_features)}}; saved to {{train_output_path}} and {{test_output_path}}")
"""


def _extract_eda_summary(stdout: str) -> str:
    """Deprecated: EDA text now comes from ``agents.eda_analysis``. Kept for compatibility."""
    text = (stdout or "").strip()
    if not text:
        return ""
    start = text.find(EDA_SUMMARY_START)
    end = text.find(EDA_SUMMARY_END)
    if start == -1 or end == -1 or end <= start:
        return ""
    body = text[start + len(EDA_SUMMARY_START):end].strip()
    return body
