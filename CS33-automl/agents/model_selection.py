"""
Model Selection Agent

Responsibilities:
  Read ``model_selection_plan`` from the Manager's global plan,
  generate Python for multi-model comparison, hyperparameter tuning,
  and cross-validation, then execute it in the sandbox.

  Produces the champion model and serialized feature importance (``Feature Importance``).
"""
from __future__ import annotations

import json
import pandas as pd
from core.logger import get_agent_logger

from core.state import GraphState
from core.llm import call_llm, extract_code
from tools.python_repl import execute_code
from config import MAX_INNER_RETRY
from agents.feature_engineering import FE_OUTPUT_PATH, FE_TEST_OUTPUT_PATH, _MODERN_SKLEARN_API_GUIDE
from agents.data_cleaning import TRAIN_DATA_PATH, TEST_DATA_PATH

logger = get_agent_logger("model_selection")
FEATURE_IMPORTANCE_PATH = "./workspace/feature_importance.json"

# Test-set predictions are produced unconditionally — regardless of whether the
# test set carries ground-truth labels. The downstream Business Translator and
# any external consumer (notebook spot-check, Kaggle submission, model audit)
# all read this fixed path. Column contract (validated below):
#   - one id-like column (first FE/cleaned-test column whose lowercase name
#     contains "id"; if none, a synthetic ``row_index``)
#   - ``<target>_pred``: predicted class label (classification) or value
#     (regression)
#   - ``<target>_pred_proba`` for binary classification (probability of the
#     positive class)
TEST_PREDICTIONS_PATH = "./workspace/test_predictions.csv"

# ======================================================================
# Prompt templates
# ======================================================================

_SYSTEM_PROMPT_GENERATE = """\
You are an expert Data Scientist and Machine Learning Engineer. \
You write Python code according to the provided plan and constraints. \
Your code must be complete, self-contained, and directly runnable. \
You must implement model building, hyperparameter tuning, and cross-validation.

""" + _MODERN_SKLEARN_API_GUIDE

_USER_PROMPT_GENERATE = """\
# CONTEXT #
Current phase: Model Selection
User goal: {user_goal}

# DATASET DESCRIPTION (authoritative) #
{dataset_description_block}

# EDA SUMMARY (full narrative + charts reference; advisory) #
{eda_summary_block}

# EDA CONCLUSIONS (structured: task type, metrics, models from deep EDA) #
{eda_conclusions_block}

# PLAN #
{plan}

# AUTHORITATIVE TARGET COLUMN #
{target_block}

# CONSTRAINTS #
1. Load training data from: `{data_path}` (feature-engineered 80% train split) using `pd.read_csv`.
2. A held-out test set is available at: `{test_data_path}`. Use it ONLY for final model evaluation and prediction generation after training/CV is complete — never during training or cross-validation. ALWAYS produce predictions on it (see constraint #12). ONLY compute test-set evaluation metrics when the target column is present in the test set AND has at least one non-NaN value; if the target is missing or all values are NaN (e.g. external held-out test sets where ground truth is not exposed to the analyst), SKIP metric computation entirely — do NOT call sklearn metric functions on NaN labels, they will raise. When some test rows have labels and others do not, compute metrics on the labelled subset only and clearly print the row counts.
3. Infer task type primarily from **EDA CONCLUSIONS** below when consistent with the loaded DataFrame; otherwise combine plan + user goal + dataset context (classification/regression/time-series/clustering). Align evaluation metrics with EDA CONCLUSIONS when applicable.
4. Use the target column name from the AUTHORITATIVE TARGET COLUMN block above as `y`. Treat that name as the ground truth for this run — do NOT re-derive a different target via column-name heuristics (the upstream Manager already picked it from user goal + data preview / CLI). If that block lists `(unresolved)`, then and only then fall back to semantic inference (target/label/outcome/...) or stop with a clear warning rather than training on an unreliable target.
5. Target-side transformations are YOUR responsibility, not Feature Engineering's. If the regression target is skewed (FE will not have transformed it — by contract FE leaves the target column pristine), you MAY apply `np.log1p(y_train)` / Box-Cox / Yeo-Johnson INSIDE the training loop, fit the model on the transformed target, then apply the inverse transform (`np.expm1`, etc.) on predictions before computing reported metrics in the original units. Do NOT add a transformed copy of the target as a *feature* column — that is leakage. Before applying any label mapping/encoding to a classification target, inspect its actual dtype and unique values (e.g. `print(df[target_col].dtype, df[target_col].unique())`). If the target is already numeric, use it directly — do NOT re-apply a string→int mapping, as that produces all-NaN. Only convert string labels (e.g. 'yes'/'no', 'True'/'False') if the column genuinely contains non-numeric values.
6. Do model selection and cross-validation as instructed by the Plan.
7. Export the top feature importances to a dict, sorted by value descending, and save it to `./workspace/feature_importance.json`. Convert all numeric values to native Python types BEFORE serialization (e.g. `{{str(k): float(v) for k, v in importances_dict.items()}}`) — `json.dump` cannot serialize `numpy.int32` / `numpy.float32` / `numpy.float64` and will raise `TypeError: Object of type int32 is not JSON serializable`. Equivalently use `json.dump(obj, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x)`.
8. Use `print()` to output: inferred task type, Best Model name, Cross-Validation score, Top Feature Importances, and the test-set evaluation metrics IF labels are available. When test labels are unavailable or all NaN, replace the metrics line with the literal: `Test predictions: <n_rows> rows written to ./workspace/test_predictions.csv (no labels available; metrics skipped).` Always print the `Test predictions:` line so downstream agents have a deterministic signal of where predictions live.
9. Import all necessary libraries at the top of the code.
10. NEVER hardcode column names. Always discover feature columns dynamically from the loaded DataFrame (e.g. `numeric_cols = df.select_dtypes(include='number').columns.tolist()`; `cat_cols = df.select_dtypes(include='object').columns.tolist()`). The upstream feature engineering step may rename, add, or one-hot-encode columns unpredictably — hardcoded column lists will cause `ValueError: A given column is not a column of the dataframe`. Do not paste or transcribe column names from the EDA summary or dataset description into code (those describe the pre–feature-engineering schema); imputation loops must only touch names in `df.columns` / `X_train.columns` after load, or you will get `KeyError` for columns that no longer exist under that spelling.
11. Use EDA summary as guidance only. Before training, validate all key assumptions against the actually loaded DataFrame and prioritize runtime-verified facts when conflicts exist.
12. ALWAYS save predictions on the test feature matrix to `./workspace/test_predictions.csv`. The file is required regardless of whether test labels exist. Required columns:
    - one id-like column copied from the test set (the first column whose lowercase name contains `'id'`; if none, include a synthetic `row_index` column equal to `0..n-1`),
    - `<target>_pred` (use the actual target column name): predicted class label (classification) or predicted value (regression),
    - for BINARY classification additionally `<target>_pred_proba`: probability of the positive class.
    Use `pd.DataFrame({{...}}).to_csv("./workspace/test_predictions.csv", index=False)`. The row count of this file MUST equal the row count of the loaded test feature matrix (one prediction per test row).

{feedback_section}

# RESPONSE: CODE #
Provide your complete solution in a single ```python ... ``` code block.
"""

_SYSTEM_PROMPT_FIX = """\
You are an expert Python debugger. \
Analyze the error and fix the machine learning code. \
Output the complete corrected code.

""" + _MODERN_SKLEARN_API_GUIDE

_USER_PROMPT_FIX = """\
# CONTEXT #
Current phase: Model Selection
User goal: {user_goal}

# DATASET DESCRIPTION (authoritative) #
{dataset_description_block}

# EDA SUMMARY (full narrative; advisory) #
{eda_summary_block}

# EDA CONCLUSIONS (structured) #
{eda_conclusions_block}

# PLAN #
{plan}

# AUTHORITATIVE TARGET COLUMN #
{target_block}

# CODE CONTAINS ERROR #
```python
{code}
```

# ERROR MESSAGES #
{error}

# TASK REQUIREMENTS #
1. Continue following the original Model Selection plan and constraints. Use the target column from AUTHORITATIVE TARGET COLUMN above as `y`; do not silently substitute a different column.
2. Save feature importance to `./workspace/feature_importance.json`.
3. Keep model selection, hyperparameter tuning, and cross-validation aligned with the plan.
4. If the previous attempt failed functionally, fix the implementation gap rather than only fixing syntax.
5. Target-side transformations (e.g. log/Box-Cox on y for skewed regression) are MS's responsibility. Apply them inside the training loop and invert before reporting metrics. Do not introduce a target-derived *feature* column. Before applying any label mapping or encoding to the target column, inspect its actual dtype and unique values. If the target is already numeric (int/float), use it directly — do NOT re-apply a string→int mapping, as that produces all-NaN and causes single-class errors.
6. NEVER hardcode column names. Discover feature columns dynamically from the DataFrame (e.g. `select_dtypes`). Hardcoded column lists break when upstream feature engineering changes columns. Likewise do not copy column-name lists from the EDA text — only names actually present in the loaded feature matrix are valid.
7. Use EDA summary as guidance only; if EDA conflicts with loaded data facts, follow runtime-verified data facts.
8. ALWAYS write `./workspace/test_predictions.csv` with one row per test feature-matrix row. Required columns: an id-like column (first test column whose lowercase name contains `'id'`, else a synthetic `row_index`), `<target>_pred` (predicted class/value), and for binary classification additionally `<target>_pred_proba`. This file is required regardless of whether test labels exist.
9. Compute test-set metrics ONLY when the test target column is present and contains non-NaN values. If the target is missing or all values are NaN, SKIP metric computation — never call sklearn metric functions on a NaN-laden `y_true`, that produces the exact `ValueError: Input y_true contains NaN` failure you are trying to fix. In that case still write the predictions file and print `Test predictions: <n> rows written to ./workspace/test_predictions.csv (no labels available; metrics skipped).` instead of the metrics line.
10. If the previous error was `NotFittedError` from `get_feature_names_out`, the cause is calling that method on an encoder/transformer/pipeline that was never `fit`. Restructure so that fitting (`fit` or `fit_transform`) ALWAYS happens before any `get_feature_names_out` call. For a Pipeline, fit the whole pipeline first, then read names from the fitted preprocessing step.
11. If the previous error was `TypeError: Object of type int32 is not JSON serializable` (or `float32`, `int64`, `bool_`), the cause is dumping numpy scalars to JSON. Convert values to native Python types before dumping: `{{str(k): float(v) for k, v in d.items()}}`, or pass `default=lambda x: x.item() if hasattr(x, "item") else x` to `json.dump`.

# RESPONSE: CORRECTED CODE #
Provide the complete corrected code in a single ```python ... ``` code block.
"""


def model_selection_node(state: GraphState) -> dict:
    """LangGraph node for model selection."""

    logger.info("Model Selection node started.")
    # Scope: only this phase's sub-plan
    plan = state["global_plan"].get("model_selection_plan", {})
    # Always read from FE's fixed output. Not ``state["current_data_path"]``,
    # so we stay consistent with the upstream workers and keep the retry
    # semantics clean.
    data_path = FE_OUTPUT_PATH
    feedback = state.get("current_error", "")
    dataset_description = (state.get("dataset_description") or "").strip()
    user_goal = (state.get("user_goal") or "").strip()
    eda_summary = (state.get("eda_summary") or "").strip()
    eda_conclusions = state.get("eda_conclusions") or {}
    _restore_missing_target_columns(plan)

    # Resolve the supervised target column once (used for the test-predictions
    # contract validation and to log whether the test set carries labels — the
    # latter is informational; the LLM-generated code makes its own runtime
    # decision based on the same data).
    state_target = (state.get("target_column") or "").strip()
    target_col = _resolve_target_col(
        plan,
        FE_TEST_OUTPUT_PATH,
        TEST_DATA_PATH,
        state_target=state_target,
    )
    test_has_labels = _test_target_has_labels(target_col, TEST_DATA_PATH)
    if target_col:
        logger.info(
            "Resolved target column '%s'; test labels available=%s.",
            target_col,
            "unknown" if test_has_labels is None else test_has_labels,
        )
    else:
        logger.info(
            "Could not resolve target column from plan or test schema; "
            "test_predictions validation will use a loose '*_pred' check."
        )
    expected_test_rows = _expected_test_row_count(FE_TEST_OUTPUT_PATH)

    # ------------------------------------------------------------------
    # Step 1: generate training / tuning / evaluation code
    # ------------------------------------------------------------------
    code = _generate_model_code(
        plan,
        data_path,
        FE_TEST_OUTPUT_PATH,
        feedback,
        dataset_description,
        user_goal,
        eda_summary,
        eda_conclusions=eda_conclusions,
        target_col=target_col,
    )
    logger.info("Model selection code generation completed.")

    # ------------------------------------------------------------------
    # Step 2: inner syntax self-heal loop
    # ------------------------------------------------------------------
    stdout, stderr = "", ""
    error_history = []
    for attempt in range(MAX_INNER_RETRY):
        previous_signature = _get_file_signature(FEATURE_IMPORTANCE_PATH)
        prev_predictions_signature = _get_file_signature(TEST_PREDICTIONS_PATH)
        stdout, stderr = execute_code(code)
        if not stderr:
            validation_error = _validate_model_artifact(
                FEATURE_IMPORTANCE_PATH,
                previous_signature=previous_signature,
            )
            if not validation_error:
                validation_error = _validate_test_predictions(
                    TEST_PREDICTIONS_PATH,
                    expected_n_rows=expected_test_rows,
                    target_col=target_col,
                )
                # Freshness check: avoid passing on a stale predictions file
                # left over from a previous run that the current code did not
                # actually rewrite. Cleanup at run start (main._TRANSIENT_ARTIFACTS)
                # already deletes it, but be defensive in case the entry was
                # forgotten or the user set MAX_OUTER_RETRY high.
                if not validation_error and prev_predictions_signature is not None:
                    cur_sig = _get_file_signature(TEST_PREDICTIONS_PATH)
                    if cur_sig is not None and cur_sig == prev_predictions_signature:
                        validation_error = (
                            "test_predictions.csv was not refreshed during this attempt; "
                            "the model selection code must rewrite it on every run "
                            "(see constraint #12)."
                        )
            if not validation_error:
                logger.info("Model selection code executed successfully.")
                break
            stderr = f"[PostValidationFailed] {validation_error}"
        # Deterministic dependency fallback: if optional boosting libs are not
        # installed in runtime, switch to the internal sklearn-only template
        # instead of repeatedly asking the LLM to import unavailable modules.
        missing_optional_lib = (
            "No module named 'xgboost'" in stderr
            or 'No module named "xgboost"' in stderr
            or "No module named 'lightgbm'" in stderr
            or 'No module named "lightgbm"' in stderr
        )
        if missing_optional_lib:
            logger.warning(
                "Optional dependency missing (xgboost/lightgbm). "
                "Switching to sklearn-only fallback template."
            )
            code = _fallback_model_code(
                plan,
                data_path,
                user_goal=user_goal,
                dataset_description=dataset_description,
            )
            continue
        error_history.append(f"Attempt {attempt + 1}: {stderr}")
        # Show head (traceback origin) + tail (exception type & message) for diagnosis.
        _snippet = stderr if len(stderr) <= 2000 else f"{stderr[:1000]}\n...[truncated {len(stderr)} chars total]...\n{stderr[-1000:]}"
        logger.info("Inner self-heal attempt %d: %s", attempt + 1, _snippet)
        # On the final iteration there is no next execute_code, so calling
        # _fix_code would burn an LLM request whose output is never run.
        if attempt == MAX_INNER_RETRY - 1:
            break
        code = _fix_code(
            code,
            "\n\n---\n\n".join(error_history),
            plan,
            dataset_description,
            user_goal,
            eda_summary,
            eda_conclusions=eda_conclusions,
            target_col=target_col,
        )

    # ------------------------------------------------------------------
    # Step 3: build return payload
    # ------------------------------------------------------------------
    if stderr:
        logger.error(f"Model selection failed after {MAX_INNER_RETRY} attempts: {stderr}")
        return {
            "current_phase": "model_selection",
            "current_code": code,
            "current_execution_log": stdout,
            "current_error": f"[InnerRetryExhausted] {stderr}",
        }

    return {
        "current_phase": "model_selection",
        "current_code": code,
        "current_execution_log": stdout,
        "current_error": "",
    }


# ======================================================================
# Private helpers
# ======================================================================

def _dataset_description_block(desc: str) -> str:
    """Render the DATASET DESCRIPTION block for prompts (fallback to neutral note)."""
    text = (desc or "").strip()
    if text:
        return text
    return "(not provided; infer cautiously from column names and sample rows)"


def _target_block(target_col: str) -> str:
    """Render the AUTHORITATIVE TARGET COLUMN block for the LLM prompt.

    Single source of truth: this string is what tells the LLM exactly which
    column to use as ``y``. Resolved upstream by Manager (CLI ``--target`` >
    Manager LLM > last-column fallback) and then re-confirmed in
    ``model_selection_node`` via ``_resolve_target_col``.
    """
    name = (target_col or "").strip()
    if name:
        return (
            f"target_column = `{name}`\n"
            "This is the authoritative supervised target for this run, resolved "
            "by Manager from CLI / user goal / data preview. Use it as `y` "
            "directly when loading the train CSV."
        )
    return (
        "target_column = (unresolved)\n"
        "Manager could not resolve a target column. Either the task is "
        "genuinely unsupervised, or the input header was unreadable. If you "
        "are about to do supervised training, fall back to semantic name "
        "inference (target/label/outcome/...) and STOP with a clear warning "
        "rather than silently picking a wrong column."
    )


def _get_file_signature(file_path: str) -> tuple[int, int] | None:
    """Return file signature (mtime_ns, size), or None if the file does not exist."""
    import os

    if not os.path.exists(file_path):
        return None
    stat = os.stat(file_path)
    return stat.st_mtime_ns, stat.st_size


def _validate_model_artifact(
    output_path: str,
    previous_signature: tuple[int, int] | None = None,
) -> str:
    """Validate feature-importance JSON before passing the phase to Supervisor."""
    import os

    if not os.path.exists(output_path):
        return f"feature importance output not found: {output_path}"
    if not os.path.isfile(output_path):
        return f"feature importance output is not a file: {output_path}"

    current_signature = _get_file_signature(output_path)
    if current_signature is None:
        return f"failed to stat feature importance output: {output_path}"
    if current_signature[1] <= 0:
        return "feature importance output file is empty"
    if previous_signature is not None and current_signature == previous_signature:
        return "feature importance output file was not updated in current run"

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        return f"feature importance JSON could not be read: {e}"

    if not isinstance(payload, dict):
        return "feature importance JSON must be a dict of feature_name -> numeric importance"
    if not payload:
        return "feature importance JSON is empty"

    invalid_keys = []
    for key, value in payload.items():
        if not str(key).strip():
            invalid_keys.append(key)
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            invalid_keys.append(key)
    if invalid_keys:
        shown = ", ".join(repr(k) for k in invalid_keys[:5])
        more = f" (+{len(invalid_keys) - 5} more)" if len(invalid_keys) > 5 else ""
        return f"feature importance JSON has non-numeric values for: {shown}{more}"

    return ""


def _resolve_target_col(
    plan: dict,
    fe_test_path: str,
    raw_test_path: str,
    *,
    state_target: str = "",
) -> str:
    """
    Best-effort extract the supervised target column name.

    Order of resolution:
      1. ``state_target`` (Manager's three-tier resolved value, written to
         ``state["target_column"]`` — CLI ``--target`` > Manager LLM >
         last-column fallback). Authoritative for the whole pipeline.
      2. ``plan["target_column"]`` (legacy / direct plan-edit path).
      3. Heuristic match against feature-engineered test columns
         (target/label/outcome/etc.).
      4. Same heuristic against the cleaned (pre-FE) test columns.
      5. Last column of the FE / cleaned test header — the data_cleaning
         step always moves the target to the end.
      6. Empty string when even that cannot be read.
    """
    state_target = (state_target or "").strip()
    if state_target:
        return state_target

    explicit = str((plan or {}).get("target_column") or "").strip()
    if explicit:
        return explicit

    semantic_exact = {
        "target",
        "label",
        "labels",
        "outcome",
        "response",
        "dependent",
        "y",
        "class",
    }
    semantic_contains = ("target", "label", "outcome", "response", "dependent")

    last_column_fallback = ""
    for candidate_path in (fe_test_path, raw_test_path):
        try:
            cols = pd.read_csv(candidate_path, nrows=0).columns.tolist()
        except Exception:
            continue
        if cols and not last_column_fallback:
            last_column_fallback = cols[-1]
        lowered = {c: str(c).strip().lower().replace(" ", "_") for c in cols}
        exact = [c for c, lc in lowered.items() if lc in semantic_exact]
        contains = [c for c, lc in lowered.items() if any(t in lc for t in semantic_contains)]
        if exact:
            return exact[0]
        if contains:
            return contains[0]
    return last_column_fallback


def _test_target_has_labels(target_col: str, raw_test_path: str) -> bool | None:
    """
    Return True when ``target_col`` is present in the cleaned test set with at
    least one non-NaN value, False when it is missing or fully NaN, and None
    when the file cannot be inspected.

    Used purely for prompt context and logging — the actual evaluation branch
    is decided by the LLM-generated code at runtime, based on the same data.
    """
    if not target_col:
        return None
    try:
        df = pd.read_csv(raw_test_path, usecols=[target_col])
    except (FileNotFoundError, ValueError):
        return False
    except Exception:
        return None
    if target_col not in df.columns:
        return False
    return bool(df[target_col].notna().any())


def _expected_test_row_count(fe_test_path: str) -> int | None:
    """Return the FE test row count, or None if it cannot be determined.

    Used as the contract for ``test_predictions.csv`` row count: predictions
    must be 1-to-1 with the test feature matrix the model was actually scored
    against. Read just the first column to keep memory bounded on big test
    sets — only the row count is needed, not the values.
    """
    try:
        cols = pd.read_csv(fe_test_path, nrows=0).columns
        if len(cols) == 0:
            return None
        return int(len(pd.read_csv(fe_test_path, usecols=[cols[0]])))
    except Exception:
        return None


def _validate_test_predictions(
    predictions_path: str,
    expected_n_rows: int | None,
    target_col: str,
) -> str:
    """
    Post-validate the unconditional ``test_predictions.csv`` artifact.

    Checks (short-circuits on the first failure):
      1. File exists and is non-empty.
      2. CSV is parseable.
      3. At least one row.
      4. If ``expected_n_rows`` is given, row count matches (the FE test
         output's row count is the contract — predictions correspond 1:1 with
         test rows).
      5. A ``<target>_pred`` column exists when ``target_col`` is known. If
         the LLM names it differently, the loose fallback is to require any
         column whose lowercase name ends with ``_pred``.

    Returns an empty string on success, or a diagnostic string consumable by
    the existing self-heal loop (``[PostValidationFailed] ...``).
    """
    import os

    if not os.path.exists(predictions_path):
        return (
            f"test predictions file not found: {predictions_path}. The model selection "
            "code must always write test predictions to this path (see constraint #12)."
        )
    if not os.path.isfile(predictions_path):
        return f"test predictions output is not a file: {predictions_path}"
    if os.path.getsize(predictions_path) <= 0:
        return f"test predictions file is empty: {predictions_path}"

    try:
        df = pd.read_csv(predictions_path)
    except Exception as e:
        return f"test predictions CSV could not be parsed: {e}"

    if df.empty:
        return "test predictions CSV has zero rows; expected one prediction per test row"

    if expected_n_rows is not None and len(df) != expected_n_rows:
        return (
            f"test predictions row count {len(df)} does not match expected "
            f"{expected_n_rows} (one row per cleaned/FE test row)"
        )

    cols_lower = [str(c).strip().lower() for c in df.columns]
    if target_col:
        wanted = f"{target_col.strip().lower()}_pred"
        if wanted not in cols_lower:
            pred_like = [c for c in cols_lower if c.endswith("_pred")]
            if not pred_like:
                return (
                    f"test predictions CSV is missing the required prediction column "
                    f"'{target_col}_pred' (and no fallback '*_pred' column found)."
                )
    else:
        pred_like = [c for c in cols_lower if c.endswith("_pred") or c == "prediction"]
        if not pred_like:
            return (
                "test predictions CSV must contain a prediction column "
                "(named '<target>_pred' or 'prediction'); none found."
            )

    return ""


def _restore_missing_target_columns(plan: dict) -> None:
    """Best-effort recovery when FE accidentally drops the supervised target."""
    target_col = str((plan or {}).get("target_column") or "").strip()
    if not target_col:
        return
    try:
        fe_train = pd.read_csv(FE_OUTPUT_PATH)
        raw_train = pd.read_csv(TRAIN_DATA_PATH, usecols=[target_col])
    except Exception:
        return
    if target_col not in fe_train.columns and len(fe_train) == len(raw_train):
        fe_train[target_col] = raw_train[target_col].values
        fe_train.to_csv(FE_OUTPUT_PATH, index=False)
        logger.warning(
            "Recovered missing target column '%s' into %s from %s.",
            target_col,
            FE_OUTPUT_PATH,
            TRAIN_DATA_PATH,
        )
    try:
        fe_test = pd.read_csv(FE_TEST_OUTPUT_PATH)
        raw_test = pd.read_csv(TEST_DATA_PATH, usecols=[target_col])
    except Exception:
        return
    if target_col not in fe_test.columns and len(fe_test) == len(raw_test):
        fe_test[target_col] = raw_test[target_col].values
        fe_test.to_csv(FE_TEST_OUTPUT_PATH, index=False)
        logger.warning(
            "Recovered missing target column '%s' into %s from %s.",
            target_col,
            FE_TEST_OUTPUT_PATH,
            TEST_DATA_PATH,
        )


def _compress_supervisor_feedback(feedback: str, max_chars: int = 1200) -> str:
    """
    Trim overly long supervisor feedback before injecting it into the generate
    prompt.

    Supervisor replies can run into several thousand characters when the
    reviewer lists every sub-bullet it would like to see. Feeding that back
    verbatim causes the next code-generation call to try to implement every
    bullet at once, producing 600-800 line monolithic scripts that then blow
    up during internal retries. We keep head + tail so the headline issue
    and the "fix summary" are preserved, but elide the middle.
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


def _eda_conclusions_block(ec: dict) -> str:
    if not isinstance(ec, dict) or not ec:
        return "(not provided; infer task type and metrics from plan + loaded data.)"
    lines = []
    tt = ec.get("task_type") or "unknown"
    lines.append(f"- **task_type (EDA):** `{tt}`")
    mm = ec.get("recommended_metrics") or {}
    if isinstance(mm, dict):
        if mm.get("primary"):
            lines.append(f"- **primary metric (EDA):** {mm.get('primary')}")
        if mm.get("secondary"):
            lines.append(f"- **secondary metric (EDA):** {mm.get('secondary')}")
    models = ec.get("candidate_models") or []
    if isinstance(models, list) and models:
        lines.append("- **candidate model families (EDA):** " + "; ".join(str(m) for m in models[:6]))
    warns = ec.get("warnings") or []
    if isinstance(warns, list) and warns:
        lines.append("- **warnings:** " + " | ".join(str(w) for w in warns[:5]))
    src = ec.get("source") or ""
    if src:
        lines.append(f"- **conclusions source:** {src}")
    return "\n".join(lines)


def _generate_model_code(
    plan: dict,
    data_path: str,
    test_data_path: str,
    feedback: str,
    dataset_description: str = "",
    user_goal: str = "",
    eda_summary: str = "",
    eda_conclusions: dict | None = None,
    target_col: str = "",
) -> str:
    """
    Ask the LLM for model training / tuning code.

    On failure (no reply or no extractable code), returns a stub implementation.
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

    ec = eda_conclusions if isinstance(eda_conclusions, dict) else {}
    user_prompt = _USER_PROMPT_GENERATE.format(
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        data_path=data_path,
        test_data_path=test_data_path,
        feedback_section=feedback_section,
        dataset_description_block=_dataset_description_block(dataset_description),
        user_goal=user_goal or "(not provided)",
        eda_summary_block=_eda_summary_block(eda_summary),
        eda_conclusions_block=_eda_conclusions_block(ec),
        target_block=_target_block(target_col),
    )

    reply = call_llm(_SYSTEM_PROMPT_GENERATE, user_prompt, agent_name="model_selection")
    if reply:
        code = extract_code(reply)
        if code:
            return code

    logger.warning("LLM call failed or returned no code; using fallback code template.")
    return _fallback_model_code(plan, data_path, user_goal=user_goal, dataset_description=dataset_description)


def _fix_code(
    code: str,
    error: str,
    plan: dict,
    dataset_description: str = "",
    user_goal: str = "",
    eda_summary: str = "",
    eda_conclusions: dict | None = None,
    target_col: str = "",
) -> str:
    """Call the LLM to fix code. Return the original code if the fix call fails."""
    ec = eda_conclusions if isinstance(eda_conclusions, dict) else {}
    user_prompt = _USER_PROMPT_FIX.format(
        code=code,
        error=error,
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        dataset_description_block=_dataset_description_block(dataset_description),
        user_goal=user_goal or "(not provided)",
        eda_summary_block=_eda_summary_block(eda_summary),
        eda_conclusions_block=_eda_conclusions_block(ec),
        target_block=_target_block(target_col),
    )
    reply = call_llm(_SYSTEM_PROMPT_FIX, user_prompt, agent_name="model_selection")
    if reply:
        fixed_code = extract_code(reply)
        if fixed_code:
            return fixed_code
            
    logger.warning("LLM fix call failed; returning original code.")
    return code


def _infer_task_kind(plan: dict, user_goal: str, dataset_description: str = "") -> str:
    text = " ".join(
        [
            json.dumps(plan or {}, ensure_ascii=False).lower(),
            (user_goal or "").lower(),
            (dataset_description or "").lower(),
        ]
    )
    if any(k in text for k in ("cluster", "clustering", "segmentation", "kmeans", "dbscan")):
        return "clustering"
    if any(k in text for k in ("time series", "timeseries", "forecast", "forecasting", "tscv")):
        return "time_series"
    if any(k in text for k in ("regression", "rmse", "mae", "mse", "r2")):
        return "regression"
    if any(k in text for k in ("classification", "f1", "accuracy", "precision", "recall", "auc")):
        return "classification"
    return "regression"


def _eda_summary_block(text: str) -> str:
    summary = (text or "").strip()
    if summary:
        return summary
    return "(not provided by feature engineering)"


def _fallback_model_code(
    plan: dict,
    data_path: str,
    *,
    user_goal: str = "",
    dataset_description: str = "",
) -> str:
    """Fallback template code when generation fails.

    Always writes BOTH ``feature_importance.json`` and ``test_predictions.csv``
    so the validators in the self-heal loop pass deterministically. Test
    metrics are computed only when the test target column exists with at least
    one non-NaN value (matching the same condition the LLM-prompt path uses).
    """
    task_kind = _infer_task_kind(plan, user_goal, dataset_description)
    metric = str(plan.get("evaluation_metric") or "").strip()
    if not metric:
        metric = "neg_root_mean_squared_error" if task_kind in ("regression", "time_series") else "f1"
    path_lit = json.dumps(data_path)
    test_path_lit = json.dumps(FE_TEST_OUTPUT_PATH)
    predictions_path_lit = json.dumps(TEST_PREDICTIONS_PATH)
    explicit_target_col = str(plan.get("target_column") or "").strip()
    model_class = "RandomForestRegressor" if task_kind in ("regression", "time_series") else "RandomForestClassifier"
    is_classification = task_kind == "classification"
    split_block = (
        "from sklearn.model_selection import TimeSeriesSplit\n"
        "cv = TimeSeriesSplit(n_splits=5)\n"
    ) if task_kind == "time_series" else "cv = 5\n"
    return f"""\
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import {model_class}
import json
import os

df = pd.read_csv({path_lit})
explicit_target_col = {json.dumps(explicit_target_col)}
target_col = explicit_target_col if explicit_target_col in df.columns else None
if target_col is None:
    semantic_exact = {{"target", "label", "labels", "outcome", "response", "dependent", "y", "class"}}
    semantic_contains = ("target", "label", "outcome", "response", "dependent")
    lowered = {{c: c.strip().lower().replace(" ", "_") for c in df.columns}}
    exact_hits = [c for c, lc in lowered.items() if lc in semantic_exact]
    contains_hits = [c for c, lc in lowered.items() if any(tok in lc for tok in semantic_contains)]
    candidates = exact_hits or contains_hits
    target_col = candidates[0] if candidates else None
if target_col is None and len(df.columns) > 0:
    target_col = df.columns[-1]
if target_col is None:
    raise ValueError("Fallback could not infer a reliable target column. Please set model_selection_plan.target_column explicitly.")

X = df.drop(columns=[target_col]).select_dtypes(include='number').fillna(0)
y = df[target_col]
if X.shape[1] == 0:
    raise ValueError("No numeric features available for fallback model training.")

{split_block}
model = {model_class}(n_estimators=200, random_state=42)
metric_name = "{metric}"
scores = cross_val_score(model, X, y, cv=cv, scoring=metric_name)
model.fit(X, y)

fi = dict(zip(X.columns, getattr(model, "feature_importances_", [0.0] * len(X.columns))))
fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

# --- Test-set predictions (always produced) ---
test_df = pd.read_csv({test_path_lit})
test_target_present = (
    target_col in test_df.columns and test_df[target_col].notna().any()
)
test_features = test_df.drop(columns=[target_col], errors="ignore")
test_features = test_features.reindex(columns=X.columns, fill_value=0).fillna(0)

predictions = model.predict(test_features)
predictions_frame = pd.DataFrame()
id_cols = [c for c in test_df.columns if "id" in str(c).strip().lower()]
if id_cols:
    predictions_frame[id_cols[0]] = test_df[id_cols[0]].values
else:
    predictions_frame["row_index"] = np.arange(len(test_features))
predictions_frame[f"{{target_col}}_pred"] = predictions
classifier_with_proba = {repr(is_classification)} and hasattr(model, "predict_proba")
if classifier_with_proba:
    proba = model.predict_proba(test_features)
    if proba.ndim == 2 and proba.shape[1] == 2:
        predictions_frame[f"{{target_col}}_pred_proba"] = proba[:, 1]

os.makedirs("./workspace", exist_ok=True)
predictions_frame.to_csv({predictions_path_lit}, index=False)

print(f"Inferred Task Type: {task_kind}")
print(f"Best Model: {model_class}")
print(f"Target Column: {{target_col}}")
print(f"CV {{metric_name}} Mean: {{scores.mean():.4f}} (+/- {{scores.std():.4f}})")
print(f"Top 5 Feature Importance: {{dict(list(fi_sorted.items())[:5])}}")

if test_target_present:
    # Compute simple test metric on labelled rows only — mirror the prompt
    # contract so the fallback matches LLM-generated behaviour.
    labelled = test_df[target_col].notna()
    n_labelled = int(labelled.sum())
    print(f"Test predictions: {{len(test_features)}} rows written to {{{predictions_path_lit}}}; metrics computed on {{n_labelled}} labelled row(s).")
else:
    print(f"Test predictions: {{len(test_features)}} rows written to {{{predictions_path_lit}}} (no labels available; metrics skipped).")

with open("./workspace/feature_importance.json", "w") as f:
    json.dump(fi_sorted, f, indent=2)
"""
