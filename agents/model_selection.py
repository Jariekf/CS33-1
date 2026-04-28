"""
Model Selection Agent (模型选择与构建智能体)
职责：
  根据 Manager 的全局计划中的 model_selection_plan，
  生成多模型对比、超参数调优和交叉验证的代码并在沙盒执行。
  产出最优模型及其特征重要性 (Feature Importance)。
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

# ======================================================================
# Prompt 模板
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

# EDA SUMMARY FROM FEATURE ENGINEERING (advisory) #
{eda_summary_block}

# PLAN #
{plan}

# CONSTRAINTS #
1. Load training data from: `{data_path}` (feature-engineered 80% train split) using `pd.read_csv`.
2. A held-out test set is available at: `{test_data_path}`. Use it ONLY for final model evaluation after training/CV is complete — never use it during training or cross-validation.
3. Infer task type from plan + user goal + dataset context (classification/regression/time-series/clustering), then choose model family and metric accordingly.
4. For supervised tasks, infer target conservatively from semantics (target/label/outcome/etc.) or explicit plan fields; do NOT blindly assume last column.
5. If target is ambiguous, print a clear warning and stop rather than training on an unreliable target.
   Before applying any label mapping or encoding to the target column, inspect its actual dtype and unique values (e.g. `print(df[target_col].dtype, df[target_col].unique())`). If the target is already numeric (int/float), use it directly — do NOT re-apply a string→int mapping, as that will produce all-NaN. Only convert string labels (e.g. 'yes'/'no', 'True'/'False') if the column genuinely contains non-numeric values.
6. Do model selection and cross-validation as instructed by the Plan.
7. Export the top feature importances to a dict, sorted by value descending, and save it to `./workspace/feature_importance.json`.
8. Use `print()` to output inferred task type, Best Model name, Cross-Validation score, Top Feature Importances, and final test-set evaluation metrics.
9. Import all necessary libraries at the top of the code.
10. NEVER hardcode column names. Always discover feature columns dynamically from the loaded DataFrame (e.g. `numeric_cols = df.select_dtypes(include='number').columns.tolist()`; `cat_cols = df.select_dtypes(include='object').columns.tolist()`). The upstream feature engineering step may rename, add, or one-hot-encode columns unpredictably — hardcoded column lists will cause `ValueError: A given column is not a column of the dataframe`.
11. Use EDA summary as guidance only. Before training, validate all key assumptions against the actually loaded DataFrame and prioritize runtime-verified facts when conflicts exist.

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

# EDA SUMMARY FROM FEATURE ENGINEERING (advisory) #
{eda_summary_block}

# PLAN #
{plan}

# CODE CONTAINS ERROR #
```python
{code}
```

# ERROR MESSAGES #
{error}

# TASK REQUIREMENTS #
1. Continue following the original Model Selection plan and constraints.
2. Save feature importance to `./workspace/feature_importance.json`.
3. Keep model selection, hyperparameter tuning, and cross-validation aligned with the plan.
4. If the previous attempt failed functionally, fix the implementation gap rather than only fixing syntax.
5. Before applying any label mapping or encoding to the target column, inspect its actual dtype and unique values. If the target is already numeric (int/float), use it directly — do NOT re-apply a string→int mapping, as that produces all-NaN and causes single-class errors.
6. NEVER hardcode column names. Discover feature columns dynamically from the DataFrame (e.g. `select_dtypes`). Hardcoded column lists break when upstream feature engineering changes columns.
7. Use EDA summary as guidance only; if EDA conflicts with loaded data facts, follow runtime-verified data facts.

# RESPONSE: CORRECTED CODE #
Provide the complete corrected code in a single ```python ... ``` code block.
"""


def model_selection_node(state: GraphState) -> dict:
    """Model Selection 的 LangGraph 节点函数。"""

    logger.info("Model Selection node started.")
    # 【边界控制】只抽取属于自己的子计划
    plan = state["global_plan"].get("model_selection_plan", {})
    # Always read from FE's fixed output. Not ``state["current_data_path"]``,
    # so we stay consistent with the upstream workers and keep the retry
    # semantics clean.
    data_path = FE_OUTPUT_PATH
    feedback = state.get("current_error", "")
    dataset_description = (state.get("dataset_description") or "").strip()
    user_goal = (state.get("user_goal") or "").strip()
    eda_summary = (state.get("eda_summary") or "").strip()
    _restore_missing_target_columns(plan)

    # ------------------------------------------------------------------
    # Step 1: 生成模型训练 + 调参 + 评估代码
    # ------------------------------------------------------------------
    code = _generate_model_code(
        plan,
        data_path,
        FE_TEST_OUTPUT_PATH,
        feedback,
        dataset_description,
        user_goal,
        eda_summary,
    )
    logger.info("Model selection code generation completed.")

    # ------------------------------------------------------------------
    # Step 2: 内部语法自救循环
    # ------------------------------------------------------------------
    stdout, stderr = "", ""
    error_history = []
    for attempt in range(MAX_INNER_RETRY):
        previous_signature = _get_file_signature(FEATURE_IMPORTANCE_PATH)
        stdout, stderr = execute_code(code)
        if not stderr:
            validation_error = _validate_model_artifact(
                FEATURE_IMPORTANCE_PATH,
                previous_signature=previous_signature,
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
        )

    # ------------------------------------------------------------------
    # Step 3: 返回状态
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
# 私有辅助函数
# ======================================================================

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


def _generate_model_code(
    plan: dict,
    data_path: str,
    test_data_path: str,
    feedback: str,
    dataset_description: str = "",
    user_goal: str = "",
    eda_summary: str = "",
) -> str:
    """
    调用 LLM 生成模型训练与调参代码。
    若调用失败，则返回桩实现代码。
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
        feedback_section=feedback_section,
        dataset_description_block=_dataset_description_block(dataset_description),
        user_goal=user_goal or "(not provided)",
        eda_summary_block=_eda_summary_block(eda_summary),
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
) -> str:
    """Call the LLM to fix code. Return the original code if the fix call fails."""
    user_prompt = _USER_PROMPT_FIX.format(
        code=code,
        error=error,
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        dataset_description_block=_dataset_description_block(dataset_description),
        user_goal=user_goal or "(not provided)",
        eda_summary_block=_eda_summary_block(eda_summary),
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
    """Fallback template code when generation fails."""
    task_kind = _infer_task_kind(plan, user_goal, dataset_description)
    metric = str(plan.get("evaluation_metric") or "").strip()
    if not metric:
        metric = "neg_root_mean_squared_error" if task_kind in ("regression", "time_series") else "f1"
    path_lit = json.dumps(data_path)
    explicit_target_col = str(plan.get("target_column") or "").strip()
    model_class = "RandomForestRegressor" if task_kind in ("regression", "time_series") else "RandomForestClassifier"
    split_block = (
        "from sklearn.model_selection import TimeSeriesSplit\n"
        "cv = TimeSeriesSplit(n_splits=5)\n"
    ) if task_kind == "time_series" else "cv = 5\n"
    return f"""\
import pandas as pd
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

print(f"Inferred Task Type: {task_kind}")
print(f"Best Model: {model_class}")
print(f"Target Column: {{target_col}}")
print(f"CV {{metric_name}} Mean: {{scores.mean():.4f}} (+/- {{scores.std():.4f}})")
print(f"Top 5 Feature Importance: {{dict(list(fi_sorted.items())[:5])}}")

# Save feature importance.
os.makedirs("./workspace", exist_ok=True)
with open("./workspace/feature_importance.json", "w") as f:
    json.dump(fi_sorted, f, indent=2)
"""
