"""
Data Cleaning Agent
Responsibilities:
  Generate Python data-cleaning code from the Manager's `data_cleaning_plan`
  and execute it in a sandbox, including an inner self-correction loop.

cleaning_contract (minimal portable schema, all keys optional):
  - max_row_drop_ratio: float in [0, 1]. Caps (input_rows - output_rows) / input_rows.
  Column dropping/renaming is allowed; we do not enforce "all columns must remain".

  vs. rules: "rules" describe procedural steps to run; cleaning_contract describes
  bounds checked on the saved output (safety net). Unknown contract keys are ignored.

  Additional post-validation on the saved CSV (see `_validate_cleaned_csv_quality`):
  - No columns that are entirely unobserved (all NaN or all empty/whitespace strings).
  - No pair of column labels that only differ by casing (e.g. `Year` and `year`).

Design references the AutoKaggle Developer Agent pattern:
  - Structured prompt (CONTEXT + PLAN + CONSTRAINTS + TASK)
  - Code extraction (regex ```python ... ```)
  - Inner debugging self-heal loop
"""
from __future__ import annotations

import json
from core.logger import get_agent_logger

from core.state import GraphState
from core.llm import call_llm, extract_code
from tools.python_repl import execute_code
from config import MAX_INNER_RETRY

logger = get_agent_logger("data_cleaning")

# Fixed per-phase output path. Every worker in the pipeline reads a fixed
# upstream output path and writes a fixed output path, so that Supervisor
# retries do not accidentally re-feed a worker's own rejected output back
# into itself. See agents/feature_engineering.py for the FE counterpart.
CLEANED_OUTPUT_PATH = "./workspace/cleaned_data.csv"

# Train/test split paths produced at the end of data_cleaning.
# Feature Engineering reads TRAIN_DATA_PATH; both are exposed to Model Selection.
TRAIN_DATA_PATH = "./workspace/train_data.csv"
TEST_DATA_PATH = "./workspace/test_data.csv"

# ----------------------------------------------------------------------------
# Pre-split mode (state["test_data_path"] non-empty):
# We concatenate the user-provided train + test CSVs into a temporary file with
# a hidden marker column so the existing LLM cleaning code can run unchanged on
# a single ``data_path``. After cleaning we slice ``cleaned_data.csv`` back into
# train/test using the marker column. Keeps schema parity with zero downstream
# changes; the marker column name is intentionally obscure to minimise the
# chance of an LLM dropping it during cleaning.
# ----------------------------------------------------------------------------
CONCAT_FOR_CLEANING_PATH = "./workspace/_concat_for_cleaning.csv"
SPLIT_MARKER_COL = "__dataset_split__"
SPLIT_MARKER_TRAIN = "train"
SPLIT_MARKER_TEST = "test"
# Minimum rows per subset after cleaning in pre-split mode (catastrophic
# row-loss guard for the train/test subsets independently).
_MIN_ROWS_PER_SUBSET_PRE_SPLIT = 25


def _infer_target_column_name(
    columns: list[str],
    model_selection_plan: dict | None = None,
) -> str | None:
    """Infer a likely supervised target column conservatively."""
    if not columns:
        return None
    explicit = ""
    if isinstance(model_selection_plan, dict):
        explicit = str(model_selection_plan.get("target_column") or "").strip()
    if explicit and explicit in columns:
        return explicit
    semantic_exact = {"target", "label", "labels", "outcome", "response", "dependent", "y", "class"}
    semantic_contains = ("target", "label", "outcome", "response", "dependent")
    lowered = {c: str(c).strip().lower().replace(" ", "_") for c in columns}
    exact_hits = [c for c, lc in lowered.items() if lc in semantic_exact]
    contains_hits = [c for c, lc in lowered.items() if any(tok in lc for tok in semantic_contains)]
    candidates = exact_hits or contains_hits
    return candidates[0] if candidates else None


def _move_target_column_to_end(
    csv_path: str,
    model_selection_plan: dict | None = None,
    *,
    target_col_override: str = "",
) -> str:
    """Best-effort reorder so a reliable target column sits at the end.

    ``target_col_override`` (when provided and present in the CSV columns)
    takes precedence over inference. Manager's three-tier resolver populates
    this from CLI / Manager-LLM / last-column fallback, so callers that pass
    the override no longer rely on the local semantic-match heuristic.
    """
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        cols = df.columns.tolist()
        target_col = (target_col_override or "").strip()
        if target_col and target_col not in cols:
            logger.warning(
                "target_col_override %r not found in %s columns; "
                "falling back to local inference.",
                target_col,
                csv_path,
            )
            target_col = ""
        if not target_col:
            target_col = _infer_target_column_name(cols, model_selection_plan) or ""
        if not target_col or target_col == cols[-1]:
            return ""
        reordered = [c for c in cols if c != target_col] + [target_col]
        df = df[reordered]
        df.to_csv(csv_path, index=False)
        logger.info("Moved target column '%s' to the end of %s.", target_col, csv_path)
        return ""
    except Exception as exc:
        return str(exc)


def data_cleaning_node(state: GraphState) -> dict:
    """LangGraph node function for Data Cleaning."""

    logger.info("Data Cleaning node started.")
    _gp = state.get("global_plan") or {}
    _dcp = _gp.get("data_cleaning_plan") if isinstance(_gp, dict) else None
    plan: dict = _dcp if isinstance(_dcp, dict) else {}
    train_input_path = state["current_data_path"]
    feedback = state.get("current_error", "")
    dataset_description = (state.get("dataset_description") or "").strip()

    # ------------------------------------------------------------------
    # Step 0: Decide ingestion mode
    # ------------------------------------------------------------------
    # Pre-split mode (state["test_data_path"] non-empty): merge train+test with
    # a hidden marker column so the existing LLM cleaning code runs unchanged
    # on a single ``data_path``. After cleaning we slice the output back into
    # train / test by the marker column and skip the random 80/20 split.
    test_data_path = (state.get("test_data_path") or "").strip()
    pre_split_mode = bool(test_data_path)
    only_in_train: list[str] = []
    only_in_test: list[str] = []
    if pre_split_mode:
        concat_path, concat_err, only_in_train, only_in_test = (
            _concat_train_test_with_marker(train_input_path, test_data_path)
        )
        if concat_err:
            logger.error("Pre-split concat failed: %s", concat_err)
            return {
                "current_phase": "data_cleaning",
                "current_code": "",
                "current_execution_log": "",
                "current_error": f"[PreSplitConcatFailed] {concat_err}",
            }
        data_path = concat_path
        logger.info(
            "Pre-split mode active: cleaning will run on concat file '%s'. "
            "Passthrough columns surfaced to LLM: only_in_train=%s, only_in_test=%s.",
            data_path,
            only_in_train,
            only_in_test,
        )
    else:
        data_path = train_input_path

    # ------------------------------------------------------------------
    # Step 1: Generate cleaning code via LLM
    # ------------------------------------------------------------------
    code = _generate_cleaning_code(
        plan,
        data_path,
        feedback,
        dataset_description,
        pre_split_mode=pre_split_mode,
        only_in_train=only_in_train,
        only_in_test=only_in_test,
    )
    logger.info("Data cleaning code generation completed.")

    # ------------------------------------------------------------------
    # Step 2: Inner self-correction loop
    # ------------------------------------------------------------------
    output_path = _get_output_path()
    stdout, stderr = "", ""
    for attempt in range(MAX_INNER_RETRY):
        previous_signature = _get_file_signature(output_path)
        stdout, stderr = execute_code(code)
        if not stderr:
            validation_error = _validate_cleaning_output(
                output_path, previous_signature=previous_signature
            )
            if not validation_error:
                validation_error = _validate_cleaned_csv_quality(
                    output_path, pre_split_mode=pre_split_mode
                )
            if not validation_error:
                validation_error = _validate_cleaning_contract(
                    plan, data_path, output_path
                )
            if not validation_error and pre_split_mode:
                validation_error = _validate_split_marker_preserved(output_path)
            if not validation_error:
                logger.info("Data cleaning code executed and validated successfully.")
                break
            stderr = f"[PostValidationFailed] {validation_error}"
        # Self-heal: feed the error back to the LLM and regenerate code
        # Show head (traceback origin) + tail (exception type & message) for diagnosis.
        _snippet = stderr if len(stderr) <= 2000 else f"{stderr[:1000]}\n...[truncated {len(stderr)} chars total]...\n{stderr[-1000:]}"
        logger.info("Inner self-heal attempt %d: %s", attempt + 1, _snippet)
        # On the final iteration there is no next execute_code, so calling
        # _fix_code would burn an LLM request whose output is never run.
        if attempt == MAX_INNER_RETRY - 1:
            break
        code = _fix_code(
            code,
            stderr,
            plan,
            dataset_description,
            pre_split_mode=pre_split_mode,
            only_in_train=only_in_train,
            only_in_test=only_in_test,
        )

    # ------------------------------------------------------------------
    # Step 3: Return state updates
    # ------------------------------------------------------------------
    if stderr:
        logger.error(f"Data cleaning failed after {MAX_INNER_RETRY} attempts: {stderr}")
        # Inner retry exhausted: still pass the latest error to Supervisor
        return {
            "current_phase": "data_cleaning",
            "current_code": code,
            "current_execution_log": stdout,
            "current_error": f"[InnerRetryExhausted] {stderr}",
        }

    # Prefer the authoritative target name from state (set by Manager's
    # three-tier resolver). Pass it directly so reorder works for any dataset
    # whose target name is not literally `target` / `label` / `y` / etc.
    state_target_col = (state.get("target_column") or "").strip()
    reorder_error = _move_target_column_to_end(
        output_path,
        (_gp.get("model_selection_plan") if isinstance(_gp, dict) else None),
        target_col_override=state_target_col,
    )
    if reorder_error:
        logger.warning("Target-column reorder skipped for cleaned CSV: %s", reorder_error)

    # ------------------------------------------------------------------
    # Step 3b: Split cleaned data into train / test
    # ------------------------------------------------------------------
    # Pre-split mode → restore the user-provided split via the marker column.
    # Single-CSV mode → random 80/20 split (legacy behaviour, unchanged).
    if pre_split_mode:
        split_error = _split_cleaned_by_marker(output_path)
        split_kind = "marker-based (user-provided split)"
    else:
        split_error = _split_cleaned_data(
            output_path,
            model_selection_plan=(_gp.get("model_selection_plan") if isinstance(_gp, dict) else None),
            target_col_override=state_target_col,
        )
        split_kind = "random 80/20"
    if split_error:
        logger.error("Train/test split failed: %s", split_error)
        return {
            "current_phase": "data_cleaning",
            "current_code": code,
            "current_execution_log": stdout,
            "current_error": f"[SplitFailed] {split_error}",
        }
    logger.info(
        "Train/test split complete (%s) -> %s / %s",
        split_kind,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
    )

    # Note: we intentionally do NOT set ``current_data_path`` here.
    # The downstream feature_engineering node reads from a fixed
    # ``TRAIN_DATA_PATH``, so on a Supervisor-driven retry the FE worker
    # will not accidentally read its own previous rejected output.
    return {
        "current_phase": "data_cleaning",
        "current_code": code,
        "current_execution_log": stdout,
        "current_error": "",
    }


# ======================================================================
# Private helper functions
# ======================================================================

# ---------- Prompt templates (patterned after AutoKaggle PROMPT_DEVELOPER) ----------

_SYSTEM_PROMPT_GENERATE = """\
You are an expert Data Cleaning developer. \
You write Python code according to the provided plan, constraints, and any CLEANING CONTRACT guardrails. \
Your code must be complete, self-contained, and directly runnable. \
Avoid redundant date columns: if you parse a datetime into parts, do not keep duplicate semantics under different names/casing (e.g. both Year and year). \
Before saving, remove columns with no observed values (all-NaN / all-empty) so downstream models do not receive useless features."""

_USER_PROMPT_GENERATE = """\
# CONTEXT #
Current phase: Data Cleaning

# DATASET DESCRIPTION (authoritative) #
{dataset_description_block}

# PLAN #
{plan}

# CLEANING CONTRACT #
{cleaning_contract_body}

# CONSTRAINTS #
1. Load data from: `{data_path}` using `pd.read_csv`.
2. Save cleaned data to: `./workspace/cleaned_data.csv` using `df.to_csv(..., index=False)`.
3. Always work on a copy of the DataFrame, not the original.
4. Use `print()` to output a summary of what was done (rows before/after, columns modified, etc.).
5. Import all necessary libraries at the top of the code.
6. Column naming consistency: if the data already has calendar fields (e.g. `Year`, `Month_Name`) and you also derive parts from a `Date` column, keep a single coherent scheme — drop or merge duplicates (do not leave both `Year` and `year`, or redundant month columns). Prefer one convention (e.g. snake_case `year`, `month`) and document drops in the print summary.
7. Before `to_csv`, drop columns that have no usable values (e.g. `df.dropna(axis=1, how="all")`, or drop columns where non-null count is 0). Do not leave placeholder columns that are entirely empty — they confuse downstream imputers.{pre_split_constraint_block}

{feedback_section}

# RESPONSE: CODE #
Provide your complete solution in a single ```python ... ``` code block.
"""

_SYSTEM_PROMPT_FIX = """\
You are an expert Python debugger. \
Analyze the error and fix the code. \
Output the complete corrected code. \
Keep the cleaning output policy: no all-empty columns in the saved CSV."""

_USER_PROMPT_FIX = """\
# CONTEXT #
Current phase: Data Cleaning

# DATASET DESCRIPTION (authoritative) #
{dataset_description_block}

# PLAN #
{plan}

# CLEANING CONTRACT #
{cleaning_contract_body}

# CODE CONTAINS ERROR #
```python
{code}
```

# ERROR MESSAGES #
{error}

# TASK REQUIREMENTS #
1. Continue following the original Data Cleaning plan and constraints.
2. Save the cleaned output to `./workspace/cleaned_data.csv`.
3. Do not leave all-empty columns in the saved CSV.
4. If the error came from post-validation, fix the cleaning logic instead of only silencing the error.{pre_split_constraint_block}

# RESPONSE: CORRECTED CODE #
Provide the complete corrected code in a single ```python ... ``` code block.
"""


def _cleaning_contract_prompt_body(plan: dict) -> str:
    """Human-readable cleaning_contract slice for the LLM (empty dict → explicit none)."""
    if not isinstance(plan, dict):
        return "(none specified; follow # PLAN # and # CONSTRAINTS # only.)"
    raw = plan.get("cleaning_contract", {})
    contract: dict = raw if isinstance(raw, dict) else {}
    if not contract:
        return "(none specified; follow # PLAN # and # CONSTRAINTS # only.)"
    return (
        json.dumps(contract, ensure_ascii=False, indent=2)
        + "\n\nHard-enforced keys (if present): max_row_drop_ratio (0..1) only. "
        "You may drop or rename columns if the plan calls for it; other contract keys are hints only."
    )


def _plan_json_for_prompt(plan: dict) -> str:
    """Full cleaning plan JSON except cleaning_contract (shown in its own section)."""
    if not isinstance(plan, dict):
        return "{}"
    body = {k: v for k, v in plan.items() if k != "cleaning_contract"}
    return json.dumps(body, ensure_ascii=False, indent=2)


def _dataset_description_block(desc: str) -> str:
    """Render the DATASET DESCRIPTION block for prompts (falls back to a neutral note)."""
    text = (desc or "").strip()
    if text:
        return text
    return "(not provided; infer cautiously from column names and sample rows)"


def _pre_split_constraint_block(
    pre_split_mode: bool,
    only_in_train: list[str] | None = None,
    only_in_test: list[str] | None = None,
) -> str:
    """
    Empty string in single-CSV mode (keeps the legacy prompt byte-identical),
    otherwise a leading-newline block that tells the LLM:
      (a) to preserve the hidden marker column the pipeline injects when the
          user supplied a pre-split test set;
      (b) about the LITERAL list of columns that exist on only one side of the
          split, with a hard "do nothing to these columns" instruction.

    Earlier iterations described the asymmetric-column rule abstractly ("for
    columns whose NaNs are concentrated on one side, do not impute…"). LLMs
    repeatedly failed to operationalise that — they could not reliably tell
    which columns in the actual data fit the description. So we now inline the
    concrete column names computed at concat time. The LLM no longer has to
    classify; it just has to leave the named columns alone.
    """
    if not pre_split_mode:
        return ""

    only_in_train = list(only_in_train or [])
    only_in_test = list(only_in_test or [])

    rule_8 = (
        f"\n8. The input CSV contains a special column named `{SPLIT_MARKER_COL}` "
        f"whose values are exactly `{SPLIT_MARKER_TRAIN}` or `{SPLIT_MARKER_TEST}`. "
        "It marks the original train/test split provided by the user. You MUST keep "
        "this column unchanged in the output CSV: do not drop, rename, transform, "
        "one-hot encode, or otherwise modify its values. Treat it as a passthrough "
        "column."
    )

    if not only_in_train and not only_in_test:
        return rule_8

    train_only_repr = ", ".join(repr(c) for c in only_in_train) if only_in_train else "(none)"
    test_only_repr = ", ".join(repr(c) for c in only_in_test) if only_in_test else "(none)"

    rule_9 = (
        "\n9. The user-provided train and test CSVs do not share identical column "
        "sets. After concat, the following columns exist on only ONE side of the "
        "split (and therefore have NaN values on the other side BY DESIGN — those "
        "NaNs are not a data-quality issue, they are how a column-union concat "
        "represents 'this column did not exist on that side'):\n"
        f"   - Columns present in train only (NaN on test rows): [{train_only_repr}]\n"
        f"   - Columns present in test only (NaN on train rows): [{test_only_repr}]\n"
        "\n"
        "   These columns are PASSTHROUGH for the cleaning stage. For each column "
        "in the two lists above you MUST:\n"
        "   - NOT call `.fillna(...)`, `SimpleImputer`, `KNNImputer`, mode/median/"
        "forward-fill, or any other imputation on it;\n"
        "   - NOT include it in any list of columns being imputed (e.g. do not "
        "add it to a `numeric_cols` / `categorical_cols` list that is then fed to "
        "an imputer);\n"
        "   - NOT drop the column;\n"
        "   - NOT drop rows because the value is NaN in that column;\n"
        "   - NOT one-hot encode, scale, normalise, label-encode, or otherwise "
        "transform it;\n"
        "   - NOT compute any per-column NaN-ratio statistic on it (any sparsity-"
        "based dropping rule must skip these columns entirely);\n"
        "   - copy its values into the output CSV unchanged (NaN cells stay NaN).\n"
        "\n"
        "   Concrete reason this matters: the supervised target column is "
        "frequently present in train but absent in test (held-out test sets where "
        "labels are intentionally not exposed). The downstream Model Selection "
        "stage detects test-side NaN target and switches to predict-only mode "
        "(writes predictions instead of fake metrics). If the cleaning stage "
        "imputes those NaNs with mode/median, it fabricates labels and breaks "
        "downstream evaluation.\n"
        "\n"
        "   Likewise, when judging 'is this column all-NaN?' or 'is this column "
        "too sparse?' for a sparsity-based drop rule, judge it on the FULL "
        f"combined frame, NOT per subset. (You can split subsets via "
        f"`{SPLIT_MARKER_COL}` if needed for stats, but the listed columns above "
        "are exempt from sparsity-based drops regardless.)"
    )
    return rule_8 + rule_9


def _generate_cleaning_code(
    plan: dict,
    data_path: str,
    feedback: str,
    dataset_description: str = "",
    pre_split_mode: bool = False,
    only_in_train: list[str] | None = None,
    only_in_test: list[str] | None = None,
) -> str:
    """
    Call the LLM to generate data-cleaning code.
    If the LLM call fails, fall back to a template implementation.
    """
    feedback_section = ""
    if feedback:
        feedback_section = f"# SUPERVISOR FEEDBACK #\nPrevious attempt was rejected. Fix based on:\n{feedback}"

    user_prompt = _USER_PROMPT_GENERATE.format(
        plan=_plan_json_for_prompt(plan),
        cleaning_contract_body=_cleaning_contract_prompt_body(plan),
        data_path=data_path,
        feedback_section=feedback_section,
        dataset_description_block=_dataset_description_block(dataset_description),
        pre_split_constraint_block=_pre_split_constraint_block(
            pre_split_mode,
            only_in_train=only_in_train or [],
            only_in_test=only_in_test or [],
        ),
    )

    reply = call_llm(_SYSTEM_PROMPT_GENERATE, user_prompt, agent_name="data_cleaning")

    if reply:
        code = extract_code(reply)
        if code:
            return code

    # Fallback: return template code
    logger.warning("LLM call failed or returned no code; using fallback code template.")
    return _fallback_cleaning_code(plan, data_path)


def _fix_code(
    code: str,
    error: str,
    plan: dict,
    dataset_description: str = "",
    pre_split_mode: bool = False,
    only_in_train: list[str] | None = None,
    only_in_test: list[str] | None = None,
) -> str:
    """
    Call the LLM to fix code based on execution errors.
    Return original code if the LLM fix call fails.
    """
    user_prompt = _USER_PROMPT_FIX.format(
        code=code,
        error=error,
        plan=_plan_json_for_prompt(plan),
        cleaning_contract_body=_cleaning_contract_prompt_body(plan),
        dataset_description_block=_dataset_description_block(dataset_description),
        pre_split_constraint_block=_pre_split_constraint_block(
            pre_split_mode,
            only_in_train=only_in_train or [],
            only_in_test=only_in_test or [],
        ),
    )
    reply = call_llm(_SYSTEM_PROMPT_FIX, user_prompt, agent_name="data_cleaning")

    if reply:
        fixed_code = extract_code(reply)
        if fixed_code:
            return fixed_code

    # Fallback: return original code
    logger.warning("LLM fix call failed; returning original code.")
    return code


def _get_output_path() -> str:
    """Return the standard output path for cleaned data."""
    return CLEANED_OUTPUT_PATH


def _get_file_signature(file_path: str) -> tuple[int, int] | None:
    """Return file signature (mtime_ns, size), or None if file does not exist."""
    import os

    if not os.path.exists(file_path):
        return None

    stat = os.stat(file_path)
    return stat.st_mtime_ns, stat.st_size


def _validate_cleaning_output(
    output_path: str, previous_signature: tuple[int, int] | None = None
) -> str:
    """
    Perform minimal hard checks on the cleaning output:
    1. Output file exists.
    2. Output path is a file, not a directory
    3. Output file size is greater than 0
    4. File signature changed in this run (avoid reusing stale output)

    Returns an empty string when valid, otherwise returns the failure reason.
    """
    import os

    if not os.path.exists(output_path):
        return f"cleaned output not found: {output_path}"

    if not os.path.isfile(output_path):
        return f"cleaned output is not a file: {output_path}"

    current_signature = _get_file_signature(output_path)
    if current_signature is None:
        return f"failed to stat cleaned output: {output_path}"

    _, file_size = current_signature
    if file_size <= 0:
        return "cleaned output file is empty"

    if previous_signature is not None:
        if current_signature == previous_signature:
            return "cleaned output file was not updated in current run"

    return ""


def _series_has_observed_value(s) -> bool:
    """
    True if this column has at least one *observed* value for validation purposes.

    - Numeric / bool / datetime / timedelta: any non-NaN counts as observed.
    - Text-like (object, string, etc.): need at least one non-NaN cell whose string form,
      after strip, is not the empty string (so a column of only ``""`` or spaces fails).
    """
    import pandas as pd

    if s.empty:
        return False
    # Typed columns: presence means "at least one non-null".
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return bool(s.notna().any())
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
        return bool(s.notna().any())
    # Object / string: ignore NaNs, then require at least one non-blank string.
    mask = s.notna()
    if not mask.any():
        return False
    text = s[mask].astype(str).str.strip()
    return bool((text != "").any())


def _validate_cleaned_csv_quality(output_csv_path: str, pre_split_mode: bool = False) -> str:
    """
    Content checks on the written CSV (after file-level validation).

    Flow (short-circuit on first failure):
      1) Load the CSV into a DataFrame.
      2) Reject if there are no column names at all.
      3) Reject if two or more columns differ only by letter case (e.g. ``Year`` and ``year``),
         because that breaks the "single naming scheme" rule and confuses downstream code.
      4) Reject if any column has zero usable values (see ``_series_has_observed_value``).

    The ``len(df) < 50`` global row-loss guard applies in both modes (combined
    train+test in pre-split mode is still well above 50 rows for any realistic
    dataset). The per-subset row-loss check (≥ 25 rows in train and in test) is
    enforced separately by ``_validate_split_marker_preserved`` so that the two
    failure modes produce distinct, actionable messages for the LLM self-heal
    loop. ``pre_split_mode`` is accepted for symmetry with the call site and
    forward-compatibility, even though no branch currently keys off it here.

    Returns an empty string when valid, otherwise a failure reason string (shown to the LLM
    in the self-heal loop as ``[PostValidationFailed] ...``).
    """
    del pre_split_mode  # currently informational only; see docstring
    import pandas as pd

    # --- Load output (caller already ensured path exists and file is non-empty) ---
    try:
        df = pd.read_csv(output_csv_path)
    except Exception as e:
        return f"cleaned CSV quality check could not read file: {e}"

    if df.columns.empty:
        return "cleaned CSV has no columns"

    # --- Hard safety net: catastrophic row loss ---
    # Regardless of cleaning_contract, reject if the output has fewer than
    # 50 rows — this almost always indicates the LLM-generated code
    # accidentally filtered out the entire dataset.
    if len(df) < 50:
        return (
            f"cleaned CSV has only {len(df)} rows, which is almost certainly a bug "
            f"in the generated cleaning code (catastrophic row loss). "
            f"The cleaning code should retain the vast majority of rows."
        )

    # --- Check 1: case-insensitive duplicate *labels* ---
    # Same logical name written twice with different casing (Year vs year) maps to the same
    # "folded" key via casefold(), so we group originals under that key and flag groups
    # with more than one distinct label.
    from collections import defaultdict

    fold_to_labels: dict[str, list[str]] = defaultdict(list)
    for c in df.columns:
        fold_to_labels[str(c).casefold()].append(str(c))
    collisions = {k: v for k, v in fold_to_labels.items() if len(v) > 1}
    if collisions:
        # Build a short human-readable message; cap how many collision groups we list.
        examples = []
        for labels in list(collisions.values())[:5]:
            examples.append(" / ".join(repr(x) for x in labels))
        suffix = f" (+{len(collisions) - 5} more)" if len(collisions) > 5 else ""
        return (
            "cleaned CSV has case-insensitive duplicate column names: "
            + "; ".join(examples)
            + (suffix if suffix else "")
        )

    # --- Check 2: columns that contain no real data (all missing or only blank text) ---
    empty_cols = [str(c) for c in df.columns if not _series_has_observed_value(df[c])] # Get all columns that have no observed values
    if empty_cols:
        shown = empty_cols[:12] # Display 12 more empty columns if there are more than 12
        more = f" (+{len(empty_cols) - 12} more)" if len(empty_cols) > 12 else ""
        return f"cleaned CSV has all-empty columns: {', '.join(repr(c) for c in shown)}{more}"

    return ""


def _validate_cleaning_contract(
    plan: dict, input_csv_path: str, output_csv_path: str
) -> str:
    """
    Hard-check cleaned CSV against data_cleaning_plan["cleaning_contract"].

    Empty/missing contract → pass (no-op). Only max_row_drop_ratio is enforced; extras ignored.

    - max_row_drop_ratio: if set and valid in [0, 1], require
      (n_in - n_out) / n_in <= ratio when n_in > 0.

    Invalid types for known keys are skipped with a warning (do not fail the run).
    If the contract only contains unknown keys, no CSV is read (forward-compatible).
    """
    if not isinstance(plan, dict):
        return ""

    raw = plan.get("cleaning_contract", {})
    if not isinstance(raw, dict) or not raw:
        return ""

    # Decide what to enforce before loading CSVs (unknown keys → no I/O).
    check_row_ratio = False
    max_ratio: float | None = None
    if "max_row_drop_ratio" in raw:
        val = raw["max_row_drop_ratio"]
        try:
            mr = float(val)
        except (TypeError, ValueError):
            logger.warning(
                "cleaning_contract.max_row_drop_ratio invalid type/value, skipping: %r",
                val,
            )
        else:
            if mr < 0 or mr > 1:
                logger.warning(
                    "cleaning_contract.max_row_drop_ratio out of [0,1], skipping: %s",
                    mr,
                )
            else:
                max_ratio = mr
                check_row_ratio = True

    if not check_row_ratio:
        return ""

    import pandas as pd

    try:
        df_in = pd.read_csv(input_csv_path)
        df_out = pd.read_csv(output_csv_path)
    except Exception as e:
        return f"cleaning_contract check could not read CSV: {e}"

    if max_ratio is not None:
        n_in, n_out = len(df_in), len(df_out)
        if n_in > 0:
            dropped_ratio = (n_in - n_out) / n_in
            if dropped_ratio > max_ratio + 1e-9:
                return (
                    f"cleaning_contract violated: row drop ratio {dropped_ratio:.4f} "
                    f"exceeds max_row_drop_ratio {max_ratio} (rows in {n_in}, out {n_out})"
                )

    return ""


def _concat_train_test_with_marker(
    train_path: str, test_path: str
) -> tuple[str, str, list[str], list[str]]:
    """
    Concatenate user-provided train and test CSVs into a single file with a
    hidden marker column so the existing LLM cleaning code can process them
    in one pass.

    Returns ``(concat_path, "", only_in_train, only_in_test)`` on success or
    ``("", error_message, [], [])`` on failure. The ``only_in_train`` and
    ``only_in_test`` lists are the column names that exist on exactly one side
    of the user-provided split — the cleaning prompt feeds them back to the
    LLM as a concrete passthrough list so the LLM never imputes a column whose
    NaNs are an artefact of the union concat (most importantly the supervised
    target column when the test set is a held-out / unlabelled holdout).

    The error_message is human-readable and surfaces upward as the data_cleaning
    node's ``current_error`` (no LLM call is made on hard failure).

    Schema policy: train and test do **not** have to have identical column sets.
    pandas concatenation takes the column union and fills NaN for any cell where
    the original side did not have that column. This is the natural behaviour for
    real-world pre-split inputs:

      - Holdout: train has the target column, test does not. After
        concat, test rows have NaN in the target column, which is the correct
        representation. (Note: downstream Model Selection currently still
        requires the target on test for evaluation; the cleaning step itself
        does not.)
      - Train-only / test-only metadata columns (e.g. submission ids on test):
        same treatment, surfaced as NaN on the side that lacks the column.

    Hard-failure cases:
      - Either CSV cannot be read by pandas.
      - Either CSV is empty (no rows).
      - Train and test share zero columns at all (clearly two different datasets).
      - Either side already contains a column literally named ``SPLIT_MARKER_COL``.
    """
    import os

    import pandas as pd

    try:
        train_df = pd.read_csv(train_path)
    except Exception as e:
        return "", f"could not read train CSV '{train_path}': {e}", [], []
    try:
        test_df = pd.read_csv(test_path)
    except Exception as e:
        return "", f"could not read test CSV '{test_path}': {e}", [], []

    if train_df.empty:
        return "", f"train CSV '{train_path}' is empty", [], []
    if test_df.empty:
        return "", f"test CSV '{test_path}' is empty", [], []

    train_set = set(train_df.columns)
    test_set = set(test_df.columns)
    if not (train_set & test_set):
        return (
            "",
            "train and test CSVs share zero columns; they do not look like the "
            "same dataset. Check the paths or use single-CSV mode.",
            [],
            [],
        )
    if SPLIT_MARKER_COL in train_set or SPLIT_MARKER_COL in test_set:
        return (
            "",
            f"input CSV already contains a column named {SPLIT_MARKER_COL!r}, "
            "which collides with the internal pre-split marker. Rename it before re-running.",
            [],
            [],
        )

    only_in_train = sorted(train_set - test_set)
    only_in_test = sorted(test_set - train_set)
    if only_in_train or only_in_test:
        logger.info(
            "Pre-split column asymmetry: only_in_train=%s, only_in_test=%s. "
            "Concat uses column union; missing cells become NaN (typical Kaggle "
            "holdout: train has the target column, test does not).",
            only_in_train,
            only_in_test,
        )

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[SPLIT_MARKER_COL] = SPLIT_MARKER_TRAIN
    test_df[SPLIT_MARKER_COL] = SPLIT_MARKER_TEST
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)

    os.makedirs(os.path.dirname(CONCAT_FOR_CLEANING_PATH), exist_ok=True)
    try:
        combined.to_csv(CONCAT_FOR_CLEANING_PATH, index=False)
    except OSError as e:
        return "", f"failed to write concat input '{CONCAT_FOR_CLEANING_PATH}': {e}", [], []

    logger.info(
        "Pre-split concat written to %s: %d train + %d test = %d rows, %d cols (incl. marker)",
        CONCAT_FOR_CLEANING_PATH,
        len(train_df),
        len(test_df),
        len(combined),
        len(combined.columns),
    )
    return CONCAT_FOR_CLEANING_PATH, "", only_in_train, only_in_test


def _validate_split_marker_preserved(output_csv_path: str) -> str:
    """
    Pre-split-mode post-validation: ensure the LLM did not drop / mutate the
    marker column, and that both subsets survived cleaning with enough rows.

    Returns an empty string on success or a diagnostic string. Messages are
    fed back into the LLM via the existing ``[PostValidationFailed]`` self-heal
    convention.
    """
    import pandas as pd

    try:
        df = pd.read_csv(output_csv_path)
    except Exception as e:
        return f"pre-split validation could not read cleaned CSV: {e}"

    if SPLIT_MARKER_COL not in df.columns:
        return (
            f"cleaned CSV is missing the required marker column "
            f"{SPLIT_MARKER_COL!r}. The cleaning code must keep this column "
            "untouched (it is required to restore the original train/test split)."
        )

    marker_values = df[SPLIT_MARKER_COL].astype(str).str.strip()
    allowed = {SPLIT_MARKER_TRAIN, SPLIT_MARKER_TEST}
    unexpected = sorted(set(marker_values.unique()) - allowed)
    if unexpected:
        return (
            f"marker column {SPLIT_MARKER_COL!r} contains unexpected values "
            f"{unexpected!r}; only {sorted(allowed)!r} are allowed. "
            "The cleaning code must not transform this column."
        )

    n_train = int((marker_values == SPLIT_MARKER_TRAIN).sum())
    n_test = int((marker_values == SPLIT_MARKER_TEST).sum())
    if n_train < _MIN_ROWS_PER_SUBSET_PRE_SPLIT:
        return (
            f"after cleaning, the train subset has only {n_train} rows "
            f"(< {_MIN_ROWS_PER_SUBSET_PRE_SPLIT}); this is almost certainly a bug "
            "in the generated cleaning code (catastrophic row loss on train)."
        )
    if n_test < _MIN_ROWS_PER_SUBSET_PRE_SPLIT:
        return (
            f"after cleaning, the test subset has only {n_test} rows "
            f"(< {_MIN_ROWS_PER_SUBSET_PRE_SPLIT}); this is almost certainly a bug "
            "in the generated cleaning code (catastrophic row loss on test)."
        )
    return ""


def _split_cleaned_by_marker(cleaned_csv_path: str) -> str:
    """
    Restore the user-provided train/test split from the cleaned CSV using the
    marker column written by ``_concat_train_test_with_marker``. Drops the
    marker column from both outputs before saving.

    Caller is responsible for having run ``_validate_split_marker_preserved``
    first; this function still asserts the marker is present so that a refactor
    error surfaces loudly rather than silently producing a single combined file.
    """
    import pandas as pd

    try:
        df = pd.read_csv(cleaned_csv_path)
    except Exception as e:
        return f"could not read cleaned CSV for marker-based split: {e}"

    if SPLIT_MARKER_COL not in df.columns:
        return (
            f"cleaned CSV is missing marker column {SPLIT_MARKER_COL!r}; "
            "cannot restore the user-provided train/test split"
        )

    marker = df[SPLIT_MARKER_COL].astype(str).str.strip()
    train_df = df[marker == SPLIT_MARKER_TRAIN].drop(columns=[SPLIT_MARKER_COL])
    test_df = df[marker == SPLIT_MARKER_TEST].drop(columns=[SPLIT_MARKER_COL])

    if train_df.empty:
        return "marker-based split produced an empty train subset"
    if test_df.empty:
        return "marker-based split produced an empty test subset"

    try:
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)
    except OSError as e:
        return f"failed to write split outputs: {e}"

    logger.info(
        "Marker-based split: %d train rows -> %s, %d test rows -> %s",
        len(train_df), TRAIN_DATA_PATH,
        len(test_df), TEST_DATA_PATH,
    )
    return ""


def _split_cleaned_data(
    cleaned_csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    model_selection_plan: dict | None = None,
    *,
    target_col_override: str = "",
) -> str:
    """
    Split the cleaned CSV into train (80%) and test (20%) sets.
    Writes TRAIN_DATA_PATH and TEST_DATA_PATH.
    Returns an empty string on success, or an error message on failure.

    ``target_col_override`` lets the caller (Manager's resolved target name
    via ``state["target_column"]``) bypass the internal semantic-name guesser,
    which only matched columns literally named ``target`` / ``label`` / etc.
    """
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(cleaned_csv_path)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        cols = train_df.columns.tolist()
        target_col = (target_col_override or "").strip()
        if target_col and target_col not in cols:
            target_col = ""
        if not target_col:
            target_col = _infer_target_column_name(cols, model_selection_plan) or ""
        if target_col and target_col in train_df.columns and target_col in test_df.columns:
            reordered = [c for c in train_df.columns if c != target_col] + [target_col]
            train_df = train_df[reordered]
            test_df = test_df[reordered]
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)
        logger.info(
            "Split: %d train rows → %s, %d test rows → %s",
            len(train_df), TRAIN_DATA_PATH,
            len(test_df), TEST_DATA_PATH,
        )
        return ""
    except Exception as exc:
        return str(exc)


def _fallback_cleaning_code(plan: dict, data_path: str) -> str:
    """Fallback template code when LLM is unavailable."""
    path_lit = json.dumps(data_path)
    return f"""\
import pandas as pd

df = pd.read_csv({path_lit})
# Run cleaning steps based on plan objectives: {plan.get('objectives', [])}
df = df.drop_duplicates()
df = df.dropna(thresh=len(df.columns) * 0.5)

output_path = "./workspace/cleaned_data.csv"
df.to_csv(output_path, index=False)
print(f"Cleaning complete. Remaining rows: {{len(df)}}. Saved to {{output_path}}")
"""
