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


def data_cleaning_node(state: GraphState) -> dict:
    """LangGraph node function for Data Cleaning."""

    logger.info("Data Cleaning node started.")
    _gp = state.get("global_plan") or {}
    _dcp = _gp.get("data_cleaning_plan") if isinstance(_gp, dict) else None
    plan: dict = _dcp if isinstance(_dcp, dict) else {}
    data_path = state["current_data_path"]
    feedback = state.get("current_error", "")

    # ------------------------------------------------------------------
    # Step 1: Generate cleaning code via LLM
    # ------------------------------------------------------------------
    code = _generate_cleaning_code(plan, data_path, feedback)
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
                validation_error = _validate_cleaned_csv_quality(output_path)
            if not validation_error:
                validation_error = _validate_cleaning_contract(
                    plan, data_path, output_path
                )
            if not validation_error:
                logger.info("Data cleaning code executed and validated successfully.")
                break
            stderr = f"[PostValidationFailed] {validation_error}"
        # Self-heal: feed the error back to the LLM and regenerate code
        logger.info("Inner self-heal attempt %d: %s", attempt + 1, stderr[:200])
        code = _fix_code(code, stderr)

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

    return {
        "current_phase": "data_cleaning",
        "current_code": code,
        "current_execution_log": stdout,
        "current_error": "",
        "current_data_path": output_path,
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
7. Before `to_csv`, drop columns that have no usable values (e.g. `df.dropna(axis=1, how="all")`, or drop columns where non-null count is 0). Do not leave placeholder columns that are entirely empty — they confuse downstream imputers.

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
# CODE CONTAINS ERROR #
```python
{code}
```

# ERROR MESSAGES #
{error}

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


def _generate_cleaning_code(plan: dict, data_path: str, feedback: str) -> str:
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
    )

    reply = call_llm(_SYSTEM_PROMPT_GENERATE, user_prompt, agent_name="data_cleaning")

    if reply:
        code = extract_code(reply)
        if code:
            return code

    # Fallback: return template code
    logger.warning("LLM call failed or returned no code; using fallback code template.")
    return _fallback_cleaning_code(plan, data_path)


def _fix_code(code: str, error: str) -> str:
    """
    Call the LLM to fix code based on execution errors.
    Return original code if the LLM fix call fails.
    """
    user_prompt = _USER_PROMPT_FIX.format(code=code, error=error)
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
    return "./workspace/cleaned_data.csv"


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


def _validate_cleaned_csv_quality(output_csv_path: str) -> str:
    """
    Content checks on the written CSV (after file-level validation).

    Flow (short-circuit on first failure):
      1) Load the CSV into a DataFrame.
      2) Reject if there are no column names at all.
      3) Reject if two or more columns differ only by letter case (e.g. ``Year`` and ``year``),
         because that breaks the "single naming scheme" rule and confuses downstream code.
      4) Reject if any column has zero usable values (see ``_series_has_observed_value``).

    Returns an empty string when valid, otherwise a failure reason string (shown to the LLM
    in the self-heal loop as ``[PostValidationFailed] ...``).
    """
    import pandas as pd

    # --- Load output (caller already ensured path exists and file is non-empty) ---
    try:
        df = pd.read_csv(output_csv_path)
    except Exception as e:
        return f"cleaned CSV quality check could not read file: {e}"

    if df.columns.empty:
        return "cleaned CSV has no columns"

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
