"""
Data Cleaning Agent
Responsibilities:
  Generate Python data-cleaning code from the Manager's `data_cleaning_plan`
  and execute it in a sandbox, including an inner self-correction loop.

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
    plan = state["global_plan"].get("data_cleaning_plan", {})
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
You write Python code according to the provided plan and constraints. \
Your code must be complete, self-contained, and directly runnable."""

_USER_PROMPT_GENERATE = """\
# CONTEXT #
Current phase: Data Cleaning

# PLAN #
{plan}

# CONSTRAINTS #
1. Load data from: `{data_path}` using `pd.read_csv`.
2. Save cleaned data to: `./workspace/cleaned_data.csv` using `df.to_csv(..., index=False)`.
3. Always work on a copy of the DataFrame, not the original.
4. Use `print()` to output a summary of what was done (rows before/after, columns modified, etc.).
5. Import all necessary libraries at the top of the code.

{feedback_section}

# RESPONSE: CODE #
Provide your complete solution in a single ```python ... ``` code block.
"""

_SYSTEM_PROMPT_FIX = """\
You are an expert Python debugger. \
Analyze the error and fix the code. \
Output the complete corrected code."""

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


def _generate_cleaning_code(plan: dict, data_path: str, feedback: str) -> str:
    """
    Call the LLM to generate data-cleaning code.
    If the LLM call fails, fall back to a template implementation.
    """
    feedback_section = ""
    if feedback:
        feedback_section = f"# SUPERVISOR FEEDBACK #\nPrevious attempt was rejected. Fix based on:\n{feedback}"

    user_prompt = _USER_PROMPT_GENERATE.format(
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        data_path=data_path,
        feedback_section=feedback_section,
    )

    reply = call_llm(_SYSTEM_PROMPT_GENERATE, user_prompt)

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
    reply = call_llm(_SYSTEM_PROMPT_FIX, user_prompt)

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


def _fallback_cleaning_code(plan: dict, data_path: str) -> str:
    """Fallback template code when LLM is unavailable."""
    return f"""\
import pandas as pd

df = pd.read_csv("{data_path}")
# Run cleaning steps based on plan objectives: {plan.get('objectives', [])}
df = df.drop_duplicates()
df = df.dropna(thresh=len(df.columns) * 0.5)

output_path = "./workspace/cleaned_data.csv"
df.to_csv(output_path, index=False)
print(f"Cleaning complete. Remaining rows: {{len(df)}}. Saved to {{output_path}}")
"""
