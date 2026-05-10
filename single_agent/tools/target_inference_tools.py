"""Target and task inference tools.

LLM-assisted inference is used when available, with deterministic fallback so
the workflow remains runnable without an API key.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from core.llm import call_llm
from tools.data_cleaning_tools import infer_target_column
from tools.model_selection_tools import infer_task_type


SYSTEM_PROMPT = """You are a data science planning assistant.
Infer the most suitable target column and task type from a user goal and dataset preview.
Return only valid JSON. Do not invent columns."""


def infer_target_and_task(
    df: pd.DataFrame,
    user_goal: str,
    data_description: str = "",
    use_llm: bool = True,
) -> dict[str, Any]:
    """Infer target column and task type with LLM fallback to rules."""

    if use_llm:
        try:
            llm_result = _infer_with_llm(df, user_goal, data_description)
            if _is_valid_result(llm_result, df.columns):
                llm_result["source"] = "llm"
                return llm_result
        except Exception as exc:
            fallback = _infer_with_rules(df, user_goal, data_description)
            fallback["fallback_error"] = str(exc)
            return fallback
    return _infer_with_rules(df, user_goal, data_description)


def _infer_with_llm(df: pd.DataFrame, user_goal: str, data_description: str = "") -> dict[str, Any]:
    """Ask the LLM for a structured target/task decision."""

    profile = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_rows": df.head(5).to_dict(orient="records"),
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(exclude="number").columns),
    }
    user_prompt = f"""# User Goal
{user_goal}

# Optional Dataset Description
{data_description or "Not provided."}

# Dataset Profile
```json
{json.dumps(profile, ensure_ascii=False, indent=2, default=str)}
```

# Required JSON
{{
  "target_column": "<one column name from the dataset, or null if unsupervised/descriptive>",
  "task_type": "classification | regression | time_series | clustering | descriptive",
  "confidence": 0.0,
  "reason": "short reason tied to user goal and available columns"
}}

Rules:
- Choose a target only if the user goal implies prediction or supervised modelling.
- Use the dataset description to resolve column meaning, target definitions, time index meaning, and leakage risks.
- For financial/business outcome prediction, prefer measurable outcome columns such as profit, sales, revenue, churn, conversion, cost, or demand if present.
- Do not choose ID, date, month, year, or grouping columns as target unless explicitly requested.
- If no reliable target exists, return null and explain why.
"""
    raw = call_llm(SYSTEM_PROMPT, user_prompt)
    return _extract_json_object(raw)


def _infer_with_rules(df: pd.DataFrame, user_goal: str, data_description: str = "") -> dict[str, Any]:
    """Deterministic fallback target/task inference."""

    combined_goal = "\n".join(part for part in [user_goal, data_description] if part)
    target_col = infer_target_column(list(df.columns), combined_goal)
    task_type = infer_task_type(combined_goal, df[target_col] if target_col else None)
    return {
        "target_column": target_col,
        "task_type": task_type if target_col else "descriptive",
        "confidence": 0.4 if target_col else 0.0,
        "reason": "deterministic fallback inference",
        "source": "rules",
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from an LLM response."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain a JSON object")
    payload = json.loads(stripped[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


def _is_valid_result(result: dict[str, Any], columns: pd.Index) -> bool:
    """Validate LLM target inference result."""

    target = result.get("target_column")
    task_type = result.get("task_type")
    if task_type not in {"classification", "regression", "time_series", "clustering", "descriptive"}:
        return False
    if target is not None and target not in set(columns):
        return False
    return True
