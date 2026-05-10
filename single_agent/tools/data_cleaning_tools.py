"""Deterministic data cleaning tools for worker agents."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from core.llm import call_llm


DUPLICATE_POLICY_SYSTEM_PROMPT = """You decide whether exact duplicate rows should be removed during data cleaning.
Return only valid JSON. Prefer preserving duplicates when they may represent legitimate repeated events or transactions."""


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to stable snake_case labels."""

    result = df.copy()
    result.columns = [
        str(col).strip().lower().replace(" ", "_").replace("__", "_")
        for col in result.columns
    ]
    return result


def infer_target_column(columns: list[str], user_goal: str = "") -> str | None:
    """Infer a likely target column from semantic column names and user goal."""

    lowered = {col: col.strip().lower().replace(" ", "_") for col in columns}
    exact_names = {"target", "label", "outcome", "response", "class", "churn", "y"}
    for col, low in lowered.items():
        if low in exact_names:
            return col
    goal = user_goal.lower()
    for col, low in lowered.items():
        if low in goal:
            return col
    financial_priority = ["profit", "sales", "gross_sales", "units_sold", "revenue"]
    if any(word in goal for word in ["financial", "business", "outcome", "recommendation", "predict"]):
        for preferred in financial_priority:
            for col, low in lowered.items():
                if low == preferred:
                    return col
    return None


def profile_dataframe(df: pd.DataFrame, target_col: str | None = None) -> dict[str, Any]:
    """Return a compact data profile for planning and validation."""

    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_ratio": df.isna().mean().round(4).to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False)
        profile["target_column"] = target_col
        profile["target_unique_values"] = int(df[target_col].nunique(dropna=True))
        profile["target_distribution"] = counts.head(20).to_dict()
    return profile


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that contain no observed values."""

    return df.dropna(axis=1, how="all")


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""

    return df.drop_duplicates().reset_index(drop=True)


def decide_duplicate_removal(
    df: pd.DataFrame,
    user_goal: str = "",
    data_description: str = "",
    target_col: str | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Decide whether exact duplicate rows should be removed."""

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count == 0:
        return {
            "remove_duplicates": False,
            "source": "deterministic",
            "reason": "No exact duplicate rows were detected.",
            "duplicate_rows_detected": 0,
        }
    context = _duplicate_policy_context(df, user_goal, data_description, target_col, duplicate_count)
    if use_llm:
        try:
            result = _extract_json_object(call_llm(DUPLICATE_POLICY_SYSTEM_PROMPT, _duplicate_policy_prompt(context)))
            if isinstance(result.get("remove_duplicates"), bool):
                return {
                    "remove_duplicates": result["remove_duplicates"],
                    "source": "llm",
                    "reason": str(result.get("reason", "")),
                    "duplicate_rows_detected": duplicate_count,
                }
        except Exception as exc:
            fallback = _fallback_duplicate_policy(context)
            fallback["fallback_error"] = str(exc)
            return fallback
    return _fallback_duplicate_policy(context)


def impute_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    target_col: str | None = None,
) -> pd.DataFrame:
    """Impute missing feature values while preserving the target column."""

    result = df.copy()
    feature_cols = [col for col in result.columns if col != target_col]
    for col in feature_cols:
        if not result[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(result[col]):
            fill_value = result[col].median() if numeric_strategy == "median" else result[col].mean()
        else:
            mode = result[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty and categorical_strategy == "mode" else "Unknown"
        result[col] = result[col].fillna(fill_value)
    return result


def parse_numeric_like_columns(
    df: pd.DataFrame,
    target_col: str | None = None,
    parse_target: bool = False,
) -> pd.DataFrame:
    """Convert currency/percent-like object columns into numeric values when safe."""

    result = df.copy()
    for conversion in numeric_like_conversion_report(result, target_col=target_col, parse_target=parse_target):
        result[conversion["column"]] = conversion["numeric_values"]
    return result


def numeric_like_conversion_report(
    df: pd.DataFrame,
    target_col: str | None = None,
    parse_target: bool = False,
) -> list[dict[str, Any]]:
    """Return safe numeric-like conversions without mutating the input frame."""

    conversions: list[dict[str, Any]] = []
    for col in df.columns:
        if (col == target_col and not parse_target) or _is_identifier_like_column(col):
            continue
        series = df[col]
        if not pd.api.types.is_object_dtype(series):
            continue
        numeric = _parse_numeric_like_series(series)
        parse_ratio = float(numeric.notna().mean())
        if parse_ratio >= 0.8:
            conversions.append({
                "column": col,
                "original_dtype": str(series.dtype),
                "converted_dtype": str(numeric.dtype),
                "parse_success_ratio": round(parse_ratio, 4),
                "numeric_values": numeric,
            })
    return conversions


def cleaning_row_loss_breakdown(
    df: pd.DataFrame,
    target_col: str | None = None,
    remove_duplicates: bool = True,
    task_type: str | None = None,
) -> dict[str, int]:
    """Report row losses caused by missing target values and duplicate rows."""

    cleaned = normalize_column_names(df)
    cleaned = drop_empty_columns(cleaned)
    cleaned = parse_numeric_like_columns(cleaned, target_col=target_col, parse_target=task_type == "regression")
    missing_target_rows_removed = 0
    if target_col and target_col in cleaned.columns:
        missing_target_rows_removed = int(cleaned[target_col].isna().sum())
        cleaned = cleaned.dropna(subset=[target_col])
    duplicate_rows_removed = int(cleaned.duplicated().sum())
    if not remove_duplicates:
        duplicate_rows_removed = 0
    return {
        "missing_target_rows_removed": missing_target_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
    }


def basic_clean_dataframe(
    df: pd.DataFrame,
    target_col: str | None = None,
    remove_duplicates: bool = True,
    task_type: str | None = None,
) -> pd.DataFrame:
    """Apply safe baseline cleaning steps."""

    cleaned = normalize_column_names(df)
    cleaned = drop_empty_columns(cleaned)
    cleaned = parse_numeric_like_columns(cleaned, target_col=target_col, parse_target=task_type == "regression")
    if target_col and target_col in cleaned.columns:
        cleaned = cleaned.dropna(subset=[target_col])
    if remove_duplicates:
        cleaned = drop_duplicate_rows(cleaned)
    return cleaned


def split_cleaned_data(
    df: pd.DataFrame,
    target_col: str | None = None,
    task_type: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split cleaned data into train/test with stratification when safe."""

    if task_type in {"time_series", "forecasting"}:
        raise ValueError(
            "Time-series tasks require chronological splitting, but data cleaning "
            "currently only supports random supervised train/test splits."
        )
    if len(df) < 2:
        raise ValueError(
            "Data cleaning produced fewer than 2 rows; at least 2 rows are required "
            "to create train/test datasets. The dataset is too small after cleaning."
        )
    n_test = math.ceil(len(df) * test_size)
    n_train = len(df) - n_test
    if n_train < 1 or n_test < 1:
        raise ValueError(
            f"Data cleaning cannot create a train/test split with {len(df)} rows "
            f"and test_size={test_size}."
        )
    stratify = None
    split_strategy = "random"
    split_warning = ""
    if target_col and task_type == "classification" and target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False)
        class_count = len(counts)
        if len(counts) > 1 and counts.min() >= 2 and n_train >= class_count and n_test >= class_count:
            stratify = df[target_col]
            split_strategy = "stratified_random"
        elif len(counts) > 1:
            split_warning = (
                "Falling back to random split because the dataset is too small "
                "to place every class in both train and test sets."
            )
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        {"split_strategy": split_strategy, "test_size": test_size, "warning": split_warning},
    )


def _is_identifier_like_column(col: str) -> bool:
    """Return whether a column name should be preserved as an identifier-like string."""

    lowered = str(col).lower()
    return any(token in lowered for token in ("id", "code", "zip", "postal", "phone"))


def _parse_numeric_like_series(series: pd.Series) -> pd.Series:
    """Parse currency/percent-like strings into numeric values."""

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"^\s*-\s*$", "0", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _duplicate_policy_context(
    df: pd.DataFrame,
    user_goal: str,
    data_description: str,
    target_col: str | None,
    duplicate_count: int,
) -> dict[str, Any]:
    """Build compact context for duplicate-removal decisions."""

    return {
        "user_goal": user_goal,
        "data_description": data_description,
        "target_column": target_col,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows_detected": duplicate_count,
        "duplicate_ratio": round(duplicate_count / max(len(df), 1), 4),
        "sample_duplicate_rows": df[df.duplicated(keep=False)].head(5).to_dict(orient="records"),
    }


def _duplicate_policy_prompt(context: dict[str, Any]) -> str:
    """Build the duplicate-removal prompt."""

    return f"""# Dataset Context
```json
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}
```

# Required JSON
{{
  "remove_duplicates": true,
  "reason": "short reason"
}}

Rules:
- Return true when exact duplicates likely represent accidental repeated records.
- Return false when exact duplicates may be legitimate repeated transactions, events, logs, time-series observations, or repeated measurements.
- Consider column names such as date, time, timestamp, transaction, order, invoice, event, session, visit, log, and amount as signals that duplicates may be legitimate.
"""


def _fallback_duplicate_policy(context: dict[str, Any]) -> dict[str, Any]:
    """Conservative deterministic duplicate policy when the LLM is unavailable."""

    columns = " ".join(str(col).lower() for col in context.get("columns", []))
    event_tokens = ("date", "time", "timestamp", "transaction", "order", "invoice", "event", "session", "visit", "log")
    preserve = any(token in columns for token in event_tokens)
    return {
        "remove_duplicates": not preserve,
        "source": "rules",
        "reason": (
            "Preserved exact duplicates because column names suggest event or transaction records."
            if preserve else
            "Removed exact duplicates because no event or transaction indicators were detected."
        ),
        "duplicate_rows_detected": int(context.get("duplicate_rows_detected", 0)),
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
