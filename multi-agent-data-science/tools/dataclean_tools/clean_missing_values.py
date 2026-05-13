import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.dataset_store import get_dataset, update_dataset
from tools.tool_trace import trace_tool_call

_NUMERIC_KINDS = "iufcb"
_ALIAS_TO_METHOD = {
    "median": "median",
    "mean": "mean",
    "mode": "mode",
    "most_frequent": "mode",
}
_SUPPORTED_FALLBACKS = {"fill_zero", "fill_median", "fill_mean", "none"}


def _is_numeric_dtype(series: pd.Series) -> bool:
    return series.dtype.kind in _NUMERIC_KINDS


def _resolve_fill_value(series: pd.Series, method: str) -> tuple[bool, Any, str]:
    if method == "constant":
        return False, None, "constant_requires_explicit_value"

    if method == "median":
        if not _is_numeric_dtype(series):
            return False, None, "non_numeric_for_median"
        value = series.median()
        if pd.isna(value):
            return False, None, "no_non_null_values_for_median"
        return True, value, ""

    if method == "mean":
        if not _is_numeric_dtype(series):
            return False, None, "non_numeric_for_mean"
        value = series.mean()
        if pd.isna(value):
            return False, None, "no_non_null_values_for_mean"
        return True, value, ""

    if method == "mode":
        mode = series.mode(dropna=True)
        if mode.empty:
            return False, None, "no_non_null_values_for_mode"
        return True, mode.iloc[0], ""

    return False, None, f"unsupported_fill_method:{method}"


def _normalize_action(action: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    strategy = str(action.get("strategy") or "").strip().lower()

    if strategy in _ALIAS_TO_METHOD:
        return True, {"operation": "fill", "method": _ALIAS_TO_METHOD[strategy]}, ""

    if strategy == "fill":
        method_raw = action.get("method")
        if method_raw is None:
            return True, {"operation": "fill", "method": "constant"}, ""
        method = str(method_raw).strip().lower()
        method = _ALIAS_TO_METHOD.get(method, method)
        return True, {"operation": "fill", "method": method}, ""

    if strategy == "drop_rows":
        return True, {"operation": "drop_rows"}, ""

    return False, {}, f"unsupported_strategy:{strategy or 'missing'}"


@tool
def clean_missing_values(
    dataset_id: str,
    actions: list[dict[str, Any]],
    fallback_strategy: str = "fill_zero",
) -> str:
    """Handle missing values with compatibility + soft warnings.

    Supported action strategies:
    - fill (constant): requires `value`
    - fill (method): method in {median, mean, mode, most_frequent}
    - drop_rows
    - median / mean / mode / most_frequent (strategy aliases)

    Unknown/invalid actions will not interrupt execution. They are reported in:
    - warnings
    - unsupported_actions
    - has_warnings

    Fallback strategy is applied only to columns that have no configured action.
    """
    tool_inputs = {
        "dataset_id": dataset_id,
        "actions": actions,
        "fallback_strategy": fallback_strategy,
    }
    filled_columns: list[str] = []
    dropped_rows = 0
    skipped_columns: list[dict[str, Any]] = []
    fallback_filled: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    unsupported_actions: list[dict[str, Any]] = []
    configured_columns: set[str] = set()

    try:
        df = get_dataset(dataset_id).copy()

        for idx, action in enumerate(actions or []):
            if not isinstance(action, dict):
                warn = {
                    "action_index": idx,
                    "reason": "action_must_be_object",
                    "action": action,
                }
                warnings.append(warn)
                unsupported_actions.append(warn)
                continue

            column = action.get("column")
            if not isinstance(column, str) or not column.strip():
                warn = {
                    "action_index": idx,
                    "reason": "missing_or_invalid_column",
                    "action": action,
                }
                warnings.append(warn)
                unsupported_actions.append(warn)
                continue

            column = column.strip()
            configured_columns.add(column)

            if column not in df.columns:
                skipped = {"column": column, "reason": "column_not_found"}
                skipped_columns.append(skipped)
                warnings.append({"action_index": idx, **skipped})
                continue

            ok, normalized, normalize_err = _normalize_action(action)
            if not ok:
                skipped = {"column": column, "reason": normalize_err}
                skipped_columns.append(skipped)
                warn = {"action_index": idx, **skipped, "action": action}
                warnings.append(warn)
                unsupported_actions.append(warn)
                continue

            if normalized["operation"] == "drop_rows":
                before_rows = len(df)
                df = df.dropna(subset=[column])
                dropped_rows += before_rows - len(df)
                continue

            method = normalized["method"]
            if method == "constant":
                if "value" not in action:
                    skipped = {"column": column, "reason": "fill_value_missing"}
                    skipped_columns.append(skipped)
                    warn = {"action_index": idx, **skipped, "action": action}
                    warnings.append(warn)
                    unsupported_actions.append(warn)
                    continue
                fill_value = action.get("value")
            else:
                can_fill, fill_value, fill_err = _resolve_fill_value(df[column], method)
                if not can_fill:
                    skipped = {"column": column, "reason": fill_err}
                    skipped_columns.append(skipped)
                    warn = {"action_index": idx, **skipped, "action": action}
                    warnings.append(warn)
                    unsupported_actions.append(warn)
                    continue

            before_nulls = int(df[column].isna().sum())
            df[column] = df[column].fillna(fill_value)
            after_nulls = int(df[column].isna().sum())
            if before_nulls > after_nulls and column not in filled_columns:
                filled_columns.append(column)

        if fallback_strategy not in _SUPPORTED_FALLBACKS:
            warnings.append(
                {
                    "reason": "unsupported_fallback_strategy",
                    "fallback_strategy": fallback_strategy,
                }
            )
            fallback_strategy = "none"

        if fallback_strategy != "none":
            fallback_columns = [
                col for col in df.columns[df.isna().any()] if col not in configured_columns
            ]
            for col in fallback_columns:
                if fallback_strategy == "fill_zero":
                    fill_value = 0 if _is_numeric_dtype(df[col]) else ""
                elif fallback_strategy == "fill_median":
                    if _is_numeric_dtype(df[col]):
                        fill_value = df[col].median()
                    else:
                        mode = df[col].mode(dropna=True)
                        fill_value = mode.iloc[0] if not mode.empty else ""
                else:  # fill_mean
                    if _is_numeric_dtype(df[col]):
                        fill_value = df[col].mean()
                    else:
                        mode = df[col].mode(dropna=True)
                        fill_value = mode.iloc[0] if not mode.empty else ""

                if pd.isna(fill_value):
                    warnings.append(
                        {
                            "column": col,
                            "reason": f"fallback_{fallback_strategy}_cannot_resolve_value",
                        }
                    )
                    continue

                df[col] = df[col].fillna(fill_value)
                fallback_filled[col] = fill_value

        update_dataset(dataset_id, df)
        report = {
            "dataset_id": dataset_id,
            "filled_columns": filled_columns,
            "dropped_rows": dropped_rows,
            "skipped_columns": skipped_columns,
            "fallback_filled": fallback_filled,
            "unsupported_actions": unsupported_actions,
            "warnings": warnings,
            "has_warnings": len(warnings) > 0,
        }
    except Exception as exc:
        report = {"error": {"type": exc.__class__.__name__, "message": str(exc)}}

    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("clean_missing_values", tool_inputs, output)
    return output
