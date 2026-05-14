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


def _pick_text(action: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = action.get(key)
        if isinstance(raw, str):
            text = raw.strip()
            if text:
                return text
    return ""


def _normalize_group_by(raw_group_by: Any) -> list[str]:
    if isinstance(raw_group_by, str):
        value = raw_group_by.strip()
        return [value] if value else []

    if isinstance(raw_group_by, list):
        normalized: list[str] = []
        for item in raw_group_by:
            if isinstance(item, str):
                value = item.strip()
                if value:
                    normalized.append(value)
        return normalized

    return []


def _extract_condition_value(action: dict[str, Any]) -> Any:
    if "condition_value" in action:
        return action.get("condition_value")
    if "when_value" in action:
        return action.get("when_value")
    if "if_value" in action:
        return action.get("if_value")
    return None


def _build_condition_mask(df: pd.DataFrame, action: dict[str, Any]) -> tuple[bool, pd.Series | None, str]:
    condition_column = _pick_text(action, "condition_column", "when_column", "if_column")
    if not condition_column:
        return False, None, "condition_column_missing"
    if condition_column not in df.columns:
        return False, None, f"condition_column_not_found:{condition_column}"

    operator = _pick_text(action, "condition_operator", "operator", "op").lower() or "eq"
    condition_value = _extract_condition_value(action)
    series = df[condition_column]

    if operator in {"eq", "=="}:
        if condition_value is None:
            return False, None, "condition_value_missing"
        return True, series == condition_value, ""

    if operator in {"ne", "!="}:
        if condition_value is None:
            return False, None, "condition_value_missing"
        return True, series != condition_value, ""

    if operator == "in":
        if not isinstance(condition_value, list):
            return False, None, "condition_value_must_be_list_for_in"
        return True, series.isin(condition_value), ""

    if operator == "not_in":
        if not isinstance(condition_value, list):
            return False, None, "condition_value_must_be_list_for_not_in"
        return True, ~series.isin(condition_value), ""

    if operator == "isna":
        return True, series.isna(), ""

    if operator == "notna":
        return True, series.notna(), ""

    return False, None, f"unsupported_condition_operator:{operator}"


def _normalize_action_aliases(action: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(action)

    column = _pick_text(action, "column", "column_name")
    if column:
        normalized["column"] = column

    strategy = _pick_text(action, "strategy", "imputation_strategy", "action").lower()
    method = _pick_text(action, "method").lower()
    if not strategy and method:
        strategy = "fill"

    if strategy:
        normalized["strategy"] = strategy

    if not method and strategy == "fill":
        method_alias = _pick_text(action, "imputation_strategy").lower()
        if method_alias and method_alias != "fill":
            method = method_alias
    if method:
        normalized["method"] = method

    if "value" not in normalized and "fill_value" in action:
        normalized["value"] = action.get("fill_value")

    condition_column = _pick_text(action, "condition_column", "when_column", "if_column")
    if condition_column:
        normalized["condition_column"] = condition_column

    condition_operator = _pick_text(action, "condition_operator", "operator", "op")
    if condition_operator:
        normalized["condition_operator"] = condition_operator

    condition_value = _extract_condition_value(action)
    if condition_value is not None and "condition_value" not in normalized:
        normalized["condition_value"] = condition_value

    flag_column = _pick_text(action, "flag_column", "indicator_column")
    if flag_column:
        normalized["flag_column"] = flag_column

    if strategy == "fill_with_constant":
        normalized["strategy"] = "fill"
        normalized["method"] = "constant"
    elif strategy == "fill_with_mode":
        normalized["strategy"] = "fill"
        normalized["method"] = "mode"
    elif strategy == "fill_with_zero":
        normalized["strategy"] = "fill"
        normalized["method"] = "constant"
        normalized.setdefault("value", 0)
    elif strategy == "fill_with_constant_or_mode":
        normalized["strategy"] = "fill"
        if "value" in normalized:
            normalized["method"] = "constant"
        else:
            normalized["method"] = "mode"
    elif strategy == "impute_group_median":
        normalized["strategy"] = "group_median"
        normalized.setdefault("fallback_method", "median")
    elif strategy == "fill_with_zero_if_no_garage":
        normalized["strategy"] = "conditional_fill"
        normalized.setdefault("condition_column", "GarageType")
        normalized.setdefault("condition_value", "NoGarage")
        normalized.setdefault("value", 0)
    elif strategy == "fill_with_constant_or_median":
        normalized["strategy"] = "conditional_fill_with_flag"
        normalized.setdefault("condition_column", "GarageType")
        normalized.setdefault("condition_value", "NoGarage")
        normalized.setdefault("condition_fill_value", 0)
        normalized.setdefault("fallback_method", "median")
        if "flag_column" not in normalized:
            raw_value = action.get("value")
            if isinstance(raw_value, str) and raw_value.strip():
                normalized["flag_column"] = raw_value.strip()

    return normalized


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

    if strategy == "group_median":
        return True, {"operation": "group_median"}, ""

    if strategy == "conditional_fill":
        return True, {"operation": "conditional_fill"}, ""

    if strategy == "conditional_fill_with_flag":
        return True, {"operation": "conditional_fill_with_flag"}, ""

    return False, {}, f"unsupported_strategy:{strategy or 'missing'}"


def _mark_filled_column(df: pd.DataFrame, column: str, filled_columns: list[str], before_nulls: int) -> None:
    after_nulls = int(df[column].isna().sum())
    if before_nulls > after_nulls and column not in filled_columns:
        filled_columns.append(column)


def _resolve_method_value_for_column(
    df: pd.DataFrame,
    column: str,
    method: str,
    action: dict[str, Any],
    *,
    constant_key: str = "value",
) -> tuple[bool, Any, str]:
    if method == "constant":
        if constant_key in action:
            return True, action.get(constant_key), ""
        if constant_key == "value":
            return False, None, "fill_value_missing"
        return False, None, f"{constant_key}_missing"

    return _resolve_fill_value(df[column], method)


@tool
def clean_missing_values(
    dataset_id: str,
    actions: list[dict[str, Any]] | None = None,
    fallback_strategy: str = "fill_zero",
) -> str:
    """Handle missing values with compatibility + soft warnings.

    Supported action strategies:
    - fill (constant): requires `value`
    - fill (method): method in {median, mean, mode, most_frequent}
    - group_median (requires `group_by`)
    - conditional_fill (requires condition + `value`)
    - conditional_fill_with_flag (condition + optional `flag_column`)
    - drop_rows

    Supported legacy aliases:
    - fill_with_constant / fill_with_mode / fill_with_zero
    - fill_with_constant_or_mode
    - impute_group_median
    - fill_with_zero_if_no_garage
    - fill_with_constant_or_median

    Unknown/invalid actions will not interrupt execution. They are reported in:
    - warnings
    - unsupported_actions
    - has_warnings

    Fallback strategy is applied only to columns that have no configured action.
    """
    tool_inputs = {
        "dataset_id": dataset_id,
        "actions": actions or [],
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

            normalized_action = _normalize_action_aliases(action)
            column = normalized_action.get("column")
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

            ok, normalized, normalize_err = _normalize_action(normalized_action)
            if not ok:
                skipped = {"column": column, "reason": normalize_err}
                skipped_columns.append(skipped)
                warn = {"action_index": idx, **skipped, "action": action}
                warnings.append(warn)
                unsupported_actions.append(warn)
                continue

            operation = normalized["operation"]
            before_nulls = int(df[column].isna().sum())

            if operation == "drop_rows":
                before_rows = len(df)
                df = df.dropna(subset=[column])
                dropped_rows += before_rows - len(df)
                continue

            if operation == "fill":
                method = normalized["method"]
                can_fill, fill_value, fill_err = _resolve_method_value_for_column(
                    df, column, method, normalized_action, constant_key="value"
                )
                if not can_fill:
                    skipped = {"column": column, "reason": fill_err}
                    skipped_columns.append(skipped)
                    warn = {"action_index": idx, **skipped, "action": action}
                    warnings.append(warn)
                    unsupported_actions.append(warn)
                    continue
                df[column] = df[column].fillna(fill_value)
                _mark_filled_column(df, column, filled_columns, before_nulls)
                continue

            if operation == "group_median":
                group_by = _normalize_group_by(normalized_action.get("group_by"))
                if not group_by:
                    skipped = {"column": column, "reason": "group_by_missing_or_invalid"}
                    skipped_columns.append(skipped)
                    warn = {"action_index": idx, **skipped, "action": action}
                    warnings.append(warn)
                    unsupported_actions.append(warn)
                    continue

                missing_group_cols = [col for col in group_by if col not in df.columns]
                if missing_group_cols:
                    skipped = {
                        "column": column,
                        "reason": f"group_by_column_not_found:{','.join(missing_group_cols)}",
                    }
                    skipped_columns.append(skipped)
                    warn = {"action_index": idx, **skipped, "action": action}
                    warnings.append(warn)
                    unsupported_actions.append(warn)
                    continue

                if not _is_numeric_dtype(df[column]):
                    skipped = {"column": column, "reason": "non_numeric_for_group_median"}
                    skipped_columns.append(skipped)
                    warn = {"action_index": idx, **skipped, "action": action}
                    warnings.append(warn)
                    unsupported_actions.append(warn)
                    continue

                grouped_median = df.groupby(group_by, dropna=False)[column].transform("median")
                missing_mask = df[column].isna()
                if missing_mask.any():
                    df.loc[missing_mask, column] = grouped_median[missing_mask]

                remaining = df[column].isna()
                if remaining.any():
                    fallback_method = _pick_text(normalized_action, "fallback_method").lower() or "median"
                    fallback_method = _ALIAS_TO_METHOD.get(fallback_method, fallback_method)
                    can_fill, fill_value, fill_err = _resolve_method_value_for_column(
                        df, column, fallback_method, normalized_action, constant_key="fallback_value"
                    )
                    if can_fill:
                        df.loc[remaining, column] = df.loc[remaining, column].fillna(fill_value)
                    else:
                        warnings.append(
                            {
                                "action_index": idx,
                                "column": column,
                                "reason": f"group_median_fallback_skipped:{fill_err}",
                            }
                        )

                _mark_filled_column(df, column, filled_columns, before_nulls)
                continue

            if operation == "conditional_fill":
                ok_mask, condition_mask, mask_err = _build_condition_mask(df, normalized_action)
                if not ok_mask or condition_mask is None:
                    skipped = {"column": column, "reason": mask_err}
                    skipped_columns.append(skipped)
                    warn = {"action_index": idx, **skipped, "action": action}
                    warnings.append(warn)
                    unsupported_actions.append(warn)
                    continue

                if "value" in normalized_action:
                    fill_value = normalized_action.get("value")
                elif "condition_fill_value" in normalized_action:
                    fill_value = normalized_action.get("condition_fill_value")
                else:
                    skipped = {"column": column, "reason": "fill_value_missing"}
                    skipped_columns.append(skipped)
                    warn = {"action_index": idx, **skipped, "action": action}
                    warnings.append(warn)
                    unsupported_actions.append(warn)
                    continue

                target_mask = df[column].isna() & condition_mask
                if target_mask.any():
                    df.loc[target_mask, column] = fill_value

                _mark_filled_column(df, column, filled_columns, before_nulls)
                continue

            if operation == "conditional_fill_with_flag":
                missing_before = df[column].isna()
                flag_column = _pick_text(normalized_action, "flag_column", "indicator_column")
                if flag_column:
                    if flag_column not in df.columns:
                        df[flag_column] = 0
                    flag_series = pd.to_numeric(df[flag_column], errors="coerce").fillna(0).astype(int)
                    flag_series.loc[missing_before] = 1
                    df[flag_column] = flag_series

                condition_mask: pd.Series | None = None
                ok_mask, condition_mask, mask_err = _build_condition_mask(df, normalized_action)
                if ok_mask and condition_mask is not None:
                    fill_value = normalized_action.get("condition_fill_value", normalized_action.get("value", 0))
                    target_mask = missing_before & condition_mask
                    if target_mask.any():
                        df.loc[target_mask, column] = fill_value
                else:
                    warnings.append(
                        {
                            "action_index": idx,
                            "column": column,
                            "reason": f"conditional_fill_skipped:{mask_err}",
                        }
                    )

                remaining = df[column].isna()
                if remaining.any():
                    fallback_method = _pick_text(normalized_action, "fallback_method").lower() or "median"
                    fallback_method = _ALIAS_TO_METHOD.get(fallback_method, fallback_method)
                    if fallback_method == "constant":
                        can_fill, fill_value, fill_err = _resolve_method_value_for_column(
                            df, column, fallback_method, normalized_action, constant_key="fallback_value"
                        )
                    else:
                        reference_series = (
                            df.loc[~condition_mask, column]
                            if condition_mask is not None
                            else df[column]
                        )
                        can_fill, fill_value, fill_err = _resolve_fill_value(reference_series, fallback_method)
                    if can_fill:
                        df.loc[remaining, column] = df.loc[remaining, column].fillna(fill_value)
                    else:
                        warnings.append(
                            {
                                "action_index": idx,
                                "column": column,
                                "reason": f"conditional_fallback_skipped:{fill_err}",
                            }
                        )

                _mark_filled_column(df, column, filled_columns, before_nulls)
                continue

            skipped = {"column": column, "reason": f"unsupported_operation:{operation}"}
            skipped_columns.append(skipped)
            warn = {"action_index": idx, **skipped, "action": action}
            warnings.append(warn)
            unsupported_actions.append(warn)

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
