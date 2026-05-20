import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.dataset_store import get_dataset, update_dataset
from tools.tool_trace import trace_tool_call


@tool
def clean_outliers(dataset_id: str, actions: list[dict[str, Any]] | None = None) -> str:
    """Handle outliers.

    Each action must contain the following fields:
      - column   (str): the column name to process; must be a numeric column.
      - strategy (str): the handling strategy; the only currently supported value is "clip_iqr" (clip values outside the IQR range to the boundary).

    Example:
      actions = [
          {"column": "Sales", "strategy": "clip_iqr"},
          {"column": "Profit", "strategy": "clip_iqr"},
      ]

    Note: strategy must be exactly "clip_iqr"; any other value (e.g. "iqr", "clip", "IQR") will cause the column to be skipped.
    """
    tool_inputs = {"dataset_id": dataset_id, "actions": actions or []}
    processed_columns = []
    skipped_columns = []
    try:
        df = get_dataset(dataset_id).copy()
        for action in actions or []:
            column = action.get("column")
            strategy = action.get("strategy")
            if column not in df.columns:
                skipped_columns.append({"column": column, "reason": "column_not_found"})
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                skipped_columns.append({"column": column, "reason": "not_numeric"})
                continue
            if strategy != "clip_iqr":
                skipped_columns.append(
                    {"column": column, "reason": f"unsupported_strategy:{strategy}"}
                )
                continue

            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            processed_columns.append(column)
        update_dataset(dataset_id, df)
        report = {
            "dataset_id": dataset_id,
            "processed_columns": processed_columns,
            "skipped_columns": skipped_columns,
        }
    except Exception as exc:
        report = {"error": {"type": exc.__class__.__name__, "message": str(exc)}}

    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("clean_outliers", tool_inputs, output)
    return output
