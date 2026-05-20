import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def compute_correlation_matrix(file_path: str, target_column: str = "") -> str:
    """Compute the Pearson correlation matrix for the numeric columns of the training set.

    Args:
      file_path (str): path to the training-set CSV file.
      target_column (str): target column name. When provided and present among the numeric columns, additionally outputs
        the absolute-value ranking of each feature's correlation with the target column (highest to lowest) to help assess feature importance.

    Returns JSON containing:
      columns: list of column names used in the computation.
      matrix: full correlation matrix (dict of dict).
      target_correlation_ranking: ranking of each feature's correlation with the target column (only present when target_column is valid).
    """
    tool_inputs = {"file_path": file_path, "target_column": target_column}
    try:
        df = pd.read_csv(file_path)
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            raise ValueError("No numeric columns in the training set; cannot compute correlation matrix.")

        corr = numeric_df.corr().round(4)
        result: dict[str, Any] = {
            "columns": list(corr.columns),
            "matrix": corr.to_dict(),
        }

        if target_column and target_column in corr.columns:
            ranking = (
                corr[target_column]
                .drop(target_column)
                .abs()
                .sort_values(ascending=False)
                .round(4)
                .to_dict()
            )
            result["target_correlation_ranking"] = ranking

        output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("compute_correlation_matrix", tool_inputs, output)
    return output
