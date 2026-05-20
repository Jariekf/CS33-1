import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.dataset_store import get_dataset, update_dataset
from core.env_utils import PROJECT_ROOT
from tools.tool_trace import trace_tool_call


@tool
def scale_features(
    dataset_id: str,
    columns: list[str],
    method: str = "standard",
    run_dir: str = "",
    test_dataset_id: str = "",
) -> str:
    """Scale the specified numeric columns. The scaler is fit on the training set and simultaneously transforms the test set (if provided).

    Args:
      dataset_id (str): ID of the in-memory training-set dataset.
      columns (list[str]): list of numeric column names to scale. Columns not in the dataset or non-numeric columns are skipped.
      method (str): scaling method.
        - "standard" (default): StandardScaler; transforms the column to mean 0, variance 1.
        - "minmax": MinMaxScaler; scales the column to the [0, 1] range.
      run_dir (str): working directory for this run; the scaler is saved under {run_dir}/features/scalers/.
      test_dataset_id (str): ID of the in-memory test-set dataset (optional).
        When provided, the test set is transformed (transform only, no re-fit) using the scaler fit on the training set.

    Returns JSON containing:
      scaled_columns: names of columns that were actually scaled.
      skipped_columns: skipped columns and their reasons.
      method: the scaling method used.
      scaler_path: path to the saved scaler pkl file.
      test_dataset_id: test-set ID (if provided and processed).
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    tool_inputs = {"dataset_id": dataset_id, "columns": columns, "method": method,
                   "run_dir": run_dir, "test_dataset_id": test_dataset_id}
    try:
        df = get_dataset(dataset_id).copy()
        scaled_columns = []
        skipped_columns = []

        valid_cols = []
        for col in columns:
            if col not in df.columns:
                skipped_columns.append({"column": col, "reason": "column_not_found"})
            elif not pd.api.types.is_numeric_dtype(df[col]):
                skipped_columns.append({"column": col, "reason": "not_numeric"})
            else:
                valid_cols.append(col)

        scaler_path = ""
        if valid_cols:
            scaler = StandardScaler() if method == "standard" else MinMaxScaler()
            df[valid_cols] = scaler.fit_transform(df[valid_cols])
            scaled_columns = valid_cols

            scaler_dir = (
                Path(run_dir) / "features" / "scalers"
                if run_dir
                else PROJECT_ROOT / "artifacts" / "features" / "scalers"
            )
            scaler_dir.mkdir(parents=True, exist_ok=True)
            scaler_path = str(scaler_dir / f"{dataset_id}_{method}_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            if test_dataset_id:
                test_df = get_dataset(test_dataset_id).copy()
                test_valid = [c for c in valid_cols if c in test_df.columns]
                if test_valid:
                    test_df[test_valid] = scaler.transform(test_df[test_valid])
                update_dataset(test_dataset_id, test_df)

        update_dataset(dataset_id, df)
        result: dict[str, Any] = {
            "dataset_id": dataset_id,
            "scaled_columns": scaled_columns,
            "skipped_columns": skipped_columns,
            "method": method,
            "scaler_path": scaler_path,
            "test_dataset_id": test_dataset_id,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("scale_features", tool_inputs, output)
    return output
