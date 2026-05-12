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
    """对指定数值列进行特征缩放。scaler 在训练集上 fit，同步 transform 测试集（若提供）。

    参数：
      dataset_id (str)：内存中的训练集数据集 ID。
      columns (list[str])：需要缩放的数值列名列表。不在数据集中或非数值的列会被跳过。
      method (str)：缩放方法。
        - "standard"（默认）：StandardScaler，将列转换为均值 0、方差 1。
        - "minmax"：MinMaxScaler，将列缩放至 [0, 1] 区间。
      run_dir (str)：本次运行的工作目录，scaler 保存在 {run_dir}/features/scalers/ 下。
      test_dataset_id (str)：内存中的测试集数据集 ID（可选）。
        提供后，使用训练集拟合的 scaler 对测试集做相同变换（transform only，不重新 fit）。

    返回 JSON，包含：
      scaled_columns：实际完成缩放的列名。
      skipped_columns：跳过的列及原因。
      method：使用的缩放方法。
      scaler_path：保存的 scaler pkl 文件路径。
      test_dataset_id：测试集 ID（若已提供并处理）。
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
