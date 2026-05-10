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
def encode_categorical(dataset_id: str, columns: list[str], method: str = "label", run_dir: str = "") -> str:
    """对类别列进行编码。

    参数：
      dataset_id (str)：内存中的数据集 ID。
      columns (list[str])：需要编码的类别列名列表。
      method (str)：编码方式。
        - "label"（默认）：LabelEncoder，将每个类别映射为整数。适合树模型（XGBoost、RandomForest）。
          拟合好的 encoder 会保存为 pkl 供测试集复用。
        - "onehot"：pd.get_dummies，将类别展开为多个 0/1 列，原列被删除。适合线性模型。
          注意：onehot 后列数会增加，确保测试集做同样处理。
      run_dir (str)：本次运行的工作目录，encoder 保存在 {run_dir}/features/encoders/ 下。

    返回 JSON，包含：
      encoded_columns：实际完成编码的列名。
      skipped_columns：跳过的列及原因。
      method：使用的编码方式。
      new_columns：onehot 编码后新增的列名列表（仅 method="onehot" 时有效）。
      encoder_paths：label 编码时每列保存的 encoder pkl 路径（仅 method="label" 时有效）。
    """
    from sklearn.preprocessing import LabelEncoder

    tool_inputs = {"dataset_id": dataset_id, "columns": columns, "method": method, "run_dir": run_dir}
    try:
        df = get_dataset(dataset_id).copy()
        encoded_columns = []
        skipped_columns = []
        new_columns: list[str] = []
        encoder_paths: dict[str, str] = {}

        for col in columns:
            if col not in df.columns:
                skipped_columns.append({"column": col, "reason": "column_not_found"})
                continue

            if method == "label":
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                encoder_dir = (
                    Path(run_dir) / "features" / "encoders"
                    if run_dir
                    else PROJECT_ROOT / "artifacts" / "features" / "encoders"
                )
                encoder_dir.mkdir(parents=True, exist_ok=True)
                enc_path = encoder_dir / f"{dataset_id}_{col}_label_encoder.pkl"
                with open(enc_path, "wb") as f:
                    pickle.dump(encoder, f)
                encoder_paths[col] = str(enc_path)
                encoded_columns.append(col)

            elif method == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col)
                new_columns.extend(dummies.columns.tolist())
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                encoded_columns.append(col)

            else:
                skipped_columns.append({"column": col, "reason": f"unsupported_method:{method}"})

        update_dataset(dataset_id, df)
        result: dict[str, Any] = {
            "dataset_id": dataset_id,
            "encoded_columns": encoded_columns,
            "skipped_columns": skipped_columns,
            "method": method,
            "new_columns": new_columns,
            "encoder_paths": encoder_paths,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("encode_categorical", tool_inputs, output)
    return output
