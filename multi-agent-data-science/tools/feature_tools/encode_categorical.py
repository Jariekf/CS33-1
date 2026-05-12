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
def encode_categorical(
    dataset_id: str,
    columns: list[str],
    method: str = "label",
    run_dir: str = "",
    test_dataset_id: str = "",
) -> str:
    """对类别列进行编码。encoder 在训练集上 fit，同步 transform 测试集（若提供）。

    参数：
      dataset_id (str)：内存中的训练集数据集 ID。
      columns (list[str])：需要编码的类别列名列表。
      method (str)：编码方式。
        - "label"（默认）：LabelEncoder，将每个类别映射为整数。适合树模型（XGBoost、RandomForest）。
          拟合好的 encoder 会保存为 pkl。测试集中未见过的类别映射为 -1。
        - "onehot"：OneHotEncoder，将类别展开为多个 0/1 列，原列被删除。适合线性模型。
          使用 sklearn OneHotEncoder（handle_unknown='ignore'），fit 在训练集上，
          确保测试集与训练集 dummy 列完全一致。
      run_dir (str)：本次运行的工作目录，encoder 保存在 {run_dir}/features/encoders/ 下。
      test_dataset_id (str)：内存中的测试集数据集 ID（可选）。
        提供后，使用训练集拟合的 encoder 对测试集做相同变换（transform only，不重新 fit）。

    返回 JSON，包含：
      encoded_columns：实际完成编码的列名。
      skipped_columns：跳过的列及原因。
      method：使用的编码方式。
      new_columns：onehot 编码后新增的列名列表（仅 method="onehot" 时有效）。
      encoder_paths：label 编码时每列保存的 encoder pkl 路径（仅 method="label" 时有效）。
      test_dataset_id：测试集 ID（若已提供并处理）。
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    tool_inputs = {"dataset_id": dataset_id, "columns": columns, "method": method,
                   "run_dir": run_dir, "test_dataset_id": test_dataset_id}
    try:
        df = get_dataset(dataset_id).copy()
        test_df = get_dataset(test_dataset_id).copy() if test_dataset_id else None

        encoded_columns = []
        skipped_columns = []
        new_columns: list[str] = []
        encoder_paths: dict[str, str] = {}

        encoder_dir = (
            Path(run_dir) / "features" / "encoders"
            if run_dir
            else PROJECT_ROOT / "artifacts" / "features" / "encoders"
        )

        for col in columns:
            if col not in df.columns:
                skipped_columns.append({"column": col, "reason": "column_not_found"})
                continue

            if method == "label":
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))

                encoder_dir.mkdir(parents=True, exist_ok=True)
                enc_path = encoder_dir / f"{dataset_id}_{col}_label_encoder.pkl"
                with open(enc_path, "wb") as f:
                    pickle.dump(encoder, f)
                encoder_paths[col] = str(enc_path)
                encoded_columns.append(col)

                if test_df is not None and col in test_df.columns:
                    known = set(encoder.classes_)
                    test_df[col] = test_df[col].astype(str).apply(
                        lambda x: encoder.transform([x])[0] if x in known else -1
                    )

            elif method == "onehot":
                encoder = OneHotEncoder(handle_unknown="ignore", dtype=float)
                encoded_train = encoder.fit_transform(df[[col]])
                if hasattr(encoded_train, "toarray"):
                    encoded_train = encoded_train.toarray()

                feature_names = encoder.get_feature_names_out([col]).tolist()
                new_columns.extend(feature_names)
                df = pd.concat(
                    [df.drop(columns=[col]),
                     pd.DataFrame(encoded_train, columns=feature_names, index=df.index)],
                    axis=1,
                )
                encoded_columns.append(col)

                encoder_dir.mkdir(parents=True, exist_ok=True)
                enc_path = encoder_dir / f"{dataset_id}_{col}_ohe_encoder.pkl"
                with open(enc_path, "wb") as f:
                    pickle.dump(encoder, f)
                encoder_paths[col] = str(enc_path)

                if test_df is not None and col in test_df.columns:
                    encoded_test = encoder.transform(test_df[[col]])
                    if hasattr(encoded_test, "toarray"):
                        encoded_test = encoded_test.toarray()
                    test_df = pd.concat(
                        [test_df.drop(columns=[col]),
                         pd.DataFrame(encoded_test, columns=feature_names, index=test_df.index)],
                        axis=1,
                    )
                elif test_df is not None:
                    # column absent in test: fill dummy columns with 0
                    for fname in feature_names:
                        test_df[fname] = 0.0

            else:
                skipped_columns.append({"column": col, "reason": f"unsupported_method:{method}"})

        update_dataset(dataset_id, df)
        if test_df is not None:
            update_dataset(test_dataset_id, test_df)

        result: dict[str, Any] = {
            "dataset_id": dataset_id,
            "encoded_columns": encoded_columns,
            "skipped_columns": skipped_columns,
            "method": method,
            "new_columns": new_columns,
            "encoder_paths": encoder_paths,
            "test_dataset_id": test_dataset_id,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("encode_categorical", tool_inputs, output)
    return output
