import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.dataset_store import get_dataset, update_dataset
from core.env_utils import PROJECT_ROOT
from tools.tool_trace import trace_tool_call

# Models that require numeric (integer) target labels and cannot handle string labels.
_MODELS_REQUIRING_ENCODING = {"XGBoost", "XGBoostRegressor"}


@tool
def encode_target_column(
    dataset_id: str,
    target_column: str,
    model_name: str,
    task_type: str = "classification",
    run_dir: str = "",
    test_dataset_id: str = "",
) -> str:
    """对目标列进行编码，使其与指定模型的要求兼容。encoder 在训练集上 fit，同步 transform 测试集（若提供）。

    工具内部自动判断是否需要编码：
      - 需要编码的模型（XGBoost、XGBoostRegressor）：
          分类任务使用 LabelEncoder 将标签映射为 0/1/2...，保存 encoder pkl，同步编码测试集目标列。
          回归任务确保目标列为 float 数值型，测试集目标列同步转换。
      - 不需要编码的模型（RandomForest、LogisticRegression、SVM、KNN 等）：
          sklearn 内部会处理字符串标签，返回 status="skipped"，目标列不变。
      - 若目标列已经是整数（如 0/1），返回 status="already_numeric"，跳过编码。

    参数：
      dataset_id (str)：内存中的训练集数据集 ID。
      target_column (str)：目标列名称，不要传入特征列。
      model_name (str)：当前模型名称，如 "XGBoost"、"RandomForest"。
      task_type (str)：任务类型。
        - "classification"（默认）：分类任务，标签编码为整数。
        - "regression"：回归任务，将目标列转换为 float。
      run_dir (str)：本次运行的工作目录，encoder 保存在 {run_dir}/features/encoders/ 下。
      test_dataset_id (str)：内存中的测试集数据集 ID（可选）。
        提供后，使用训练集拟合的 encoder 对测试集目标列做相同变换。

    返回 JSON，包含：
      dataset_id：数据集 ID。
      target_column：目标列名。
      encoded：是否执行了编码（True/False）。
      status：执行状态。
        - "encoded"：成功完成 LabelEncoder 编码。
        - "already_numeric"：目标列已为整数，跳过编码。
        - "converted_to_float"：回归任务转换为 float 成功。
        - "skipped"：当前模型不需要目标列编码。
      encoder_path：LabelEncoder pkl 路径（status="encoded" 时有值，其余为空字符串）。
      original_classes：编码前的原始类别列表（status="encoded" 时有值）。
      test_dataset_id：测试集 ID（若已提供并处理）。
    """
    from sklearn.preprocessing import LabelEncoder

    tool_inputs = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "model_name": model_name,
        "task_type": task_type,
        "run_dir": run_dir,
        "test_dataset_id": test_dataset_id,
    }
    try:
        df = get_dataset(dataset_id).copy()
        test_df = get_dataset(test_dataset_id).copy() if test_dataset_id else None

        if target_column not in df.columns:
            output = json.dumps(
                {
                    "error": {
                        "type": "ColumnNotFound",
                        "message": f"目标列 '{target_column}' 不存在于数据集中",
                    }
                },
                ensure_ascii=False,
            )
            trace_tool_call("encode_target_column", tool_inputs, output)
            return output

        result: dict[str, Any] = {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "encoded": False,
            "status": "skipped",
            "encoder_path": "",
            "original_classes": [],
            "test_dataset_id": test_dataset_id,
        }

        if task_type == "regression":
            df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
            update_dataset(dataset_id, df)
            result["encoded"] = True
            result["status"] = "converted_to_float"
            if test_df is not None and target_column in test_df.columns:
                test_df[target_column] = pd.to_numeric(test_df[target_column], errors="coerce")
                update_dataset(test_dataset_id, test_df)

        elif task_type == "classification":
            col = df[target_column]
            already_int = pd.api.types.is_integer_dtype(col) or (
                pd.api.types.is_float_dtype(col)
                and col.dropna().apply(lambda x: float(x).is_integer()).all()
            )

            if already_int:
                result["status"] = "already_numeric"
            elif model_name in _MODELS_REQUIRING_ENCODING:
                encoder = LabelEncoder()
                original_classes = col.unique().tolist()
                df[target_column] = encoder.fit_transform(col.astype(str))

                encoder_dir = (
                    Path(run_dir) / "features" / "encoders"
                    if run_dir
                    else PROJECT_ROOT / "artifacts" / "features" / "encoders"
                )
                encoder_dir.mkdir(parents=True, exist_ok=True)
                enc_path = encoder_dir / f"{dataset_id}_{model_name}_target_encoder.pkl"
                with open(enc_path, "wb") as f:
                    pickle.dump(encoder, f)

                update_dataset(dataset_id, df)
                result["encoded"] = True
                result["status"] = "encoded"
                result["encoder_path"] = str(enc_path)
                result["original_classes"] = [str(c) for c in original_classes]

                if test_df is not None and target_column in test_df.columns:
                    known = set(encoder.classes_)
                    test_df[target_column] = test_df[target_column].astype(str).apply(
                        lambda x: encoder.transform([x])[0] if x in known else -1
                    )
                    update_dataset(test_dataset_id, test_df)
            # else: other models (RandomForest, LogisticRegression, etc.) → status="skipped"

        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("encode_target_column", tool_inputs, output)
    return output
