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
    """Encode the target column to make it compatible with the requirements of the specified model. The encoder is fit on the training set and simultaneously transforms the test set (if provided).

    The tool automatically determines whether encoding is needed:
      - Models that require encoding (XGBoost, XGBoostRegressor):
          For classification tasks, LabelEncoder maps labels to 0/1/2...; saves the encoder pkl; simultaneously encodes the test-set target column.
          For regression tasks, ensures the target column is a float numeric type; the test-set target column is converted simultaneously.
      - Models that do not require encoding (RandomForest, LogisticRegression, SVM, KNN, etc.):
          sklearn handles string labels internally; returns status="skipped" and leaves the target column unchanged.
      - If the target column is already integer (e.g. 0/1), returns status="already_numeric" and skips encoding.

    Args:
      dataset_id (str): ID of the in-memory training-set dataset.
      target_column (str): target column name; do not pass a feature column.
      model_name (str): current model name, e.g. "XGBoost", "RandomForest".
      task_type (str): task type.
        - "classification" (default): classification task; labels are encoded as integers.
        - "regression": regression task; convert the target column to float.
      run_dir (str): working directory for this run; the encoder is saved under {run_dir}/features/encoders/.
      test_dataset_id (str): ID of the in-memory test-set dataset (optional).
        When provided, the test-set target column is transformed using the encoder fit on the training set.

    Returns JSON containing:
      dataset_id: dataset ID.
      target_column: target column name.
      encoded: whether encoding was performed (True/False).
      status: execution status.
        - "encoded": LabelEncoder encoding completed successfully.
        - "already_numeric": target column is already integer; encoding skipped.
        - "converted_to_float": regression task conversion to float succeeded.
        - "skipped": current model does not require target column encoding.
      encoder_path: path to the LabelEncoder pkl (has a value when status="encoded"; otherwise an empty string).
      original_classes: list of original classes before encoding (has a value when status="encoded").
      test_dataset_id: test-set ID (if provided and processed).
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
                        "message": f"Target column '{target_column}' does not exist in the dataset",
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
