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
    """Encode categorical columns. The encoder is fit on the training set and simultaneously transforms the test set (if provided).

    Args:
      dataset_id (str): ID of the in-memory training-set dataset.
      columns (list[str]): list of categorical column names to encode.
      method (str): encoding method.
        - "label" (default): LabelEncoder; maps each category to an integer. Suitable for tree models (XGBoost, RandomForest).
          The fitted encoder is saved as a pkl. Unseen categories in the test set are mapped to -1.
        - "onehot": OneHotEncoder; expands categories into multiple 0/1 columns; the original column is deleted. Suitable for linear models.
          Uses sklearn OneHotEncoder (handle_unknown='ignore'); fit on the training set;
          ensures the test set has exactly the same dummy columns as the training set.
      run_dir (str): working directory for this run; the encoder is saved under {run_dir}/features/encoders/.
      test_dataset_id (str): ID of the in-memory test-set dataset (optional).
        When provided, the test set is transformed (transform only, no re-fit) using the encoder fit on the training set.

    Returns JSON containing:
      encoded_columns: names of columns that were actually encoded.
      skipped_columns: skipped columns and their reasons.
      method: the encoding method used.
      new_columns: list of newly added column names after onehot encoding (only valid when method="onehot").
      encoder_paths: paths to the encoder pkl saved for each column during label encoding (only valid when method="label").
      test_dataset_id: test-set ID (if provided and processed).
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
