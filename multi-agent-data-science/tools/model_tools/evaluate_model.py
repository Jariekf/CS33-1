import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


def _preprocess_test(
    df: pd.DataFrame,
    target_column: str,
    scaler_path: str,
    scaled_columns: list[str],
    encoder_paths: dict[str, str],
    encoding_method: str,
    encoded_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    """对测试集做与训练集一致的预处理，对齐到 feature_columns。"""
    df = df.copy()

    if scaler_path and Path(scaler_path).is_file() and scaled_columns:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        valid = [c for c in scaled_columns if c in df.columns]
        if valid:
            df[valid] = scaler.transform(df[valid])

    if encoding_method == "label" and encoder_paths:
        for col, enc_path in encoder_paths.items():
            if col in df.columns and Path(enc_path).is_file():
                with open(enc_path, "rb") as f:
                    encoder = pickle.load(f)
                known = set(encoder.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in known else -1
                )
    elif encoding_method == "onehot" and encoded_columns:
        for col in encoded_columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    X = pd.DataFrame(index=df.index)
    for col in feature_columns:
        X[col] = df[col] if col in df.columns else 0
    return X


@tool
def evaluate_model(
    model_path: str,
    feature_columns_path: str,
    test_data_path: str,
    model_name: str,
    task_type: str,
    target_column: str,
    scaler_path: str = "",
    scaled_columns: list[str] | None = None,
    encoder_paths: dict[str, str] | None = None,
    encoding_method: str = "",
    encoded_columns: list[str] | None = None,
    target_encoder_path: str = "",
    evaluation_metrics: list[str] | None = None,
    primary_metric: str = "",
) -> str:
    """在测试集上评估训练好的模型，返回评估指标。

    参数：
      model_path (str)：train_model 返回的模型 pkl 路径。
      feature_columns_path (str)：train_model 返回的特征列名 JSON 路径，用于对齐测试集列。
      test_data_path (str)：原始测试集 CSV 路径（state 中的 testing_data_path）。
      model_name (str)：模型名称。
      task_type (str)："classification" 或 "regression"。
      target_column (str)：目标列名。
      scaler_path (str)：scale_features 返回的 scaler pkl 路径，无缩放则留空。
      scaled_columns (list)：被缩放的列名列表。
      encoder_paths (dict)：label 编码时各列的 encoder pkl 路径，无则留空 {}。
      encoding_method (str)："label"、"onehot" 或 ""。
      encoded_columns (list)：被编码的列名列表。
      target_encoder_path (str)：encode_target_column 保存的目标列 LabelEncoder pkl 路径。
        XGBoost 等模型训练时对目标列做了 0/1 编码，评估时必须用同一 encoder 对测试集目标列做相同转换，
        否则 y_test 仍是字符串，导致指标计算错误。无目标列编码则留空 ""。
      evaluation_metrics (list)：EDA Agent 指定的综合评估指标列表（如 ["roc_auc", "f1_weighted"]）。
        工具始终计算所有基础指标，此参数额外触发 roc_auc（分类）的计算，并在返回结果中标注。
      primary_metric (str)：EDA Agent 指定的主要指标（CV 选型所用），写入结果便于下游追踪。


    返回 JSON，包含：
      model_name：模型名称。
      model_path：模型文件路径。
      primary_metric：主要指标名称。
      metrics：全量评估指标 dict。
        - 分类基础：accuracy、f1_macro、f1_weighted、confusion_matrix（始终计算）；
          若 evaluation_metrics 含 roc_auc/recall/precision 则额外计算对应指标。
        - 回归：rmse、mae、r2
    """
    eval_metrics_set = set(evaluation_metrics or [])
    tool_inputs = {
        "model_path": model_path,
        "test_data_path": test_data_path,
        "model_name": model_name,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "evaluation_metrics": sorted(eval_metrics_set),
    }
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        feature_columns: list[str] = json.loads(Path(feature_columns_path).read_text())

        test_df = pd.read_csv(test_data_path)
        if target_column not in test_df.columns:
            raise ValueError(f"目标列 '{target_column}' 不在测试集中。")
        y_test = test_df[target_column]

        if target_encoder_path and Path(target_encoder_path).is_file():
            with open(target_encoder_path, "rb") as f:
                target_encoder = pickle.load(f)
            known = set(target_encoder.classes_)
            y_test = y_test.astype(str).apply(
                lambda x: target_encoder.transform([x])[0] if x in known else -1
            )

        X_test = _preprocess_test(
            test_df,
            target_column=target_column,
            scaler_path=scaler_path,
            scaled_columns=scaled_columns or [],
            encoder_paths=encoder_paths or {},
            encoding_method=encoding_method,
            encoded_columns=encoded_columns or [],
            feature_columns=feature_columns,
        )

        y_pred = model.predict(X_test)

        metrics: dict[str, Any] = {}
        if task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score
            metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            metrics["f1_macro"] = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
            metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
            if "recall" in eval_metrics_set:
                from sklearn.metrics import recall_score
                metrics["recall"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
            if "precision" in eval_metrics_set:
                from sklearn.metrics import precision_score
                metrics["precision"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
            if "roc_auc" in eval_metrics_set:
                try:
                    from sklearn.metrics import roc_auc_score
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)
                        n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 1
                        if n_classes == 2:
                            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                        else:
                            metrics["roc_auc"] = round(float(roc_auc_score(
                                y_test, y_prob, multi_class="ovr", average="weighted"
                            )), 4)
                except Exception:
                    metrics["roc_auc"] = None

            # 混淆矩阵：始终计算，以嵌套列表形式返回
            from sklearn.metrics import confusion_matrix
            metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        else:
            import math
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            metrics["rmse"] = round(float(math.sqrt(mean_squared_error(y_test, y_pred))), 4)
            metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)

        result = {
            "model_name": model_name,
            "model_path": model_path,
            "primary_metric": primary_metric or "",
            "metrics": metrics,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("evaluate_model", tool_inputs, output)
    return output
