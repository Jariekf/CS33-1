import json
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def evaluate_model(
    model_path: str,
    feature_columns_path: str,
    featured_test_data_path: str,
    model_name: str,
    task_type: str,
    target_column: str,
    evaluation_metrics: list[str] | None = None,
    primary_metric: str = "",
) -> str:
    """在预处理好的测试集上评估训练好的模型，返回评估指标。

    工作流位置：train_model（全量重训）→ **evaluate_model（测试集评估）**

    前置条件：feature agent 已对测试集做了与训练集完全一致的预处理（缩放、编码，含目标列编码），
    并将结果保存为 CSV，路径存入 state.testing_feature_dataset_ids[model_name]。

    参数：
      model_path (str)：train_model 返回的模型 pkl 路径。
      feature_columns_path (str)：train_model 返回的特征列名 JSON 路径，用于从测试集中提取对齐的特征列。
      featured_test_data_path (str)：feature agent 输出的预处理后测试集 CSV 路径
        (state.testing_feature_dataset_ids[model_name])。
      model_name (str)：模型名称。
      task_type (str)："classification" 或 "regression"。
      target_column (str)：目标列名，从测试集中提取 y_test。
      evaluation_metrics (list)：EDA Agent 指定的重点评估指标列表（如 ["roc_auc", "f1_weighted"]）。
        写入结果 JSON，供 model agent 复盘和 supervisor agent 审查时划定关注范围。
        工具内部始终计算全量指标（不以此列表为 if-gate）：
          分类：accuracy、f1_macro、f1_weighted、recall、precision、roc_auc（均计算）。
          回归：rmse、mae、r2（均计算）。
      primary_metric (str)：EDA Agent 指定的主要指标（CV 选型所用），写入结果便于下游追踪。

    返回 JSON，包含：
      model_name：模型名称。
      model_path：模型文件路径。
      primary_metric：主要指标名称。
      evaluation_metrics：原样透传的重点指标列表。
      metrics：标量评估指标 dict（不含混淆矩阵）。
      confusion_matrix：混淆矩阵嵌套列表（仅分类任务，否则为 null）。
      resource_metrics：资源感知指标 dict。
        - inference_latency_ms_per_sample：预热后单样本平均推理延迟（毫秒）。
        - model_size_kb：模型 pkl 文件大小（KB）。
    """
    tool_inputs = {
        "model_path": model_path,
        "featured_test_data_path": featured_test_data_path,
        "model_name": model_name,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "evaluation_metrics": sorted(set(evaluation_metrics or [])),
    }
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        feature_columns: list[str] = json.loads(Path(feature_columns_path).read_text())

        test_df = pd.read_csv(featured_test_data_path)
        if target_column not in test_df.columns:
            raise ValueError(f"目标列 '{target_column}' 不在测试集中。")

        y_test = test_df[target_column]

        # 按训练时确定的列序提取 X_test，缺失列填 0
        X_test = pd.DataFrame(index=test_df.index)
        for col in feature_columns:
            X_test[col] = test_df[col] if col in test_df.columns else 0

        # ── 推理延迟计时（预热 1 次后正式计时，取均值）────────────────────────
        model.predict(X_test)  # warm-up
        _t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        _elapsed_ms = (time.perf_counter() - _t0) * 1000
        n_samples = len(X_test)
        inference_latency_ms_per_sample = round(_elapsed_ms / n_samples, 4) if n_samples > 0 else None

        metrics: dict[str, Any] = {}
        confusion_matrix_result = None

        if task_type == "classification":
            from sklearn.metrics import (
                accuracy_score,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
            )
            metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            metrics["f1_macro"] = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
            metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
            metrics["recall"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
            metrics["precision"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
            # roc_auc requires predict_proba; set None + reason if model doesn't support it
            # or if y_test contains only one class (e.g. skewed test split)
            if hasattr(model, "predict_proba"):
                try:
                    from sklearn.metrics import roc_auc_score
                    y_prob = model.predict_proba(X_test)
                    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 1
                    if n_classes == 2:
                        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                    else:
                        metrics["roc_auc"] = round(float(roc_auc_score(
                            y_test, y_prob, multi_class="ovr", average="weighted"
                        )), 4)
                except ValueError as exc:
                    metrics["roc_auc"] = None
                    metrics["roc_auc_skipped_reason"] = str(exc)
                except Exception as exc:
                    metrics["roc_auc"] = None
                    metrics["roc_auc_skipped_reason"] = f"unexpected error: {exc}"
            else:
                metrics["roc_auc"] = None
                metrics["roc_auc_skipped_reason"] = "model does not support predict_proba"
            confusion_matrix_result = confusion_matrix(y_test, y_pred).tolist()
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            metrics["rmse"] = round(float(math.sqrt(mean_squared_error(y_test, y_pred))), 4)
            metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)

        # ── 资源感知指标 ──────────────────────────────────────────────────────
        model_size_kb = round(os.path.getsize(model_path) / 1024, 2) if Path(model_path).is_file() else None

        result = {
            "task_type": task_type,
            "model_name": model_name,
            "model_path": model_path,
            "primary_metric": primary_metric or "",
            "evaluation_metrics": sorted(set(evaluation_metrics or [])),
            "metrics": metrics,
            "confusion_matrix": confusion_matrix_result,
            "resource_metrics": {
                "inference_latency_ms_per_sample": inference_latency_ms_per_sample,
                "model_size_kb": model_size_kb,
            },
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("evaluate_model", tool_inputs, output)
    return output
