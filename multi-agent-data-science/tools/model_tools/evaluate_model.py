import json
import math
import os
import pickle
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
    """Evaluate a trained model on a preprocessed test set and return evaluation metrics.

    Pipeline position: train_model (full retrain) -> **evaluate_model (test-set evaluation)**

    Precondition: the feature agent has applied preprocessing identical to that of the training set
    (scaling, encoding, including target-column encoding) to the test set, saved the result as a CSV,
    and stored the path in state.testing_feature_dataset_ids[model_name].

    Args:
      model_path (str): path to the model pkl returned by train_model.
      feature_columns_path (str): path to the feature-column-names JSON returned by train_model,
        used to extract the aligned feature columns from the test set.
      featured_test_data_path (str): path to the preprocessed test-set CSV output by the feature agent
        (state.testing_feature_dataset_ids[model_name]).
      model_name (str): name of the model.
      task_type (str): "classification" or "regression".
      target_column (str): target column name; used to extract y_test from the test set.
      evaluation_metrics (list): list of key evaluation metrics specified by the EDA Agent
        (e.g. ["roc_auc", "f1_weighted"]). Written to the result JSON so that the model agent
        and supervisor agent know which metrics to focus on during review.
        The tool always computes the full set of metrics internally (this list is not an if-gate):
          classification: accuracy, f1_macro, f1_weighted, recall, precision, roc_auc (all computed).
          regression: rmse, mae, r2 (all computed).
      primary_metric (str): the primary metric specified by the EDA Agent (used for CV model selection);
        written to the result for downstream tracking.

    Returns JSON containing:
      model_name: name of the model.
      model_path: path to the model file.
      primary_metric: name of the primary metric.
      evaluation_metrics: the key metrics list, passed through as-is.
      evaluation_metrics: the key metrics list, passed through as-is.
      metrics: dict of scalar evaluation metrics (excluding the confusion matrix).
      confusion_matrix: nested-list confusion matrix (classification only; null for regression).
      resource_metrics: resource-aware metrics dict.
        - inference_latency_ms_per_sample: average per-sample inference latency after warm-up (ms).
        - model_size_kb: size of the model pkl file (KB).
      test_predictions_path: path to the predictions CSV (under run_dir/predictions/ alongside the model).
        Column order: feature columns (X_test) -> y_true -> y_pred -> y_prob_{class} (classification only).
        Includes all feature columns so that each sample's input features and prediction can be traced directly during review.
        null when writing fails (non-fatal; does not affect the metrics result).
    """
    tool_inputs = {
        "model_path": model_path,
        "featured_test_data_path": featured_test_data_path,
        "model_name": model_name,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "evaluation_metrics": sorted(set(evaluation_metrics or [])),
        "evaluation_metrics": sorted(set(evaluation_metrics or [])),
    }
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        feature_columns: list[str] = json.loads(Path(feature_columns_path).read_text())

        test_df = pd.read_csv(featured_test_data_path)
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in test set.")

        y_test = test_df[target_column]

        # Extract X_test using the column order determined during training; fill missing columns with 0.
        X_test = pd.DataFrame(index=test_df.index)
        for col in feature_columns:
            X_test[col] = test_df[col] if col in test_df.columns else 0

        # -- Inference latency timing (warm up once, then time formally and average) --
        model.predict(X_test)  # warm-up
        _t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        _elapsed_ms = (time.perf_counter() - _t0) * 1000
        n_samples = len(X_test)
        inference_latency_ms_per_sample = round(_elapsed_ms / n_samples, 4) if n_samples > 0 else None

        metrics: dict[str, Any] = {}
        confusion_matrix_result = None
        n_classes_eval: int | None = None
        y_proba = None  # declared here so prediction-saving block can access it

        if task_type == "classification":
            import numpy as np
            from sklearn.metrics import (
                accuracy_score,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
            )
            metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            # f1_macro: equal weight per class → sensitive to minority class performance
            # f1_weighted: sample-count weighted → reflects overall accuracy on the actual distribution
            metrics["f1_macro"] = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
            metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
            metrics["recall"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
            metrics["precision"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)

            # ── Compute y_proba once; reused by roc_auc / log_loss / top-k ──────
            n_classes_eval = int(y_test.nunique())

            # sensitivity: binary → recall of positive class (TP / (TP+FN))
            #              multi-class → macro recall (unweighted mean per-class recall)
            if n_classes_eval == 2:
                metrics["sensitivity"] = round(float(recall_score(y_test, y_pred, average="binary", zero_division=0)), 4)
            else:
                metrics["sensitivity"] = round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4)
            _proba_err: str = ""
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception as exc:
                    _proba_err = f"predict_proba failed: {exc}"
            else:
                _proba_err = "model does not support predict_proba"

            # ── roc_auc ───────────────────────────────────────────────────────
            if y_proba is not None:
                try:
                    from sklearn.metrics import roc_auc_score
                    n_cls = y_proba.shape[1] if y_proba.ndim > 1 else 1
                    if n_cls == 2:
                        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba[:, 1])), 4)
                    else:
                        metrics["roc_auc"] = round(float(roc_auc_score(
                            y_test, y_proba, multi_class="ovr", average="weighted"
                        )), 4)
                except ValueError as exc:
                    metrics["roc_auc"] = None
                    metrics["roc_auc_skipped_reason"] = str(exc)
                except Exception as exc:
                    metrics["roc_auc"] = None
                    metrics["roc_auc_skipped_reason"] = f"unexpected error: {exc}"
            else:
                metrics["roc_auc"] = None
                metrics["roc_auc_skipped_reason"] = _proba_err

            # ── log_loss: continuous & probability-calibration-sensitive ──────
            if y_proba is not None:
                try:
                    from sklearn.metrics import log_loss
                    metrics["log_loss"] = round(float(log_loss(y_test, y_proba)), 4)
                except Exception as exc:
                    metrics["log_loss"] = None
                    metrics["log_loss_skipped_reason"] = str(exc)
            else:
                metrics["log_loss"] = None
                metrics["log_loss_skipped_reason"] = _proba_err

            # ── top-k accuracy: auto-computed when n_classes > 10 ─────────────
            if n_classes_eval > 10:
                k = min(5, n_classes_eval - 1)
                if y_proba is not None:
                    try:
                        from sklearn.metrics import top_k_accuracy_score
                        metrics[f"top_{k}_accuracy"] = round(
                            float(top_k_accuracy_score(y_test, y_proba, k=k)), 4
                        )
                    except Exception as exc:
                        metrics[f"top_{k}_accuracy"] = None
                        metrics[f"top_{k}_accuracy_skipped_reason"] = str(exc)
                else:
                    metrics[f"top_{k}_accuracy"] = None
                    metrics[f"top_{k}_accuracy_skipped_reason"] = _proba_err

            confusion_matrix_result = confusion_matrix(y_test, y_pred).tolist()
        else:
            import numpy as np
            from sklearn.metrics import (
                max_error as sklearn_max_error,
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )
            metrics["rmse"] = round(float(math.sqrt(mean_squared_error(y_test, y_pred))), 4)
            metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)
            metrics["max_error"] = round(float(sklearn_max_error(y_test, y_pred)), 4)
            # mape: undefined when y_true contains zeros → degrade gracefully
            try:
                from sklearn.metrics import mean_absolute_percentage_error
                mape_val = float(mean_absolute_percentage_error(y_test, y_pred))
                metrics["mape"] = round(mape_val, 4) if np.isfinite(mape_val) else None
                if not np.isfinite(mape_val):
                    metrics["mape_skipped_reason"] = "y_true contains zero(s); MAPE is undefined"
            except Exception as exc:
                metrics["mape"] = None
                metrics["mape_skipped_reason"] = str(exc)
            # rmsle: clip predictions to ≥ 0 to avoid log domain errors
            try:
                from sklearn.metrics import mean_squared_log_error
                metrics["rmsle"] = round(
                    float(np.sqrt(mean_squared_log_error(y_test, np.clip(y_pred, 0, None)))), 4
                )
            except Exception as exc:
                metrics["rmsle"] = None
                metrics["rmsle_skipped_reason"] = str(exc)

        # -- Resource-aware metrics --
        model_size_kb = round(os.path.getsize(model_path) / 1024, 2) if Path(model_path).is_file() else None

        # -- Persist predictions to disk (traceability) --
        # Column order: feature columns -> y_true -> y_pred -> y_prob_{class} (classification only)
        # Includes X_test feature columns so each sample's input features and prediction can be traced directly during review.
        # model_path = .../run_dir/models/ModelName.pkl -> predictions/ is a sibling directory
        test_predictions_path: str | None = None
        try:
            pred_dir = Path(model_path).parent.parent / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_df = X_test.copy()
            pred_df["y_true"] = y_test.values
            pred_df["y_pred"] = y_pred
            if task_type == "classification" and y_proba is not None:
                classes = (
                    model.classes_ if hasattr(model, "classes_") else range(y_proba.shape[1])
                )
                for i, cls in enumerate(classes):
                    pred_df[f"y_prob_{cls}"] = y_proba[:, i]
            pred_path = pred_dir / f"{model_name}_test_predictions.csv"
            pred_df.to_csv(pred_path, index=True)
            test_predictions_path = str(pred_path)
        except Exception as _pred_exc:
            test_predictions_path = None  # non-fatal, do not abort evaluation

        result = {
            "task_type": task_type,
            "model_name": model_name,
            "model_path": model_path,
            "primary_metric": primary_metric or "",
            "evaluation_metrics": sorted(set(evaluation_metrics or [])),
            "n_classes": n_classes_eval if task_type == "classification" else None,
            "metrics": metrics,
            "confusion_matrix": confusion_matrix_result,
            "resource_metrics": {
                "inference_latency_ms_per_sample": inference_latency_ms_per_sample,
                "model_size_kb": model_size_kb,
            },
            "test_predictions_path": test_predictions_path,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("evaluate_model", tool_inputs, output)
    return output
