"""Model Selection worker node."""

from __future__ import annotations

import os

import pandas as pd

from core.progress import log_step
from core.state_schema import FEATURE_IMPORTANCE_PATH, MODEL_REPORT_PATH, WorkflowState
from tools.model_tuning_tools import tune_and_evaluate_model
from tools.validation_tools import write_json


def model_selection_node(state: WorkflowState) -> dict:
    """Train candidate models and write model reports."""

    task_summary = state["global_plan"]["task_summary"]
    target_col = task_summary.get("target_column")
    task_type = task_summary.get("task_type", "regression")
    candidate_models = task_summary.get("candidate_models", [])
    feature_pipelines = state["stage_reports"].get("feature_engineering", {}).get("pipelines", {})
    if not target_col:
        raise ValueError("Supervised model selection requires a target column")
    if not candidate_models:
        raise ValueError("Model selection requires candidate_models in task_summary")
    candidate_results = []
    best_result, best_importance = None, {}
    metric = task_summary.get("primary_metric")
    for model_plan in candidate_models:
        model_id = model_plan["model_id"]
        pipeline = feature_pipelines.get(model_id)
        if not pipeline:
            raise ValueError(f"Missing feature engineering pipeline for model_id: {model_id}")
        with log_step(f"model_selection candidate: {model_id}"):
            train_df = pd.read_csv(pipeline["train_path"])
            test_df = pd.read_csv(pipeline["test_path"])
            _, model_report, importance = tune_and_evaluate_model(
                train_df,
                test_df,
                target_col,
                task_type,
                model_id,
                model_plan=model_plan,
            )
            model_dir = os.path.dirname(pipeline["train_path"])
            model_report_path = os.path.join(model_dir, "model_report.json")
            importance_path = os.path.join(model_dir, "feature_importance.json")
            write_json(model_report_path, model_report)
            write_json(importance_path, importance)
        result = {
            "model_id": model_id,
            "model_report_path": model_report_path,
            "feature_importance_path": importance_path,
            "feature_pipeline": pipeline,
            **model_report,
        }
        candidate_results.append(result)
        if best_result is None or model_report["cv_score"] > best_result["cv_score"]:
            best_result, best_importance = result, importance
    if best_result is None:
        raise ValueError("No candidate model could be evaluated")
    report = {
        "task_type": task_type,
        "target_column": target_col,
        "candidate_model_ids": [model["model_id"] for model in candidate_models],
        "metric": metric or best_result["metric"],
        "candidate_results": candidate_results,
        "selected_model_id": best_result["model_id"],
        "best_model": best_result["model_id"],
        "best_cv_score": best_result["cv_score"],
        "test_metrics": best_result["test_metrics"],
        "tuning": {
            "enabled": True,
            "strategy": "LLM-assisted tuning plan with deterministic validated GridSearchCV execution",
            "selected_model_tuning": best_result.get("tuning", {}),
            "candidate_tuning_status": {
                result["model_id"]: result.get("tuning", {}).get("status", "unknown")
                for result in candidate_results
            },
        },
        "selected_model_report_path": best_result["model_report_path"],
        "selected_feature_importance_path": best_result["feature_importance_path"],
    }
    write_json(MODEL_REPORT_PATH, report)
    write_json(FEATURE_IMPORTANCE_PATH, best_importance)
    stage_reports = dict(state.get("stage_reports", {}))
    stage_reports["model_selection"] = report
    return {"stage_reports": stage_reports, "current_stage": "model_selection", "current_error": ""}
