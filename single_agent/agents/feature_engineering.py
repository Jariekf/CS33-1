"""Feature Engineering worker node."""

from __future__ import annotations

import os

import pandas as pd

from core.progress import log_step
from core.state_schema import (
    FE_TEST_PATH,
    FE_TRAIN_PATH,
    FEATURE_REPORT_PATH,
    PIPELINES_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    WorkflowState,
)
from tools.feature_engineering_tools import apply_model_feature_policy
from tools.validation_tools import write_json


def feature_engineering_node(state: WorkflowState) -> dict:
    """Apply the selected FE policy and write model-ready train/test artifacts."""

    task_summary = state["global_plan"]["task_summary"]
    policy = state["stage_reports"]["feature_decision"]["feature_engineering_policy"]
    candidate_models = task_summary.get("candidate_models", [])
    target_col = task_summary.get("target_column")
    if not candidate_models:
        raise ValueError("Feature engineering requires candidate_models in task_summary")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    pipelines: dict = {}
    first_train, first_test = None, None
    for model_plan in candidate_models:
        model_id = model_plan["model_id"]
        with log_step(f"feature_engineering pipeline: {model_id}"):
            fe_train, fe_test, execution = apply_model_feature_policy(
                train_df,
                test_df,
                model_plan,
                target_col,
                policy=policy,
            )
            model_dir = os.path.join(PIPELINES_DIR, model_id)
            os.makedirs(model_dir, exist_ok=True)
            train_path = os.path.join(model_dir, "fe_train.csv")
            test_path = os.path.join(model_dir, "fe_test.csv")
            fe_train.to_csv(train_path, index=False)
            fe_test.to_csv(test_path, index=False)
        pipelines[model_id] = {
            "model_id": model_id,
            "train_path": train_path,
            "test_path": test_path,
            "execution": execution,
            "train_shape": list(fe_train.shape),
            "test_shape": list(fe_test.shape),
            "target_column": target_col,
        }
        if first_train is None:
            first_train, first_test = fe_train, fe_test
    if first_train is not None and first_test is not None:
        first_train.to_csv(FE_TRAIN_PATH, index=False)
        first_test.to_csv(FE_TEST_PATH, index=False)
    report = {
        "stage": "feature_engineering",
        "status": "completed",
        "policy": policy,
        "pipelines": pipelines,
        "execution": {"pipeline_count": len(pipelines), "model_ids": list(pipelines.keys())},
        "train_shape": list(first_train.shape) if first_train is not None else [0, 0],
        "test_shape": list(first_test.shape) if first_test is not None else [0, 0],
        "target_column": target_col,
    }
    write_json(FEATURE_REPORT_PATH, report)
    stage_reports = dict(state.get("stage_reports", {}))
    stage_reports["feature_engineering"] = report
    return {"stage_reports": stage_reports, "current_stage": "feature_engineering", "current_error": ""}
