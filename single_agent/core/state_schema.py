"""Shared workflow state and artifact paths for demo2.

The state is intentionally lightweight: agents read and update a dictionary-like
object, while deterministic tools write artifacts into ``workspace/``.
"""

from __future__ import annotations

from typing import Any, TypedDict


WORKSPACE_DIR = "workspace"
PIPELINES_DIR = f"{WORKSPACE_DIR}/pipelines"
CLEANED_DATA_PATH = f"{WORKSPACE_DIR}/cleaned_data.csv"
TRAIN_DATA_PATH = f"{WORKSPACE_DIR}/train_data.csv"
TEST_DATA_PATH = f"{WORKSPACE_DIR}/test_data.csv"
FE_TRAIN_PATH = f"{WORKSPACE_DIR}/fe_train.csv"
FE_TEST_PATH = f"{WORKSPACE_DIR}/fe_test.csv"
CLEANING_REPORT_PATH = f"{WORKSPACE_DIR}/cleaning_report.json"
EDA_REPORT_PATH = f"{WORKSPACE_DIR}/eda_report.json"
EDA_SUMMARY_PATH = f"{WORKSPACE_DIR}/eda_summary.md"
FEATURE_POLICY_REPORT_PATH = f"{WORKSPACE_DIR}/feature_policy_report.json"
FEATURE_REPORT_PATH = f"{WORKSPACE_DIR}/feature_report.json"
MODEL_REPORT_PATH = f"{WORKSPACE_DIR}/model_report.json"
FEATURE_IMPORTANCE_PATH = f"{WORKSPACE_DIR}/feature_importance.json"
PLAN_PATCH_LOG_PATH = f"{WORKSPACE_DIR}/plan_patches.json"
MANAGER_PLAN_PATH = f"{WORKSPACE_DIR}/manager_plan.json"
BUSINESS_REPORT_PATH = f"{WORKSPACE_DIR}/business_report.md"


class WorkflowState(TypedDict, total=False):
    """Mutable state passed between workflow nodes."""

    user_goal: str
    data_description: str
    data_path: str
    test_data_path: str
    input_mode: str
    global_plan: dict[str, Any]
    stage_reports: dict[str, dict[str, Any]]
    current_stage: str
    current_error: str
    plan_version: int
    plan_patches: list[dict[str, Any]]
    manager_plan_path: str
    business_report_path: str
