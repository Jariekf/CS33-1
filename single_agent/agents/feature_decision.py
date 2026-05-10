"""Feature Decision node.

This node decides whether feature engineering should be skipped, light, standard,
or task-specific based on the cleaned data profile.
"""

from __future__ import annotations

from core.state_schema import FEATURE_POLICY_REPORT_PATH, WorkflowState
from tools.feature_engineering_tools import recommend_feature_engineering_policy
from tools.validation_tools import write_json


def feature_decision_node(state: WorkflowState) -> dict:
    """Create the final feature engineering policy after cleaning."""

    task_summary = state["global_plan"]["task_summary"]
    cleaning_report = state["stage_reports"]["data_cleaning"]
    eda_report = state["stage_reports"].get("eda_analysis", {})
    policy = recommend_feature_engineering_policy(
        task_type=task_summary["task_type"],
        candidate_models=task_summary.get("candidate_models", []),
        data_profile=cleaning_report["profile_after"],
    )
    policy["eda_recommendations"] = eda_report.get("feature_engineering_recommendations", [])
    report = {"stage": "feature_decision", "feature_engineering_policy": policy}
    write_json(FEATURE_POLICY_REPORT_PATH, report)
    stage_reports = dict(state.get("stage_reports", {}))
    stage_reports["feature_decision"] = report
    return {"stage_reports": stage_reports, "current_stage": "feature_decision", "current_error": ""}
