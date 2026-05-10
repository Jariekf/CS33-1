"""EDA analysis node."""

from __future__ import annotations

import pandas as pd

from core.state_schema import (
    CLEANED_DATA_PATH,
    EDA_REPORT_PATH,
    EDA_SUMMARY_PATH,
    TRAIN_DATA_PATH,
    WorkflowState,
)
from tools.eda_tools import build_eda_report, build_eda_summary
from tools.validation_tools import write_json


def eda_analysis_node(state: WorkflowState) -> dict:
    """Run leakage-aware EDA after data cleaning."""

    task_summary = state["global_plan"]["task_summary"]
    target_col = task_summary.get("target_column")
    task_type = task_summary.get("task_type", "regression")
    cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    report = build_eda_report(cleaned_df, train_df, target_col, task_type)
    report["data_description_used"] = bool(state.get("data_description", ""))
    report["data_description"] = state.get("data_description", "")
    write_json(EDA_REPORT_PATH, report)
    with open(EDA_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(build_eda_summary(report))
    stage_reports = dict(state.get("stage_reports", {}))
    stage_reports["eda_analysis"] = report
    return {"stage_reports": stage_reports, "current_stage": "eda_analysis", "current_error": ""}
