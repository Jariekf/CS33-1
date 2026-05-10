"""Data Cleaning worker node.

This node uses deterministic cleaning tools and writes structured artifacts for
downstream nodes and Supervisor validation.
"""

from __future__ import annotations

import pandas as pd

from core.state_schema import (
    CLEANED_DATA_PATH,
    CLEANING_REPORT_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    WorkflowState,
)
from tools.data_cleaning_tools import (
    basic_clean_dataframe,
    cleaning_row_loss_breakdown,
    decide_duplicate_removal,
    normalize_column_names,
    numeric_like_conversion_report,
    profile_dataframe,
    split_cleaned_data,
)
from tools.validation_tools import ensure_workspace, write_json


def data_cleaning_node(state: WorkflowState) -> dict:
    """Clean raw data, split train/test, and write a cleaning report."""

    ensure_workspace()
    plan = state["global_plan"]["task_summary"] #task_summary is not showed in WorkflowState class
    target_col = plan.get("target_column")
    task_type = plan.get("task_type")
    input_mode = plan.get("input_mode", state.get("input_mode", "single_file_split"))
    data_reading_options = plan.get("data_reading_options", {})
    raw_df = normalize_column_names(pd.read_csv(state["data_path"], **data_reading_options))
    before = profile_dataframe(raw_df, target_col)
    duplicate_policy = decide_duplicate_removal(
        raw_df,
        user_goal=state.get("user_goal", ""),
        data_description=state.get("data_description", ""),
        target_col=target_col,
    )
    row_loss_breakdown = cleaning_row_loss_breakdown(
        raw_df,
        target_col,
        remove_duplicates=duplicate_policy["remove_duplicates"],
        task_type=task_type,
    )
    numeric_conversions = [
        {key: value for key, value in conversion.items() if key != "numeric_values"}
        for conversion in numeric_like_conversion_report(
            raw_df,
            target_col=target_col,
            parse_target=task_type == "regression",
        )
    ]
    cleaned_df = basic_clean_dataframe(
        raw_df,
        target_col,
        remove_duplicates=duplicate_policy["remove_duplicates"],
        task_type=task_type,
    )
    after = profile_dataframe(cleaned_df, target_col)
    if input_mode == "provided_train_test":
        test_df, split_report = _clean_provided_test_data(state, plan, target_col, task_type)
        train_df = cleaned_df
        combined_profile = {
            "train": profile_dataframe(train_df, target_col),
            "test": profile_dataframe(test_df, target_col),
        }
    else:
        train_df, test_df, split_report = split_cleaned_data(cleaned_df, target_col, task_type)
        combined_profile = None
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    report = {
        "stage": "data_cleaning",
        "status": "completed",
        "target_column": target_col,
        "input_mode": input_mode,
        "data_description_used": bool(state.get("data_description", "")),
        "profile_before": before,
        "profile_after": after,
        "provided_train_test_profile": combined_profile,
        "row_loss": before["rows"] - after["rows"],
        "row_loss_breakdown": row_loss_breakdown,
        "duplicate_policy": duplicate_policy,
        "numeric_like_conversions": numeric_conversions,
        "split": split_report,
    }
    write_json(CLEANING_REPORT_PATH, report)
    stage_reports = dict(state.get("stage_reports", {}))
    stage_reports["data_cleaning"] = report
    return {"stage_reports": stage_reports, "current_stage": "data_cleaning", "current_error": ""}


def _clean_provided_test_data(
    state: WorkflowState,
    plan: dict,
    target_col: str | None,
    task_type: str | None,
) -> tuple[pd.DataFrame, dict]:
    """Read and clean a user-provided test dataset without splitting train."""

    test_path = state.get("test_data_path")
    if not test_path:
        raise ValueError("input_mode=provided_train_test requires test_data_path")
    test_options = plan.get("test_data_reading_options") or plan.get("data_reading_options", {})
    test_raw_df = normalize_column_names(pd.read_csv(test_path, **test_options))
    if target_col and target_col not in test_raw_df.columns:
        raise ValueError(
            "Provided test data does not include the target column. "
            "Inference-only test sets are not supported yet; provide a labelled test set for evaluation."
        )
    test_cleaned_df = basic_clean_dataframe(
        test_raw_df,
        target_col,
        remove_duplicates=False,
        task_type=task_type,
    )
    return test_cleaned_df, {
        "split_strategy": "provided_train_test",
        "test_size": None,
        "warning": "",
        "test_source_path": test_path,
        "test_rows_before": int(len(test_raw_df)),
        "test_rows_after": int(len(test_cleaned_df)),
        "test_has_target": bool(target_col and target_col in test_cleaned_df.columns),
    }
