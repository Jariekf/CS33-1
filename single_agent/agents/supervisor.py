"""Supervisor node with deterministic artifact validation."""

from __future__ import annotations

from core.state_schema import (
    CLEANING_REPORT_PATH,
    BUSINESS_REPORT_PATH,
    CLEANED_DATA_PATH,
    EDA_REPORT_PATH,
    EDA_SUMMARY_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_POLICY_REPORT_PATH,
    FEATURE_REPORT_PATH,
    FE_TEST_PATH,
    FE_TRAIN_PATH,
    MODEL_REPORT_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    WorkflowState,
)
from tools.validation_tools import validate_csv, validate_json_object, validate_matching_columns


def supervisor_node(state: WorkflowState, stage: str) -> dict:
    """Validate required artifacts for a workflow stage."""

    checks = _stage_checks(stage)
    failures = []
    for check in checks:
        ok, message = check()
        if not ok:
            failures.append(message)
    report = {"stage": stage, "passed": not failures, "failures": failures}
    return {"current_stage": f"supervisor:{stage}", "current_error": "; ".join(failures), "supervisor_report": report}


def _stage_checks(stage: str):
    """Return deterministic validation checks for a stage."""

    if stage == "data_cleaning":
        return [
            lambda: validate_csv(CLEANED_DATA_PATH),
            lambda: validate_csv(TRAIN_DATA_PATH),
            lambda: validate_csv(TEST_DATA_PATH),
            lambda: validate_json_object(CLEANING_REPORT_PATH),
        ]
    if stage == "feature_engineering":
        return [
            lambda: validate_matching_columns(FE_TRAIN_PATH, FE_TEST_PATH),
            lambda: validate_json_object(FEATURE_POLICY_REPORT_PATH),
            lambda: validate_json_object(FEATURE_REPORT_PATH),
        ]
    if stage == "eda_analysis":
        return [
            lambda: validate_json_object(EDA_REPORT_PATH),
            lambda: _validate_markdown_report(EDA_SUMMARY_PATH, min_length=100, required_sections=["EDA Summary"]),
        ]
    if stage == "model_selection":
        return [
            lambda: validate_json_object(MODEL_REPORT_PATH),
            lambda: validate_json_object(FEATURE_IMPORTANCE_PATH),
        ]
    if stage == "business_translator":
        return [lambda: _validate_markdown_report(BUSINESS_REPORT_PATH)]
    return [lambda: (False, f"unknown stage: {stage}")]


def _validate_markdown_report(
    path: str,
    min_length: int = 200,
    required_sections: list[str] | None = None,
) -> tuple[bool, str]:
    """Validate that a markdown report exists and has useful content."""

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except FileNotFoundError:
        return False, f"missing markdown report: {path}"
    if len(content) < min_length:
        return False, f"markdown report is too short: {path}"
    required_sections = required_sections or ["Executive", "Evidence", "Recommendations", "Limitations"]
    missing = [section for section in required_sections if section not in content]
    if missing:
        return False, f"markdown report missing sections: {missing}"
    return True, ""
