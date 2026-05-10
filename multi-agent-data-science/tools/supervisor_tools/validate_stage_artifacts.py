from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


REQUIRED_FIELDS_BY_STAGE: dict[str, tuple[str, ...]] = {
    "cleaning_plan_created": ("dataset_profile", "task_type", "target_column", "cleaning_plan"),
    "cleaning_saved": ("cleaned_data_path", "cleaning_report", "cleaning_passed"),
    "split_done": ("training_data_path", "testing_data_path", "split_report"),
    "eda_done": ("eda_report", "candidate_models", "feature_engineering_plan"),
    "feature_saved": ("feature_dataset_ids",),
    "model_evaluated": ("model_results",),
    "compare_done": ("best_model",),
}


REPORT_BY_STAGE: dict[str, str] = {
    "cleaning_plan_created": "agent_manager_report.md",
    "cleaning_saved": "data_clean_agent_report.md",
    "eda_done": "eda_agent_report.md",
    "feature_saved": "feature_agent_report.md",
    "model_evaluated": "model_agent_report.md",
}


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {} or value is False


def _check_path(path_value: str, *, label: str, expected: str = "file") -> dict[str, Any]:
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    check: dict[str, Any] = {
        "label": label,
        "path": str(resolved),
        "expected": expected,
        "exists": resolved.exists(),
        "passed": False,
    }

    if not resolved.exists():
        check["message"] = "路径不存在"
        return check

    if expected == "dir":
        check["is_dir"] = resolved.is_dir()
        check["passed"] = resolved.is_dir()
        check["message"] = "目录存在" if check["passed"] else "路径存在但不是目录"
        return check

    check["is_file"] = resolved.is_file()
    if not resolved.is_file():
        check["message"] = "路径存在但不是文件"
        return check

    check["size_bytes"] = resolved.stat().st_size
    if check["size_bytes"] <= 0:
        check["message"] = "文件为空"
        return check

    if resolved.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(resolved)
            check["rows"] = int(df.shape[0])
            check["cols"] = int(df.shape[1])
            check["columns"] = [str(col) for col in df.columns[:50]]
            check["passed"] = df.shape[0] > 0 and df.shape[1] > 0
            check["message"] = "CSV 可读且非空" if check["passed"] else "CSV 可读但没有有效行列"
        except Exception as exc:
            check["message"] = f"CSV 读取失败：{exc}"
        return check

    check["passed"] = True
    check["message"] = "文件存在且非空"
    return check


def _add_if_path(checks: list[dict[str, Any]], state: dict[str, Any], key: str) -> None:
    value = state.get(key)
    if isinstance(value, str) and value.strip():
        checks.append(_check_path(value, label=key))


def _check_report_path(checks: list[dict[str, Any]], state: dict[str, Any], stage: str) -> None:
    run_dir = state.get("run_dir")
    report_name = REPORT_BY_STAGE.get(stage)
    if not run_dir or not report_name:
        return
    checks.append(_check_path(str(Path(run_dir) / "reports" / report_name), label=f"report:{report_name}"))


def _check_feature_artifacts(checks: list[dict[str, Any]], state: dict[str, Any]) -> None:
    feature_dataset_ids = state.get("feature_dataset_ids") or {}
    if isinstance(feature_dataset_ids, dict):
        for model_name, path in feature_dataset_ids.items():
            if isinstance(path, str) and path:
                checks.append(_check_path(path, label=f"feature_dataset:{model_name}"))

    feature_preprocessors = state.get("feature_preprocessors") or {}
    if not isinstance(feature_preprocessors, dict):
        return

    for model_name, preprocessors in feature_preprocessors.items():
        if not isinstance(preprocessors, dict):
            continue
        for key in ("target_encoder_path", "scaler_path"):
            path = preprocessors.get(key)
            if isinstance(path, str) and path:
                checks.append(_check_path(path, label=f"preprocessor:{model_name}:{key}"))
        encoder_paths = preprocessors.get("encoder_paths") or {}
        if isinstance(encoder_paths, dict):
            for col, path in encoder_paths.items():
                if isinstance(path, str) and path:
                    checks.append(_check_path(path, label=f"encoder:{model_name}:{col}"))


def _check_model_artifacts(checks: list[dict[str, Any]], state: dict[str, Any]) -> None:
    model_results = state.get("model_results") or {}
    if isinstance(model_results, dict):
        for model_name, result in model_results.items():
            if not isinstance(result, dict):
                continue
            model_path = result.get("model_path")
            if isinstance(model_path, str) and model_path:
                checks.append(_check_path(model_path, label=f"model:{model_name}"))

    best_model = state.get("best_model") or {}
    if isinstance(best_model, dict):
        model_path = best_model.get("model_path")
        if isinstance(model_path, str) and model_path:
            checks.append(_check_path(model_path, label="best_model"))


def _build_checks(stage: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    _add_if_path(checks, state, "data_path")

    if stage in {"cleaning_saved", "split_done", "eda_done", "feature_saved", "model_evaluated", "compare_done"}:
        _add_if_path(checks, state, "cleaned_data_path")

    if stage in {"split_done", "eda_done", "feature_saved", "model_evaluated", "compare_done"}:
        _add_if_path(checks, state, "training_data_path")
        _add_if_path(checks, state, "testing_data_path")

    if stage in {"feature_saved", "model_evaluated", "compare_done"}:
        _check_feature_artifacts(checks, state)

    if stage in {"model_evaluated", "compare_done"}:
        _check_model_artifacts(checks, state)

    _check_report_path(checks, state, stage)
    return checks


@tool
def validate_stage_artifacts(current_stage: str, state_json: str) -> str:
    """校验当前阶段的轻量中间产物。

    参数：
      current_stage：当前要验收的阶段，例如 cleaning_saved、split_done、eda_done。
      state_json：GlobalState 摘要的 JSON 字符串。

    返回 JSON，包含：
      passed：必需字段和关键路径是否通过确定性校验。
      missing_fields：缺失的 state 字段。
      checks：每个文件/目录的存在性、可读性、基础元信息。
      failures：失败项摘要，供 Supervisor LLM 决定是否重跑原 agent。
    """
    tool_inputs = {"current_stage": current_stage}
    try:
        state = json.loads(state_json)
        required_fields = REQUIRED_FIELDS_BY_STAGE.get(current_stage, ())
        missing_fields = [key for key in required_fields if _is_missing(state.get(key))]
        checks = _build_checks(current_stage, state)
        failures = [
            {
                "label": check.get("label"),
                "path": check.get("path"),
                "message": check.get("message"),
            }
            for check in checks
            if not check.get("passed")
        ]
        passed = not missing_fields and not failures
        result = {
            "stage": current_stage,
            "passed": passed,
            "missing_fields": missing_fields,
            "checks": checks,
            "failures": failures,
            "summary": "artifact validation passed" if passed else "artifact validation failed",
        }
    except Exception as exc:
        result = {
            "stage": current_stage,
            "passed": False,
            "missing_fields": [],
            "checks": [],
            "failures": [{"label": "validate_stage_artifacts", "message": str(exc)}],
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
        }

    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("validate_stage_artifacts", tool_inputs, output)
    return output
