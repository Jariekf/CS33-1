"""
Manager prompt evaluation reporting.

This module writes an automated run record for the current manager prompt
configuration after a normal pipeline execution. The output is intended for
prompt iteration: after manually editing manager-related system prompts, rerun
the program and compare the reports under workspace/.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

from config import WORKSPACE_DIR

MANAGER_EVAL_DIR = os.path.join(WORKSPACE_DIR, "manager_eval")
MANAGER_EVAL_LATEST_DIR = os.path.join(MANAGER_EVAL_DIR, "latest")
MANAGER_EVAL_HISTORY_DIR = os.path.join(MANAGER_EVAL_DIR, "history")


def write_manager_prompt_eval_report(
    initial_state: dict,
    final_state: dict,
    run_started_at: datetime,
) -> None:
    """Write timestamped and latest manager prompt evaluation reports."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(MANAGER_EVAL_LATEST_DIR, exist_ok=True)
    os.makedirs(MANAGER_EVAL_HISTORY_DIR, exist_ok=True)

    metrics = _collect_metrics(
        initial_state=initial_state,
        final_state=final_state,
        run_started_at=run_started_at,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"manager_prompt_eval_{timestamp}"

    md_path = os.path.join(MANAGER_EVAL_HISTORY_DIR, f"{base_name}.md")
    csv_path = os.path.join(MANAGER_EVAL_HISTORY_DIR, f"{base_name}.csv")
    latest_md_path = os.path.join(MANAGER_EVAL_LATEST_DIR, "manager_prompt_eval_latest.md")
    latest_csv_path = os.path.join(MANAGER_EVAL_LATEST_DIR, "manager_prompt_eval_latest.csv")

    markdown = _render_markdown_report(metrics)
    csv_row = _csv_row(metrics)

    _write_text(md_path, markdown)
    _write_text(latest_md_path, markdown)
    _write_csv(csv_path, csv_row)
    _write_csv(latest_csv_path, csv_row)


def _collect_metrics(initial_state: dict, final_state: dict, run_started_at: datetime) -> dict:
    global_plan = final_state.get("global_plan", {}) or {}
    eval_meta = global_plan.get("_manager_plan_eval", {}) or {}
    summary_meta = global_plan.get("_manager_summary_eval", {}) or {}
    fusion_meta = global_plan.get("_manager_fusion_eval", {}) or {}

    artifacts = _artifact_status(run_started_at=run_started_at)
    if final_state.get("final_business_report", ""):
        artifacts["business_report"] = True
    reports_count = sum(
        1 for key in ("data_cleaning_report", "feature_engineering_report", "model_selection_report")
        if artifacts[key]
    )

    pipeline_success = all(
        artifacts[key] for key in ("cleaned_data", "fe_output", "feature_importance", "business_report")
    ) and reports_count == 3
    summary_json_parse_ok = bool(summary_meta.get("json_parse_ok", False))
    summary_required_fields_ok = bool(summary_meta.get("required_fields_ok", False))
    summary_empty_rate = float(summary_meta.get("empty_rate", 1.0))
    fusion_empty_rate = float(fusion_meta.get("empty_rate", 1.0))
    fusion_repetition_rate = float(fusion_meta.get("repetition_rate", 0.0))
    fusion_code_leakage_flag = bool(fusion_meta.get("code_leakage_flag", False))
    fusion_rap_enabled = bool(fusion_meta.get("rap_enabled", False))
    fusion_arxiv_raw_chars = int(fusion_meta.get("arxiv_raw_chars", 0))
    fusion_web_raw_chars = int(fusion_meta.get("web_raw_chars", 0))
    fusion_pwc_raw_chars = int(fusion_meta.get("pwc_raw_chars", 0))
    fusion_kaggle_raw_chars = int(fusion_meta.get("kaggle_raw_chars", 0))
    failure_label, failure_owner, failure_reason = _infer_failure_attribution(
        json_parse_ok=bool(eval_meta.get("json_parse_ok", False)),
        schema_ok=bool(eval_meta.get("schema_ok", False)),
        used_fallback=bool(eval_meta.get("used_fallback", True)),
        pipeline_success=pipeline_success,
        summary_json_parse_ok=summary_json_parse_ok,
        summary_required_fields_ok=summary_required_fields_ok,
        summary_empty_rate=summary_empty_rate,
        fusion_empty_rate=fusion_empty_rate,
        fusion_code_leakage_flag=fusion_code_leakage_flag,
    )

    return {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_path": initial_state.get("data_path", ""),
        "user_goal": initial_state.get("user_goal", ""),
        "manager_cleaning_hints": initial_state.get("manager_cleaning_hints", ""),
        "prompt_hash": eval_meta.get("prompt_hash", ""),
        "system_prompt": eval_meta.get("system_prompt", ""),
        "user_prompt_preview": eval_meta.get("user_prompt_preview", ""),
        "raw_reply_preview": eval_meta.get("raw_reply_preview", ""),
        "json_parse_ok": eval_meta.get("json_parse_ok", False),
        "schema_ok": eval_meta.get("schema_ok", False),
        "used_fallback": eval_meta.get("used_fallback", True),
        "summary_json_parse_ok": summary_json_parse_ok,
        "summary_required_fields_ok": summary_required_fields_ok,
        "summary_empty_rate": summary_empty_rate,
        "fusion_empty_rate": fusion_empty_rate,
        "fusion_repetition_rate": fusion_repetition_rate,
        "fusion_code_leakage_flag": fusion_code_leakage_flag,
        "fusion_rap_enabled": fusion_rap_enabled,
        "fusion_arxiv_raw_chars": fusion_arxiv_raw_chars,
        "fusion_web_raw_chars": fusion_web_raw_chars,
        "fusion_pwc_raw_chars": fusion_pwc_raw_chars,
        "fusion_kaggle_raw_chars": fusion_kaggle_raw_chars,
        "faithfulness_warning_count": int(summary_meta.get("faithfulness_warning_count", 0)),
        "failure_attribution_label": failure_label,
        "failure_owner": failure_owner,
        "failure_reason": failure_reason,
        "pipeline_success": pipeline_success,
        "outer_retry_total": final_state.get("outer_retry_total", 0),
        "artifacts": artifacts,
    }


def _artifact_status(run_started_at: datetime) -> dict[str, bool]:
    threshold = run_started_at.timestamp()

    def _fresh(path: str) -> bool:
        return os.path.exists(path) and os.path.getmtime(path) >= threshold

    return {
        "cleaned_data": _fresh(os.path.join(WORKSPACE_DIR, "cleaned_data.csv")),
        "fe_output": _fresh(os.path.join(WORKSPACE_DIR, "fe_output.csv")),
        "feature_importance": _fresh(os.path.join(WORKSPACE_DIR, "feature_importance.json")),
        "business_report": _fresh(os.path.join(WORKSPACE_DIR, "business_insight_report.md")),
        "data_cleaning_report": _fresh(os.path.join(WORKSPACE_DIR, "data_cleaning_report.md")),
        "feature_engineering_report": _fresh(
            os.path.join(WORKSPACE_DIR, "feature_engineering_report.md")
        ),
        "model_selection_report": _fresh(os.path.join(WORKSPACE_DIR, "model_selection_report.md")),
    }


def _render_markdown_report(metrics: dict) -> str:
    metric_rows = [
        ("run_timestamp", metrics["run_timestamp"]),
        ("prompt_hash", metrics["prompt_hash"]),
        ("json_parse_ok", metrics["json_parse_ok"]),
        ("schema_ok", metrics["schema_ok"]),
        ("used_fallback", metrics["used_fallback"]),
        ("summary_json_parse_ok", metrics["summary_json_parse_ok"]),
        ("summary_required_fields_ok", metrics["summary_required_fields_ok"]),
        ("summary_empty_rate", metrics["summary_empty_rate"]),
        ("fusion_empty_rate", metrics["fusion_empty_rate"]),
        ("fusion_repetition_rate", metrics["fusion_repetition_rate"]),
        ("fusion_code_leakage_flag", metrics["fusion_code_leakage_flag"]),
        ("fusion_rap_enabled", metrics["fusion_rap_enabled"]),
        ("fusion_arxiv_raw_chars", metrics["fusion_arxiv_raw_chars"]),
        ("fusion_web_raw_chars", metrics["fusion_web_raw_chars"]),
        ("fusion_pwc_raw_chars", metrics["fusion_pwc_raw_chars"]),
        ("fusion_kaggle_raw_chars", metrics["fusion_kaggle_raw_chars"]),
        ("faithfulness_warning_count", metrics["faithfulness_warning_count"]),
        ("failure_attribution_label", metrics["failure_attribution_label"]),
        ("failure_owner", metrics["failure_owner"]),
        ("outer_retry_total", metrics["outer_retry_total"]),
        ("pipeline_success", metrics["pipeline_success"]),
    ]

    metric_table = "\n".join(f"| {name} | {value} |" for name, value in metric_rows)
    artifact_rows = "\n".join(
        f"| {name} | {status} |" for name, status in metrics["artifacts"].items()
    )

    return f"""# Manager Prompt Eval Report

## Run Context

- Data path: `{metrics["data_path"]}`
- User goal: {metrics["user_goal"]}
- Cleaning hints: {metrics["manager_cleaning_hints"] or "(none)"}
- Prompt hash: `{metrics["prompt_hash"]}`

## Metric Record Table

| Metric | Value |
| --- | --- |
{metric_table}

## Artifact Status

| Artifact | Exists |
| --- | --- |
{artifact_rows}

## Metric Definitions

| Metric | What It Measures |
| --- | --- |
| `json_parse_ok` | Whether the raw `manager_plan` reply could be parsed into JSON successfully. |
| `schema_ok` | Whether the parsed plan contains the required top-level sections and expected core types. |
| `used_fallback` | Whether the pipeline had to fall back to the built-in default plan because the LLM output was unusable. |
| `summary_json_parse_ok` | Whether `manager_summary` output can be parsed as JSON. |
| `summary_required_fields_ok` | Whether all required summary fields are present and non-empty. |
| `summary_empty_rate` | Empty-field ratio in required summary fields (0 means all present). |
| `fusion_empty_rate` | Whether fused RAP context is empty (1.0 empty, 0.0 non-empty). |
| `fusion_repetition_rate` | Ratio of duplicated lines in fusion output (lower is better). |
| `fusion_code_leakage_flag` | Whether fusion output leaks code-like content (e.g. fences/import/def/class). |
| `fusion_rap_enabled` | Whether RAP retrieval was enabled for this run. |
| `fusion_*_raw_chars` | Raw character counts gathered from arXiv, web, PapersWithCode, and Kaggle before fusion. |
| `faithfulness_warning_count` | Warning count from lightweight faithfulness checks; for observation only. |
| `failure_attribution_label` | Primary failure attribution label for this run. |
| `failure_owner` | Suggested owner to triage the primary failure cause. |
| `outer_retry_total` | Total number of supervisor-triggered outer retries. Lower is better. |
| `pipeline_success` | Whether the full workflow produced the expected intermediate artifacts and stage reports successfully. |
| `artifact status` | A supporting reference only. It shows which output files were produced during this run. |

## Failure Attribution

- Label: `{metrics["failure_attribution_label"]}`
- Owner: `{metrics["failure_owner"]}`
- Reason: {metrics["failure_reason"]}

## Prompt Snapshot

```text
{metrics["system_prompt"]}
```

## Raw Reply Preview

```text
{metrics["raw_reply_preview"] or "(empty reply)"}
```

## User Prompt Preview

```text
{metrics["user_prompt_preview"]}
```
"""


def _csv_row(metrics: dict) -> list[dict[str, str]]:
    return [
        {
            "run_timestamp": metrics["run_timestamp"],
            "prompt_hash": metrics["prompt_hash"],
            "json_parse_ok": str(metrics["json_parse_ok"]),
            "schema_ok": str(metrics["schema_ok"]),
            "used_fallback": str(metrics["used_fallback"]),
            "summary_json_parse_ok": str(metrics["summary_json_parse_ok"]),
            "summary_required_fields_ok": str(metrics["summary_required_fields_ok"]),
            "summary_empty_rate": str(metrics["summary_empty_rate"]),
            "fusion_empty_rate": str(metrics["fusion_empty_rate"]),
            "fusion_repetition_rate": str(metrics["fusion_repetition_rate"]),
            "fusion_code_leakage_flag": str(metrics["fusion_code_leakage_flag"]),
            "fusion_rap_enabled": str(metrics["fusion_rap_enabled"]),
            "fusion_arxiv_raw_chars": str(metrics["fusion_arxiv_raw_chars"]),
            "fusion_web_raw_chars": str(metrics["fusion_web_raw_chars"]),
            "fusion_pwc_raw_chars": str(metrics["fusion_pwc_raw_chars"]),
            "fusion_kaggle_raw_chars": str(metrics["fusion_kaggle_raw_chars"]),
            "faithfulness_warning_count": str(metrics["faithfulness_warning_count"]),
            "failure_attribution_label": metrics["failure_attribution_label"],
            "failure_owner": metrics["failure_owner"],
            "outer_retry_total": str(metrics["outer_retry_total"]),
            "pipeline_success": str(metrics["pipeline_success"]),
        }
    ]


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_csv(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _infer_failure_attribution(
    *,
    json_parse_ok: bool,
    schema_ok: bool,
    used_fallback: bool,
    pipeline_success: bool,
    summary_json_parse_ok: bool,
    summary_required_fields_ok: bool,
    summary_empty_rate: float,
    fusion_empty_rate: float,
    fusion_code_leakage_flag: bool,
) -> tuple[str, str, str]:
    """Infer primary owner/tag from minimal health metrics."""
    if pipeline_success:
        return ("none", "none", "pipeline_success=true")
    if (not json_parse_ok) or (not schema_ok) or used_fallback:
        return (
            "manager_input_issue",
            "manager",
            "manager_plan parse/schema/fallback indicates prompt-input instability",
        )
    if (
        (not summary_json_parse_ok)
        or (not summary_required_fields_ok)
        or (summary_empty_rate > 0.0)
        or (fusion_empty_rate > 0.0)
        or fusion_code_leakage_flag
    ):
        return (
            "manager_midlayer_issue",
            "manager",
            "summary/fusion health metrics indicate midlayer quality degradation",
        )
    return (
        "downstream_execution_issue",
        "downstream",
        "manager structure looks healthy; likely downstream execution or review instability",
    )


def write_manager_midlayer_eval_report(
    *,
    data_path: str,
    user_goal: str,
    manager_cleaning_hints: str = "",
) -> dict:
    """Standalone summary/fusion evaluation entry independent from full pipeline."""
    from agents.manager import _get_data_preview
    from agents.manager_retrieval import (
        retrieve_and_fuse_rap_context_with_meta,
        summarize_and_extract_slots_with_meta,
    )

    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(MANAGER_EVAL_LATEST_DIR, exist_ok=True)
    os.makedirs(MANAGER_EVAL_HISTORY_DIR, exist_ok=True)
    df_preview = _get_data_preview(data_path)
    summary, slots, summary_meta = summarize_and_extract_slots_with_meta(
        user_goal=user_goal,
        df_preview=df_preview,
        cleaning_hints=manager_cleaning_hints,
    )
    fusion_text, fusion_meta = retrieve_and_fuse_rap_context_with_meta(
        user_requirement_summary=summary,
        slots=slots,
        user_goal=user_goal,
    )
    metrics = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_path": data_path,
        "user_goal": user_goal,
        "manager_cleaning_hints": manager_cleaning_hints,
        "summary_json_parse_ok": bool(summary_meta.get("json_parse_ok", False)),
        "summary_required_fields_ok": bool(summary_meta.get("required_fields_ok", False)),
        "summary_empty_rate": float(summary_meta.get("empty_rate", 1.0)),
        "fusion_empty_rate": float(fusion_meta.get("empty_rate", 1.0)),
        "fusion_repetition_rate": float(fusion_meta.get("repetition_rate", 0.0)),
        "fusion_code_leakage_flag": bool(fusion_meta.get("code_leakage_flag", False)),
        "fusion_rap_enabled": bool(fusion_meta.get("rap_enabled", False)),
        "fusion_arxiv_raw_chars": int(fusion_meta.get("arxiv_raw_chars", 0)),
        "fusion_web_raw_chars": int(fusion_meta.get("web_raw_chars", 0)),
        "fusion_pwc_raw_chars": int(fusion_meta.get("pwc_raw_chars", 0)),
        "fusion_kaggle_raw_chars": int(fusion_meta.get("kaggle_raw_chars", 0)),
        "faithfulness_warning_count": int(summary_meta.get("faithfulness_warning_count", 0)),
        "summary_preview": summary[:1000],
        "fusion_preview": fusion_text[:1200],
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"manager_midlayer_eval_{timestamp}"
    md_path = os.path.join(MANAGER_EVAL_HISTORY_DIR, f"{base_name}.md")
    csv_path = os.path.join(MANAGER_EVAL_HISTORY_DIR, f"{base_name}.csv")
    latest_md_path = os.path.join(MANAGER_EVAL_LATEST_DIR, "manager_midlayer_eval_latest.md")
    latest_csv_path = os.path.join(MANAGER_EVAL_LATEST_DIR, "manager_midlayer_eval_latest.csv")
    markdown = _render_midlayer_markdown_report(metrics)
    csv_row = _csv_row_midlayer(metrics)
    _write_text(md_path, markdown)
    _write_text(latest_md_path, markdown)
    _write_csv(csv_path, csv_row)
    _write_csv(latest_csv_path, csv_row)
    return metrics


def _render_midlayer_markdown_report(metrics: dict) -> str:
    rows = [
        ("run_timestamp", metrics["run_timestamp"]),
        ("summary_json_parse_ok", metrics["summary_json_parse_ok"]),
        ("summary_required_fields_ok", metrics["summary_required_fields_ok"]),
        ("summary_empty_rate", metrics["summary_empty_rate"]),
        ("fusion_empty_rate", metrics["fusion_empty_rate"]),
        ("fusion_repetition_rate", metrics["fusion_repetition_rate"]),
        ("fusion_code_leakage_flag", metrics["fusion_code_leakage_flag"]),
        ("fusion_rap_enabled", metrics["fusion_rap_enabled"]),
        ("fusion_arxiv_raw_chars", metrics["fusion_arxiv_raw_chars"]),
        ("fusion_web_raw_chars", metrics["fusion_web_raw_chars"]),
        ("fusion_pwc_raw_chars", metrics["fusion_pwc_raw_chars"]),
        ("fusion_kaggle_raw_chars", metrics["fusion_kaggle_raw_chars"]),
        ("faithfulness_warning_count", metrics["faithfulness_warning_count"]),
    ]
    metric_table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return f"""# Manager Midlayer Eval Report

## Run Context

- Data path: `{metrics["data_path"]}`
- User goal: {metrics["user_goal"]}
- Cleaning hints: {metrics["manager_cleaning_hints"] or "(none)"}

## Metric Record Table

| Metric | Value |
| --- | --- |
{metric_table}

## Summary Preview

```text
{metrics["summary_preview"] or "(empty summary)"}
```

## Fusion Preview

```text
{metrics["fusion_preview"] or "(empty fusion)"}
```
"""


def _csv_row_midlayer(metrics: dict) -> list[dict[str, str]]:
    return [
        {
            "run_timestamp": metrics["run_timestamp"],
            "summary_json_parse_ok": str(metrics["summary_json_parse_ok"]),
            "summary_required_fields_ok": str(metrics["summary_required_fields_ok"]),
            "summary_empty_rate": str(metrics["summary_empty_rate"]),
            "fusion_empty_rate": str(metrics["fusion_empty_rate"]),
            "fusion_repetition_rate": str(metrics["fusion_repetition_rate"]),
            "fusion_code_leakage_flag": str(metrics["fusion_code_leakage_flag"]),
            "fusion_rap_enabled": str(metrics["fusion_rap_enabled"]),
            "fusion_arxiv_raw_chars": str(metrics["fusion_arxiv_raw_chars"]),
            "fusion_web_raw_chars": str(metrics["fusion_web_raw_chars"]),
            "fusion_pwc_raw_chars": str(metrics["fusion_pwc_raw_chars"]),
            "fusion_kaggle_raw_chars": str(metrics["fusion_kaggle_raw_chars"]),
            "faithfulness_warning_count": str(metrics["faithfulness_warning_count"]),
        }
    ]
