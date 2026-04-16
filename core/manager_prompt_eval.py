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


def write_manager_prompt_eval_report(
    initial_state: dict,
    final_state: dict,
    run_started_at: datetime,
) -> None:
    """Write timestamped and latest manager prompt evaluation reports."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    metrics = _collect_metrics(
        initial_state=initial_state,
        final_state=final_state,
        run_started_at=run_started_at,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"manager_prompt_eval_{timestamp}"

    md_path = os.path.join(WORKSPACE_DIR, f"{base_name}.md")
    csv_path = os.path.join(WORKSPACE_DIR, f"{base_name}.csv")
    latest_md_path = os.path.join(WORKSPACE_DIR, "manager_prompt_eval_latest.md")
    latest_csv_path = os.path.join(WORKSPACE_DIR, "manager_prompt_eval_latest.csv")

    markdown = _render_markdown_report(metrics)
    csv_row = _csv_row(metrics)

    _write_text(md_path, markdown)
    _write_text(latest_md_path, markdown)
    _write_csv(csv_path, csv_row)
    _write_csv(latest_csv_path, csv_row)


def _collect_metrics(initial_state: dict, final_state: dict, run_started_at: datetime) -> dict:
    global_plan = final_state.get("global_plan", {}) or {}
    eval_meta = global_plan.get("_manager_plan_eval", {}) or {}

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
| `outer_retry_total` | Total number of supervisor-triggered outer retries. Lower is better. |
| `pipeline_success` | Whether the full workflow produced the expected intermediate artifacts and stage reports successfully. |
| `artifact status` | A supporting reference only. It shows which output files were produced during this run. |

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
