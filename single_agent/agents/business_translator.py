"""Business Translator agent.

This agent converts structured workflow outputs into a concise business-facing
report. It uses an LLM when configured and falls back to deterministic markdown.
"""

from __future__ import annotations

import json
from typing import Any

from core.llm import call_llm
from core.state_schema import (
    BUSINESS_REPORT_PATH,
    EDA_REPORT_PATH,
    FEATURE_IMPORTANCE_PATH,
    MODEL_REPORT_PATH,
    WorkflowState,
)
from tools.validation_tools import read_json


SYSTEM_PROMPT = """You are a senior business analytics consultant.
Translate data science workflow results into concise, evidence-based business recommendations.
Do not invent metrics or claims. If evidence is weak, state the limitation clearly."""


def business_translator_node(state: WorkflowState) -> dict:
    """Generate the final business insight report."""

    context = _build_business_context(state)
    try:
        report = call_llm(SYSTEM_PROMPT, _build_user_prompt(context))
        source = "llm"
    except Exception as exc:
        report = _fallback_report(context, str(exc))
        source = "fallback"
    with open(BUSINESS_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    stage_reports = dict(state.get("stage_reports", {}))
    stage_reports["business_translator"] = {
        "stage": "business_translator",
        "status": "completed",
        "source": source,
        "report_path": BUSINESS_REPORT_PATH,
    }
    return {
        "stage_reports": stage_reports,
        "business_report_path": BUSINESS_REPORT_PATH,
        "current_stage": "business_translator",
        "current_error": "",
    }


def _build_business_context(state: WorkflowState) -> dict[str, Any]:
    """Collect structured evidence for report generation."""

    model_report = _safe_read_json(MODEL_REPORT_PATH)
    eda_report = _safe_read_json(EDA_REPORT_PATH)
    feature_importance = _safe_read_json(FEATURE_IMPORTANCE_PATH)
    top_features = dict(list(feature_importance.items())[:10])
    return {
        "user_goal": state.get("user_goal", ""),
        "data_description": state.get("data_description", ""),
        "plan_version": state.get("plan_version", 1),
        "task_summary": state.get("global_plan", {}).get("task_summary", {}),
        "plan_patches": state.get("plan_patches", []),
        "stage_reports": state.get("stage_reports", {}),
        "eda_report": eda_report,
        "model_report": model_report,
        "top_features": top_features,
    }


def _build_user_prompt(context: dict[str, Any]) -> str:
    """Build the LLM prompt from structured workflow evidence."""

    return f"""# USER GOAL
{context["user_goal"]}

# OPTIONAL DATASET DESCRIPTION
{context.get("data_description") or "Not provided."}

# STRUCTURED WORKFLOW EVIDENCE
```json
{json.dumps(context, indent=2, ensure_ascii=False)}
```

# REPORT REQUIREMENTS
Write a concise business report in markdown with exactly these sections:

1. Executive Answer
2. Key Evidence
3. Model Performance
4. Business Recommendations
5. Limitations and Next Steps

Rules:
- Use only the provided evidence.
- Include the most relevant EDA findings when they are available.
- Mention the model task type, best model, metric, and test performance when available.
- Explain top features in business language.
- Keep recommendations actionable and tied to evidence.
- If the workflow used fallback or evidence is incomplete, clearly state that limitation.
"""


def _fallback_report(context: dict[str, Any], reason: str) -> str:
    """Create a deterministic report when the LLM is unavailable."""

    task = context.get("task_summary", {})
    model = context.get("model_report", {})
    eda = context.get("eda_report", {})
    target_analysis = eda.get("target_analysis", {})
    relationships = eda.get("feature_target_relationships", {})
    top_features = context.get("top_features", {})
    lines = [
        "# Business Insight Report",
        "",
        "## Executive Answer",
        f"The workflow completed the analysis for: {context.get('user_goal', '')}.",
        "",
        "## Key Evidence",
        f"- Task type: {task.get('task_type', 'unknown')}",
        f"- Target column: {task.get('target_column', 'not specified')}",
        f"- Dataset description provided: {bool(context.get('data_description'))}",
        f"- Plan version: {context.get('plan_version', 1)}",
        f"- EDA target summary: {target_analysis.get('summary', 'not available')}",
        "",
        "## Model Performance",
        f"- Best model: {model.get('best_model', 'not available')}",
        f"- Primary metric: {model.get('metric', task.get('primary_metric', 'not available'))}",
        f"- Test metrics: {model.get('test_metrics', {})}",
        "",
        "## Business Recommendations",
        "- Review the top drivers and validate whether they are actionable in the business process.",
        "- Use the model output as decision support, not as an automatic decision rule.",
        "- Re-run the workflow with richer metadata if the target definition or dataset meaning is unclear.",
        "",
        "## Top Features",
    ]
    if top_features:
        lines.extend([f"- {name}: {value:.4f}" for name, value in top_features.items()])
    else:
        lines.append("- No feature importance was available.")
    lines.extend(["", "## EDA Findings"])
    top_numeric = relationships.get("top_numeric_relationships", [])[:3]
    top_categorical = relationships.get("top_categorical_relationships", [])[:3]
    if top_numeric or top_categorical:
        for item in top_numeric + top_categorical:
            lines.append(f"- {item.get('feature')}: {item.get('score_name')}={item.get('score')}")
    else:
        lines.append("- No EDA feature-target relationships were available.")
    lines.extend([
        "",
        "## Limitations and Next Steps",
        f"- LLM report generation was unavailable, so this fallback report was used. Reason: {reason}",
        "- Confirm metric suitability and evaluate the model on more representative data before deployment.",
    ])
    return "\n".join(lines)


def _safe_read_json(path: str) -> dict[str, Any]:
    """Read a JSON object and return an empty dict if unavailable."""

    try:
        return read_json(path)
    except Exception:
        return {}
