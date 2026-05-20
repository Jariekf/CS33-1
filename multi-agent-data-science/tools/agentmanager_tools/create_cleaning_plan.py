import json
from typing import Any, Literal

from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def create_cleaning_plan(
    target: Literal["train", "test"],
    summary: str = "",
    column_name_actions: list[dict[str, Any]] | None = None,
    type_conversions: list[dict[str, Any]] | None = None,
    missing_value_actions: list[dict[str, Any]] | None = None,
    outlier_actions: list[dict[str, Any]] | None = None,
    drop_columns: list[str] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Record the data cleaning plan for the training or test set. Must be called twice: target="train" and target="test".

    target: "train" denotes the training-set cleaning plan; "test" denotes the test-set cleaning plan.
    notes: optional list of supplementary notes; each note is a short string. Must be a list of strings, not bare text.
    """
    normalized_notes: list[str]
    if notes is None:
        normalized_notes = []
    elif isinstance(notes, list):
        normalized_notes = [str(item) for item in notes]
    else:
        normalized_notes = [str(notes)]

    result = {
        "target": target,
        "summary": str(summary or ""),
        "column_name_actions": column_name_actions or [],
        "type_conversions": type_conversions or [],
        "missing_value_actions": missing_value_actions or [],
        "outlier_actions": outlier_actions or [],
        "drop_columns": drop_columns or [],
        "notes": normalized_notes,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    trace_tool_call("create_cleaning_plan", result, output)
    return output
