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
    """记录针对训练集或测试集的数据清洗计划。需调用两次：target="train" 和 target="test"。

    target: "train" 表示训练集清洗计划，"test" 表示测试集清洗计划。
    notes: 可选的补充说明列表，每条说明为一个简短字符串。必须是字符串列表，不可以是裸文本。
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
