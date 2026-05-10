import json
from typing import Any

from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def create_cleaning_plan(
    summary: str,
    column_name_actions: list[dict[str, Any]],
    type_conversions: list[dict[str, Any]],
    missing_value_actions: list[dict[str, Any]],
    outlier_actions: list[dict[str, Any]],
    drop_columns: list[str],
    notes: list[str] | None = None,
) -> str:
    """记录 Agent 根据数据画像和任务类型制定的数据清洗计划。

    notes: 可选的补充说明列表，每条说明为一个简短字符串（例如 ["保留冗余列作为特征工程选项", "Year列异常值为正常数据"]）。
           必须是字符串列表，不可以是裸文本。
    """
    result = {
        "summary": summary,
        "column_name_actions": column_name_actions,
        "type_conversions": type_conversions,
        "missing_value_actions": missing_value_actions,
        "outlier_actions": outlier_actions,
        "drop_columns": drop_columns,
        "notes": notes or [],
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    trace_tool_call("create_cleaning_plan", result, output)
    return output
