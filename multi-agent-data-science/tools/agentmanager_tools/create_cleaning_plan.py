import json
from typing import Any, Literal

from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def create_cleaning_plan(
    target: Literal["train", "test"],
    summary: str,
    column_name_actions: list[dict[str, Any]],
    type_conversions: list[dict[str, Any]],
    missing_value_actions: list[dict[str, Any]],
    outlier_actions: list[dict[str, Any]],
    drop_columns: list[str],
    notes: list[str] | None = None,
) -> str:
    """记录针对训练集或测试集的数据清洗计划。需调用两次：target="train" 和 target="test"。

    target: "train" 表示训练集清洗计划，"test" 表示测试集清洗计划。
    notes: 可选的补充说明列表，每条说明为一个简短字符串。必须是字符串列表，不可以是裸文本。
    """
    result = {
        "target": target,
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
