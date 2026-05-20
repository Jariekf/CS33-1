import json
from typing import Literal

from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def infer_task(
    task_type: Literal["regression", "classification", "unknown"],
    target_column: str,
    confidence: float,
    reason: str,
) -> str:
    """Record the modeling task type inferred by the Agent based on the user question, data description, and data profile."""
    result = {
        "task_type": task_type,
        "target_column": target_column,
        "confidence": confidence,
        "reason": reason,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    trace_tool_call("infer_task", result, output)
    return output
