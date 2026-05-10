import json
import os
from typing import Any

from dotenv import load_dotenv


load_dotenv(override=True)


def _format_value(value: Any, max_chars: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)

    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... <truncated {len(text) - max_chars} chars>"


def trace_tool_call(
    tool_name: str,
    tool_inputs: dict[str, Any],
    tool_output: Any,
    max_chars: int = 4000,
) -> None:
    """统一打印 tool 调用名称、入参和返回值摘要。"""
    if os.getenv("TRACE_TOOLS", "false").lower() != "true":
        return

    print(f"\n[tool call] {tool_name}")
    print("[tool input]")
    print(_format_value(tool_inputs, max_chars=max_chars))
    print("[tool output]")
    print(_format_value(tool_output, max_chars=max_chars))
    print("[tool end]\n")
