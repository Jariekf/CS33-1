"""
Data Cleaning Agent (数据清洗智能体)
职责：
  根据 Manager 的全局计划中的 data_cleaning_plan，
  生成 Python 清洗代码并在沙盒中执行（含内部自救机制）。

设计参照 AutoKaggle Developer Agent 的核心模式：
  - 结构化 Prompt (CONTEXT + PLAN + CONSTRAINTS + TASK)
  - 代码提取 (regex ```python...```)
  - 内部 Debug 自救循环
"""
from __future__ import annotations

import json
import logging

from core.state import GraphState
from core.llm import call_llm, extract_code
from tools.python_repl import execute_code
from config import MAX_INNER_RETRY

logger = logging.getLogger(__name__)


def data_cleaning_node(state: GraphState) -> dict:
    """Data Cleaning 的 LangGraph 节点函数。"""

    plan = state["global_plan"].get("data_cleaning_plan", {})
    data_path = state["current_data_path"]
    feedback = state.get("current_error", "")

    # ------------------------------------------------------------------
    # Step 1: 调用 LLM 生成清洗代码
    # ------------------------------------------------------------------
    code = _generate_cleaning_code(plan, data_path, feedback)

    # ------------------------------------------------------------------
    # Step 2: 内部语法自救循环 (Inner Self-Correction)
    # ------------------------------------------------------------------
    stdout, stderr = "", ""
    for attempt in range(MAX_INNER_RETRY):
        stdout, stderr = execute_code(code)
        if not stderr:
            break
        # 自救：将报错信息喂回 LLM 重新生成代码
        logger.info("内部自救第 %d 次: %s", attempt + 1, stderr[:200])
        code = _fix_code(code, stderr)

    # ------------------------------------------------------------------
    # Step 3: 返回状态更新
    # ------------------------------------------------------------------
    if stderr:
        # 内部自救失败，仍然将最后一次的错误传给 Supervisor
        return {
            "current_phase": "data_cleaning",
            "current_code": code,
            "current_execution_log": stdout,
            "current_error": f"[InnerRetryExhausted] {stderr}",
        }

    return {
        "current_phase": "data_cleaning",
        "current_code": code,
        "current_execution_log": stdout,
        "current_error": "",
        "current_data_path": _get_output_path(),
    }


# ======================================================================
# 私有辅助函数
# ======================================================================

# ---------- Prompt 模板 (参照 AutoKaggle PROMPT_DEVELOPER 结构) ----------

_SYSTEM_PROMPT_GENERATE = """\
You are an expert Data Cleaning developer. \
You write Python code according to the provided plan and constraints. \
Your code must be complete, self-contained, and directly runnable."""

_USER_PROMPT_GENERATE = """\
# CONTEXT #
Current phase: Data Cleaning

# PLAN #
{plan}

# CONSTRAINTS #
1. Load data from: `{data_path}` using `pd.read_csv`.
2. Save cleaned data to: `./workspace/cleaned_data.csv` using `df.to_csv(..., index=False)`.
3. Always work on a copy of the DataFrame, not the original.
4. Use `print()` to output a summary of what was done (rows before/after, columns modified, etc.).
5. Import all necessary libraries at the top of the code.

{feedback_section}

# RESPONSE: CODE #
Provide your complete solution in a single ```python ... ``` code block.
"""

_SYSTEM_PROMPT_FIX = """\
You are an expert Python debugger. \
Analyze the error and fix the code. \
Output the complete corrected code."""

_USER_PROMPT_FIX = """\
# CODE CONTAINS ERROR #
```python
{code}
```

# ERROR MESSAGES #
{error}

# RESPONSE: CORRECTED CODE #
Provide the complete corrected code in a single ```python ... ``` code block.
"""


def _generate_cleaning_code(plan: dict, data_path: str, feedback: str) -> str:
    """
    调用 LLM 生成清洗代码。
    参照 AutoKaggle Developer._execute 的多轮 Prompt 构建逻辑，
    此处简化为单轮结构化 Prompt。
    若 LLM 调用失败则降级返回桩代码模板。
    """
    feedback_section = ""
    if feedback:
        feedback_section = f"# SUPERVISOR FEEDBACK #\nPrevious attempt was rejected. Fix based on:\n{feedback}"

    user_prompt = _USER_PROMPT_GENERATE.format(
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        data_path=data_path,
        feedback_section=feedback_section,
    )

    reply = call_llm(_SYSTEM_PROMPT_GENERATE, user_prompt)

    if reply:
        code = extract_code(reply)
        if code:
            return code

    # 降级：返回桩代码模板
    logger.warning("LLM 调用失败或未返回代码，使用桩代码模板。")
    return _fallback_cleaning_code(plan, data_path)


def _fix_code(code: str, error: str) -> str:
    """
    调用 LLM 根据报错修复代码。
    参照 AutoKaggle PROMPT_DEVELOPER_DEBUG_FIX 的简化版。
    若 LLM 不可用则返回原代码。
    """
    user_prompt = _USER_PROMPT_FIX.format(code=code, error=error)
    reply = call_llm(_SYSTEM_PROMPT_FIX, user_prompt)

    if reply:
        fixed_code = extract_code(reply)
        if fixed_code:
            return fixed_code

    # 降级：返回原代码
    logger.warning("LLM 修复调用失败，返回原代码。")
    return code


def _get_output_path() -> str:
    """返回清洗后数据的标准输出路径。"""
    return "./workspace/cleaned_data.csv"


def _fallback_cleaning_code(plan: dict, data_path: str) -> str:
    """LLM 不可用时的降级桩代码模板。"""
    return f"""\
import pandas as pd

df = pd.read_csv("{data_path}")
# 基于计划执行清洗: {plan.get('objectives', [])}
df = df.drop_duplicates()
df = df.dropna(thresh=len(df.columns) * 0.5)

output_path = "./workspace/cleaned_data.csv"
df.to_csv(output_path, index=False)
print(f"清洗完成，剩余 {{len(df)}} 行，已保存至 {{output_path}}")
"""
