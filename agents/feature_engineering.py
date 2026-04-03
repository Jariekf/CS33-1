"""
Feature Engineering Agent (特征工程智能体)
职责：
  根据 Manager 的全局计划中的 feature_engineering_plan，
  生成 Python 特征工程代码并在沙盒中执行（含内部自救机制）。

设计参照 AutoKaggle Developer Agent 的核心模式：
  - 结构化 Prompt (CONTEXT + PLAN + CONSTRAINTS + TASK)
  - 上一阶段摘要注入上下文 (previous_summary)
  - 代码提取 (regex ```python...```)

The column names in the output CSV should be readable by the downstream "Feature Importance / Business Report": retain the business column names, and use meaningful names for new features;

It is forbidden to change them to anonymous column names such as x0, x1, feature_0, etc. without justification (enforced by the agent's Prompt constraints).
"""
from __future__ import annotations

import json
from core.logger import get_agent_logger

from core.state import GraphState
from core.llm import call_llm, extract_code
from tools.python_repl import execute_code
from config import MAX_INNER_RETRY

logger = get_agent_logger("feature_engineering")


def feature_engineering_node(state: GraphState) -> dict:
    """Feature Engineering 的 LangGraph 节点函数。"""

    logger.info("Feature Engineering node started.")
    # 【边界控制】只抽取属于自己的子计划
    plan = state["global_plan"].get("feature_engineering_plan", {})
    data_path = state["current_data_path"]
    feedback = state.get("current_error", "")
    # 只读取最近的 Memory 摘要 (上一阶段清洗成果)
    previous_summary = state["memory"][-1] if state.get("memory") else ""

    # ------------------------------------------------------------------
    # Step 1: 生成特征工程代码
    # ------------------------------------------------------------------
    code = _generate_fe_code(plan, data_path, previous_summary, feedback)
    logger.info("Feature engineering code generation completed.")

    # ------------------------------------------------------------------
    # Step 2: 内部语法自救循环
    # ------------------------------------------------------------------
    stdout, stderr = "", ""
    for attempt in range(MAX_INNER_RETRY):
        stdout, stderr = execute_code(code)
        if not stderr:
            logger.info("Feature engineering code executed successfully.")
            break
        logger.info("内部自救第 %d 次: %s", attempt + 1, stderr[:200])
        code = _fix_code(code, stderr)

    # ------------------------------------------------------------------
    # Step 3: 返回状态更新
    # ------------------------------------------------------------------
    if stderr:
        logger.error(f"Feature engineering failed after {MAX_INNER_RETRY} attempts: {stderr}")
        return {
            "current_phase": "feature_engineering",
            "current_code": code,
            "current_execution_log": stdout,
            "current_error": f"[InnerRetryExhausted] {stderr}",
        }

    return {
        "current_phase": "feature_engineering",
        "current_code": code,
        "current_execution_log": stdout,
        "current_error": "",
        "current_data_path": "./workspace/fe_output.csv",
    }


# ======================================================================
# 私有辅助函数
# ======================================================================

# ---------- Prompt 模板 (参照 AutoKaggle PROMPT_DEVELOPER 结构) ----------

_SYSTEM_PROMPT_GENERATE = """\
You are an expert Feature Engineering developer. \
You write Python code according to the provided plan and constraints. \
Your code must be complete, self-contained, and directly runnable. \
Column names in the saved CSV must stay human-interpretable for downstream model reports: \
keep original names where columns are retained; name new features descriptively (snake_case). \
Never bulk-rename features to generic x0, x1, x2, or feature_0 unless the prompt explicitly requires it."""

_USER_PROMPT_GENERATE = """\
# CONTEXT #
Current phase: Feature Engineering

# PREVIOUS PHASE SUMMARY #
{previous_summary}

# PLAN #
{plan}

# CONSTRAINTS #
1. Load data from: `{data_path}` using `pd.read_csv`.
2. Save feature-engineered data to: `./workspace/fe_output.csv` using `df.to_csv(..., index=False)`.
3. Always work on a copy of the DataFrame, not the original.
4. Apply consistent modifications (e.g., scaling, encoding) to ensure reproducibility.
5. Use `print()` to output a summary (feature count before/after, new features created, etc.).
6. Import all necessary libraries at the top of the code.
7. Common feature engineering operations to consider:
   - Derived features (interaction, polynomial, aggregation)
   - Category encoding (Label Encoding, One-Hot, Target Encoding)
   - Feature scaling (StandardScaler, MinMaxScaler)
   - Handling of datetime features (extraction of year/month/day/hour etc.)
8. Column names for `./workspace/fe_output.csv` must remain meaningful for business-facing reports:
   - Keep original input column names when those columns still exist (do not rename to x0/x1/...).
   - New or transformed columns: use descriptive snake_case names, e.g. `units_sold_log`, `gross_sales_sqrt`, `country_Germany` (one-hot suffix), `order_date_year`.
   - Do NOT replace all column names with sequential generics (`x0`, `x1`, `feature_0`, `col_0`, etc.).
9. If you drop columns, that is fine; if you one-hot or encode, use clear prefixes/suffixes so names trace back to the source column.

{feedback_section}

# RESPONSE: CODE #
Provide your complete solution in a single ```python ... ``` code block.
"""

_SYSTEM_PROMPT_FIX = """\
You are an expert Python debugger. \
Analyze the error and fix the code. \
Output the complete corrected code. \
Preserve the same column-naming rules as feature engineering: descriptive names, no generic x0/x1 renames."""

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


def _generate_fe_code(
    plan: dict, data_path: str, previous_summary: str, feedback: str
) -> str:
    """
    调用 LLM 生成特征工程代码。
    参照 AutoKaggle Developer._execute 的 Prompt 构建逻辑，注入上一阶段摘要。
    若 LLM 调用失败则降级返回桩代码模板。
    """
    feedback_section = ""
    if feedback:
        feedback_section = f"# SUPERVISOR FEEDBACK #\nPrevious attempt was rejected. Fix based on:\n{feedback}"

    user_prompt = _USER_PROMPT_GENERATE.format(
        plan=json.dumps(plan, ensure_ascii=False, indent=2),
        data_path=data_path,
        previous_summary=previous_summary if previous_summary else "无上一阶段摘要",
        feedback_section=feedback_section,
    )

    reply = call_llm(_SYSTEM_PROMPT_GENERATE, user_prompt, agent_name="feature_engineering")

    if reply:
        code = extract_code(reply)
        if code:
            return code

    # 降级：返回桩代码模板
    logger.warning("LLM 调用失败或未返回代码，使用桩代码模板。")
    return _fallback_fe_code(plan, data_path, previous_summary)


def _fix_code(code: str, error: str) -> str:
    """
    调用 LLM 根据报错修复代码。
    参照 AutoKaggle PROMPT_DEVELOPER_DEBUG_FIX 的简化版。
    若 LLM 不可用则返回原代码。
    """
    user_prompt = _USER_PROMPT_FIX.format(code=code, error=error)
    reply = call_llm(_SYSTEM_PROMPT_FIX, user_prompt, agent_name="feature_engineering")

    if reply:
        fixed_code = extract_code(reply)
        if fixed_code:
            return fixed_code

    logger.warning("LLM 修复调用失败，返回原代码。")
    return code


def _fallback_fe_code(plan: dict, data_path: str, previous_summary: str) -> str:
    """LLM 不可用时的降级桩代码模板。"""
    return f"""\
import pandas as pd

df = pd.read_csv("{data_path}")
# 特征工程计划: {plan.get('objectives', [])}
# 上一阶段摘要: {previous_summary}

output_path = "./workspace/fe_output.csv"
df.to_csv(output_path, index=False)
print(f"特征工程完成，特征数: {{df.shape[1]}}，已保存至 {{output_path}}")
"""
