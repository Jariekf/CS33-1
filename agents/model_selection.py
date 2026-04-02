"""
Model Selection Agent (模型选择与构建智能体)
职责：
  根据 Manager 的全局计划中的 model_selection_plan，
  生成多模型对比、超参数调优和交叉验证的代码并在沙盒执行。
  产出最优模型及其特征重要性 (Feature Importance)。
"""
from __future__ import annotations

import json
from core.logger import get_agent_logger

from core.state import GraphState
from core.llm import call_llm, extract_code
from tools.python_repl import execute_code
from config import MAX_INNER_RETRY

logger = get_agent_logger("model_selection")

# ======================================================================
# Prompt 模板
# ======================================================================

_SYSTEM_PROMPT_GENERATE = """\
You are an expert Data Scientist and Machine Learning Engineer. \
You write Python code according to the provided plan and constraints. \
Your code must be complete, self-contained, and directly runnable. \
You must implement model building, hyperparameter tuning, and cross-validation."""

_USER_PROMPT_GENERATE = """\
# CONTEXT #
Current phase: Model Selection

# PLAN #
{plan}

# CONSTRAINTS #
1. Load data from: `{data_path}` using `pd.read_csv`.
2. Assuming the target variable is known or inferable, if not mentioned, assume the last column is the target `y` and other numeric/categorical columns are features `X`.
3. Do model selection and cross-validation as instructed by the Plan.
4. Export the top feature importances to a dict, sorted by value descending, and save it to `./workspace/feature_importance.json`.
5. Use `print()` to output the Best Model name, Cross-Validation score, and Top Feature Importances.
6. Import all necessary libraries at the top of the code.

{feedback_section}

# RESPONSE: CODE #
Provide your complete solution in a single ```python ... ``` code block.
"""

_SYSTEM_PROMPT_FIX = """\
You are an expert Python debugger. \
Analyze the error and fix the machine learning code. \
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


def model_selection_node(state: GraphState) -> dict:
    """Model Selection 的 LangGraph 节点函数。"""

    logger.info("Model Selection node started.")
    # 【边界控制】只抽取属于自己的子计划
    plan = state["global_plan"].get("model_selection_plan", {})
    data_path = state["current_data_path"]
    feedback = state.get("current_error", "")

    # ------------------------------------------------------------------
    # Step 1: 生成模型训练 + 调参 + 评估代码
    # ------------------------------------------------------------------
    code = _generate_model_code(plan, data_path, feedback)
    logger.info("Model selection code generation completed.")

    # ------------------------------------------------------------------
    # Step 2: 内部语法自救循环
    # ------------------------------------------------------------------
    stdout, stderr = "", ""
    for attempt in range(MAX_INNER_RETRY):
        stdout, stderr = execute_code(code)
        if not stderr:
            logger.info("Model selection code executed successfully.")
            break
        logger.info("内部自救第 %d 次: %s", attempt + 1, stderr[:200])
        code = _fix_code(code, stderr)

    # ------------------------------------------------------------------
    # Step 3: 返回状态
    # ------------------------------------------------------------------
    if stderr:
        logger.error(f"Model selection failed after {MAX_INNER_RETRY} attempts: {stderr}")
        return {
            "current_phase": "model_selection",
            "current_code": code,
            "current_execution_log": stdout,
            "current_error": f"[InnerRetryExhausted] {stderr}",
        }

    return {
        "current_phase": "model_selection",
        "current_code": code,
        "current_execution_log": stdout,
        "current_error": "",
    }


# ======================================================================
# 私有辅助函数
# ======================================================================

def _generate_model_code(plan: dict, data_path: str, feedback: str) -> str:
    """
    调用 LLM 生成模型训练与调参代码。
    若调用失败，则返回桩实现代码。
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

    logger.warning("LLM 调用失败或未返回代码，使用桩代码模板。")
    return _fallback_model_code(plan, data_path)


def _fix_code(code: str, error: str) -> str:
    """调用 LLM 修复代码。若失败则原样返回。"""
    user_prompt = _USER_PROMPT_FIX.format(code=code, error=error)
    reply = call_llm(_SYSTEM_PROMPT_FIX, user_prompt)
    if reply:
        fixed_code = extract_code(reply)
        if fixed_code:
            return fixed_code
            
    logger.warning("LLM 修复调用失败，返回原代码。")
    return code


def _fallback_model_code(plan: dict, data_path: str) -> str:
    """失败时的桩代码模板。"""
    metric = plan.get("evaluation_metric", "f1")
    return f"""\
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import json
import os

df = pd.read_csv("{data_path}")
# 假设最后一列是目标变量
X = df.iloc[:, :-1].select_dtypes(include='number').fillna(0)
y = df.iloc[:, -1]

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='{metric}')
model.fit(X, y)

fi = dict(zip(X.columns, model.feature_importances_))
fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

print(f"Best Model: RandomForest")
print(f"CV {'{metric}'} Mean: {{scores.mean():.4f}} (+/- {{scores.std():.4f}})")
print(f"Top 5 Feature Importance: {{dict(list(fi_sorted.items())[:5])}}")

# 保存特征重要性
os.makedirs("./workspace", exist_ok=True)
with open("./workspace/feature_importance.json", "w") as f:
    json.dump(fi_sorted, f, indent=2)
"""

