"""
Manager Agent (统筹与规划智能体)
职责：
  1. 解析用户目标，生成 arxiv 检索查询词
  2. 通过 RAP (Retrieval-Augmented Planning) 检索文献获取先验知识
  3. 结合用户目标与学术背景，生成结构化的全局执行计划 (Global Execution Plan)
  4. 初始化全局 State 并启动流水线
"""
from __future__ import annotations

import os
import json
import re
import pandas as pd

from core.state import GraphState
from core.llm import call_llm
from tools.arxiv_tool import search_arxiv
from config import WORKSPACE_DIR
from core.logger import get_agent_logger

logger = get_agent_logger("manager")


def manager_node(state: GraphState) -> dict:
    """
    Manager Agent 的 LangGraph 节点函数。
    输入: 包含 data_path 和 user_goal 的初始 State。
    输出: 更新 global_plan, current_phase, current_data_path 等字段。
    """

    user_goal = state["user_goal"]
    data_path = state["data_path"]

    logger.info(f"Starting manager_node with user_goal: '{user_goal}' and data_path: '{data_path}'")

    # ------------------------------------------------------------------
    # Step 1: 提取数据元信息 (列名、行数等)，让 LLM 感知数据轮廓
    # ------------------------------------------------------------------
    df_preview = _get_data_preview(data_path)
    logger.info("Data preview generated.")

    # ------------------------------------------------------------------
    # Step 2: 利用 LLM 将用户目标转化为学术检索查询词
    # ------------------------------------------------------------------
    search_queries = _generate_search_queries(user_goal, df_preview)
    logger.info(f"Generated {len(search_queries)} search queries for RAP: {search_queries}")

    # ------------------------------------------------------------------
    # Step 3: RAP — 调用 arxiv 工具检索相关文献
    # ------------------------------------------------------------------
    retrieved_docs = []
    for query in search_queries:
        docs = search_arxiv(query)
        retrieved_docs.extend(docs)
    logger.info(f"RAP search complete. Gathered {len(retrieved_docs)} documents.")

    # ------------------------------------------------------------------
    # Step 4: 结合用户目标 + 文献先验 + 数据概览，生成结构化执行计划
    # ------------------------------------------------------------------
    global_plan = _generate_global_plan(user_goal, df_preview, retrieved_docs)
    logger.info("Global structured plan generated successfully.")

    # ------------------------------------------------------------------
    # Step 5: 初始化运行时状态并返回
    # ------------------------------------------------------------------
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    return {
        "global_plan": global_plan,
        "current_phase": "data_cleaning",
        "current_data_path": data_path,
        "current_code": "",
        "current_execution_log": "",
        "current_error": "",
        "retry_count": 0,
        "current_review": None,
        "memory": [],
        "final_business_report": "",
    }


# ======================================================================
# 私有辅助函数 (待接入真实 LLM 时替换桩逻辑)
# ======================================================================

def _get_data_preview(data_path: str) -> str:
    """读取数据集的基础元信息，返回文本摘要。"""
    try:
        df = pd.read_csv(data_path, nrows=5)
        info = (
            f"Columns ({len(df.columns)}): {list(df.columns)}\n"
            f"Shape: {df.shape}\n"
            f"Dtypes:\n{df.dtypes.to_string()}\n"
            f"Head:\n{df.head(3).to_string()}"
        )
        return info
    except Exception as e:
        return f"[Warning] 无法预览数据: {e}"


def _generate_search_queries(user_goal: str, df_preview: str) -> list[str]:
    """调用 LLM 将用户目标与数据概览转化为 arxiv 检索查询词。"""
    sys_prompt = "You are a senior Data Science Manager. Generate precise academic search queries."
    user_prompt = f"""
User Goal: {user_goal}
Data Preview:
{df_preview}

Based on the goal and data structure, generate 1 to 2 English search queries for an academic database (like Arxiv) to discover machine learning methodologies or feature engineering techniques relevant to this dataset and objective.
Return ONLY a JSON array of strings, e.g., ["query 1", "query 2"]
"""
    reply = call_llm(sys_prompt, user_prompt, agent_name="manager")
    if reply:
        try:
            match = re.search(r'\[.*\]', reply, re.DOTALL)
            if match:
                queries = json.loads(match.group(0))
                if isinstance(queries, list) and queries:
                    return queries
        except Exception:
            pass
            
    # Fallback
    return [f"{user_goal} machine learning methodology"]


def _generate_global_plan(
    user_goal: str, df_preview: str, retrieved_docs: list[dict]
) -> dict:
    """调用 LLM 生成结构化执行计划 (JSON)。

    `data_cleaning_plan.cleaning_contract` — Minimal general mode (keys are optional; default or `{}` indicates no validation):

    - `max_row_drop_ratio`: A floating-point number from 0 to 1, representing the maximum percentage of rows allowed to be deleted/dropped relative to the number of rows in the input CSV.

    Column retention is not enforced: cleaning may drop or rename columns per objectives.

    Differences from `rules`:

    - `rules`: Plans which cleaning steps (processes) to perform, for subsequent rule-based execution or suggestions.

    - `cleaning_contract`: Task-level safety boundaries (results), hard-validated by the Data Cleaning node after data is written to disk.
    """
    docs_context = "\n".join(
        [f"- {d.get('title', 'N/A')}: {d.get('summary', '')[:200]}" for d in retrieved_docs]
    ) if retrieved_docs else "无检索结果"

    sys_prompt = "You are a Data Science Engineering Director generating an execution plan."
    user_prompt = f"""
User Goal: {user_goal}
Data Preview:
{df_preview}

Literature Context (RAP):
{docs_context}

Please formulate a high-level Data Science global execution plan tailored to the objective. 
You must output ONLY a valid JSON object with EXACTLY the following structure (no markdown fences around it if possible, or strictly enclosed in ```json...```):
{{
    "data_cleaning_plan": {{
        "objectives": ["obj1", "obj2"],
        "constraints": "string constraint",
        "rules": [],
        "cleaning_contract": {{}}
    }},
    "feature_engineering_plan": {{
        "objectives": ["obj1"],
        "constraints": "string constraint"
    }},
    "model_selection_plan": {{
        "objectives": ["model1", "model2"],
        "evaluation_metric": "e.g., f1, accuracy"
    }}
}}

Notes for data_cleaning_plan:
- "rules": procedural steps to perform (what to run). Optional list of objects like {{"rule": "drop_duplicates", "params": {{}}}}. Use [] if none. Not validated as bounds.
- "cleaning_contract": post-execution guardrails. Prefer {{}} or only "max_row_drop_ratio". Do not list "protected_columns" or imply all columns must survive cleaning — dropping columns is allowed.
    * "max_row_drop_ratio" (number 0..1, optional): only enforced key; max allowed fraction of input rows removed, i.e. (rows_in - rows_out) / rows_in.
"""
    reply = call_llm(sys_prompt, user_prompt, agent_name="manager")
    if reply:
        try:
            match = re.search(r'\{.*\}', reply, re.DOTALL)
            if match:
                plan = json.loads(match.group(0))
                if isinstance(plan, dict) and "data_cleaning_plan" in plan:
                    _normalize_data_cleaning_plan(plan)
                    plan["_rap_context"] = docs_context
                    return plan
        except Exception:
            pass

    # Fallback
    fallback_plan = {
        "data_cleaning_plan": {
            "objectives": ["检查并处理缺失值", "移除重复行", "修正数据类型"],
            "constraints": "删除行数不超过总数据 5%",
            "rules": [],
            "cleaning_contract": {},
        },
        "feature_engineering_plan": {
            "objectives": ["基于 RAP 文献构造关键衍生特征", "类别编码", "特征缩放"],
            "constraints": "特征维度控制在合理范围",
        },
        "model_selection_plan": {
            "objectives": ["测试 XGBoost / LightGBM / RandomForest", "交叉验证"],
            "evaluation_metric": "f1",
        },
        "_rap_context": docs_context,
    }
    _normalize_data_cleaning_plan(fallback_plan)
    return fallback_plan


def _normalize_data_cleaning_plan(plan: dict) -> None:
    """Ensure data_cleaning_plan has rules + cleaning_contract with safe defaults."""
    dcp = plan.get("data_cleaning_plan")
    if not isinstance(dcp, dict):
        plan["data_cleaning_plan"] = {
            "objectives": [],
            "constraints": "",
            "rules": [],
            "cleaning_contract": {},
        }
        return
    if not isinstance(dcp.get("rules"), list):
        dcp["rules"] = []
    if not isinstance(dcp.get("cleaning_contract"), dict):
        dcp["cleaning_contract"] = {}
    # Drop legacy / LLM-hallucinated key: we do not enforce "must keep these columns".
    dcp["cleaning_contract"].pop("protected_columns", None)
