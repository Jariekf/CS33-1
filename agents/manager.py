"""
Manager Agent (统筹与规划智能体)
职责：
  1. 解析用户目标（与可选 cleaning hints），生成 user_requirement_summary（LLM）
  2. 通过 RAP (Retrieval-Augmented Planning) 多源轻量检索并融合为 rap_context
  3. 结合摘要、RAP 与数据概览，生成结构化的全局执行计划 global_plan
  4. 初始化全局 State 并启动流水线

注（manager_node 内执行顺序）：
  Step 1a — 数据轮廓：读 CSV 元信息（列、dtype、样例行），无 LLM。
  Step 1b — 需求理解：LLM 产出 user_requirement_summary + 检索槽位（manager_retrieval.summarize_and_extract_slots）。
  Step 2 — RAP：多源轻量检索并融合（manager_retrieval.retrieve_and_fuse_rap_context）。
  Step 3 — 规划：LLM 产出 global_plan；另附只读元数据 _rap_context、_user_requirement_summary。
  Step 4 — 初始化工作区与运行时状态并返回。
"""
from __future__ import annotations

import os
import json
import re
import pandas as pd

from core.state import GraphState
from core.llm import call_llm
from config import WORKSPACE_DIR
from core.logger import get_agent_logger

from agents.manager_retrieval import (
    generate_arxiv_search_queries,
    retrieve_and_fuse_rap_context,
    summarize_and_extract_slots,
)

logger = get_agent_logger("manager")


def manager_node(state: GraphState) -> dict:
    """
    Manager Agent 的 LangGraph 节点函数。
    输入: 包含 data_path 和 user_goal 的初始 State。
    输出: 更新 global_plan, current_phase, current_data_path 等字段。
    """

    user_goal = state["user_goal"]
    data_path = state["data_path"]
    cleaning_hints = (state["manager_cleaning_hints"] or "").strip()

    logger.info(f"Starting manager_node with user_goal: '{user_goal}' and data_path: '{data_path}'")

    # ------------------------------------------------------------------
    # Step 1a — 数据轮廓（非 LLM）
    # 从 CSV 读取列名、dtype、shape、前几行样例；不是「关键词提取」，而是结构化表头/类型事实，供 1b 与规划 LLM 使用。
    # ------------------------------------------------------------------
    df_preview = _get_data_preview(data_path)
    logger.info("Data preview generated.")

    # ------------------------------------------------------------------
    # Step 1b — 需求摘要与检索槽位（LLM）
    # manager_cleaning_hints：可选「输入便签」——用户/队友给的清洗约定草稿等，只参与本段摘要 prompt；
    # 产出物是 user_requirement_summary（叙述）+ 槽位；不等于 global_plan 里的 rules/cleaning_contract（那些由 Step 3 单独生成）。
    # ------------------------------------------------------------------
    req_summary, slots = summarize_and_extract_slots(
        user_goal,
        df_preview,
        cleaning_hints,
    )
    logger.info("User requirement summary and retrieval slots ready.")

    # ------------------------------------------------------------------
    # Step 2 — RAP：检索 + 融合（与 1b 分离，便于理解与测试）
    # ------------------------------------------------------------------
    rap_context = retrieve_and_fuse_rap_context(req_summary, slots, user_goal)
    logger.info("RAP context built (arXiv + optional web/PWC/Kaggle).")

    # ------------------------------------------------------------------
    # Step 3 — 结构化 global_plan（LLM）
    # ------------------------------------------------------------------
    global_plan = _generate_global_plan(
        user_goal,
        df_preview,
        rap_context,
        user_requirement_summary=req_summary,
    )
    logger.info("Global structured plan generated successfully.")

    # ------------------------------------------------------------------
    # Step 4 — 初始化运行时状态并返回
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
    """
    读取 CSV 的结构化轮廓（列名、 dtypes、shape、前几行），拼成一段纯文本。
    不调用 LLM、不做关键词抽取；仅为后续 LLM（Step 1b 摘要、Step 3 规划）提供可引用的数据事实。
    """
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
    """兼容测试与外部调用：基于摘要+槽位生成 arXiv 检索词。"""
    return generate_arxiv_search_queries(user_goal, df_preview)


def _generate_global_plan(
    user_goal: str,
    df_preview: str,
    rap_context: str,
    *,
    user_requirement_summary: str = "",
) -> dict:
    """调用 LLM 生成结构化执行计划 (JSON)。

    与旧版参数 `retrieved_docs: list[dict]` 的对应关系：
    - 旧：在函数内把多篇 {title, summary} 拼成一段「文献上下文」字符串。
    - 现：由 `retrieve_and_fuse_rap_context` 预先产出 `rap_context`（已含多源融合后的叙述），本函数只消费字符串。
    - `plan["_rap_context"]` 存的是注入规划 prompt 的那份 RAP 文本，便于审计与复现。

    `user_requirement_summary` 与 `manager_cleaning_hints`：
    - hints 仅在 Step 1b 写入摘要 prompt；本参数收到的是已生成的摘要（可已内含 hints 的语义）。
    - 规划阶段同时看到 `user_goal`、摘要块、`df_preview` 与 `rap_context`，四者角色不同，勿混同。

    `data_cleaning_plan.cleaning_contract` — Minimal general mode (keys are optional; default or `{}` indicates no validation):

    - `max_row_drop_ratio`: A floating-point number from 0 to 1, representing the maximum percentage of rows allowed to be deleted/dropped relative to the number of rows in the input CSV.

    Column retention is not enforced: cleaning may drop or rename columns per objectives.

    Differences from `rules`:

    - `rules`: Plans which cleaning steps (processes) to perform, for subsequent rule-based execution or suggestions.

    - `cleaning_contract`: Task-level safety boundaries (results), hard-validated by the Data Cleaning node after data is written to disk.
    """
    summary_block = user_requirement_summary.strip() or user_goal
    docs_context = rap_context.strip() if rap_context.strip() else "无检索结果"

    sys_prompt = "You are a Data Science Engineering Director generating an execution plan."
    user_prompt = f"""
User Goal: {user_goal}
User requirement summary (for alignment):
{summary_block}

Data Preview:
{df_preview}

Literature / external knowledge (RAP):
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
    reply = call_llm(sys_prompt, user_prompt, agent_name="manager_plan")
    if reply:
        try:
            match = re.search(r'\{.*\}', reply, re.DOTALL)
            if match:
                plan = json.loads(match.group(0))
                if isinstance(plan, dict) and "data_cleaning_plan" in plan:
                    _normalize_data_cleaning_plan(plan)
                    plan["_rap_context"] = docs_context
                    plan["_user_requirement_summary"] = summary_block
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
        "_user_requirement_summary": summary_block,
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
