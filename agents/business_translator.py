"""
Business Action Translator Agent (商业建议转化智能体)
职责：
  1. 从模型阶段的落盘报告中提取核心指标与特征重要性
  2. 基于特征重要性构造学术检索词，调用 arxiv RAG 获取行业策略文献
  3. 结合内部数据表征 + 外部学术增强，生成专业化商业洞察报告
"""
from __future__ import annotations

import json
import os
import re

from core.state import GraphState
from core.llm import call_llm
from tools.arxiv_tool import search_arxiv
from tools.file_reader import read_file
from config import WORKSPACE_DIR
from core.logger import get_agent_logger

logger = get_agent_logger("business_translator")


def business_translator_node(state: GraphState) -> dict:
    """Business Action Translator 的 LangGraph 节点函数。"""

    user_goal = state["user_goal"]
    memory = state.get("memory", [])
    dataset_description = (state.get("dataset_description") or "").strip()
    logger.info("Business Translator node started.")

    # ------------------------------------------------------------------
    # Step 1: 提取模型核心结论 (从落盘文件中按需读取)
    # ------------------------------------------------------------------
    fi_path = os.path.join(WORKSPACE_DIR, "feature_importance.json")
    feature_importance = _load_feature_importance(fi_path)
    model_summary = memory[-1] if memory else "无模型阶段摘要"
    logger.info(f"Loaded feature importance for {len(feature_importance)} features.")

    # ------------------------------------------------------------------
    # Step 2: 构造 RAG 检索查询词
    # ------------------------------------------------------------------
    top_features = list(feature_importance.keys())[:5] if feature_importance else []
    search_queries = _build_rag_queries(user_goal, top_features)
    logger.info(f"Generated {len(search_queries)} RAG queries for external strategy literature.")

    # ------------------------------------------------------------------
    # Step 3: 调用 arxiv 检索外部文献
    # ------------------------------------------------------------------
    retrieved_docs = []
    for q in search_queries:
        retrieved_docs.extend(search_arxiv(q))
    logger.info(f"Arxiv search complete. Retrieved {len(retrieved_docs)} strategy documents.")

    # ------------------------------------------------------------------
    # Step 4: 调用 LLM 生成商业分析报告 (TODO: 接入LLM)
    # ------------------------------------------------------------------
    report = _generate_business_report(
        user_goal,
        model_summary,
        feature_importance,
        top_features,
        retrieved_docs,
        dataset_description=dataset_description,
    )
    logger.info("Business report generation complete.")

    return {"final_business_report": report}


# ======================================================================
# 私有辅助函数
# ======================================================================

def _load_feature_importance(path: str) -> dict:
    """
    Load the feature-importance artifact produced by model_selection.

    The contract is a flat JSON object ``{feature_name: importance}``, but in
    practice the LLM-generated model-selection code drifts into two other
    shapes from time to time:

      1. List of pairs:        ``[["duration", 0.39], ["poutcome", 0.15], ...]``
      2. List of dict entries: ``[{"feature": "duration", "importance": 0.39}, ...]``
         (also seen: ``"name"``/``"column"`` instead of ``"feature"``,
         ``"score"``/``"value"`` instead of ``"importance"``)

    We accept all three, normalise into a dict, and sort descending by value
    so ``list(...)[:5]`` in the caller still gets the top-K. Returning a
    plain dict keeps the downstream ``.keys()`` / ``.items()`` call sites
    unchanged.
    """
    try:
        content = read_file(path)
        raw = json.loads(content)
    except Exception:
        return {}

    normalised: dict[str, float] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                normalised[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name, score = item[0], item[1]
            elif isinstance(item, dict):
                name = (
                    item.get("feature")
                    or item.get("name")
                    or item.get("column")
                )
                score = (
                    item.get("importance")
                    if "importance" in item
                    else item.get("score", item.get("value"))
                )
            else:
                continue
            if name is None or score is None:
                continue
            try:
                normalised[str(name)] = float(score)
            except (TypeError, ValueError):
                continue
    else:
        return {}

    # Sort descending so list(d.keys())[:5] gives top-5 regardless of how the
    # producer ordered the file.
    return dict(
        sorted(normalised.items(), key=lambda kv: kv[1], reverse=True)
    )


def _build_rag_queries(user_goal: str, top_features: list[str]) -> list[str]:
    """调用 LLM 根据特征与目标交叉生成检索词。"""
    if not top_features:
        return [f"{user_goal} business analytics"]

    features_str = ", ".join(top_features[:3])
    sys_prompt = "You are an expert data strategy consultant. Your task is to generate robust search queries for academic literature (e.g. Arxiv)."
    user_prompt = f"""
Goal: {user_goal}
Top Features from Model: {features_str}

Please generate 2 to 3 precise English search queries for an academic database that combine the user's goal with the top features to find actionable business strategies or methodologies.
Return ONLY a JSON array of strings, e.g.:
["query 1", "query 2"]
"""
    reply = call_llm(sys_prompt, user_prompt, agent_name="business_translator")
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
    queries = []
    for feature in top_features[:3]:
        queries.append(f"{user_goal} {feature} business strategy intervention")
    return queries


def _generate_business_report(
    user_goal: str,
    model_summary: str,
    feature_importance: dict,
    top_features: list[str],
    retrieved_docs: list[dict],
    *,
    dataset_description: str = "",
) -> str:
    """调用 LLM 生成最终商业报告。"""
    docs_text = "\n".join(
        [f"- {d.get('title', 'N/A')}: {d.get('summary', '')[:150]}..." for d in retrieved_docs]
    ) if retrieved_docs else "暂无外部文献支撑"

    fi_text = "\n".join(
        [f"  - {k}: {v:.4f}" for k, v in list(feature_importance.items())[:5]]
    ) if feature_importance else "  暂无特征权重数据"

    desc_text = dataset_description.strip() if dataset_description else ""
    dataset_description_block = (
        desc_text
        if desc_text
        else "(not provided; base the report on the user goal and model outputs only.)"
    )

    sys_prompt = "You are an elite Data Strategy Consultant and Business Analyst. Your role is to interpret machine learning results and synthesize them into a highly professional, expert-level business insight report. CRITICAL INSTRUCTION: Be concise, strictly avoid repeating any sentences, phrases, or ideas. DO NOT fall into an infinite generation loop."

    user_prompt = f"""
Please generate a structured Business Strategy & Insight Report in Markdown format (always output in Chinese).

User Goal: {user_goal}

Dataset Description (authoritative; use this to interpret features and business context):
{dataset_description_block}

Model Validation Summary:
{model_summary}

Top Features & Importance:
{fi_text}

External Literature Support (RAG retrieved academic context):
{docs_text}

Requirements:
1. Executive Summary (Overview of the project goal and machine learning validation approach).
2. Business Interpretation of Top Features (Why do these features matter? Explain their real-world impact based on their importance).
3. Literature Syntheses (How does external research validate or enhance our findings? Emphasize the strategies from the retrieved docs).
4. Actionable Business Recommendations (Highly specific, practical intervention strategies the business can apply).
Please output professional, structured Markdown. PREVENT REPETITION: Do NOT say the same thing more than once.
"""
    reply = call_llm(sys_prompt, user_prompt, agent_name="business_translator")
    if reply:
        return reply

    # Fallback
    report = f"""# 商业洞察分析报告

## 1. 执行摘要
用户目标: {user_goal}
模型验证结果: {model_summary}

## 2. 核心特征的业务解读
以下特征对预测目标影响最大:
{fi_text}

## 3. 外部文献与行业参考 (RAG 增强)
{docs_text}

## 4. 可执行商业建议
（本节依赖 LLM 生成；本次调用未返回有效正文，故仅保留模板。请检查 `business_translator` 所用 API、配额或 `STUB_MODE`，并参考上文特征与文献做业务决策。）
"""
    return report
