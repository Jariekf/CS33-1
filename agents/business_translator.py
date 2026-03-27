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


def business_translator_node(state: GraphState) -> dict:
    """Business Action Translator 的 LangGraph 节点函数。"""

    user_goal = state["user_goal"]
    memory = state.get("memory", [])

    # ------------------------------------------------------------------
    # Step 1: 提取模型核心结论 (从落盘文件中按需读取)
    # ------------------------------------------------------------------
    fi_path = os.path.join(WORKSPACE_DIR, "feature_importance.json")
    feature_importance = _load_feature_importance(fi_path)
    model_summary = memory[-1] if memory else "无模型阶段摘要"

    # ------------------------------------------------------------------
    # Step 2: 构造 RAG 检索查询词
    # ------------------------------------------------------------------
    top_features = list(feature_importance.keys())[:5] if feature_importance else []
    search_queries = _build_rag_queries(user_goal, top_features)

    # ------------------------------------------------------------------
    # Step 3: 调用 arxiv 检索外部文献
    # ------------------------------------------------------------------
    retrieved_docs = []
    for q in search_queries:
        retrieved_docs.extend(search_arxiv(q))

    # ------------------------------------------------------------------
    # Step 4: 调用 LLM 生成商业分析报告 (TODO: 接入LLM)
    # ------------------------------------------------------------------
    report = _generate_business_report(
        user_goal, model_summary, feature_importance, top_features, retrieved_docs
    )

    return {"final_business_report": report}


# ======================================================================
# 私有辅助函数
# ======================================================================

def _load_feature_importance(path: str) -> dict:
    """加载模型选择阶段落盘的特征重要性文件。"""
    try:
        content = read_file(path)
        return json.loads(content)
    except Exception:
        return {}


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
    reply = call_llm(sys_prompt, user_prompt)
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
) -> str:
    """调用 LLM 生成最终商业报告。"""
    docs_text = "\n".join(
        [f"- {d.get('title', 'N/A')}: {d.get('summary', '')[:150]}..." for d in retrieved_docs]
    ) if retrieved_docs else "暂无外部文献支撑"

    fi_text = "\n".join(
        [f"  - {k}: {v:.4f}" for k, v in list(feature_importance.items())[:5]]
    ) if feature_importance else "  暂无特征权重数据"

    sys_prompt = "You are an elite Data Strategy Consultant and Business Analyst. Your role is to interpret machine learning results and synthesize them into a highly professional, expert-level business insight report. CRITICAL INSTRUCTION: Be concise, strictly avoid repeating any sentences, phrases, or ideas. DO NOT fall into an infinite generation loop."
    
    user_prompt = f"""
Please generate a structured Business Strategy & Insight Report in Markdown format (always output in Chinese). 

User Goal: {user_goal}
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
    reply = call_llm(sys_prompt, user_prompt)
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
(由于平台限制未生成详细内容，请参考以上特征进行业务干预)
"""
    return report
