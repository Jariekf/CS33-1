"""
Business Action Translator Agent (商业建议转化智能体)
职责：
  1. 从模型阶段的落盘报告中提取核心指标与特征重要性
  2. 检索麦肯锡风格知识库（mckinsey_kb）作为主要外部素材
     - 按 source_type 分三桶：mckinsey / equity_research / textbook
     - 命中数不足时回退到 arxiv 学术检索
  3. 结合内部数据表征 + 外部素材，按 McKinsey Pyramid Principle 输出商业洞察报告
"""
from __future__ import annotations

import json
import os
import re

from core.state import GraphState
from core.llm import call_llm
from tools.arxiv_tool import search_arxiv
from tools.file_reader import read_file
from rag.service import retrieve_knowledge
from config import WORKSPACE_DIR
from core.logger import get_agent_logger

logger = get_agent_logger("business_translator")

# Independent collection so the McKinsey corpus does not pollute the generic
# rag_collection used by future RAG consumers.
MCKINSEY_COLLECTION = os.getenv("MCKINSEY_COLLECTION", "mckinsey_kb")

# Three-bucket retrieval keeps the LLM aware of *what kind* of evidence each
# snippet is, so it can apply McKinsey style to consulting reports, financial
# framing to equity research, and structural lessons to the textbook.
MCKINSEY_SOURCE_TYPES = ("mckinsey", "equity_research", "textbook")
MCKINSEY_PER_BUCKET_TOP_K = int(os.getenv("MCKINSEY_PER_BUCKET_TOP_K", "3"))

# RAG-first, arxiv-fallback. If aggregated McKinsey hits stay below this
# threshold we still call arxiv so the report has *some* external grounding.
MCKINSEY_MIN_HITS = int(os.getenv("MCKINSEY_MIN_HITS", "4"))

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE_SKELETONS_PATH = os.path.join(
    PROJECT_ROOT_DIR, "data", "mckinsey_kb", "style_skeletons.md"
)


def business_translator_node(state: GraphState) -> dict:
    """Business Action Translator 的 LangGraph 节点函数。"""

    user_goal = state["user_goal"]
    memory = state.get("memory", [])
    dataset_description = (state.get("dataset_description") or "").strip()
    eda_summary = (state.get("eda_summary") or "").strip()
    logger.info("Business Translator node started.")

    # ------------------------------------------------------------------
    # Step 1: 提取模型核心结论 (从落盘文件中按需读取)
    # ------------------------------------------------------------------
    fi_path = os.path.join(WORKSPACE_DIR, "feature_importance.json")
    feature_importance = _load_feature_importance(fi_path)
    model_summary = memory[-1] if memory else "No model-stage summary available."
    logger.info(f"Loaded feature importance for {len(feature_importance)} features.")

    # ------------------------------------------------------------------
    # Step 2: 构造 RAG 检索查询词
    # ------------------------------------------------------------------
    top_features = list(feature_importance.keys())[:5] if feature_importance else []
    search_queries = _build_rag_queries(user_goal, top_features)
    logger.info(f"Generated {len(search_queries)} RAG queries for external strategy literature.")

    # ------------------------------------------------------------------
    # Step 3: McKinsey RAG 检索（主路径）
    # ------------------------------------------------------------------
    mckinsey_buckets = _retrieve_mckinsey_context(search_queries)
    total_mckinsey_hits = sum(len(v) for v in mckinsey_buckets.values())
    logger.info(
        "McKinsey RAG retrieval: total_hits=%d (mckinsey=%d, equity_research=%d, textbook=%d)",
        total_mckinsey_hits,
        len(mckinsey_buckets.get("mckinsey", [])),
        len(mckinsey_buckets.get("equity_research", [])),
        len(mckinsey_buckets.get("textbook", [])),
    )

    # ------------------------------------------------------------------
    # Step 4: arxiv fallback（仅当本地 RAG 命中不足时）
    # ------------------------------------------------------------------
    retrieved_arxiv: list[dict] = []
    if total_mckinsey_hits < MCKINSEY_MIN_HITS:
        logger.info(
            "McKinsey hits %d < threshold %d; falling back to arxiv search",
            total_mckinsey_hits,
            MCKINSEY_MIN_HITS,
        )
        for q in search_queries:
            retrieved_arxiv.extend(search_arxiv(q))
        logger.info(f"Arxiv fallback retrieved {len(retrieved_arxiv)} doc(s).")
    else:
        logger.info("Sufficient McKinsey hits; skipping arxiv fallback.")

    # ------------------------------------------------------------------
    # Step 5: 加载风格骨架锚点
    # ------------------------------------------------------------------
    style_skeletons = _load_style_skeletons()

    # ------------------------------------------------------------------
    # Step 6: 调用 LLM 生成商业分析报告
    # ------------------------------------------------------------------
    report = _generate_business_report(
        user_goal,
        model_summary,
        feature_importance,
        top_features,
        retrieved_arxiv,
        mckinsey_buckets=mckinsey_buckets,
        style_skeletons=style_skeletons,
        dataset_description=dataset_description,
        eda_summary=eda_summary,
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

    return dict(
        sorted(normalised.items(), key=lambda kv: kv[1], reverse=True)
    )


def _build_rag_queries(user_goal: str, top_features: list[str]) -> list[str]:
    """调用 LLM 根据特征与目标交叉生成检索词。"""
    if not top_features:
        return [f"{user_goal} business analytics"]

    features_str = ", ".join(top_features[:3])
    sys_prompt = "You are an expert data strategy consultant. Your task is to generate robust search queries for both consulting reports (McKinsey, equity research) and academic literature (Arxiv)."
    user_prompt = f"""
Goal: {user_goal}
Top Features from Model: {features_str}

Please generate 2 to 3 precise English search queries that combine the user's goal with the top features to surface actionable business strategies, market context, or methodologies.
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

    queries = []
    for feature in top_features[:3]:
        queries.append(f"{user_goal} {feature} business strategy intervention")
    return queries


def _retrieve_mckinsey_context(queries: list[str]) -> dict[str, list[dict]]:
    """Retrieve top-k chunks per source_type for each query, dedupe, and bucket.

    Returns a dict keyed by source_type with a list of
    ``{"content", "metadata"}`` items. Empty-list values are kept for
    downstream prompt formatting.
    """
    buckets: dict[str, list[dict]] = {st: [] for st in MCKINSEY_SOURCE_TYPES}
    seen_keys: dict[str, set] = {st: set() for st in MCKINSEY_SOURCE_TYPES}

    for query in queries:
        for source_type in MCKINSEY_SOURCE_TYPES:
            try:
                hits = retrieve_knowledge(
                    query=query,
                    top_k=MCKINSEY_PER_BUCKET_TOP_K,
                    collection_name=MCKINSEY_COLLECTION,
                    metadata_filter={"source_type": source_type},
                )
            except Exception as e:
                logger.warning(
                    "McKinsey RAG retrieval failed for query=%r, source_type=%s: %s",
                    query, source_type, e,
                )
                continue

            for hit in hits:
                content = (hit.get("content") or "").strip()
                if not content:
                    continue
                # Dedupe per-bucket using the first 200 chars as a cheap hash.
                key = content[:200]
                if key in seen_keys[source_type]:
                    continue
                seen_keys[source_type].add(key)
                buckets[source_type].append(hit)

    return buckets


def _load_style_skeletons() -> str:
    try:
        return read_file(STYLE_SKELETONS_PATH).strip()
    except Exception as e:
        logger.warning("Style skeletons unavailable (%s); proceeding without.", e)
        return ""


def _format_mckinsey_bucket(label: str, items: list[dict]) -> str:
    if not items:
        return f"### {label}\n(no matching context retrieved.)\n"
    blocks: list[str] = [f"### {label}"]
    for i, item in enumerate(items, start=1):
        meta = item.get("metadata") or {}
        title = meta.get("title", "Untitled")
        section = meta.get("section_path") or ""
        year = meta.get("year")
        head = f"[{i}] {title}"
        if year:
            head += f" ({year})"
        if section:
            head += f"  -  {section}"
        body = (item.get("content") or "").strip()
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks) + "\n"


def _generate_business_report(
    user_goal: str,
    model_summary: str,
    feature_importance: dict,
    top_features: list[str],
    retrieved_docs: list[dict],
    *,
    mckinsey_buckets: dict[str, list[dict]] | None = None,
    style_skeletons: str = "",
    dataset_description: str = "",
    eda_summary: str = "",
) -> str:
    """调用 LLM 生成最终商业报告（McKinsey 风格）。"""
    mckinsey_buckets = mckinsey_buckets or {}

    arxiv_text = "\n".join(
        [f"- {d.get('title', 'N/A')}: {d.get('summary', '')[:150]}..." for d in retrieved_docs]
    ) if retrieved_docs else "(no arxiv fallback used.)"

    fi_text = "\n".join(
        [f"  - {k}: {v:.4f}" for k, v in list(feature_importance.items())[:5]]
    ) if feature_importance else "  No feature-importance data available."

    desc_text = dataset_description.strip() if dataset_description else ""
    dataset_description_block = (
        desc_text
        if desc_text
        else "(not provided; base the report on the user goal and model outputs only.)"
    )
    eda_block = eda_summary if eda_summary else "(not provided by feature engineering)"

    mck_block = _format_mckinsey_bucket("McKinsey Reports", mckinsey_buckets.get("mckinsey", []))
    eq_block = _format_mckinsey_bucket("Equity Research & Strategy", mckinsey_buckets.get("equity_research", []))
    tb_block = _format_mckinsey_bucket("Management Textbook", mckinsey_buckets.get("textbook", []))
    skeletons_block = style_skeletons or "(no style skeletons available; rely on the McKinsey style rules below.)"

    sys_prompt = (
        "You are an elite Data Strategy Consultant writing in McKinsey consulting style. "
        "Apply the Pyramid Principle (top-down, answer-first), open with SCQA (Situation - Complication - Question - Answer), "
        "structure findings into 3-5 MECE themes, and close with concrete sequenced actions. "
        "CRITICAL: be concise; do NOT repeat any sentence, phrase, or idea; do NOT enter generation loops; "
        "ground every claim in either the model evidence or the retrieved external context."
    )

    user_prompt = f"""
Produce a Business Strategy & Insight Report in Markdown, English only.

# User Goal
{user_goal}

# Dataset Description (authoritative business context for feature interpretation)
{dataset_description_block}

# EDA Summary from Feature Engineering
{eda_block}

# Model Validation Summary
{model_summary}

# Top Features & Importance
{fi_text}

# External Context - McKinsey Knowledge Base (primary)
{mck_block}
{eq_block}
{tb_block}

# External Context - Arxiv (fallback, may be empty)
{arxiv_text}

# Reference Skeletons (structural anchors only - imitate the *shape*, not the content)
{skeletons_block}

# Required Output Structure (McKinsey-style, in this exact order)
1. **Executive Summary**  -  the bottom-line answer in one paragraph plus three bullet takeaways (Pyramid Principle: answer first).
2. **Situation & Complication**  -  current state of the problem and why it matters now (SCQA opening).
3. **Key Findings**  -  3 to 5 MECE themes; each theme: claim -> evidence (cite features, model results, or external sources by their bracketed index like [1]) -> "so what" implication.
4. **Strategic Implications**  -  what these findings mean for the business; reference the equity research / textbook framings where relevant.
5. **Recommended Actions**  -  sequenced, concrete 90-day priorities; no vague language ("consider exploring" is forbidden); each action specifies owner / metric / timeframe where possible.

Style rules:
- Match the tone and structure of the McKinsey Reports section above.
- Quote or cite external context using [1], [2] referencing the bracketed indices in the McKinsey/Equity/Textbook sections.
- Do NOT invent statistics that are not in the model summary or the retrieved external context.
- PREVENT REPETITION: do NOT say the same thing more than once.
"""
    reply = call_llm(sys_prompt, user_prompt, agent_name="business_translator")
    if reply:
        return reply

    return f"""# Business Insight Analysis Report

## 1. Executive Summary
User goal: {user_goal}
Model validation result: {model_summary}

## 2. Business Interpretation of Key Features
The following features have the strongest impact on the prediction target:
{fi_text}

## 3. External Context (McKinsey RAG primary; arxiv fallback)
{mck_block}
{eq_block}
{tb_block}

Arxiv fallback:
{arxiv_text}

## 4. Actionable Business Recommendations
This section depends on LLM generation. The current call did not return valid content, so only the fallback template is provided. Check the API, quota, or `STUB_MODE` used by `business_translator`, and use the feature and external evidence above for decision-making.
"""
