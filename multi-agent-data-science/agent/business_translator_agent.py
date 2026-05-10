"""Business Action Translator — McKinsey RAG + arXiv fallback → Markdown 业务报告."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import END, StateGraph

from core.global_state import GlobalState
from core.my_llm import call_llm
from rag.service import retrieve_knowledge
from tools.business_tools.arxiv_tool import search_arxiv
from tools.business_tools.file_reader import read_file

logger = logging.getLogger(__name__)

MCKINSEY_COLLECTION = os.getenv("MCKINSEY_COLLECTION", "mckinsey_kb")
MCKINSEY_SOURCE_TYPES = ("mckinsey", "equity_research", "textbook")
MCKINSEY_PER_BUCKET_TOP_K = int(os.getenv("MCKINSEY_PER_BUCKET_TOP_K", "3"))
MCKINSEY_MIN_HITS = int(os.getenv("MCKINSEY_MIN_HITS", "4"))

STYLE_SKELETONS_PATH = PROJECT_ROOT / "data" / "mckinsey_kb" / "style_skeletons.md"


def _build_model_validation_summary(state: GlobalState) -> str:
    """用 best_model + model_results 拼出模型阶段摘要（无 feature_importance 文件时）。"""
    best = state.get("best_model") or {}
    results = state.get("model_results") or {}
    lines: list[str] = []
    if best:
        lines.append(
            f"**Best model:** {best.get('model_name', 'n/a')}  "
            f"task_type={best.get('task_type', 'n/a')}"
        )
        m = best.get("metrics") or {}
        if m:
            lines.append("**Best model metrics (test):** " + json.dumps(m, ensure_ascii=False))
    if results:
        lines.append("**All candidate results:**")
        for name, r in results.items():
            lines.append(
                f"- {name}: metrics={json.dumps(r.get('metrics', {}), ensure_ascii=False)}"
            )
    return "\n".join(lines) if lines else "No model validation summary available."


def _build_rag_queries(user_goal: str, top_features: list[str]) -> list[str]:
    if not top_features:
        return [f"{user_goal} business analytics strategy"]

    features_str = ", ".join(top_features[:3])
    sys_prompt = (
        "You are an expert data strategy consultant. "
        "Generate robust search queries for consulting reports and academic literature."
    )
    user_prompt = f"""
Goal: {user_goal}
Top Features from Model: {features_str}

Please generate 2 to 3 precise English search queries that combine the user's goal with the top features.
Return ONLY a JSON array of strings, e.g.:
["query 1", "query 2"]
"""
    reply = call_llm(sys_prompt, user_prompt, agent_name="business_translator")
    if reply:
        try:
            match = re.search(r"\[.*\]", reply, re.DOTALL)
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
                    query,
                    source_type,
                    e,
                )
                continue

            for hit in hits:
                content = (hit.get("content") or "").strip()
                if not content:
                    continue
                key = content[:200]
                if key in seen_keys[source_type]:
                    continue
                seen_keys[source_type].add(key)
                buckets[source_type].append(hit)

    return buckets


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


def _load_style_skeletons() -> str:
    try:
        return read_file(str(STYLE_SKELETONS_PATH)).strip()
    except Exception as e:
        logger.warning("Style skeletons unavailable (%s); proceeding without.", e)
        return ""


def _format_eda_placeholder(eda_report: Any) -> str:
    if not eda_report:
        return "(not provided — upstream did not export structured EDA summary.)"
    try:
        return json.dumps(eda_report, ensure_ascii=False, indent=2)[:12000]
    except Exception:
        return str(eda_report)[:12000]


def _generate_business_report(
    user_goal: str,
    model_summary: str,
    retrieved_arxiv: list[dict],
    *,
    mckinsey_buckets: dict[str, list[dict]],
    style_skeletons: str,
    dataset_description: str,
    eda_block: str,
) -> str:
    arxiv_text = (
        "\n".join(
            f"- {d.get('title', 'N/A')}: {str(d.get('summary', ''))[:150]}..."
            for d in retrieved_arxiv
        )
        if retrieved_arxiv
        else "(no arxiv fallback used.)"
    )

    desc_text = dataset_description.strip() if dataset_description else ""
    dataset_description_block = (
        desc_text
        if desc_text
        else "(not provided; base the report on the user goal and model outputs only.)"
    )

    mck_block = _format_mckinsey_bucket("McKinsey Reports", mckinsey_buckets.get("mckinsey", []))
    eq_block = _format_mckinsey_bucket(
        "Equity Research & Strategy", mckinsey_buckets.get("equity_research", [])
    )
    tb_block = _format_mckinsey_bucket("Management Textbook", mckinsey_buckets.get("textbook", []))
    skeletons_block = (
        style_skeletons or "(no style skeletons available; rely on McKinsey style rules below.)"
    )

    sys_prompt = (
        "You are an elite Data Strategy Consultant writing in McKinsey consulting style. "
        "Apply the Pyramid Principle (top-down, answer-first), open with SCQA, "
        "structure findings into 3-5 MECE themes, and close with concrete sequenced actions. "
        "CRITICAL: be concise; ground every claim in model evidence or retrieved external context. "
        "Output MUST be plain Markdown in English (no YAML front matter)."
    )

    user_prompt = f"""
Produce a **Business Strategy & Insight Report** as Markdown only (English).

# User Goal
{user_goal}

# Dataset Description (business context)
{dataset_description_block}

# EDA / profiling context (may be JSON from upstream agents)
{eda_block}

# Model Validation Summary
{model_summary}

# Top Features & Importance
(not provided — feature_importance.json was not produced in this pipeline run; do not invent scores.)

# Test-Set Predictions Summary
(not provided — test_predictions.csv was not produced in this pipeline run.)

# External Context - McKinsey Knowledge Base (primary)
{mck_block}
{eq_block}
{tb_block}

# External Context - Arxiv (fallback, may be empty)
{arxiv_text}

# Reference Skeletons (structural anchors only - imitate the *shape*, not the content)
{skeletons_block}

# Required Output Structure (in this exact order)
1. **Executive Summary** — bottom-line answer + three bullet takeaways.
2. **Situation & Complication** — SCQA opening.
3. **Key Findings** — 3 to 5 MECE themes; claim → evidence → implication.
4. **Strategic Implications** — business meaning; cite external sources as [1], [2] where applicable.
5. **Recommended Actions** — sequenced 90-day priorities; concrete owners/metrics/timeframe where possible.

Style rules:
- Match the tone of the McKinsey Reports section where relevant.
- Do NOT invent statistics absent from the model summary or retrieved context.
"""

    reply = call_llm(sys_prompt, user_prompt, agent_name="business_translator")
    if reply:
        return reply

    return f"""# Business Insight Analysis Report

## Executive Summary
User goal: {user_goal}

## Model Validation
{model_summary}

## External Context
{mck_block}
{eq_block}
{tb_block}

Arxiv fallback:
{arxiv_text}

*(LLM returned empty content; check API / quota / STUB_MODE.)*
"""


def business_translator_node(state: GlobalState) -> dict[str, Any]:
    """LangGraph 节点：生成业务报告并写入 ``{{run_dir}}/reports/business_report.md``。"""
    user_goal = state.get("user_request") or ""
    dataset_description = (state.get("description") or "").strip()
    eda_block = _format_eda_placeholder(state.get("eda_report"))
    run_dir = state.get("run_dir") or ""

    model_summary = _build_model_validation_summary(state)
    top_features: list[str] = []

    search_queries = _build_rag_queries(user_goal, top_features)
    logger.info("Generated %d RAG queries.", len(search_queries))

    mckinsey_buckets = _retrieve_mckinsey_context(search_queries)
    total_mckinsey_hits = sum(len(v) for v in mckinsey_buckets.values())

    retrieved_arxiv: list[dict] = []
    if total_mckinsey_hits < MCKINSEY_MIN_HITS:
        for q in search_queries:
            retrieved_arxiv.extend(search_arxiv(q))
    style_skeletons = _load_style_skeletons()

    report = _generate_business_report(
        user_goal,
        model_summary,
        retrieved_arxiv,
        mckinsey_buckets=mckinsey_buckets,
        style_skeletons=style_skeletons,
        dataset_description=dataset_description,
        eda_block=eda_block,
    )

    if run_dir:
        reports_dir = Path(run_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / "business_report.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"[business_translator] report saved → {out_path}")

    return {
        "business_report": report,
        "current_stage": "business_report_done",
    }


def _show_graph_if_needed(compiled, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "business_translator_graph.png"
        image_bytes = compiled.get_graph().draw_mermaid_png()
        image_path.write_bytes(image_bytes)
        print(f"Graph 图片已保存: {image_path}")
    except Exception as exc:
        print(f"图渲染失败: {exc}")


def build_business_translator_agent(show_graph: bool = False):
    """单节点子图，与 agent_manager 等保持一致的编译接口。"""
    graph = StateGraph(GlobalState)
    graph.add_node("business_translator", business_translator_node)
    graph.set_entry_point("business_translator")
    graph.add_edge("business_translator", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled
