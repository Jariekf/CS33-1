"""LangChain @tool wrappers for Business Translator Agent (RAG, arXiv, style skeletons, finalize)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.tools import tool

from rag.service import retrieve_knowledge
from tools.business_tools.arxiv_tool import search_arxiv
from tools.business_tools.file_reader import read_file

logger = logging.getLogger(__name__)

MCKINSEY_COLLECTION = os.getenv("MCKINSEY_COLLECTION", "mckinsey_kb")
MCKINSEY_SOURCE_TYPES = ("mckinsey", "equity_research", "textbook")
MCKINSEY_PER_BUCKET_TOP_K = int(os.getenv("MCKINSEY_PER_BUCKET_TOP_K", "3"))

# Resolved relative to multi-agent-data-science package root (parent of tools/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STYLE_SKELETONS_PATH = os.path.join(_PROJECT_ROOT, "data", "mckinsey_kb", "style_skeletons.md")


def retrieve_mckinsey_context_impl(queries: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Shared retrieval logic: dedupe by content prefix per source_type bucket."""
    buckets: dict[str, list[dict]] = {st: [] for st in MCKINSEY_SOURCE_TYPES}
    seen_keys: dict[str, set[str]] = {st: set() for st in MCKINSEY_SOURCE_TYPES}

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


@tool
def retrieve_mckinsey_knowledge(queries: list[str]) -> str:
    """Retrieve relevant chunks from the McKinsey-style knowledge base (3 buckets: reports, equity research, textbook).

    Call with 2–4 English search queries derived from the user goal and top predictive features.
    Returns JSON with bucket hit counts and short previews for planning next steps.
    """
    if not queries:
        return json.dumps(
            {"error": {"type": "ValidationError", "message": "queries must be a non-empty list of strings."}},
            ensure_ascii=False,
        )
    buckets = retrieve_mckinsey_context_impl(queries)
    total = sum(len(v) for v in buckets.values())
    preview_lines: list[str] = []
    for st, items in buckets.items():
        for item in items[:1]:
            meta = (item.get("metadata") or {}).get("title", "")
            body = (item.get("content") or "")[:160].replace("\n", " ")
            preview_lines.append(f"[{st}] {meta}: {body}...")
    return json.dumps(
        {
            "buckets": buckets,
            "total_hits": total,
            "queries_used": queries,
            "preview": "\n".join(preview_lines) if preview_lines else "(no hits)",
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def search_arxiv_papers(queries: list[str]) -> str:
    """Search arXiv for academic papers as a fallback when McKinsey KB context is thin.

    Pass 1–3 focused English queries. Results append to state for the final report.
    """
    if not queries:
        return json.dumps(
            {"error": {"type": "ValidationError", "message": "queries must be non-empty."}},
            ensure_ascii=False,
        )
    all_papers: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for q in queries[:5]:
        for paper in search_arxiv(q):
            t = str(paper.get("title", ""))
            if t and t not in seen_titles:
                seen_titles.add(t)
                all_papers.append(paper)
    return json.dumps(
        {
            "papers": all_papers[:20],
            "count": len(all_papers[:20]),
            "queries_used": queries,
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def load_report_style_skeletons() -> str:
    """Load structural outline templates (McKinsey-style skeletons) from disk for report shaping."""
    try:
        text = read_file(STYLE_SKELETONS_PATH).strip()
    except Exception as e:
        return json.dumps(
            {"error": {"type": e.__class__.__name__, "message": str(e)}, "skeletons": ""},
            ensure_ascii=False,
        )
    preview = text[:4000] + ("..." if len(text) > 4000 else "")
    return json.dumps(
        {"skeletons": text, "char_count": len(text), "preview": preview},
        ensure_ascii=False,
    )


@tool
def generate_business_report() -> str:
    """Generate the final English Markdown business strategy report from pipeline state and prior tool results.

    Call this AFTER retrieve_mckinsey_knowledge and optionally search_arxiv_papers and load_report_style_skeletons.
    The orchestrator compiles metrics, EDA, RAG context, and skeletons into the report.
    """
    return json.dumps(
        {
            "status": "finalize_requested",
            "message": "Report drafting is executed by the graph handler using GlobalState.",
        },
        ensure_ascii=False,
    )


bat_tools = [
    retrieve_mckinsey_knowledge,
    search_arxiv_papers,
    load_report_style_skeletons,
    generate_business_report,
]
