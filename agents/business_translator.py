"""
Business Action Translator Agent

Responsibilities:
  1. Pull core metrics and feature importance from the model stage's on-disk artefacts.
  2. Retrieve McKinsey-flavoured corpus material (``mckinsey_kb``) as primary external context.
     Buckets by ``source_type``: mckinsey / equity_research / textbook.
     When hit count is low, fall back to ArXiv search.
  3. Blend internal data signals with external snippets and draft a business insight
     report following McKinsey-style pyramid structure.
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

# Where Model Selection unconditionally drops test-set predictions.
# Mirrors ``agents.model_selection.TEST_PREDICTIONS_PATH``; we re-declare the
# string here instead of importing to avoid pulling sklearn / pandas at module
# import time for environments that only run business_translator.
TEST_PREDICTIONS_FILENAME = "test_predictions.csv"

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
    """LangGraph node for business report generation."""
    user_goal = state["user_goal"]
    memory = state.get("memory", [])
    dataset_description = (state.get("dataset_description") or "").strip()
    eda_summary = (state.get("eda_summary") or "").strip()
    eda_charts: list[dict] = state.get("eda_charts") or []
    logger.info("Business Translator node started.")

    # ------------------------------------------------------------------
    # Step 1: extract model artefacts (lazy reads where needed)
    # ------------------------------------------------------------------
    fi_path = os.path.join(WORKSPACE_DIR, "feature_importance.json")
    feature_importance = _load_feature_importance(fi_path)
    model_summary = memory[-1] if memory else "No model-stage summary available."
    logger.info(f"Loaded feature importance for {len(feature_importance)} features.")

    # Test-set predictions are written unconditionally by Model Selection (see
    # agents/model_selection.py constraint #12). Surfacing a brief summary
    # gives the report something concrete to say when the test set carried no
    # ground-truth labels and there are no test metrics
    # in the model summary to talk about. When labels were present the summary
    # is still useful as a sanity check on prediction distribution.
    predictions_summary = _summarise_test_predictions(
        os.path.join(WORKSPACE_DIR, TEST_PREDICTIONS_FILENAME)
    )
    if predictions_summary["available"]:
        logger.info(
            "Loaded test predictions summary: %d rows, target_pred_col=%r, has_proba=%s",
            predictions_summary["n_rows"],
            predictions_summary.get("pred_col"),
            predictions_summary.get("has_proba"),
        )
    else:
        logger.info(
            "No test_predictions.csv found at expected path; report will omit prediction distribution block."
        )

    # ------------------------------------------------------------------
    # Step 2: build retrieval queries for RAG
    # ------------------------------------------------------------------
    top_features = list(feature_importance.keys())[:5] if feature_importance else []
    search_queries = _build_rag_queries(user_goal, top_features)
    logger.info(f"Generated {len(search_queries)} RAG queries for external strategy literature.")

    # ------------------------------------------------------------------
    # Step 3: McKinsey corpus RAG retrieval (happy path)
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
    # Step 4: Arxiv fallback (only when local RAG lacks coverage)
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
    # Step 5: load style-outline anchors
    # ------------------------------------------------------------------
    style_skeletons = _load_style_skeletons()

    # ------------------------------------------------------------------
    # Step 6: invoke the LLM to draft the narrative report
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
        eda_charts=eda_charts,
        predictions_summary=predictions_summary,
    )
    logger.info("Business report generation complete.")

    return {"final_business_report": report}


# ======================================================================
# Private helpers
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
    """Ask the LLM to weave goal + strongest features into RAG-ready queries."""
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


def _summarise_test_predictions(path: str) -> dict:
    """
    Compute a small, prompt-friendly summary of the test-predictions artifact.

    Returns a dict with stable keys regardless of whether the file exists:
      - ``available``: bool
      - ``path``: str (absolute or workspace-relative path attempted)
      - ``n_rows``: int (0 when unavailable)
      - ``pred_col``: str (the ``<target>_pred`` column name found, or "")
      - ``has_proba``: bool (binary-classification probability column present)
      - ``proba_col``: str
      - ``positive_rate``: float | None (mean of the proba column when present)
      - ``class_distribution``: dict[str, int] | None (small dict for
        classification ``_pred`` columns, capped to top 10)
      - ``regression_stats``: dict | None (count/mean/std/min/max for numeric
        prediction columns when classification doesn't apply)
      - ``error``: str | None (parse failure reason, surfaced in the prompt)
    """
    summary: dict = {
        "available": False,
        "path": path,
        "n_rows": 0,
        "pred_col": "",
        "has_proba": False,
        "proba_col": "",
        "positive_rate": None,
        "class_distribution": None,
        "regression_stats": None,
        "error": None,
    }
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return summary

    try:
        import pandas as pd  # local import to keep BT lean if unused
    except ImportError as e:
        summary["error"] = f"pandas unavailable: {e}"
        return summary

    try:
        df = pd.read_csv(path)
    except Exception as e:
        summary["error"] = f"failed to parse predictions CSV: {e}"
        return summary

    summary["available"] = True
    summary["n_rows"] = int(len(df))
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    pred_col_key = next(
        (k for k in cols_lower if k.endswith("_pred")),
        "prediction" if "prediction" in cols_lower else "",
    )
    if pred_col_key:
        pred_col = cols_lower[pred_col_key]
        summary["pred_col"] = pred_col
        proba_key = next((k for k in cols_lower if k.endswith("_pred_proba")), "")
        if proba_key:
            proba_col = cols_lower[proba_key]
            summary["has_proba"] = True
            summary["proba_col"] = proba_col
            try:
                summary["positive_rate"] = float(df[proba_col].mean())
            except Exception:
                summary["positive_rate"] = None
        try:
            series = df[pred_col]
            unique_count = int(series.nunique(dropna=True))
            if unique_count <= 20:
                vc = series.value_counts(dropna=False).head(10)
                summary["class_distribution"] = {
                    str(k): int(v) for k, v in vc.items()
                }
            else:
                desc = series.describe()
                summary["regression_stats"] = {
                    "count": int(desc.get("count", 0)),
                    "mean": float(desc.get("mean", 0.0)),
                    "std": float(desc.get("std", 0.0)),
                    "min": float(desc.get("min", 0.0)),
                    "max": float(desc.get("max", 0.0)),
                }
        except Exception:
            pass
    return summary


def _format_predictions_summary_block(summary: dict) -> str:
    """Render the test-predictions summary as a Markdown block for the prompt."""
    if not summary or not summary.get("available"):
        return (
            "(no test_predictions.csv found; do not invent prediction figures. "
            "If the model selection stage did not write predictions, omit any "
            "claims about test-set predictions.)"
        )
    lines = [
        f"- **rows**: {summary.get('n_rows', 0)}",
        f"- **prediction column**: `{summary.get('pred_col') or 'unknown'}`",
    ]
    if summary.get("has_proba"):
        rate = summary.get("positive_rate")
        rate_text = f"{rate:.3f}" if isinstance(rate, (int, float)) else "n/a"
        lines.append(f"- **probability column**: `{summary['proba_col']}`")
        lines.append(f"- **mean predicted positive-class probability**: {rate_text}")
    dist = summary.get("class_distribution")
    if dist:
        items = ", ".join(f"`{k}`={v}" for k, v in dist.items())
        lines.append(f"- **predicted class distribution**: {items}")
    rstats = summary.get("regression_stats")
    if rstats:
        lines.append(
            "- **predicted value stats**: count="
            f"{rstats['count']}, mean={rstats['mean']:.4f}, "
            f"std={rstats['std']:.4f}, min={rstats['min']:.4f}, max={rstats['max']:.4f}"
        )
    return "\n".join(lines)


def _format_eda_charts_block(charts: list[dict]) -> str:
    if not charts:
        return (
            "(no static EDA chart files; generation may have been skipped or failed — "
            "do not invent image paths.)"
        )
    lines: list[str] = []
    for c in charts:
        rel = (c.get("rel_path") or "").strip()
        desc = (c.get("desc") or c.get("name") or "chart").strip()
        lines.append(f"- `{rel}` — {desc}  (when relevant: `![{desc}]({rel})`)")
    return "\n".join(lines)


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
    eda_charts: list[dict] | None = None,
    predictions_summary: dict | None = None,
) -> str:
    """Invoke the LLM to produce the final McKinsey-styled briefing."""
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
    eda_block = eda_summary if eda_summary else "(not provided by EDA stage)"
    eda_charts_block = _format_eda_charts_block(eda_charts or [])
    predictions_block = _format_predictions_summary_block(predictions_summary or {})

    mck_block = _format_mckinsey_bucket("McKinsey Reports", mckinsey_buckets.get("mckinsey", []))
    eq_block = _format_mckinsey_bucket("Equity Research & Strategy", mckinsey_buckets.get("equity_research", []))
    tb_block = _format_mckinsey_bucket("Management Textbook", mckinsey_buckets.get("textbook", []))
    skeletons_block = style_skeletons or "(no style skeletons available; rely on the McKinsey style rules below.)"

    sys_prompt = (
        "You are an elite Data Strategy Consultant writing in McKinsey consulting style. "
        "Apply the Pyramid Principle (top-down, answer-first), open with SCQA (Situation - Complication - Question - Answer), "
        "structure findings into 3-5 MECE themes, and close with concrete sequenced actions. "
        "CRITICAL: be concise; do NOT repeat any sentence, phrase, or idea; do NOT enter generation loops; "
        "ground every claim in either the model evidence or the retrieved external context. "
        "Your output is a Quarto (.qmd) source document: it MUST start with a YAML front matter block "
        "(see user instructions), then Markdown body. Image paths must reference only files listed under EDA chart files."
    )

    user_prompt = f"""
Produce a Business Strategy & Insight Report as a Quarto (.qmd) document: YAML front matter first, then Markdown body, English only.

# Document shell (required)
Your response MUST begin with this YAML front matter (adjust title only if needed):
---
title: "Business Strategy & Insight Report"
format:
  html:
    toc: true
    theme: cosmo
  pdf:
    documentclass: article
---

After the closing `---`, write the report sections in Markdown.

# User Goal
{user_goal}

# Dataset Description (authoritative business context for feature interpretation)
{dataset_description_block}

# EDA Summary (static analysis + chart manifest)
{eda_block}

# EDA chart files (embed when they support a claim; paths relative to project root / Quarto file location)
{eda_charts_block}

Use `![short caption](./eda_figs/<filename>.png)` in **Situation & Complication** and/or **Key Findings** where an image strengthens the narrative (typically 1--3 images). Do not reference images that are not listed above.

# Model Validation Summary
{model_summary}

# Test-Set Predictions Summary
{predictions_block}

When the Model Validation Summary above does NOT contain test-set evaluation metrics (e.g. the test set arrived without ground-truth labels and the model selection stage skipped metric computation), do NOT invent test metrics. Instead anchor evaluation claims in the cross-validation score from the Model Validation Summary, and use the Test-Set Predictions Summary above to characterise the model's behaviour on the held-out set (e.g. predicted class distribution, mean positive-class probability, predicted value range). Be explicit when only CV evidence is available — it is more honest and more useful than fabricated holdout numbers.

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
