"""
Lightweight Manager RAP (retrieval-augmented planning) layer.

Analogous to AutoML-style retrieve_knowledge, without PDF ingestion or BM25.

- One LLM call builds ``user_requirement_summary`` plus structured retrieval slots (task/domain/area);
  optional Cleaning hints are injected.
- Retrieval pulls summary-level snippets only: arXiv abstracts, DuckDuckGo snippets, optional local
  PapersWithCode JSON, optional Kaggle kernel metadata.
- One LLM call fuses snippets into planner-ready prose.

Dependencies: ``arxiv`` is required; optional ``duckduckgo-search``, ``kaggle`` + credentials.
PapersWithCode participates only when ``datasets.json`` and ``evaluation-tables.json`` exist under
``config.PWC_DATA_DIR``; otherwise that source behaves as disabled.

----
Data flow (with ``manager.py``)

``user_goal``, ``df_preview``, ``cleaning_hints``
    → summarize_and_extract_slots → (user_requirement_summary, slots)
    → retrieve_and_fuse_rap_context(summary, slots, user_goal) → rap_context string
    → manager._generate_global_plan(..., rap_context, user_requirement_summary=...)
    → ``global_plan`` (JSON plus meta ``_rap_context`` / ``_user_requirement_summary``)

This module does not touch LangGraph ``GraphState``. ``manager_node`` calls it locally and merges
results into ``global_plan``.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from config import ARXIV_MAX_RESULTS, PWC_DATA_DIR, RAP_ENABLED, RAP_FUSE_MAX_CHARS, RAP_KAGGLE_ENABLED, RAP_PWC_ENABLED, RAP_WEB_ENABLED, WEB_MAX_RESULTS
from core.llm import call_llm
from core.logger import get_agent_logger
from tools.arxiv_tool import search_arxiv

logger = get_agent_logger("manager")

# Summary vs slots: both come from the same LLM JSON. ``user_requirement_summary`` is narrative text for humans
# and downstream prompts (_fuse_sources, planning). ``slots`` holds short phrases for keyword-style backends
# (ArXiv query strings, PWC text filter, Kaggle search). ``df_preview`` is only input context to that LLM.


def generate_arxiv_search_queries(user_goal: str, df_preview: str) -> list[str]:
    """Build ArXiv search queries from ``user_goal`` and ``df_preview`` (same slots path as summarize; use ``build_rap_context`` internally to skip a redundant LLM call)."""
    _, slots = summarize_and_extract_slots(user_goal, df_preview, cleaning_hints="")
    return _queries_from_slots(slots, user_goal)


_SUMMARY_REQUIRED_KEYS = ("downstream_task", "application_domain", "area")


def _faithfulness_warning_count(
    cleaning_hints: str,
    summary: str,
) -> int:
    """Lightweight signal: hint clauses that do not appear (substring) in the summary."""
    hints = (cleaning_hints or "").strip()
    if not hints:
        return 0
    summary_lower = summary.lower()
    n = 0
    for chunk in re.split(r"[;\n]+", hints):
        chunk = chunk.strip().lower()
        if len(chunk) >= 6 and chunk not in summary_lower:
            n += 1
    return min(n, 12)


def _summarize_slots_core(
    user_goal: str,
    df_preview: str,
    cleaning_hints: str,
    dataset_description: str = "",
) -> tuple[str, dict[str, str], str | None, bool]:
    """
    Returns ``(summary, slots, raw_reply_or_none, json_parse_ok)``.
    ``json_parse_ok`` is False on empty LLM reply or JSON parse failure (fallback path).
    """
    hints_block = ""
    if cleaning_hints and cleaning_hints.strip():
        hints_block = f"""
Additional cleaning / contract hints from the team (must be reflected in the summary and kept consistent for retrieval):
{cleaning_hints.strip()}
"""
    desc_block = ""
    if dataset_description and dataset_description.strip():
        desc_block = f"""
Dataset description (authoritative; align the summary with this context when consistent with the data preview):
{dataset_description.strip()}
"""
    sys_prompt = "You are a senior data science project manager. Output only valid JSON, no markdown."
    user_prompt = f"""
User goal:
{user_goal}

Data preview:
{df_preview}
{desc_block}
{hints_block}

Return a single JSON object with exactly these string fields:
- "user_requirement_summary": one concise paragraph (English or Chinese matching the user goal) describing the ML task, data, and objectives; if cleaning hints were given, weave them in as constraints.
- "downstream_task": short phrase for search (e.g. churn prediction, demand forecasting).
- "application_domain": short phrase (e.g. telecom, e-commerce).
- "area": ML angle for search (e.g. tabular classification, gradient boosting).

Example: {{"user_requirement_summary":"...","downstream_task":"churn prediction","application_domain":"telecom","area":"tabular supervised learning"}}
"""
    reply = call_llm(sys_prompt, user_prompt, agent_name="manager_summary")
    default_slots = {
        "downstream_task": user_goal[:120],
        "application_domain": "general",
        "area": "machine learning",
    }
    if not reply:
        summary = user_goal if not cleaning_hints else f"{user_goal}\nConstraints: {cleaning_hints}"
        return summary.strip(), default_slots, None, False

    try:
        match = re.search(r"\{[^{}]*\}", reply, re.DOTALL)
        if not match:
            match = re.search(r"\{.*\}", reply, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            summary = str(obj.get("user_requirement_summary", user_goal)).strip()
            slots = {
                "downstream_task": str(obj.get("downstream_task", default_slots["downstream_task"])).strip(),
                "application_domain": str(obj.get("application_domain", default_slots["application_domain"])).strip(),
                "area": str(obj.get("area", default_slots["area"])).strip(),
            }
            return (summary or user_goal), slots, reply, True
    except (json.JSONDecodeError, TypeError):
        pass

    summary = user_goal if not cleaning_hints else f"{user_goal}\nConstraints: {cleaning_hints}"
    return summary.strip(), default_slots, reply if reply else None, False


def summarize_and_extract_slots(
    user_goal: str,
    df_preview: str,
    cleaning_hints: str,
    dataset_description: str = "",
) -> tuple[str, dict[str, str]]:
    """Return ``user_requirement_summary`` and retrieval slots ``downstream_task``, ``application_domain``, ``area``.

    cleaning_hints — optional teammate/user cleaning rules wired into summary for retrieval alignment.
    dataset_description — optional ``--desc`` prose injected into the summary LLM.
    """
    summary, slots, _, _ = _summarize_slots_core(
        user_goal, df_preview, cleaning_hints, dataset_description
    )
    return summary, slots


def summarize_and_extract_slots_with_meta(
    user_goal: str,
    df_preview: str,
    cleaning_hints: str,
    dataset_description: str = "",
) -> tuple[str, dict[str, str], dict]:
    """
    Same as ``summarize_and_extract_slots`` but returns evaluation metadata for
    ``global_plan['_manager_summary_eval']`` (prompt-iteration / diagnostics).
    """
    summary, slots, _raw, json_parse_ok = _summarize_slots_core(
        user_goal, df_preview, cleaning_hints, dataset_description
    )
    field_values = [summary] + [str(slots.get(k, "")).strip() for k in _SUMMARY_REQUIRED_KEYS]
    nonempty = sum(1 for v in field_values if v)
    empty_rate = 1.0 - (nonempty / len(field_values)) if field_values else 1.0
    required_fields_ok = json_parse_ok and nonempty == len(field_values)
    meta = {
        "json_parse_ok": json_parse_ok,
        "required_fields_ok": required_fields_ok,
        "empty_rate": float(empty_rate),
        "faithfulness_warning_count": _faithfulness_warning_count(cleaning_hints, summary),
    }
    return summary, slots, meta


def _queries_from_slots(slots: dict[str, str], user_goal: str) -> list[str]:
    t = slots.get("downstream_task", "").strip()
    d = slots.get("application_domain", "").strip()
    a = slots.get("area", "").strip()
    q1 = f"{t} {d}".strip()
    q2 = f"{a} {t}".strip()
    out: list[str] = []
    if q1:
        out.append(q1)
    if q2 and q2 not in out:
        out.append(q2)
    return out[:2] if out else [f"{user_goal} machine learning"]


def _format_arxiv_context(queries: list[str], max_per_query: int | None) -> str:
    max_per_query = max_per_query or ARXIV_MAX_RESULTS
    lines: list[str] = []
    seen: set[str] = set()
    for q in queries:
        for doc in search_arxiv(q, max_results=max_per_query):
            title = doc.get("title", "N/A")
            key = title
            if key in seen:
                continue
            seen.add(key)
            summ = (doc.get("summary") or "")[:600].replace("\n", " ")
            lines.append(f"- {title}: {summ}")
    return "\n".join(lines) if lines else ""


def _retrieve_web_light(user_requirement_summary: str) -> str:
    if not RAP_WEB_ENABLED:
        return ""
    query_reply = call_llm(
        "You generate short web search queries for ML engineering. Reply with a single line, at most 12 words, no quotes.",
        f"Problem summary:\n{user_requirement_summary}\n\nGive one English search query.",
        agent_name="manager_summary",
    )
    search_query = (query_reply or user_requirement_summary)[:200].strip()
    try:
        from duckduckgo_search import DDGS
        # Web search uses DuckDuckGo (`duckduckgo-search`) — lightweight, usually no paid Google Search API key.
        rows: list[str] = []
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=WEB_MAX_RESULTS):
                title = r.get("title") or ""
                body = (r.get("body") or "")[:400]
                href = r.get("href") or ""
                if title or body:
                    rows.append(f"- {title} | {body} | {href}")
        return "\n".join(rows)
    except ImportError:
        logger.warning(
            "duckduckgo-search not installed; skipping web retrieval. pip install duckduckgo-search"
        )
        return ""
    except Exception as e:
        logger.warning("Web retrieval failed: %s", e)
        return ""


def _retrieve_pwc_light(slots: dict[str, str], user_requirement_summary: str, top_k: int = 8) -> str:
    """Optional local PapersWithCode mirror: missing directory or JSON files yields empty text (non-fatal)."""
    if not RAP_PWC_ENABLED:
        return ""
    base = Path(PWC_DATA_DIR)
    ds_path = base / "datasets.json"
    tb_path = base / "evaluation-tables.json"
    if not ds_path.is_file() or not tb_path.is_file():
        logger.info("PapersWithCode local cache missing or incomplete; skipping (path=%s)", base)
        return ""

    user_task = slots.get("downstream_task", "").lower().replace("-", " ").replace("_", " ")
    user_area = slots.get("area", "").lower().replace("-", " ").replace("_", " ")

    try:
        datasets = json.loads(ds_path.read_text(encoding="utf-8"))
        datasets = [x for x in datasets if isinstance(x, dict) and len(x.get("data_loaders") or []) > 0]
        loaders: list[str] = []
        for dataset in datasets:
            desc = (dataset.get("description") or "").lower()
            tasks = [str(t.get("task", "")).lower() for t in (dataset.get("tasks") or []) if isinstance(t, dict)]
            if user_task and (user_task in desc or user_task in tasks or user_area in desc):
                name = dataset.get("name", "")
                loaders.append(
                    f"DATASET: {name}\nDESC: {dataset.get('description', '')[:500]}\nTASKS: {tasks[:5]}\nLOADERS: {dataset.get('data_loaders', [])[:3]}"
                )

        tables = json.loads(tb_path.read_text(encoding="utf-8"))
        tables = [t for t in tables if isinstance(t, dict) and len(t.get("datasets") or []) > 0]
        bench: list[str] = []
        for table in tables:
            desc = (table.get("description") or "").lower()
            task = str(table.get("task", "")).lower()
            cats = [str(c).lower() for c in (table.get("categories") or [])]
            if user_task and (
                user_task in desc
                or user_task == task
                or user_area in desc
                or any(user_area in c for c in cats)
            ):
                bench.append(
                    f"TASK: {table.get('task')}\nDESC: {table.get('description', '')[:400]}\nDATASETS: {str(table.get('datasets'))[:800]}"
                )

        if not loaders and not bench:
            return ""

        if len(loaders) > top_k:
            loaders = random.sample(loaders, top_k)
        if len(bench) > top_k:
            bench = random.sample(bench, top_k)

        blob = "\n---\n".join(loaders + bench)
        return blob[: RAP_FUSE_MAX_CHARS]
    except Exception as e:
        logger.warning("PapersWithCode retrieval failed: %s", e)
        return ""


def _retrieve_kaggle_light(slots: dict[str, str], top_k: int = 6) -> str:
    if not RAP_KAGGLE_ENABLED:
        return ""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.warning(
            "`kaggle` package not installed; skipping Kaggle retrieval. pip install kaggle"
        )
        return ""

    task = slots.get("downstream_task", "").replace("-", " ").replace("_", " ").strip()
    domain = slots.get("application_domain", "").replace("-", " ").replace("_", " ").strip()
    search = f"{task} {domain}".strip() or "machine learning"

    try:
        api = KaggleApi()
        api.authenticate()
        kernels = api.kernels_list(search=search, sort_by="relevance", page=1, page_size=top_k)
        lines: list[str] = []
        for k in kernels:
            title = getattr(k, "title", None) or str(k)
            ref = getattr(k, "ref", "")
            lines.append(f"- {title} ({ref})")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(
            "Kaggle retrieval skipped (~/.kaggle/kaggle.json credentials required): %s", e
        )
        return ""


def _fuse_sources(
    user_requirement_summary: str,
    parts: dict[str, str],
) -> str:
    """Merge per-source excerpts into fused planning notes (single LLM pass)."""
    non_empty = {k: v for k, v in parts.items() if v and v.strip()}
    if not non_empty:
        return ""

    if len(non_empty) == 1:
        only = next(iter(non_empty.values()))
        if len(only) <= 2500:
            return only.strip()

    blocks = []
    for name, text in non_empty.items():
        clipped = text[: RAP_FUSE_MAX_CHARS]
        blocks.append(f"### {name}\n{clipped}")

    bundle = "\n\n".join(blocks)
    sys_prompt = (
        "You are an ML/AI research consultant. Merge the following noisy snippets into one coherent list of "
        "actionable suggestions for downstream planning (data cleaning, features, modeling). "
        "No code. Prefer concrete method names and task-relevant practices. Use the same language as the user summary when possible."
    )
    user_prompt = f"""
User requirement summary:
{user_requirement_summary}

Sources:
{bundle}

Produce a bullet list of key insights (5–12 bullets) the planning agent can use.
"""
    fused = call_llm(sys_prompt, user_prompt, agent_name="manager_fusion")
    if fused and fused.strip():
        return fused.strip()
    return bundle[:6000]


def _fusion_health_meta(rap_context: str) -> dict:
    """Derived metrics on fused RAP text (manager prompt-eval contract)."""
    text = (rap_context or "").strip()
    empty_rate = 0.0 if text else 1.0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    repetition_rate = 0.0
    if len(lines) > 1:
        repetition_rate = max(0.0, (len(lines) - len(set(lines))) / len(lines))
    lower = text.lower()
    code_leakage_flag = ("```" in text) or ("import " in lower and "def " in lower)
    return {
        "empty_rate": float(empty_rate),
        "repetition_rate": float(repetition_rate),
        "code_leakage_flag": bool(code_leakage_flag),
    }


def _retrieve_and_fuse_rap_context_core(
    user_requirement_summary: str,
    slots: dict[str, str],
    user_goal: str,
) -> tuple[str, dict]:
    """
    Returns ``(rap_context, gather_stats)`` where ``gather_stats`` holds raw
    retrieval sizes and ``rap_enabled`` (for eval); health fields are added in
    ``retrieve_and_fuse_rap_context_with_meta``.
    """
    if not RAP_ENABLED:
        logger.info("RAP is disabled by RAP_ENABLED=false. Skipping external retrieval.")
        text = (
            "No external retrieval results; RAP is disabled, so planning uses only "
            "the user goal and data preview."
        )
        return text, {
            "rap_enabled": False,
            "arxiv_raw_chars": 0,
            "web_raw_chars": 0,
            "pwc_raw_chars": 0,
            "kaggle_raw_chars": 0,
        }

    queries = _queries_from_slots(slots, user_goal)

    arxiv_raw = _format_arxiv_context(queries, ARXIV_MAX_RESULTS)
    web_raw = _retrieve_web_light(user_requirement_summary)
    pwc_raw = _retrieve_pwc_light(slots, user_requirement_summary)
    kaggle_raw = _retrieve_kaggle_light(slots)

    parts = {
        "arXiv (abstracts)": arxiv_raw,
        "Web search snippets": web_raw,
        "PapersWithCode (local)": pwc_raw,
        "Kaggle kernels (titles)": kaggle_raw,
    }
    rap_context = _fuse_sources(user_requirement_summary, parts)
    if not rap_context.strip():
        rap_context = arxiv_raw or (
            "No external retrieval results; planning uses only the user goal and data preview."
        )

    gather = {
        "rap_enabled": True,
        "arxiv_raw_chars": len(arxiv_raw),
        "web_raw_chars": len(web_raw),
        "pwc_raw_chars": len(pwc_raw),
        "kaggle_raw_chars": len(kaggle_raw),
    }
    logger.info(
        "RAP gather | raw_chars: arxiv=%d web=%d pwc=%d kaggle=%d | flags: WEB=%s PWC=%s KAGGLE=%s | fused_chars=%d",
        gather["arxiv_raw_chars"],
        gather["web_raw_chars"],
        gather["pwc_raw_chars"],
        gather["kaggle_raw_chars"],
        RAP_WEB_ENABLED,
        RAP_PWC_ENABLED,
        RAP_KAGGLE_ENABLED,
        len(rap_context),
    )
    return rap_context, gather


def retrieve_and_fuse_rap_context(
    user_requirement_summary: str,
    slots: dict[str, str],
    user_goal: str,
) -> str:
    """
    RAP step 2: multi-source retrieval + fusion from Step-1b summary and slots.
    Decoupled from requirement understanding for easier testing or strategy swaps.
    """
    rap_context, _gather = _retrieve_and_fuse_rap_context_core(
        user_requirement_summary, slots, user_goal
    )
    return rap_context


def retrieve_and_fuse_rap_context_with_meta(
    user_requirement_summary: str,
    slots: dict[str, str],
    user_goal: str,
) -> tuple[str, dict]:
    """Like ``retrieve_and_fuse_rap_context`` but returns fusion eval metadata."""
    rap_context, gather = _retrieve_and_fuse_rap_context_core(
        user_requirement_summary, slots, user_goal
    )
    health = _fusion_health_meta(rap_context)
    return rap_context, {**gather, **health}


def build_rap_context(
    user_goal: str,
    df_preview: str,
    *,
    cleaning_hints: str = "",
) -> tuple[str, str]:
    """
    Convenience: run Step 1b (summary + slots) then Step 2 (retrieve + fuse).

    Returns:
        user_requirement_summary — for logs or follow-up retrieval.
        rap_context — fused string for `_generate_global_plan`.
    """
    summary, slots = summarize_and_extract_slots(user_goal, df_preview, cleaning_hints)
    rap_context = retrieve_and_fuse_rap_context(summary, slots, user_goal)
    return summary, rap_context
