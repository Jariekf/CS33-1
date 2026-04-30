"""
Lightweight Manager RAP retrieval layer, aligned with AutoML-style
retrieve_knowledge without PDF parsing or BM25.

- One LLM call generates `user_requirement_summary` plus structured retrieval
  slots such as task, domain, and area.
- Optional `cleaning_hints` and `dataset_description` shape this summary step.
- Multiple sources collect summary-level text only: arXiv abstracts,
  DuckDuckGo snippets, local PapersWithCode JSON, and Kaggle kernel metadata.
- One LLM call fuses the sources into planning-ready knowledge text.

Dependencies: arxiv is available; `duckduckgo-search`, `kaggle`, and Kaggle API
credentials are optional. PWC participates only when `datasets.json` and
`evaluation-tables.json` exist under `config.PWC_DATA_DIR`; an empty directory
is equivalent to disabling that source.

Data flow connected to manager.py:
  user_goal, df_preview, cleaning_hints, dataset_description
      -> summarize_and_extract_slots -> (user_requirement_summary, slots)
      -> retrieve_and_fuse_rap_context(summary, slots, user_goal) -> rap_context: str
      -> manager._generate_global_plan(..., rap_context,
           user_requirement_summary=..., dataset_description=...)
      -> global_plan with planning metadata such as _rap_context /
         _user_requirement_summary

This module does not read or write LangGraph GraphState. It is called by
manager_node, and the returned values remain local to manager before being
written into global_plan.
"""
from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path

from config import ARXIV_MAX_RESULTS, PWC_DATA_DIR, RAP_ENABLED, RAP_FUSE_MAX_CHARS, RAP_KAGGLE_ENABLED, RAP_PWC_ENABLED, RAP_WEB_ENABLED, WEB_MAX_RESULTS
from core.llm import call_llm
from core.logger import get_agent_logger
from tools.arxiv_tool import search_arxiv

logger = get_agent_logger("manager")
_SUMMARY_REQUIRED_FIELDS = (
    "user_requirement_summary",
    "downstream_task",
    "application_domain",
    "area",
)
_KEYWORD_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "using",
    "user", "goal", "data", "preview", "need", "model", "predict",
}

"""
Summary vs slots:
- user_requirement_sum is natural language for humans and downstream LLMs.
- slots are short structured phrases used by keyword-style retrieval
  such as arXiv queries, PWC text filtering, and Kaggle kernel search.
Both come from the same LLM JSON. The summary handles narrative alignment,
while slots provide a retrieval interface. df_preview is input context, not a slot.
"""
def generate_arxiv_search_queries(user_goal: str, df_preview: str) -> list[str]:
    """Generate arXiv search queries from the goal and data preview."""
    _, slots = summarize_and_extract_slots(user_goal, df_preview, cleaning_hints="")
    return _queries_from_slots(slots, user_goal)


def summarize_and_extract_slots(
    user_goal: str,
    df_preview: str,
    cleaning_hints: str,
    dataset_description: str = "",
) -> tuple[str, dict[str, str]]:
    """
    Generate user_requirement_summary and extract downstream_task, application_domain, and area.

    `cleaning_hints` narrows the summary toward cleaning constraints.
    `dataset_description` provides authoritative dataset context when available.
    """
    summary, slots, _ = summarize_and_extract_slots_with_meta(
        user_goal=user_goal,
        df_preview=df_preview,
        cleaning_hints=cleaning_hints,
        dataset_description=dataset_description,
    )
    return summary, slots


def summarize_and_extract_slots_with_meta(
    user_goal: str,
    df_preview: str,
    cleaning_hints: str,
    dataset_description: str = "",
) -> tuple[str, dict[str, str], dict]:
    """
    Generate summary/slots with lightweight health and faithfulness metrics.

    This is the only step that consumes raw `cleaning_hints`; later planning
    sees the resulting summary instead. `dataset_description` is treated as
    authoritative context when present.
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
Authoritative dataset documentation (treat as ground truth; prefer this over guesses inferred from column names):
{dataset_description.strip()}
"""
    sys_prompt = "You are a senior data science project manager. Output only valid JSON, no markdown."
    user_prompt = f"""
User goal:
{user_goal}

Data preview:
{df_preview}
{desc_block}{hints_block}

Return a single JSON object with exactly these string fields:
- "user_requirement_summary": one concise paragraph (English or Chinese matching the user goal) describing the ML task, data, and objectives; if cleaning hints were given, weave them in as constraints.
- "downstream_task": short phrase for search (e.g. churn prediction, demand forecasting).
- "application_domain": short phrase (e.g. telecom, e-commerce).
- "area": ML angle for search (e.g. tabular classification, gradient boosting).

Example: {{"user_requirement_summary":"...","downstream_task":"churn prediction","application_domain":"telecom","area":"tabular supervised learning"}}
"""
    reply = call_llm(sys_prompt, user_prompt, agent_name="manager_summary")
    return _parse_summary_and_slots(
        reply=reply,
        user_goal=user_goal,
        df_preview=df_preview,
        cleaning_hints=cleaning_hints,
    )


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
        """DuckDuckGo search usually works without an official Google Search API key."""

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
        logger.warning("duckduckgo-search is not installed; skipping web retrieval. pip install duckduckgo-search")
        return ""
    except Exception as e:
        logger.warning("Web retrieval failed: %s", e)
        return ""


def _retrieve_pwc_light(slots: dict[str, str], user_requirement_summary: str, top_k: int = 8) -> str:
    """Use optional local PWC data; missing JSON files return an empty string without error."""
    if not RAP_PWC_ENABLED:
        return ""
    base = Path(PWC_DATA_DIR)
    ds_path = base / "datasets.json"
    tb_path = base / "evaluation-tables.json"
    if not ds_path.is_file() or not tb_path.is_file():
        logger.info("PapersWithCode local data is unavailable; skipping path: %s", base)
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
        logger.warning("PWC lightweight retrieval failed: %s", e)
        return ""


def _retrieve_kaggle_light(slots: dict[str, str], top_k: int = 6) -> str:
    if not RAP_KAGGLE_ENABLED:
        return ""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.warning("kaggle package is not installed; skipping Kaggle retrieval. pip install kaggle")
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
        logger.warning("Kaggle retrieval skipped; credentials such as ~/.kaggle/kaggle.json may be required: %s", e)
        return ""


def _fuse_sources(
    user_requirement_summary: str,
    parts: dict[str, str],
) -> str:
    """Fuse raw source snippets into planning-ready knowledge with one LLM call."""
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

Produce a bullet list of key insights (5-12 bullets) the planning agent can use.
"""
    fused = call_llm(sys_prompt, user_prompt, agent_name="manager_fusion")
    if fused and fused.strip():
        return fused.strip()
    return bundle[:6000]


def _fuse_sources_with_meta(
    user_requirement_summary: str,
    parts: dict[str, str],
) -> tuple[str, dict]:
    fused = _fuse_sources(user_requirement_summary, parts)
    return fused, _build_fusion_eval_meta(
        fused,
        rap_enabled=True,
        arxiv_raw_chars=len(parts.get("arXiv (abstracts)", "")),
        web_raw_chars=len(parts.get("Web search snippets", "")),
        pwc_raw_chars=len(parts.get("PapersWithCode (local)", "")),
        kaggle_raw_chars=len(parts.get("Kaggle kernels (titles)", "")),
    )


def retrieve_and_fuse_rap_context(
    user_requirement_summary: str,
    slots: dict[str, str],
    user_goal: str,
) -> str:
    """
    Step 2 (RAP): use the Step 1b summary and slots for multi-source retrieval,
    then fuse the results into one text block.
    """
    rap_context, _ = retrieve_and_fuse_rap_context_with_meta(
        user_requirement_summary=user_requirement_summary,
        slots=slots,
        user_goal=user_goal,
    )
    return rap_context


def retrieve_and_fuse_rap_context_with_meta(
    user_requirement_summary: str,
    slots: dict[str, str],
    user_goal: str,
) -> tuple[str, dict]:
    """Step 2 RAP with lightweight health metrics for standalone evaluation."""
    if not RAP_ENABLED:
        logger.info("RAP is disabled by RAP_ENABLED=false. Skipping external retrieval.")
        rap_context = "No external retrieval results; RAP is disabled, so planning uses only the user goal and data preview."
        return rap_context, _build_fusion_eval_meta(rap_context, rap_enabled=False)

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
    rap_context, eval_meta = _fuse_sources_with_meta(user_requirement_summary, parts)
    if not rap_context.strip():
        rap_context = arxiv_raw or "No external retrieval results; planning uses only the user goal and data preview."
        eval_meta = _build_fusion_eval_meta(
            rap_context,
            rap_enabled=True,
            arxiv_raw_chars=len(arxiv_raw),
            web_raw_chars=len(web_raw),
            pwc_raw_chars=len(pwc_raw),
            kaggle_raw_chars=len(kaggle_raw),
        )

    logger.info(
        "RAP gather | raw_chars: arxiv=%d web=%d pwc=%d kaggle=%d | flags: WEB=%s PWC=%s KAGGLE=%s | fused_chars=%d",
        len(arxiv_raw),
        len(web_raw),
        len(pwc_raw),
        len(kaggle_raw),
        RAP_WEB_ENABLED,
        RAP_PWC_ENABLED,
        RAP_KAGGLE_ENABLED,
        len(rap_context),
    )

    return rap_context, eval_meta


def build_rap_context(
    user_goal: str,
    df_preview: str,
    *,
    cleaning_hints: str = "",
    dataset_description: str = "",
) -> tuple[str, str]:
    """
    Convenience wrapper that runs Step 1b (summary + slots) and Step 2 (retrieval + fusion).

    Returns:
        user_requirement_summary: reusable by logs, later nodes, or another retrieval pass.
        rap_context: fused RAP text that can be injected directly into
            ``manager._generate_global_plan``.
    """
    summary, slots = summarize_and_extract_slots(
        user_goal, df_preview, cleaning_hints, dataset_description=dataset_description
    )
    rap_context = retrieve_and_fuse_rap_context(summary, slots, user_goal)
    return summary, rap_context


def _parse_summary_and_slots(
    *,
    reply: str | None,
    user_goal: str,
    df_preview: str,
    cleaning_hints: str,
) -> tuple[str, dict[str, str], dict]:
    default_slots = {
        "downstream_task": user_goal[:120],
        "application_domain": "general",
        "area": "machine learning",
    }
    fallback_summary = user_goal if not cleaning_hints else f"{user_goal}\nConstraints: {cleaning_hints}"
    parsed_obj: dict | None = None
    json_parse_ok = False

    if reply:
        try:
            match = re.search(r"\{[^{}]*\}", reply, re.DOTALL)
            if not match:
                match = re.search(r"\{.*\}", reply, re.DOTALL)
            if match:
                maybe_obj = json.loads(match.group(0))
                if isinstance(maybe_obj, dict):
                    parsed_obj = maybe_obj
                    json_parse_ok = True
        except (json.JSONDecodeError, TypeError):
            parsed_obj = None

    raw_fields = {}
    for key in _SUMMARY_REQUIRED_FIELDS:
        value = ""
        if parsed_obj is not None:
            value = str(parsed_obj.get(key, "")).strip()
        raw_fields[key] = value
    empty_count = sum(1 for v in raw_fields.values() if not v)
    empty_rate = round(empty_count / float(len(_SUMMARY_REQUIRED_FIELDS)), 4)
    required_fields_ok = empty_count == 0

    summary = raw_fields["user_requirement_summary"] or fallback_summary
    slots = {
        "downstream_task": raw_fields["downstream_task"] or default_slots["downstream_task"],
        "application_domain": raw_fields["application_domain"] or default_slots["application_domain"],
        "area": raw_fields["area"] or default_slots["area"],
    }

    faithfulness = _build_faithfulness_warnings(
        summary=summary,
        user_goal=user_goal,
        df_preview=df_preview,
        cleaning_hints=cleaning_hints,
    )
    meta = {
        "json_parse_ok": json_parse_ok,
        "required_fields_ok": required_fields_ok,
        "empty_rate": empty_rate,
        "raw_reply_preview": (reply or "")[:1000],
        **faithfulness,
    }
    return summary.strip(), slots, meta


def _tokenize_keywords(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", text or "")
        if len(token) >= 3
    }
    return {token for token in tokens if token not in _KEYWORD_STOPWORDS}


def _extract_preview_columns(df_preview: str) -> set[str]:
    match = re.search(r"Columns\s*\(\d+\):\s*(\[[^\]]*\])", df_preview or "")
    if not match:
        return set()
    try:
        cols = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return set()
    if not isinstance(cols, list):
        return set()
    return {str(col).strip() for col in cols if str(col).strip()}


def _build_faithfulness_warnings(
    *,
    summary: str,
    user_goal: str,
    df_preview: str,
    cleaning_hints: str,
) -> dict:
    goal_keywords = _tokenize_keywords(user_goal)
    summary_keywords = _tokenize_keywords(summary)
    goal_keyword_hit = bool(goal_keywords.intersection(summary_keywords)) if goal_keywords else True

    preview_columns = _extract_preview_columns(df_preview)
    explicit_refs = re.findall(r"[`'\"]([A-Za-z_][A-Za-z0-9_]*)[`'\"]", summary or "")
    unknown_refs = []
    if preview_columns:
        lowered = {col.lower() for col in preview_columns}
        unknown_refs = [ref for ref in explicit_refs if ref.lower() not in lowered]
    column_reference_ok = len(unknown_refs) == 0

    hints_coverage_ok = True
    if cleaning_hints.strip():
        hints_keywords = _tokenize_keywords(cleaning_hints)
        hints_coverage_ok = bool(hints_keywords.intersection(summary_keywords)) if hints_keywords else True

    warning_count = sum(
        1
        for status in (goal_keyword_hit, column_reference_ok, hints_coverage_ok)
        if not status
    )
    return {
        "faithfulness_goal_keyword_hit": goal_keyword_hit,
        "faithfulness_column_reference_ok": column_reference_ok,
        "faithfulness_hints_coverage_ok": hints_coverage_ok,
        "faithfulness_warning_count": warning_count,
    }


def _build_fusion_eval_meta(
    text: str,
    *,
    rap_enabled: bool,
    arxiv_raw_chars: int = 0,
    web_raw_chars: int = 0,
    pwc_raw_chars: int = 0,
    kaggle_raw_chars: int = 0,
) -> dict:
    content = (text or "").strip()
    if not content:
        return {
            "empty_rate": 1.0,
            "repetition_rate": 0.0,
            "code_leakage_flag": False,
            "raw_reply_preview": "",
            "rap_enabled": rap_enabled,
            "arxiv_raw_chars": arxiv_raw_chars,
            "web_raw_chars": web_raw_chars,
            "pwc_raw_chars": pwc_raw_chars,
            "kaggle_raw_chars": kaggle_raw_chars,
        }

    lines = [line.strip().lower() for line in content.splitlines() if line.strip()]
    repetition_rate = 0.0
    if lines:
        repetition_rate = round((len(lines) - len(set(lines))) / float(len(lines)), 4)
    leakage_pattern = re.compile(r"```|^\s*(import|from|def|class)\b", re.IGNORECASE | re.MULTILINE)
    code_leakage_flag = bool(leakage_pattern.search(content))
    return {
        "empty_rate": 0.0,
        "repetition_rate": repetition_rate,
        "code_leakage_flag": code_leakage_flag,
        "raw_reply_preview": content[:1000],
        "rap_enabled": rap_enabled,
        "arxiv_raw_chars": arxiv_raw_chars,
        "web_raw_chars": web_raw_chars,
        "pwc_raw_chars": pwc_raw_chars,
        "kaggle_raw_chars": kaggle_raw_chars,
    }
