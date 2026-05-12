"""Manager RAP: 摘要 + 检索槽位 + arXiv / 可选 Web，再融合为清洗规划参考文本。"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.tools import tool

from core.my_llm import call_llm
from tools.tool_trace import trace_tool_call

_SUMMARY_REQUIRED_KEYS = ("downstream_task", "application_domain", "area")
_RAP_FUSE_MAX_CHARS = int(os.getenv("RAP_FUSE_MAX_CHARS", "6000"))
_RAP_WEB_ENABLED = os.getenv("RAP_WEB_ENABLED", "false").lower() == "true"
_RAP_ENABLED = os.getenv("RAP_ENABLED", "true").lower() == "true"
_WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "3"))


def _dataset_preview(dataset_profile: dict[str, Any] | None) -> str:
    if not dataset_profile:
        return "(dataset profile unavailable)"
    dataset = dataset_profile.get("dataset") or {}
    sample = dataset_profile.get("sample_rows") or []
    preview = {
        "dataset": {
            "rows": dataset.get("rows"),
            "cols": dataset.get("cols"),
            "columns": dataset.get("columns", [])[:40],
            "dtypes": dataset.get("dtypes", {}),
        },
        "missing_values": dataset_profile.get("missing_values", {}),
        "outliers": dataset_profile.get("outliers", {}),
        "type_issues": dataset_profile.get("type_issues", {}),
        "drop_candidates": dataset_profile.get("drop_candidates", {}),
        "sample_rows": sample[:3] if isinstance(sample, list) else [],
    }
    return json.dumps(preview, ensure_ascii=False, indent=2, default=str)[:12000]


def _summary_field_meta(summary: str, slots: dict[str, str]) -> dict[str, Any]:
    field_values = [str(summary).strip()] + [str(slots.get(k, "")).strip() for k in _SUMMARY_REQUIRED_KEYS]
    nonempty = sum(1 for v in field_values if v)
    empty_rate = 1.0 - (nonempty / len(field_values)) if field_values else 1.0
    required_ok = nonempty == len(field_values)
    return {
        "summary_required_fields_ok": required_ok,
        "summary_empty_rate": float(empty_rate),
    }


def _fusion_output_health(rap_context: str) -> dict[str, Any]:
    text = (rap_context or "").strip()
    empty_rate = 0.0 if text else 1.0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    repetition_rate = 0.0
    if len(lines) > 1:
        repetition_rate = max(0.0, (len(lines) - len(set(lines))) / len(lines))
    lower = text.lower()
    code_leakage_flag = ("```" in text) or ("import " in lower and "def " in lower)
    return {
        "fusion_text_empty_rate": float(empty_rate),
        "fusion_line_repetition_rate": float(repetition_rate),
        "fusion_code_leakage_flag": bool(code_leakage_flag),
        "fusion_output_chars": len(text),
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    for pattern in (r"\{[^{}]*\}", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _fallback_summary_and_slots(
    user_request: str,
    task_type: str,
    target_column: str,
) -> tuple[str, dict[str, str]]:
    task = task_type or "supervised learning"
    target = f" target={target_column}" if target_column else ""
    summary = f"{user_request.strip() or 'Run a data science workflow'} ({task}{target})."
    slots = {
        "downstream_task": user_request[:120] or task,
        "application_domain": "general",
        "area": f"tabular {task}".strip(),
    }
    return summary, slots


def _summarize_and_extract_slots(
    user_request: str,
    dataset_profile: dict[str, Any] | None,
    description: str,
    task_type: str,
    target_column: str,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    fallback_summary, fallback_slots = _fallback_summary_and_slots(
        user_request, task_type, target_column
    )
    sys_prompt = "You are a senior data science project manager. Output only valid JSON."
    user_prompt = f"""
User request:
{user_request}

Dataset description:
{description or "(not provided)"}

Task metadata:
task_type={task_type or "unknown"}
target_column={target_column or "(unknown)"}

Dataset profile:
{_dataset_preview(dataset_profile)}

Return one JSON object with exactly these string fields:
- "user_requirement_summary": concise paragraph describing the ML task, data, target, and business goal.
- "downstream_task": short search phrase for the supervised task.
- "application_domain": short search phrase for the business/domain.
- "area": short search phrase for the ML method angle.
"""
    try:
        raw_reply = call_llm(sys_prompt, user_prompt, agent_name="manager_summary")
    except Exception as exc:
        sm = _summary_field_meta(fallback_summary, fallback_slots)
        return fallback_summary, fallback_slots, {
            "summary_json_parse_ok": False,
            "summary_error": f"{exc.__class__.__name__}: {exc}",
            **sm,
        }

    obj = _extract_json_object(raw_reply)
    if not obj:
        sm = _summary_field_meta(fallback_summary, fallback_slots)
        return fallback_summary, fallback_slots, {"summary_json_parse_ok": False, **sm}

    summary = str(obj.get("user_requirement_summary") or fallback_summary).strip()
    slots = {
        key: str(obj.get(key) or fallback_slots[key]).strip()
        for key in _SUMMARY_REQUIRED_KEYS
    }
    sm = _summary_field_meta(summary, slots)
    return summary, slots, {"summary_json_parse_ok": True, **sm}


def _queries_from_slots(slots: dict[str, str], user_request: str) -> list[str]:
    downstream_task = slots.get("downstream_task", "").strip()
    domain = slots.get("application_domain", "").strip()
    area = slots.get("area", "").strip()
    queries = [
        f"{downstream_task} {domain}".strip(),
        f"{area} {downstream_task}".strip(),
    ]
    unique: list[str] = []
    for query in queries:
        if query and query not in unique:
            unique.append(query)
    return unique[:2] if unique else [f"{user_request} machine learning"]


def _search_arxiv_light(query: str) -> list[dict[str, Any]]:
    if os.getenv("STUB_MODE", "false").lower() == "true":
        return [
            {
                "title": "Stub Paper on " + query,
                "summary": "Dummy retrieval result for early pipeline tests.",
                "url": "http://arxiv.org/abs/0000.0000",
            }
        ]
    try:
        import arxiv
    except ImportError:
        return []

    max_results = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [
        {
            "title": paper.title,
            "summary": paper.summary,
            "url": paper.entry_id,
        }
        for paper in client.results(search)
    ]


def _format_arxiv_context(queries: list[str]) -> tuple[str, int, list[str]]:
    lines: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for query in queries:
        try:
            papers = _search_arxiv_light(query)
        except Exception as exc:
            errors.append(f"arxiv:{query}: {exc.__class__.__name__}: {exc}")
            continue
        for paper in papers:
            title = str(paper.get("title") or "Untitled")
            if title in seen:
                continue
            seen.add(title)
            summary = str(paper.get("summary") or "")[:600].replace("\n", " ")
            url = str(paper.get("url") or "")
            lines.append(f"- {title}: {summary} {url}".strip())
    text = "\n".join(lines)
    return text, len(text), errors


def _retrieve_web_light(user_requirement_summary: str) -> tuple[str, int, list[str]]:
    if not _RAP_WEB_ENABLED:
        return "", 0, []
    errors: list[str] = []
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "", 0, ["duckduckgo-search is not installed"]
    try:
        rows = []
        with DDGS() as ddgs:
            for result in ddgs.text(user_requirement_summary[:200], max_results=_WEB_MAX_RESULTS):
                title = result.get("title") or ""
                body = (result.get("body") or "")[:400]
                href = result.get("href") or ""
                if title or body:
                    rows.append(f"- {title} | {body} | {href}")
        text = "\n".join(rows)
        return text, len(text), errors
    except Exception as exc:
        return "", 0, [f"web: {exc.__class__.__name__}: {exc}"]


def _fuse_sources(user_requirement_summary: str, parts: dict[str, str]) -> tuple[str, dict[str, Any]]:
    non_empty = {name: text for name, text in parts.items() if text.strip()}
    if not non_empty:
        return (
            "No external retrieval results; planning uses only the user goal, dataset description, and data profile.",
            {"fusion_empty_rate": 1.0, "fallback_used": True},
        )

    bundle = "\n\n".join(
        f"### {name}\n{text[:_RAP_FUSE_MAX_CHARS]}" for name, text in non_empty.items()
    )
    sys_prompt = (
        "You are an ML planning advisor. Fuse noisy retrieval snippets into concise, "
        "actionable cleaning-planning guidance. Focus on data quality, leakage risks, "
        "feature preservation, and task-specific caveats. Do not write code."
    )
    user_prompt = f"""
User requirement summary:
{user_requirement_summary}

Retrieved sources:
{bundle}

Produce 5-10 bullets. These are advisory only and must not override observed dataset facts.
"""
    try:
        fused = call_llm(sys_prompt, user_prompt, agent_name="manager_fusion")
    except Exception:
        fused = ""
    if fused and fused.strip():
        return fused.strip(), {"fusion_empty_rate": 0.0, "fallback_used": False}
    clipped = bundle[:_RAP_FUSE_MAX_CHARS]
    return clipped, {"fusion_empty_rate": 0.0, "fallback_used": True}


@tool
def build_manager_rap_context(
    user_request: str,
    description: str = "",
    task_type: str = "",
    target_column: str = "",
    dataset_profile: dict[str, Any] | None = None,
) -> str:
    """在 train/test 清洗计划之前构建 Manager RAP context。"""
    tool_inputs = {
        "user_request": user_request,
        "description": description,
        "task_type": task_type,
        "target_column": target_column,
        "has_dataset_profile": bool(dataset_profile),
    }
    errors: list[str] = []
    summary, slots, summary_meta = _summarize_and_extract_slots(
        user_request=user_request,
        dataset_profile=dataset_profile,
        description=description,
        task_type=task_type,
        target_column=target_column,
    )

    if not _RAP_ENABLED:
        context = (
            "No external retrieval results; RAP is disabled, so cleaning planning uses only local dataset facts."
        )
        meta = {
            "rap_enabled": False,
            **summary_meta,
            "arxiv_raw_chars": 0,
            "web_raw_chars": 0,
            "retrieval_bundle_empty_rate": 1.0,
            "fusion_empty_rate": 1.0,
            "fallback_used": True,
            "errors": [],
        }
        meta.update(_fusion_output_health(context))
    else:
        queries = _queries_from_slots(slots, user_request)
        arxiv_text, arxiv_chars, arxiv_errors = _format_arxiv_context(queries)
        web_text, web_chars, web_errors = _retrieve_web_light(summary)
        errors.extend(arxiv_errors)
        errors.extend(web_errors)
        bundle_empty = not (arxiv_text.strip() or web_text.strip())
        context, fusion_meta = _fuse_sources(
            summary,
            {"arXiv abstracts": arxiv_text, "Web snippets": web_text},
        )
        meta = {
            "rap_enabled": True,
            **summary_meta,
            "queries_used": queries,
            "arxiv_raw_chars": arxiv_chars,
            "web_raw_chars": web_chars,
            "retrieval_bundle_empty_rate": 1.0 if bundle_empty else 0.0,
            **fusion_meta,
            "errors": errors,
        }
        meta.update(_fusion_output_health(context))

    result = {
        "user_requirement_summary": summary,
        "rap_retrieval_slots": slots,
        "manager_rap_context": context,
        "manager_rap_meta": meta,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("build_manager_rap_context", tool_inputs, output)
    return output
