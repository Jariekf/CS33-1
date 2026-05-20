"""Agent report writer & pure-function node logging utilities."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# When each agent calls save_agent_report(..., run_dir=...), reports are written to {run_dir}/reports;
# if run_dir is empty (e.g., running a subgraph standalone), falls back to this directory to avoid path errors.
_DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "reports"
_RAP_REPORT_SUMMARY_CHARS = int(os.getenv("MANAGER_RAP_REPORT_SUMMARY_CHARS", "800"))
_RAP_REPORT_CONTEXT_CHARS = int(os.getenv("MANAGER_RAP_REPORT_CONTEXT_CHARS", "1200"))

# Only business_translator_agent may produce business recommendations. This maintains a blacklist
# of section headings that must be stripped from non-BAT reports (case-insensitive, trims whitespace and trailing punctuation).
_BUSINESS_HEADING_BLACKLIST: frozenset[str] = frozenset(
    {
        # 中文
        "业务建议",
        "业务影响",
        "业务洞察",
        "业务推荐",
        "业务价值",
        "业务意义",
        "行动建议",
        "推荐操作",
        "策略建议",
        "战略建议",
        "下一步",
        "下一步行动",
        "下一步操作",
        "下一步执行",
        "后续行动",
        "特征工程建议",
        "建模建议",
        "推荐模型",
        "类别不平衡处理",
        # English (When normalizing, remove the final "s" and convert to lowercase)
        "business recommendation",
        "business insight",
        "business action",
        "business implication",
        "business value",
        "recommendation",
        "recommended action",
        "next step",
        "action item",
        "strategic implication",
        "strategic recommendation",
        "feature engineering recommendation",
        "feature engineering suggestion",
        "modeling recommendation",
        "modeling suggestion",
        "recommended model",
    }
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*```")
_PUNCT_TAIL_RE = re.compile(r"[\s:：。，,\.！!?？]+$")
# Chapter prefix: emoji / Roman numerals / Arabic numerals / Chinese numerals + number separator
_HEADING_PREFIX_RE = re.compile(
    r"^[\s\W\d_]*(?:[一二三四五六七八九十百千]+、)?[\s\W\d_]*"
)
_HEADING_SUFFIX_RE = re.compile(r"[\s\W_]+$")


def _normalise_heading(text: str) -> str:
    """Normalise a Markdown heading for blacklist matching.

    Steps:
    1. Lowercase, strip markdown emphasis (``*_` ``) from both ends;
    2. Strip prefix: emoji, numbering (Arabic/Roman/Chinese e.g. 一、二、…), whitespace/punctuation;
    3. Take the main heading before a colon (e.g. "Next step: feature engineering" → "next step");
    4. Strip trailing whitespace/punctuation/emoji;
    5. If ends with plural 's' and the de-pluralised form is in the blacklist, normalise to singular.
    """
    t = text.strip().lower()
    t = t.strip("*_`")
    t = _PUNCT_TAIL_RE.sub("", t)
    t = _HEADING_PREFIX_RE.sub("", t)
    t = re.split(r"[:：]", t, maxsplit=1)[0]
    t = _HEADING_SUFFIX_RE.sub("", t)
    t = t.strip()
    if t.endswith("s") and not t.endswith("ss"):
        candidate = t[:-1]
        if candidate in _BUSINESS_HEADING_BLACKLIST:
            return candidate
    return t


def _strip_non_bat_recommendations(content: str) -> str:
    """Strip business-recommendation sections from non-BAT reports.

    Scans Markdown text:
    - Skips content inside fenced code blocks (``` ... ```);
    - When a ``# ... ######`` heading matches the blacklist, removes all lines from that
      heading up to the next heading of the same or higher level (or EOF);
    - Cleans up orphaned ``---`` dividers and runs of 3+ blank lines left after removal.
    """
    if not content:
        return content

    lines = content.split("\n")
    out: list[str] = []
    in_code = False
    skip_until_level: int | None = None  # heading level of the section currently being skipped

    for line in lines:
        if _FENCE_RE.match(line):
            in_code = not in_code
            if skip_until_level is None:
                out.append(line)
            continue

        if in_code:
            if skip_until_level is None:
                out.append(line)
            continue

        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading = _normalise_heading(m.group(2))
            if skip_until_level is not None:
                # currently skipping: stop when we hit a heading at the same or higher level
                if level <= skip_until_level:
                    skip_until_level = None
                    # if this heading also matches the blacklist, keep skipping
                    if heading in _BUSINESS_HEADING_BLACKLIST:
                        skip_until_level = level
                        continue
                    out.append(line)
                # sub-headings still inside the skip range, do not output
                continue
            if heading in _BUSINESS_HEADING_BLACKLIST:
                skip_until_level = level
                continue
            out.append(line)
            continue

        if skip_until_level is None:
            out.append(line)

    # cleanup: remove orphaned `---` dividers (nothing before/after) and excess blank lines
    cleaned: list[str] = []
    for i, line in enumerate(out):
        stripped = line.strip()
        if stripped == "---":
            prev_blank = (not cleaned) or cleaned[-1].strip() == ""
            next_idx = i + 1
            while next_idx < len(out) and out[next_idx].strip() == "":
                next_idx += 1
            next_blank_or_end = next_idx >= len(out)
            if prev_blank and next_blank_or_end:
                continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def save_agent_report(agent_name: str, message: Any, model: str = "", run_dir: str = "") -> str:
    """Append the agent's last message to {run_dir}/reports/{agent_name}_report.md.

    For loop agents like feature_agent / model_agent, pass the model parameter to label the round.
    Returns the path of the written file.

    Business recommendations (Business Recommendation/Next Steps and equivalent sections) are only
    permitted in ``business_insight_report.md`` produced by ``business_translator_agent``;
    other agents' reports are filtered through ``_strip_non_bat_recommendations`` before being saved.
    """
    reports_dir = Path(run_dir) / "reports" if run_dir else _DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{agent_name}_report.md"

    content = str(message.content) if hasattr(message, "content") else str(message)

    if agent_name != "business_translator":
        filtered = _strip_non_bat_recommendations(content)
        if not filtered.strip():
            filtered = "_(Business recommendation sections have been filtered; this agent is not responsible for business recommendations.)_"
        content = filtered

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"## {agent_name}" + (f" — {model}" if model else "") + f"  `{timestamp}`"
    section = f"{title}\n\n{content}\n\n---\n\n"

    with open(path, "a", encoding="utf-8") as f:
        f.write(section)

    print(f"[{agent_name}] report saved → {path}")
    return str(path)


def append_manager_rap_midlayer_to_report(state: dict[str, Any]) -> str | None:
    """Append a concise RAP mid-layer diagnostics section to the agent_manager report (reads RAP fields from state)."""
    meta = state.get("manager_rap_meta") if isinstance(state.get("manager_rap_meta"), dict) else {}
    summary = str(state.get("user_requirement_summary") or "")
    context = str(state.get("manager_rap_context") or "")
    if not meta and not summary and not context:
        return None

    run_dir = (state.get("run_dir") or "").strip()
    reports_dir = Path(run_dir) / "reports" if run_dir else _DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "agent_manager_report.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _clip(text: str, n: int) -> str:
        text = text.strip()
        if len(text) <= n:
            return text
        return text[:n] + f"\n\n… (truncated, total {len(text)} characters)"

    queries = meta.get("queries_used") or []
    queries_s = ", ".join(str(q) for q in queries) if queries else "(none)"
    err_list = meta.get("errors") or []

    lines = [
        f"## Manager RAP · Mid-layer Diagnostics `{ts}`",
        "",
        "### Metrics (`manager_rap_meta`)",
        "",
        f"- **rap_enabled**: {meta.get('rap_enabled', '')}",
        f"- **summary_json_parse_ok**: {meta.get('summary_json_parse_ok', '')}",
        f"- **summary_required_fields_ok**: {meta.get('summary_required_fields_ok', '')}",
        f"- **summary_empty_rate**: {meta.get('summary_empty_rate', '')}",
        f"- **retrieval_bundle_empty_rate**: {meta.get('retrieval_bundle_empty_rate', '')}",
        f"- **arxiv_raw_chars / web_raw_chars**: {meta.get('arxiv_raw_chars', '')} / {meta.get('web_raw_chars', '')}",
        f"- **queries_used**: {queries_s}",
        f"- **fusion_empty_rate** (1.0 when no retrieval chunks available for fusion): {meta.get('fusion_empty_rate', '')}",
        f"- **fusion_fallback_used**: {meta.get('fallback_used', '')}",
        f"- **fusion_text_empty_rate**: {meta.get('fusion_text_empty_rate', '')}",
        f"- **fusion_line_repetition_rate**: {meta.get('fusion_line_repetition_rate', '')}",
        f"- **fusion_code_leakage_flag**: {meta.get('fusion_code_leakage_flag', '')}",
        f"- **fusion_output_chars**: {meta.get('fusion_output_chars', '')}",
    ]
    if meta.get("summary_error"):
        lines.append(f"- **summary_error**: {meta.get('summary_error')}")
    if err_list:
        lines.append("- **errors (top 3)**: " + " | ".join(str(e) for e in err_list[:3]))
    lines.extend(
        [
            "",
            "### Preview · `user_requirement_summary`",
            "",
            "```text",
            _clip(summary, _RAP_REPORT_SUMMARY_CHARS) or "(empty)",
            "```",
            "",
            "### Preview · `manager_rap_context`",
            "",
            "```text",
            _clip(context, _RAP_REPORT_CONTEXT_CHARS) or "(empty)",
            "```",
            "",
            "---",
            "",
        ]
    )
    section = "\n".join(lines)
    with open(path, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"[agent_manager] RAP diagnostics appended → {path}")
    return str(path)


def save_state_snapshot(state: dict[str, Any]) -> str:
    """Save a timestamped JSON snapshot of the global state (excluding messages).

    Filename format: YYYY-MM-DD_HHMMSS_state.json, saved under {run_dir}/reports/.
    Returns the path of the written file.
    """
    run_dir = state.get("run_dir") or ""
    reports_dir = Path(run_dir) / "reports" if run_dir else _DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = reports_dir / f"{timestamp}_state.json"

    snapshot = {k: v for k, v in state.items() if k != "messages"}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)

    print(f"[state snapshot] saved → {path}")
    return str(path)


def log_node(node_name: str, success: bool, detail: str = "") -> None:
    """Log success/failure for a pure-function node, printing directly to the console."""
    status = "✓" if success else "✗"
    msg = f"[{node_name}] {status}"
    if detail:
        msg += f"  {detail}"
    print(msg)
