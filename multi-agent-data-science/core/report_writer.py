"""Agent 报告写入 & 纯函数节点日志工具。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# 当各 agent 调用 save_agent_report(..., run_dir=...) 时，报告写入 ``{run_dir}/reports``；
# run_dir 为空（例如仅单独跑某子图）时回退到此目录，避免抛路径错误。
_DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "reports"
_RAP_REPORT_SUMMARY_CHARS = int(os.getenv("MANAGER_RAP_REPORT_SUMMARY_CHARS", "800"))
_RAP_REPORT_CONTEXT_CHARS = int(os.getenv("MANAGER_RAP_REPORT_CONTEXT_CHARS", "1200"))


def save_agent_report(agent_name: str, message: Any, model: str = "", run_dir: str = "") -> str:
    """将 agent 最后一条消息追加写入 {run_dir}/reports/{agent_name}_report.md。

    对 feature_agent / model_agent 这类循环 agent，传入 model 参数以标注是哪一轮。
    返回写入的文件路径。
    """
    reports_dir = Path(run_dir) / "reports" if run_dir else _DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{agent_name}_report.md"

    content = str(message.content) if hasattr(message, "content") else str(message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"## {agent_name}" + (f" — {model}" if model else "") + f"  `{timestamp}`"
    section = f"{title}\n\n{content}\n\n---\n\n"

    with open(path, "a", encoding="utf-8") as f:
        f.write(section)

    print(f"[{agent_name}] report saved → {path}")
    return str(path)


def append_manager_rap_midlayer_to_report(state: dict[str, Any]) -> str | None:
    """在 agent_manager 报告末尾追加精简 RAP 中层检测（依赖 state 中的 RAP 字段）。"""
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
        return text[:n] + f"\n\n…（已截断，共 {len(text)} 字符）"

    queries = meta.get("queries_used") or []
    queries_s = ", ".join(str(q) for q in queries) if queries else "(none)"
    err_list = meta.get("errors") or []

    lines = [
        f"## Manager RAP · 中层检测 `{ts}`",
        "",
        "### 指标（`manager_rap_meta`）",
        "",
        f"- **rap_enabled**: {meta.get('rap_enabled', '')}",
        f"- **summary_json_parse_ok**: {meta.get('summary_json_parse_ok', '')}",
        f"- **summary_required_fields_ok**: {meta.get('summary_required_fields_ok', '')}",
        f"- **summary_empty_rate**: {meta.get('summary_empty_rate', '')}",
        f"- **retrieval_bundle_empty_rate**: {meta.get('retrieval_bundle_empty_rate', '')}",
        f"- **arxiv_raw_chars / web_raw_chars**: {meta.get('arxiv_raw_chars', '')} / {meta.get('web_raw_chars', '')}",
        f"- **queries_used**: {queries_s}",
        f"- **fusion_empty_rate**（无检索片段可供融合时为 1.0）: {meta.get('fusion_empty_rate', '')}",
        f"- **fusion_fallback_used**: {meta.get('fallback_used', '')}",
        f"- **fusion_text_empty_rate**: {meta.get('fusion_text_empty_rate', '')}",
        f"- **fusion_line_repetition_rate**: {meta.get('fusion_line_repetition_rate', '')}",
        f"- **fusion_code_leakage_flag**: {meta.get('fusion_code_leakage_flag', '')}",
        f"- **fusion_output_chars**: {meta.get('fusion_output_chars', '')}",
    ]
    if meta.get("summary_error"):
        lines.append(f"- **summary_error**: {meta.get('summary_error')}")
    if err_list:
        lines.append("- **errors（前 3 条）**: " + " | ".join(str(e) for e in err_list[:3]))
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
    """将全局 state 快照（去除 messages）保存为带时间戳的 JSON 文件。

    文件名格式：YYYY-MM-DD_HHMMSS_state.json，保存在 {run_dir}/reports/ 下。
    返回写入的文件路径。
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
    """纯函数节点的成功/失败日志，直接 print 到控制台。"""
    status = "✓" if success else "✗"
    msg = f"[{node_name}] {status}"
    if detail:
        msg += f"  {detail}"
    print(msg)
