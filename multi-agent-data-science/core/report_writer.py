"""Agent 报告写入 & 纯函数节点日志工具。"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# 当各 agent 调用 save_agent_report(..., run_dir=...) 时，报告写入 ``{run_dir}/reports``；
# run_dir 为空（例如仅单独跑某子图）时回退到此目录，避免抛路径错误。
_DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "reports"
_RAP_REPORT_SUMMARY_CHARS = int(os.getenv("MANAGER_RAP_REPORT_SUMMARY_CHARS", "800"))
_RAP_REPORT_CONTEXT_CHARS = int(os.getenv("MANAGER_RAP_REPORT_CONTEXT_CHARS", "1200"))

# 只有 business_translator_agent 可以产出业务建议。这里维护一份非 BAT 报告
# 中必须剥离的章节标题黑名单（大小写不敏感、去除首尾空白与尾部冒号/标点）。
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
        # 英文（normalise 时去除尾部 s，且小写）
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
# 章节前缀：emoji / 罗马数字 / 阿拉伯数字 / 中文数字 + 序号分隔符
_HEADING_PREFIX_RE = re.compile(
    r"^[\s\W\d_]*(?:[一二三四五六七八九十百千]+、)?[\s\W\d_]*"
)
_HEADING_SUFFIX_RE = re.compile(r"[\s\W_]+$")


def _normalise_heading(text: str) -> str:
    """归一化 Markdown 标题用于黑名单匹配。

    流程：
    1. 小写、剥两端 markdown emphasis（``*_` ``）；
    2. 剥前缀：emoji、序号（阿拉伯/罗马/中文「一、二、…」）、空白/标点；
    3. 取冒号 / 中文冒号前的主标题（"下一步：特征工程与建模" → "下一步"）；
    4. 剥尾部空白/标点/emoji；
    5. 若以英文复数 s 结尾且去 s 命中黑名单，则归一化为单数。
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
    """从非 BAT 报告中剥离业务建议类章节。

    扫描 Markdown 文本：
    - 跳过 fenced code block（``` ... ```）内的内容；
    - 当遇到 ``# ... ###### Heading`` 且标题在黑名单时，删除从该标题行开始到
      下一个**同级或更高级**标题（或 EOF）之间的全部行；
    - 清理删除后可能遗留的孤立 ``---`` 分隔线及 3 行以上连续空行。
    """
    if not content:
        return content

    lines = content.split("\n")
    out: list[str] = []
    in_code = False
    skip_until_level: int | None = None  # 当前正在跳过的章节起始级别

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
                # 已经处于跳过状态：只要遇到同级或更高级标题就停止跳过
                if level <= skip_until_level:
                    skip_until_level = None
                    # 当前标题本身是否也命中黑名单？继续跳过
                    if heading in _BUSINESS_HEADING_BLACKLIST:
                        skip_until_level = level
                        continue
                    out.append(line)
                # 否则仍在 skip 区间内的子标题，不输出
                continue
            if heading in _BUSINESS_HEADING_BLACKLIST:
                skip_until_level = level
                continue
            out.append(line)
            continue

        if skip_until_level is None:
            out.append(line)

    # 清理：移除孤立的 `---`（前后无内容）以及多余空行
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
    """将 agent 最后一条消息追加写入 {run_dir}/reports/{agent_name}_report.md。

    对 feature_agent / model_agent 这类循环 agent，传入 model 参数以标注是哪一轮。
    返回写入的文件路径。

    业务建议（Business Recommendation/Next Steps 等同义章节）只允许出现在
    ``business_translator_agent`` 产出的 ``business_insight_report.md`` 中；
    其他 agent 的报告会在落盘前经 ``_strip_non_bat_recommendations`` 过滤。
    """
    reports_dir = Path(run_dir) / "reports" if run_dir else _DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{agent_name}_report.md"

    content = str(message.content) if hasattr(message, "content") else str(message)

    if agent_name != "business_translator":
        filtered = _strip_non_bat_recommendations(content)
        if not filtered.strip():
            filtered = "_(Business recommendation sections were filtered; this agent is not responsible for business insights.)_"
        content = filtered

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
        return text[:n] + f"\n\n... (truncated, {len(text)} chars total)"

    queries = meta.get("queries_used") or []
    queries_s = ", ".join(str(q) for q in queries) if queries else "(none)"
    err_list = meta.get("errors") or []

    lines = [
        f"## Manager RAP · Mid-layer Audit `{ts}`",
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
        f"- **fusion_empty_rate** (1.0 when no retrieval fragments available): {meta.get('fusion_empty_rate', '')}",
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
