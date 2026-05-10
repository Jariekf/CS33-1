"""Agent 报告写入 & 纯函数节点日志工具。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

_DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "reports"


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
