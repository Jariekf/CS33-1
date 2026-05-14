"""Supervisor nodes: rule-check paths written by each agent stage, generate LLM hints on failure.

Usage in graph.py:
    from agent.supervisor import make_supervisor
    graph.add_node("supervisor_split",        make_supervisor("split"))
    graph.add_node("supervisor_clean_train",  make_supervisor("clean_train"))
    graph.add_node("supervisor_clean_test",   make_supervisor("clean_test"))
    graph.add_node("supervisor_feature",      make_supervisor("feature"))

Each stage key also drives route helpers:
    from agent.supervisor import make_supervisor_router
    graph.add_conditional_edges("supervisor_split", make_supervisor_router("split", max_retry=2), ...)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from core.global_state import GlobalState
from core.my_llm import llm_for


# ---------------------------------------------------------------------------
# Rule checkers — return list[str] of issue descriptions (empty = pass)
# ---------------------------------------------------------------------------

def _check_split(state: GlobalState) -> list[str]:
    issues: list[str] = []
    train = (state.get("training_data_path") or "").strip()
    test = (state.get("testing_data_path") or "").strip()
    if not train:
        issues.append("training_data_path 未写入 state")
    elif not Path(train).exists():
        issues.append(f"训练集文件不存在：{train}")
    if not test:
        issues.append("testing_data_path 未写入 state")
    elif not Path(test).exists():
        issues.append(f"测试集文件不存在：{test}")
    return issues


def _check_clean_train(state: GlobalState) -> list[str]:
    issues: list[str] = []
    path = (state.get("train_cleaned_data_path") or "").strip()
    passed = bool(state.get("train_cleaning_passed"))
    if not passed:
        issues.append("train_cleaning_passed 为 False 或未写入")
    if not path:
        issues.append("train_cleaned_data_path 未写入 state")
    elif not Path(path).exists():
        issues.append(f"清洗后训练集文件不存在：{path}")
    report = state.get("train_cleaning_report") or {}
    steps = report.get("steps", [])
    saved = any(s.get("tool") == "save_dataset_from_store" for s in steps)
    if steps and not saved:
        issues.append("train_cleaning_report 中未记录 save_dataset_from_store 调用")
    return issues


def _check_clean_test(state: GlobalState) -> list[str]:
    issues: list[str] = []
    path = (state.get("test_cleaned_data_path") or "").strip()
    passed = bool(state.get("test_cleaning_passed"))
    if not passed:
        issues.append("test_cleaning_passed 为 False 或未写入")
    if not path:
        issues.append("test_cleaned_data_path 未写入 state")
    elif not Path(path).exists():
        issues.append(f"清洗后测试集文件不存在：{path}")
    report = state.get("test_cleaning_report") or {}
    steps = report.get("steps", [])
    saved = any(s.get("tool") == "save_dataset_from_store" for s in steps)
    if steps and not saved:
        issues.append("test_cleaning_report 中未记录 save_dataset_from_store 调用")
    return issues


def _check_eda(state: GlobalState) -> list[str]:
    issues: list[str] = []
    candidate_models = state.get("candidate_models") or []
    if not candidate_models:
        issues.append(
            "candidate_models 未写入，EDA 未完成 select_candidate_models_and_metrics 工具调用"
        )
    if not (state.get("primary_metric") or "").strip():
        issues.append("primary_metric 未写入，EDA 未完成指标选择")
    return issues


def _last_saved_feature_model(state: GlobalState) -> str | None:
    """Return the most recently saved model (last key in feature_dataset_ids)."""
    feature_ids = state.get("feature_dataset_ids") or {}
    return next(reversed(feature_ids), None) if feature_ids else None


def _check_feature(state: GlobalState) -> list[str]:
    issues: list[str] = []
    feature_ids = state.get("feature_dataset_ids") or {}
    if not feature_ids:
        return issues  # nothing saved yet — nothing to verify

    model = next(reversed(feature_ids))
    train_path = feature_ids.get(model, "")

    if train_path == "skipped":
        return issues  # already marked skipped, nothing to verify

    if not train_path:
        issues.append(f"feature_dataset_ids[{model}] 未写入")
    elif not Path(train_path).exists():
        issues.append(f"模型 {model} 训练集特征文件不存在：{train_path}")

    if state.get("testing_data_path"):
        test_ids = state.get("testing_feature_dataset_ids") or {}
        test_path = test_ids.get(model, "")
        if not test_path:
            issues.append(f"testing_feature_dataset_ids[{model}] 未写入")
        elif not Path(test_path).exists():
            issues.append(f"模型 {model} 测试集特征文件不存在：{test_path}")

    return issues


_RULE_CHECKERS: dict[str, Callable[[GlobalState], list[str]]] = {
    "split": _check_split,
    "clean_train": _check_clean_train,
    "clean_test": _check_clean_test,
    "eda": _check_eda,
    "feature": _check_feature,
}

_STAGE_LABELS: dict[str, str] = {
    "split": "数据集切分",
    "clean_train": "训练集数据清洗",
    "clean_test": "测试集数据清洗",
    "eda": "EDA 探索性数据分析",
    "feature": "特征工程",
}

# EDA 阶段 hint 中必须提及的 4 个工具，以防 LLM 跳过工具调用
_EDA_TOOLS_REMINDER = (
    "EDA Agent 必须依次调用以下 4 个工具（缺一不可）："
    "profile_training_data → compute_correlation_matrix → "
    "generate_eda_charts → select_candidate_models_and_metrics。"
    "请在建议中明确要求完成全部 4 个工具调用。"
)


# ---------------------------------------------------------------------------
# LLM hint generator — single short sentence for the upstream agent
# ---------------------------------------------------------------------------

def _llm_generate_hint(stage: str, state: GlobalState, issues: list[str]) -> str:
    issues_text = "；".join(issues)
    label = _STAGE_LABELS.get(stage, stage)

    model_ctx = ""
    if stage == "feature":
        model = _last_saved_feature_model(state)
        if model:
            model_ctx = f"当前模型：{model}。"

    run_dir = state.get("run_dir") or ""
    source_ctx = ""
    if stage == "split":
        source_ctx = f"原始数据路径：{state.get('data_path') or '未知'}。"
    elif stage == "clean_train":
        source_ctx = f"原始训练集路径：{state.get('raw_training_data_path') or state.get('training_data_path') or '未知'}。"
    elif stage == "clean_test":
        source_ctx = f"原始测试集路径：{state.get('raw_testing_data_path') or state.get('testing_data_path') or '未知'}。"
    elif stage == "eda":
        source_ctx = _EDA_TOOLS_REMINDER

    system = (
        "你是一个数据科学流水线的 Supervisor，负责将校验失败的信息简明转化为对上游 Agent 的修复建议。"
        "要求：\n"
        "1. 只输出 1-2 句话的中文建议，面向上游 Agent；\n"
        "2. 告诉 Agent 具体哪里不对（文件不存在、保存步骤未执行等）；\n"
        "3. 告诉 Agent 只补做缺失的那一步，不要重新执行已完成的步骤；\n"
        "4. 不要输出任何分析或解释，直接输出建议文本。"
    )
    user = (
        f"阶段：{label}。{model_ctx}{source_ctx}运行目录：{run_dir or '未知'}。\n"
        f"检测到的问题：{issues_text}。\n"
        "请输出针对上述问题的修复建议（1-2 句）："
    )
    try:
        llm = llm_for("supervisor")
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        hint = (getattr(resp, "content", None) or "").strip()
        return hint if hint else issues_text
    except Exception as exc:
        return f"{issues_text}（hint 生成失败：{exc}）"


# ---------------------------------------------------------------------------
# Supervisor report writer
# ---------------------------------------------------------------------------

def _append_supervisor_report(
    run_dir: str,
    stage: str,
    status: str,
    issues: list[str],
    hint: str,
    retry: int,
) -> None:
    from core.report_writer import _DEFAULT_REPORTS_DIR  # type: ignore[attr-defined]

    reports_dir = Path(run_dir) / "reports" if run_dir else _DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "supervisor_report.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = _STAGE_LABELS.get(stage, stage)
    icon = "✓" if status == "pass" else "✗"

    lines = [
        f"## {icon} Supervisor · {label}  `{ts}`  retry={retry}",
        "",
    ]
    if status == "pass":
        lines.append("所有检查通过，继续下游节点。")
    else:
        lines.append(f"**问题列表：**")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")
        lines.append(f"**生成的 hint：** {hint}")
    lines.extend(["", "---", ""])

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[supervisor] {icon} {label} (retry={retry}) → {path}")


# ---------------------------------------------------------------------------
# Stage key helpers
# ---------------------------------------------------------------------------

def _hint_key(stage: str, state: GlobalState) -> str:
    """Return the supervisor_hints dict key for the current stage."""
    if stage == "feature":
        model = _last_saved_feature_model(state) or "unknown"
        return f"feature:{model}"
    return stage


def _retry_key(stage: str, state: GlobalState) -> str:
    """Return the supervisor_retry dict key for the current stage."""
    return _hint_key(stage, state)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_supervisor(stage: str, max_retry: int = 2) -> Callable[[GlobalState], GlobalState]:
    """Return a LangGraph node function that checks paths for *stage*."""

    checker = _RULE_CHECKERS.get(stage)
    if checker is None:
        raise ValueError(f"Unknown supervisor stage: {stage!r}. Valid: {list(_RULE_CHECKERS)}")

    def _node(state: GlobalState) -> GlobalState:
        issues = checker(state)
        h_key = _hint_key(stage, state)
        r_key = _retry_key(stage, state)
        retry = (state.get("supervisor_retry") or {}).get(r_key, 0)
        run_dir = state.get("run_dir") or ""

        if not issues:
            _append_supervisor_report(run_dir, stage, "pass", [], "", retry)
            hints = dict(state.get("supervisor_hints") or {})
            hints[h_key] = ""
            return {"supervisor_hints": hints}

        # Generate LLM hint
        hint = _llm_generate_hint(stage, state, issues)
        _append_supervisor_report(run_dir, stage, "fail", issues, hint, retry)

        hints = dict(state.get("supervisor_hints") or {})
        hints[h_key] = hint
        retries = dict(state.get("supervisor_retry") or {})
        retries[r_key] = retry + 1

        return {
            "supervisor_hints": hints,
            "supervisor_retry": retries,
            "current_stage": f"supervisor_{stage.replace('_', '')}_failed",
        }

    _node.__name__ = f"supervisor_{stage}"
    return _node


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def make_supervisor_router(
    stage: str,
    upstream: str,
    downstream: str,
    max_retry: int = 2,
) -> Callable[[GlobalState], str]:
    """Return a conditional-edge function: pass → downstream, fail → upstream or END."""

    from langgraph.graph import END

    def _router(state: GlobalState) -> str:
        h_key = _hint_key(stage, state)
        r_key = _retry_key(stage, state)
        hint = (state.get("supervisor_hints") or {}).get(h_key, "")
        retry = (state.get("supervisor_retry") or {}).get(r_key, 0)

        if not hint:
            return downstream
        if retry <= max_retry:
            return upstream
        return END

    _router.__name__ = f"route_supervisor_{stage}"
    return _router
