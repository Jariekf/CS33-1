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
        issues.append("training_data_path was not written in state")
    elif not Path(train).exists():
        issues.append(f"The training set file does not exist: {train}")
    if not test:
        issues.append("testing_data_path was not written in state")
    elif not Path(test).exists():
        issues.append(f"The test set file does not exist:{test}")
    return issues


def _check_clean_train(state: GlobalState) -> list[str]:
    issues: list[str] = []
    path = (state.get("train_cleaned_data_path") or "").strip()
    passed = bool(state.get("train_cleaning_passed"))
    if not passed:
        issues.append("train_cleaning_passed Set to False or not written")
    if not path:
        issues.append("train_cleaned_data_path was not written in state")
    elif not Path(path).exists():
        issues.append(f"The training set file does not exist after cleaning: {path}")
    report = state.get("train_cleaning_report") or {}
    steps = report.get("steps", [])
    saved = any(s.get("tool") == "save_dataset_from_store" for s in steps)
    if steps and not saved:
        issues.append("The call to `save_dataset_from_store` is not logged in `train_cleaning_report`")
    return issues


def _check_clean_test(state: GlobalState) -> list[str]:
    issues: list[str] = []
    path = (state.get("test_cleaned_data_path") or "").strip()
    passed = bool(state.get("test_cleaning_passed"))
    if not passed:
        issues.append("test_cleaning_passed Set to False or not written")
    if not path:
        issues.append("test_cleaned_data_path was not written in state")
    elif not Path(path).exists():
        issues.append(f"The test set file does not exist after cleaning: {path}")
    report = state.get("test_cleaning_report") or {}
    steps = report.get("steps", [])
    saved = any(s.get("tool") == "save_dataset_from_store" for s in steps)
    if steps and not saved:
        issues.append("The call to `save_dataset_from_store` is not logged in `test_cleaning_report`")
    return issues


def _check_eda(state: GlobalState) -> list[str]:
    issues: list[str] = []
    candidate_models = state.get("candidate_models") or []
    if not candidate_models:
        issues.append(
            "candidate_models has not been written; EDA is incomplete; the select_candidate_models_and_metrics tool has not been called"
        )
    if not (state.get("primary_metric") or "").strip():
        issues.append("primary_metric not written; EDA metric selection incomplete")
    return issues


def _failed_feature_models(state: GlobalState) -> list[tuple[str, list[str]]]:
    candidate_models = state.get("candidate_models") or []
    feature_ids = state.get("feature_dataset_ids") or {}
    test_ids = state.get("testing_feature_dataset_ids") or {}
    has_test = bool(state.get("testing_data_path"))
    failed = []
    
    for model in candidate_models:
        train_path = feature_ids.get(model, "")
        if train_path == "skipped":
            continue
            
        problems = []
        if not train_path:
            problems.append("The training set feature file was not written to state")
        elif not Path(train_path).exists():
            problems.append(f"The training set feature file does not exist: {train_path}")
            
        if has_test:
            test_path = test_ids.get(model, "")
            if not test_path:
                problems.append("The test set feature file was not written state")
            elif not Path(test_path).exists():
                problems.append(f"The test set feature file does not exist: {test_path}")
                
        if problems:
            failed.append((model, problems))
            
    return failed


def _check_feature(state: GlobalState) -> list[str]:
    issues: list[str] = []
    for model, problems in _failed_feature_models(state):
        for p in problems:
            issues.append(f"[{model}] {p}")
    return issues


_RULE_CHECKERS: dict[str, Callable[[GlobalState], list[str]]] = {
    "split": _check_split,
    "clean_train": _check_clean_train,
    "clean_test": _check_clean_test,
    "eda": _check_eda,
    "feature": _check_feature,
}

_STAGE_LABELS: dict[str, str] = {
    "split": "Dataset Split",
    "clean_train": "Training Set Data Cleaning",
    "clean_test": "Test Set Data Cleaning",
    "eda": "EDA Exploratory Data Analysis",
    "feature": "Feature Engineering",
}

# 4 Tools That Must Be Mentioned in EDA Phase Hints to Prevent LLMs from Skipping Tool Calls
_EDA_TOOLS_REMINDER = (
    "The EDA Agent must call the following four tools in sequence (all are required):"
    "profile_training_data → compute_correlation_matrix → "
    "generate_eda_charts → select_candidate_models_and_metrics."
    "Please explicitly require that all four tools be called in the recommendation."
)


# ---------------------------------------------------------------------------
# LLM hint generator — single short sentence for the upstream agent
# ---------------------------------------------------------------------------

def _llm_generate_hint(stage: str, state: GlobalState, issues: list[str]) -> str:
    issues_text = "; ".join(issues)
    label = _STAGE_LABELS.get(stage, stage)

    model_ctx = ""

    run_dir = state.get("run_dir") or ""
    source_ctx = ""
    if stage == "split":
        source_ctx = f"Original data path: {state.get('data_path') or 'unknown'}. "
    elif stage == "clean_train":
        source_ctx = f"Original training set path: {state.get('raw_training_data_path') or state.get('training_data_path') or 'unknown'}. "
    elif stage == "clean_test":
        source_ctx = f"Original test set path: {state.get('raw_testing_data_path') or state.get('testing_data_path') or 'unknown'}. "
    elif stage == "eda":
        source_ctx = _EDA_TOOLS_REMINDER

    system = (
        "You are a Supervisor of a data science pipeline, responsible for concisely converting validation failure information into fix recommendations for upstream agents. "
        "Requirements:\n"
        "1. Output only 1-2 sentences of recommendations directed at the upstream agent;\n"
        "2. Tell the agent specifically what is wrong (file does not exist, save step not executed, etc.);\n"
        "3. Tell the agent to only redo the missing step, not re-execute already completed steps;\n"
        "4. Do not output any analysis or explanation, output only the recommendation text."
    )
    user = (
        f"Stage: {label}. {model_ctx}{source_ctx}Run directory: {run_dir or 'unknown'}.\n"
        f"Detected issues: {issues_text}.\n"
        "Please output fix recommendations for the above issues (1-2 sentences):"
    )
    try:
        llm = llm_for("supervisor")
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        hint = (getattr(resp, "content", None) or "").strip()
        return hint if hint else issues_text
    except Exception as exc:
        return f"{issues_text} (hint generation failed: {exc})"


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
        lines.append("All checks passed, proceeding to downstream node.")
    else:
        lines.append("**Issue list:**")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")
        lines.append(f"**Generated hint:** {hint}")
    lines.extend(["", "---", ""])

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[supervisor] {icon} {label} (retry={retry}) → {path}")


# ---------------------------------------------------------------------------
# Stage key helpers
# ---------------------------------------------------------------------------

def _hint_key(stage: str, state: GlobalState) -> str:
    """Return the supervisor_hints dict key for the current stage."""
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
        if stage == "feature":
            from agent.feature_agent import _get_current_model
            if _get_current_model(state) is not None:
                return {}

        issues = checker(state)
        run_dir = state.get("run_dir") or ""

        if stage == "feature":
            failed_models = [m for m, _ in _failed_feature_models(state)]
            
            if not issues:
                _append_supervisor_report(run_dir, stage, "pass", [], "", 0)
                # Ensure all hints are cleared if passing? We can just leave them as they pass the stage
                return {}
                
            retries_dict = dict(state.get("supervisor_retry") or {})
            max_r = 0
            for m in failed_models:
                max_r = max(max_r, retries_dict.get(f"feature:{m}", 0))
            
            hint = _llm_generate_hint(stage, state, issues)
            _append_supervisor_report(run_dir, stage, "fail", issues, hint, max_r)
            
            hints = dict(state.get("supervisor_hints") or {})
            for m in failed_models:
                hints[f"feature:{m}"] = hint
                retries_dict[f"feature:{m}"] = retries_dict.get(f"feature:{m}", 0) + 1
                
            return {
                "supervisor_hints": hints,
                "supervisor_retry": retries_dict,
                "current_stage": "supervisor_feature_failed",
            }
        
        # Original logic for other stages
        h_key = _hint_key(stage, state)
        r_key = _retry_key(stage, state)
        retry = (state.get("supervisor_retry") or {}).get(r_key, 0)

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
