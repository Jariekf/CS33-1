from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from core.global_state import GlobalState
from core.my_llm import llm_deepseek
from core.report_writer import save_agent_report
from tools.supervisor_tools.validate_stage_artifacts import validate_stage_artifacts


SUPERVISOR_PASS_THRESHOLD = 3
DEFAULT_MAX_SUPERVISOR_RETRIES = 2

# 不会触发 supervisor 验收的阶段，避免对纯函数节点反复重跑。
DETERMINISTIC_STAGES = {"split_done", "compare_done"}


def _compact_state_for_review(state: GlobalState) -> dict[str, Any]:
    excluded = {"messages", "supervisor_artifact_validation"}
    return {key: value for key, value in state.items() if key not in excluded}


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json|JSON)?\s*", "", stripped, count=1)
    stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    return stripped.strip()


def _extract_json_object(text: str) -> str | None:
    text = _strip_markdown_fences(text)
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_json_object(content: str) -> dict[str, Any]:
    payload = _extract_json_object(content)
    if payload is None:
        raise ValueError("Supervisor LLM response does not contain a JSON object.")
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", payload))


def _max_retry_count(state: GlobalState, stage: str) -> int:
    max_retries = state.get("max_retries") or {}
    return int(max_retries.get(stage, DEFAULT_MAX_SUPERVISOR_RETRIES))


def _next_retry_counts(state: GlobalState, stage: str, passed: bool) -> dict[str, int]:
    retry_counts = dict(state.get("retry_counts") or {})
    if passed:
        retry_counts[stage] = 0
    else:
        retry_counts[stage] = int(retry_counts.get(stage, 0)) + 1
    return retry_counts


def _run_artifact_validation(state: GlobalState, current_stage: str) -> dict[str, Any]:
    """直接调用 deterministic 校验工具，不需要 LLM 决定。"""
    state_summary = _compact_state_for_review(state)
    raw = validate_stage_artifacts.invoke(
        {
            "current_stage": current_stage,
            "state_json": json.dumps(state_summary, ensure_ascii=False, default=str),
        }
    )
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError as exc:
        return {
            "stage": current_stage,
            "passed": False,
            "missing_fields": [],
            "checks": [],
            "failures": [{"label": "validate_stage_artifacts", "message": str(exc)}],
        }


def _normalise_review(
    review_data: dict[str, Any], state: GlobalState, validation: dict[str, Any]
) -> dict[str, Any]:
    # stage 强制对齐 current_stage，不信任 LLM 返回的 reviewed_stage，避免 retry/路由错乱。
    stage = str(state.get("current_stage") or "")
    validation_passed = bool(validation.get("passed", False))
    score = int(review_data.get("score", 1))
    score = max(1, min(5, score))
    passed = bool(review_data.get("passed", False)) and score >= SUPERVISOR_PASS_THRESHOLD

    feedback = str(review_data.get("feedback", "") or "")
    if not validation_passed:
        passed = False
        score = min(score, 2)
        failures = validation.get("failures") or []
        missing = validation.get("missing_fields") or []
        feedback = feedback or (
            "Artifact 校验未通过。"
            f" 缺失字段：{missing or '无'}。"
            f" 失败项：{failures or '无'}。"
        )

    new_retry_counts = _next_retry_counts(state, stage, passed)
    retry_count = new_retry_counts.get(stage, 0)
    max_retries = _max_retry_count(state, stage)
    exhausted = (not passed) and retry_count >= max_retries
    next_action = str(review_data.get("next_action") or ("continue" if passed else "retry"))
    if exhausted:
        next_action = "continue"
        feedback = (
            f"{feedback}\n当前阶段 {stage} 已达到 supervisor 最大重试次数 "
            f"({retry_count}/{max_retries})，停止自动重跑并按防死锁策略强制放行。"
        )

    return {
        "reviewed_stage": stage,
        "score": score,
        "passed": passed,
        "next_action": next_action,
        "feedback": feedback,
        "summary": review_data.get("summary", ""),
        "risks": review_data.get("risks", []),
        "retry_count": retry_count,
        "max_retries": max_retries,
        "retry_exhausted": exhausted,
        "artifact_validation": validation,
        "_retry_counts_after": new_retry_counts,
    }


def _fallback_review(
    state: GlobalState, error: Exception, validation: dict[str, Any]
) -> dict[str, Any]:
    stage = str(state.get("current_stage") or "")
    validation_passed = bool(validation.get("passed", False))
    passed = validation_passed
    new_retry_counts = _next_retry_counts(state, stage, passed)
    retry_count = new_retry_counts.get(stage, 0)
    max_retries = _max_retry_count(state, stage)
    exhausted = (not passed) and retry_count >= max_retries

    feedback = f"Supervisor LLM 审查解析失败：{error}"
    if not validation_passed:
        feedback += (
            f"\nArtifact 校验失败：{validation.get('failures') or validation.get('missing_fields')}"
        )
    if exhausted:
        feedback += (
            f"\n当前阶段 {stage} 已达到最大重试次数 ({retry_count}/{max_retries})，"
            "按防死锁策略强制放行。"
        )

    return {
        "reviewed_stage": stage,
        "score": 3 if passed else 2,
        "passed": passed,
        "next_action": "continue" if (passed or exhausted) else "retry",
        "feedback": "" if passed else feedback,
        "summary": "使用 artifact 校验结果生成的 fallback 审查结果。",
        "risks": ["LLM 审查结果不可解析，语义质量判断不完整。"],
        "retry_count": retry_count,
        "max_retries": max_retries,
        "retry_exhausted": exhausted,
        "artifact_validation": validation,
        "_retry_counts_after": new_retry_counts,
    }


def _invalid_labels_with_prefix(review: dict[str, Any], prefix: str) -> list[str]:
    validation = review.get("artifact_validation") or {}
    failures = validation.get("failures") or []
    names: list[str] = []
    for failure in failures:
        label = failure.get("label", "")
        if isinstance(label, str) and label.startswith(prefix):
            names.append(label.removeprefix(prefix))
    return names


def _cleanup_failed_stage_outputs(state: GlobalState, review: dict[str, Any]) -> dict[str, Any]:
    """失败重跑前清掉会误导 worker 的过期产物；强制放行时不清理。"""
    if review["passed"] or review.get("retry_exhausted"):
        return {}

    stage = review.get("reviewed_stage", "")
    updates: dict[str, Any] = {}

    if stage == "cleaning_saved":
        updates["cleaned_data_path"] = ""
        updates["cleaning_passed"] = False

    if stage == "feature_saved":
        feature_dataset_ids = dict(state.get("feature_dataset_ids") or {})
        if feature_dataset_ids:
            invalid_models = _invalid_labels_with_prefix(review, "feature_dataset:")
            if invalid_models:
                for model_name in invalid_models:
                    feature_dataset_ids.pop(model_name, None)
            else:
                feature_dataset_ids.pop(next(reversed(feature_dataset_ids)), None)
            updates["feature_dataset_ids"] = feature_dataset_ids

    if stage == "model_evaluated":
        model_results = dict(state.get("model_results") or {})
        if model_results:
            invalid_models = _invalid_labels_with_prefix(review, "model:")
            if invalid_models:
                for model_name in invalid_models:
                    model_results.pop(model_name, None)
            else:
                model_results.pop(next(reversed(model_results)), None)
            updates["model_results"] = model_results

    return updates


def _supervisor_node(state: GlobalState) -> GlobalState:
    current_stage = state.get("current_stage") or "unknown"

    validation = _run_artifact_validation(state, current_stage)

    state_summary = _compact_state_for_review(state)
    system_message = SystemMessage(
        content=(
            "你是 multi-agent data science pipeline 的 Supervisor Agent。"
            "你要基于 artifact 工具校验结果和 GlobalState 摘要，判断当前阶段是否可以放行。"
            "如果工具校验失败，必须判定不通过。必须只返回合法 JSON。"
        )
    )
    human_message = HumanMessage(
        content=(
            f"当前阶段：{current_stage}\n\n"
            "Artifact 工具校验结果：\n"
            f"{json.dumps(validation, ensure_ascii=False, indent=2, default=str)}\n\n"
            "GlobalState 摘要：\n"
            f"{json.dumps(state_summary, ensure_ascii=False, indent=2, default=str)}\n\n"
            "返回 JSON 格式如下：\n"
            "{\n"
            '  "score": <1到5的整数>,\n'
            '  "passed": <true或false>,\n'
            '  "next_action": "<continue|retry|stop>",\n'
            '  "feedback": "<如果不通过，给出具体修复建议；通过则可为空>",\n'
            '  "summary": "<简短总结>",\n'
            '  "risks": ["<剩余风险>"]\n'
            "}\n"
            "评分标准：5 完整可靠；4 可通过但有小风险；3 勉强可继续；"
            "2 需要重试；1 严重失败。"
        )
    )

    try:
        response = llm_deepseek.invoke([system_message, human_message])
        review = _normalise_review(_parse_json_object(str(response.content)), state, validation)
    except Exception as exc:
        review = _fallback_review(state, exc, validation)

    review_message = AIMessage(
        content=json.dumps(review, ensure_ascii=False, indent=2, default=str)
    )
    save_agent_report("supervisor_agent", review_message, run_dir=state.get("run_dir") or "")

    public_review = {k: v for k, v in review.items() if not k.startswith("_")}
    updates: dict[str, Any] = {
        "supervisor_artifact_validation": validation,
        "supervisor_review": public_review,
        "supervisor_passed": review["passed"],
        "supervisor_feedback": review["feedback"],
        "next_action": review["next_action"],
        "retry_counts": review["_retry_counts_after"],
    }
    updates.update(_cleanup_failed_stage_outputs(state, review))
    return updates


def _show_graph_if_needed(agent_graph, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "supervisor_agent_graph.png"
        image_bytes = agent_graph.get_graph().draw_mermaid_png()
        image_path.write_bytes(image_bytes)
        print(f"Graph 图片已保存: {image_path}")
    except Exception as exc:
        print(f"图渲染失败: {exc}")


def build_supervisor_agent(show_graph: bool = False):
    graph = StateGraph(GlobalState)
    graph.add_node("supervisor_agent", _supervisor_node)
    graph.set_entry_point("supervisor_agent")
    graph.add_edge("supervisor_agent", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


supervisor_agent = build_supervisor_agent()


if __name__ == "__main__":
    supervisor_agent = build_supervisor_agent(show_graph=True)
    events = supervisor_agent.stream(
        {
            "messages": [HumanMessage(content="请审查当前流水线阶段。")],
            "current_stage": "eda_done",
            "task_type": "classification",
            "target_column": "y",
            "candidate_models": ["LogisticRegression", "RandomForest", "XGBoost"],
            "feature_engineering_plan": {
                "LogisticRegression": {"scaling": True, "encoding": "onehot"},
                "RandomForest": {"scaling": False, "encoding": "label"},
                "XGBoost": {"scaling": False, "encoding": "label"},
            },
            "retry_counts": {},
            "max_retries": {"eda_done": 2},
        },
        stream_mode="values",
    )
    for event in events:
        review = event.get("supervisor_review")
        if review:
            print(json.dumps(review, ensure_ascii=False, indent=2, default=str))
