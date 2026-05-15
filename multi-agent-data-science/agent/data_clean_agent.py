from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

import time

from core.global_state import GlobalState
from core.my_llm import llm_for
from core.report_writer import save_agent_report
from tools.dataclean_tools.clean_data_types import clean_data_types
from tools.dataclean_tools.clean_duplicates import clean_duplicates
from tools.dataclean_tools.clean_drop_columns import clean_drop_columns
from tools.dataclean_tools.clean_missing_values import clean_missing_values
from tools.dataclean_tools.clean_outliers import clean_outliers
from tools.dataclean_tools.load_dataset_to_store import load_dataset_to_store
from tools.dataclean_tools.save_dataset_from_store import save_dataset_from_store

tools = [
    load_dataset_to_store,
    save_dataset_from_store,
    clean_data_types,
    clean_missing_values,
    clean_outliers,
    clean_drop_columns,
    clean_duplicates,
]
llm_with_tools = llm_for("data_clean").bind_tools(tools)


_CLEAN_TOOL_NAMES = {
    "load_dataset_to_store",
    "save_dataset_from_store",
    "clean_data_types",
    "clean_missing_values",
    "clean_outliers",
    "clean_drop_columns",
    "clean_duplicates",
}


def _trim_messages_to_clean_round(messages: list[Any]) -> list[Any]:
    """裁剪历史 messages，只保留本轮 data_clean 子图相关的工具调用/响应链。

    避免上游 (agent_manager) 阶段的长篇 Markdown 总结（含 cleaning plan、
    业务建议、EDA/特征/建模建议等）污染上下文，导致 data_clean LLM
    走神变成「复读机」，把上游报告整段抄进 data_clean_agent_report.md。
    """
    if not messages:
        return []

    last_clean_tool_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            if msg.name in _CLEAN_TOOL_NAMES:
                last_clean_tool_idx = i
            break

    if last_clean_tool_idx is None:
        return []

    start = last_clean_tool_idx
    for j in range(last_clean_tool_idx - 1, -1, -1):
        msg = messages[j]
        if isinstance(msg, ToolMessage) and msg.name in _CLEAN_TOOL_NAMES:
            start = j
            continue
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            names = {tc.get("name") for tc in msg.tool_calls}
            if names & _CLEAN_TOOL_NAMES:
                start = j
                continue
        break

    return messages[start:]


def _merge_cleaning_report(
    state: GlobalState, tool_name: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = state.get("cleaning_target", "train")
    report_key = "train_cleaning_report" if target == "train" else "test_cleaning_report"
    existing = dict(state.get(report_key) or {})
    steps = list(existing.get("steps", []))
    steps.append({"tool": tool_name, "result": payload})
    existing["steps"] = steps
    return report_key, existing


def _current_report_key(state: GlobalState) -> str:
    target = state.get("cleaning_target", "train")
    return "train_cleaning_report" if target == "train" else "test_cleaning_report"


def _normalize_path_text(path: str) -> str:
    raw = (path or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve())
    except Exception:
        try:
            return str(Path(raw).expanduser())
        except Exception:
            return raw


def _expected_source_path(state: GlobalState) -> str:
    target = state.get("cleaning_target", "train")
    if target == "train":
        source = (
            (state.get("raw_training_data_path") or "").strip()
            or (state.get("training_data_path") or "").strip()
            or (state.get("data_path") or "").strip()
        )
    else:
        source = (
            (state.get("raw_testing_data_path") or "").strip()
            or (state.get("testing_data_path") or "").strip()
        )
    return _normalize_path_text(source)


def _opposite_split_prefix(target: str) -> str:
    return "cleaned_test_" if target == "train" else "cleaned_train_"


def _payload_dataset_id(payload: dict[str, Any]) -> str:
    return str(payload.get("dataset_id") or "").strip()


def _validate_load_payload(state: GlobalState, payload: dict[str, Any]) -> tuple[bool, str]:
    expected_source = _expected_source_path(state)
    got_source = _normalize_path_text(str(payload.get("source_path") or ""))
    if expected_source and got_source and got_source != expected_source:
        target = state.get("cleaning_target", "train")
        return (
            False,
            f"Current target={target} expects source_path={expected_source}, "
            f"but tool loaded source_path={got_source}.",
        )
    return True, ""


def _validate_dataset_id_alignment(state: GlobalState, payload: dict[str, Any]) -> tuple[bool, str]:
    expected_dataset_id = str(state.get("dataset_id") or "").strip()
    got_dataset_id = _payload_dataset_id(payload)
    if expected_dataset_id and got_dataset_id and got_dataset_id != expected_dataset_id:
        return (
            False,
            f"Expected dataset_id={expected_dataset_id} in current stage, "
            f"but got dataset_id={got_dataset_id}.",
        )
    return True, ""


def _validate_save_payload(state: GlobalState, payload: dict[str, Any]) -> tuple[bool, str]:
    ok, msg = _validate_dataset_id_alignment(state, payload)
    if not ok:
        return ok, msg

    target = state.get("cleaning_target", "train")
    output_path = str(payload.get("output_path") or "")
    name = Path(output_path).name.lower() if output_path else ""
    opposite_prefix = _opposite_split_prefix(target)
    if name.startswith(opposite_prefix):
        return (
            False,
            f"Current target={target} but output_path looks like opposite split: {output_path}",
        )
    return True, ""


def _save_payload_matches_stage_for_recovery(state: GlobalState, payload: dict[str, Any]) -> bool:
    valid, _ = _validate_save_payload(state, payload)
    return valid


def _build_rejected_tool_update(
    state: GlobalState,
    tool_name: str,
    stage: str,
    error_type: str,
    message: str,
    payload: dict[str, Any],
) -> GlobalState:
    rejected_payload = {
        "error": {"type": error_type, "message": message},
        "rejected_payload": payload,
    }
    report_key, report = _merge_cleaning_report(state, tool_name, rejected_payload)
    return {
        report_key: report,
        "current_stage": stage,
        "errors": (state.get("errors") or [])
        + [{"stage": stage.replace("_failed", ""), "tool": tool_name, "type": error_type, "message": message}],
    }


def _ensure_save_step_recorded(
    state: GlobalState, payload: dict[str, Any]
) -> tuple[str, dict[str, Any], bool]:
    report_key = _current_report_key(state)
    existing = dict(state.get(report_key) or {})
    steps = list(existing.get("steps", []))
    payload_dataset_id = _payload_dataset_id(payload)
    payload_output_path = str(payload.get("output_path") or "").strip()
    already_saved = any(
        isinstance(step, dict)
        and step.get("tool") == "save_dataset_from_store"
        and _payload_dataset_id(step.get("result") or {}) == payload_dataset_id
        and str((step.get("result") or {}).get("output_path") or "").strip() == payload_output_path
        for step in steps
    )
    if already_saved:
        return report_key, existing, False
    steps.append({"tool": "save_dataset_from_store", "result": payload})
    existing["steps"] = steps
    return report_key, existing, True


def _normalize_action_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _pick_text(action: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = action.get(key)
        if isinstance(raw, str):
            text = raw.strip()
            if text:
                return text
    return ""


def _normalize_type_conversions(value: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for action in _normalize_action_list(value):
        mapped = dict(action)
        column = _pick_text(action, "column", "column_name")
        target_type = _pick_text(action, "target_type", "new_type", "to", "conversion_type")
        if column:
            mapped["column"] = column
        if target_type:
            mapped["target_type"] = target_type
        normalized.append(mapped)
    return normalized


def _normalize_missing_value_actions(value: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for action in _normalize_action_list(value):
        mapped = dict(action)
        column = _pick_text(action, "column", "column_name")
        strategy = _pick_text(action, "strategy", "imputation_strategy", "action").lower()
        method = _pick_text(action, "method")
        if not method and strategy == "fill":
            method_alias = _pick_text(action, "imputation_strategy")
            if method_alias.lower() != "fill":
                method = method_alias

        if column:
            mapped["column"] = column
        if strategy:
            mapped["strategy"] = strategy
        if method:
            mapped["method"] = method
        if "value" not in mapped and "fill_value" in action:
            mapped["value"] = action.get("fill_value")

        if strategy == "fill_with_zero_if_no_garage":
            mapped.setdefault("condition_column", "GarageType")
            mapped.setdefault("condition_value", "NoGarage")
            mapped.setdefault("value", 0)

        if strategy == "fill_with_constant_or_median":
            mapped.setdefault("condition_column", "GarageType")
            mapped.setdefault("condition_value", "NoGarage")
            mapped.setdefault("condition_fill_value", 0)
            mapped.setdefault("fallback_method", "median")
            raw_value = mapped.get("value")
            if isinstance(raw_value, str) and raw_value.strip():
                mapped.setdefault("flag_column", raw_value.strip())

        normalized.append(mapped)
    return normalized


def _normalize_outlier_actions(value: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for action in _normalize_action_list(value):
        mapped = dict(action)
        column = _pick_text(action, "column", "column_name")
        if column:
            mapped["column"] = column
        normalized.append(mapped)
    return normalized


def _normalize_drop_columns(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            col = item.strip()
            if col:
                normalized.append(col)
            continue
        if isinstance(item, dict):
            col = _pick_text(item, "column", "column_name")
            if col:
                normalized.append(col)
    return normalized


def _normalize_cleaning_plan(cleaning_plan: Any) -> dict[str, Any]:
    if not isinstance(cleaning_plan, dict):
        return {
            "summary": "",
            "column_name_actions": [],
            "type_conversions": [],
            "missing_value_actions": [],
            "outlier_actions": [],
            "drop_columns": [],
            "notes": [],
        }

    notes_raw = cleaning_plan.get("notes")
    if isinstance(notes_raw, list):
        notes = [str(item) for item in notes_raw]
    elif notes_raw:
        notes = [str(notes_raw)]
    else:
        notes = []

    normalized = dict(cleaning_plan)
    normalized["summary"] = str(cleaning_plan.get("summary") or "")
    normalized["column_name_actions"] = _normalize_action_list(cleaning_plan.get("column_name_actions"))
    normalized["type_conversions"] = _normalize_type_conversions(cleaning_plan.get("type_conversions"))
    normalized["missing_value_actions"] = _normalize_missing_value_actions(
        cleaning_plan.get("missing_value_actions")
    )
    normalized["outlier_actions"] = _normalize_outlier_actions(cleaning_plan.get("outlier_actions"))
    normalized["drop_columns"] = _normalize_drop_columns(cleaning_plan.get("drop_columns"))
    normalized["notes"] = notes
    return normalized


def _data_clean_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    target = state.get("cleaning_target", "train")
    run_dir = state.get("run_dir") or ""
    dataset_id = state.get("dataset_id")
    description = state.get("description", "")
    user_request = state.get("user_request", "")

    dataset_name = (state.get("dataset_name") or "").strip()

    if target == "train":
        cleaning_plan = state.get("train_cleaning_plan") or state.get("cleaning_plan")
        source_path = (
            (state.get("raw_training_data_path") or "").strip()
            or (state.get("training_data_path") or "").strip()
            or (state.get("data_path") or "").strip()
        )
        if not dataset_name and source_path:
            dataset_name = Path(source_path).stem
        prefix = f"cleaned_train_{dataset_name or 'dataset'}"
    else:
        cleaning_plan = state.get("test_cleaning_plan") or state.get("cleaning_plan")
        source_path = (
            (state.get("raw_testing_data_path") or "").strip()
            or (state.get("testing_data_path") or "").strip()
        )
        if not dataset_name and source_path:
            dataset_name = Path(source_path).stem
        prefix = f"cleaned_test_{dataset_name or 'dataset'}"

    normalized_plan = _normalize_cleaning_plan(cleaning_plan) if cleaning_plan else None
    plan_text = (
        json.dumps(normalized_plan, ensure_ascii=False, indent=2, default=str)
        if normalized_plan is not None
        else "No cleaning_plan provided. Infer safe cleaning steps from the dataset profile."
    )

    if run_dir:
        default_save_path = f"{run_dir}/clean/{prefix}.csv"
    else:
        default_save_path = f"artifacts/clean/{prefix}.csv"

    supervisor_hint = (state.get("supervisor_hints") or {}).get(f"clean_{target}", "").strip()

    system_parts = [
        f"You are a data cleaning execution agent (ReAct). Current target: {target}.",
        f"Source file path: {source_path or 'N/A'}.",
        f"Current dataset_id in memory: {dataset_id or 'not loaded yet'}.",
        f"Dataset description: {description or 'N/A'}.",
        f"User request: {user_request or 'N/A'}.",
        f"Run directory: {run_dir or 'N/A'}. All output files must be saved under this directory.",
        *(
            [
                f"[Supervisor hint] Previous check failed: {supervisor_hint}\n"
                "Only fix the reported problem. Do not redo already completed steps.\n"
            ]
            if supervisor_hint
            else []
        ),
        "If dataset is not loaded, call load_dataset_to_store(file_path=...) first.",
        "Then execute cleaning tools according to the cleaning plan in this order as needed: "
        "clean_data_types, clean_missing_values, clean_outliers, clean_drop_columns, clean_duplicates.",
        "All tools must use the exact dataset_id returned by load_dataset_to_store.",
        "Strict split guard: in target=train, only load/clean/save the training source path; "
        "in target=test, only load/clean/save the testing source path.",
        "Never load the opposite split in the current run, and never summarize the opposite split "
        "in this stage's final response.",
        "If drop_columns exists in plan, call clean_drop_columns(dataset_id=..., columns=[...]).",
        f"After cleaning, call save_dataset_from_store and write CSV to output_path={default_save_path}.",
        "Tool-call contract: always pass explicit list args, even when empty.",
        "You are in data-clean stage only. Never call model/EDA/feature tools "
        "(e.g., train_model, tune_and_select_model, evaluate_model, encode_categorical, scale_features).",
        "Use the normalized cleaning_plan below as the source of truth for tool arguments.",
        "Mapping: type_conversions -> clean_data_types(dataset_id=..., conversions=[...]); "
        "missing_value_actions -> clean_missing_values(dataset_id=..., actions=[...], fallback_strategy='fill_zero'); "
        "outlier_actions -> clean_outliers(dataset_id=..., actions=[...]); "
        "drop_columns -> clean_drop_columns(dataset_id=..., columns=[...]).",
        "If a plan section is missing, call the corresponding tool with an empty list instead of omitting params.",
        f"\n--- cleaning_plan ({target}) ---\n",
        plan_text,
        "\nAfter all cleaning steps and save are complete, your final text response should summarize "
        "what was executed (columns touched, row/column shape changes, etc.). Do not mention EDA, "
        "feature engineering, or modeling. Only summarize the current target split.",
        "\nFinal-response strict format: ONLY a flat list of <=8 bullet points "
        "(plain '- ' lines). Do NOT output Markdown headings (no '#', no '##', no bold-titled "
        "sections). Do NOT echo any prior plan, business advice, recommended models, next steps, "
        "feature engineering advice, or modeling advice. Do NOT include emojis. "
        "Each bullet must describe a concrete data_clean action on the current target split only.",
    ]

    trimmed_history = _trim_messages_to_clean_round(messages)
    messages = [SystemMessage(content="".join(system_parts)), *trimmed_history]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    messages = state.get("messages") or []
    if not messages:
        return {}

    last_msg = messages[-1]
    if not isinstance(last_msg, ToolMessage):
        return {}

    handlers = {
        "load_dataset_to_store": _handle_load_dataset,
        "save_dataset_from_store": _handle_save_dataset,
        "clean_data_types": _handle_clean_step,
        "clean_missing_values": _handle_clean_step,
        "clean_outliers": _handle_clean_step,
        "clean_drop_columns": _handle_clean_step,
        "clean_duplicates": _handle_clean_step,
    }
    trailing_tool_messages: list[ToolMessage] = []
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        trailing_tool_messages.append(msg)
    trailing_tool_messages.reverse()

    running_state: GlobalState = dict(state)
    merged_update: dict[str, Any] = {}
    for msg in trailing_tool_messages:
        if not msg.name:
            continue
        handler = handlers.get(msg.name)
        if handler is None:
            continue
        delta = handler(running_state, msg)
        if not delta:
            continue
        running_state = {**running_state, **delta}
        merged_update = {**merged_update, **delta}
        if str(delta.get("current_stage") or "").endswith("_failed"):
            # Stop at first failed tool update in this batch to avoid cascading contamination.
            break

    return merged_update


def _parse_tool_payload(tool_message: ToolMessage) -> dict[str, Any]:
    raw_content = str(tool_message.content)
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raw_preview = raw_content if len(raw_content) <= 1000 else f"{raw_content[:1000]}... <truncated>"
        return {
            "error": {
                "type": exc.__class__.__name__,
                "message": "工具输出不是合法 JSON",
                "raw_content_preview": raw_preview,
            }
        }


def _handle_load_dataset(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    if "error" in payload:
        report_key, report = _merge_cleaning_report(state, tool_message.name, payload)
        return {
            report_key: report,
            "current_stage": "cleaning_load_failed",
            "errors": (state.get("errors") or []) + [{"stage": "cleaning_load", **payload["error"]}],
        }

    valid, reason = _validate_load_payload(state, payload)
    if not valid:
        return _build_rejected_tool_update(
            state=state,
            tool_name=tool_message.name,
            stage="cleaning_load_failed",
            error_type="StageSourceMismatch",
            message=reason,
            payload=payload,
        )

    loaded_dataset_id = _payload_dataset_id(payload)
    if not loaded_dataset_id:
        return _build_rejected_tool_update(
            state=state,
            tool_name=tool_message.name,
            stage="cleaning_load_failed",
            error_type="MissingDatasetId",
            message="load_dataset_to_store returned empty dataset_id.",
            payload=payload,
        )

    report_key, report = _merge_cleaning_report(state, tool_message.name, payload)
    return {
        "dataset_id": loaded_dataset_id,
        report_key: report,
        "current_stage": "dataset_loaded",
    }


def _handle_save_dataset(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    target = state.get("cleaning_target", "train")
    if "error" in payload:
        report_key, report = _merge_cleaning_report(state, tool_message.name, payload)
        return {
            report_key: report,
            "current_stage": "cleaning_save_failed",
            "errors": (state.get("errors") or []) + [{"stage": "cleaning_save", **payload["error"]}],
        }

    valid, reason = _validate_save_payload(state, payload)
    if not valid:
        return _build_rejected_tool_update(
            state=state,
            tool_name=tool_message.name,
            stage="cleaning_save_failed",
            error_type="StageDatasetMismatch",
            message=reason,
            payload=payload,
        )

    output_path = str(payload.get("output_path") or "").strip()
    if not output_path:
        return _build_rejected_tool_update(
            state=state,
            tool_name=tool_message.name,
            stage="cleaning_save_failed",
            error_type="MissingOutputPath",
            message="save_dataset_from_store returned empty output_path.",
            payload=payload,
        )

    report_key, report = _merge_cleaning_report(state, tool_message.name, payload)
    if target == "train":
        return {
            "train_cleaned_data_path": output_path,
            "training_data_path": output_path,  # update so downstream agents use cleaned data
            "cleaned_data_path": output_path,   # legacy compat
            report_key: report,
            "train_cleaning_passed": True,
            "cleaning_passed": True,
            "current_stage": "train_cleaning_saved",
            "next_action": "",
        }
    return {
        "test_cleaned_data_path": output_path,
        "testing_data_path": output_path,  # update so downstream agents use cleaned data
        report_key: report,
        "test_cleaning_passed": True,
        "cleaning_passed": True,
        "current_stage": "test_cleaning_saved",
        "next_action": "",
    }


def _handle_clean_step(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    if "error" in payload:
        report_key, report = _merge_cleaning_report(state, tool_message.name, payload)
        return {
            report_key: report,
            "current_stage": "cleaning_step_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "cleaning_step", "tool": tool_message.name, **payload["error"]}],
        }

    valid, reason = _validate_dataset_id_alignment(state, payload)
    if not valid:
        return _build_rejected_tool_update(
            state=state,
            tool_name=tool_message.name,
            stage="cleaning_step_failed",
            error_type="StageDatasetMismatch",
            message=reason,
            payload=payload,
        )

    report_key, report = _merge_cleaning_report(state, tool_message.name, payload)
    return {
        report_key: report,
        "current_stage": "cleaning_step_done",
    }


def _timer_start_clean(state: GlobalState) -> GlobalState:
    target = state.get("cleaning_target", "train")
    key = f"data_clean_{target}"
    timing = dict(state.get("timing") or {})
    timing[key] = {"start": time.time()}
    return {"timing": timing}


def _timer_end_clean(state: GlobalState) -> GlobalState:
    target = state.get("cleaning_target", "train")
    key = f"data_clean_{target}"
    timing = dict(state.get("timing") or {})
    entry = dict(timing.get(key) or {})
    now = time.time()
    entry["end"] = now
    elapsed = now - entry.get("start", now)
    print(f"[timer] {key:<28} {elapsed:6.1f}s")
    timing[key] = entry
    return {"timing": timing}


def _route_after_agent(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "self_check"


_MAX_INTERNAL_RETRY = 2


def _clean_self_check(state: GlobalState) -> GlobalState:
    target = state.get("cleaning_target", "train")
    internal_retry = dict(state.get("clean_internal_retry") or {})
    retry_count = internal_retry.get(target, 0)

    if target == "train":
        path = (state.get("train_cleaned_data_path") or "").strip()
        passed = bool(state.get("train_cleaning_passed"))
    else:
        path = (state.get("test_cleaned_data_path") or "").strip()
        passed = bool(state.get("test_cleaning_passed"))

    ok = passed and bool(path) and Path(path).exists()

    # If state wasn't updated but save tool actually ran (multi-tool-batch edge case),
    # recover the path directly from the most recent save ToolMessage.
    recovered: dict = {}
    if not ok:
        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, ToolMessage) and msg.name == "save_dataset_from_store":
                try:
                    payload = json.loads(str(msg.content))
                    saved_path = (payload.get("output_path") or "").strip()
                    if (
                        saved_path
                        and "error" not in payload
                        and Path(saved_path).exists()
                        and _save_payload_matches_stage_for_recovery(state, payload)
                    ):
                        ok = True
                        path = saved_path
                        report_key, report, changed = _ensure_save_step_recorded(state, payload)
                        if changed:
                            recovered[report_key] = report
                        if target == "train":
                            recovered = {
                                **recovered,
                                "train_cleaned_data_path": saved_path,
                                "training_data_path": saved_path,
                                "cleaned_data_path": saved_path,
                                "train_cleaning_passed": True,
                                "cleaning_passed": True,
                            }
                        else:
                            recovered = {
                                **recovered,
                                "test_cleaned_data_path": saved_path,
                                "testing_data_path": saved_path,
                                "test_cleaning_passed": True,
                                "cleaning_passed": True,
                            }
                except Exception:
                    pass
                if ok:
                    break

    if ok:
        save_agent_report("data_clean_agent", state["messages"][-1], run_dir=state.get("run_dir") or "")
        hints = dict(state.get("supervisor_hints") or {})
        hints[f"clean_{target}"] = ""
        internal_retry[target] = 0
        return {**recovered, "supervisor_hints": hints, "clean_internal_retry": internal_retry}

    if retry_count >= _MAX_INTERNAL_RETRY:
        save_agent_report("data_clean_agent", state["messages"][-1], run_dir=state.get("run_dir") or "")
        # Signal exhausted so router goes to timer_end
        internal_retry[target] = _MAX_INTERNAL_RETRY + 1
        return {"clean_internal_retry": internal_retry}

    issues: list[str] = []
    if not passed:
        issues.append(f"{target}_cleaning_passed is False or missing in state")
    if not path:
        issues.append(f"{target}_cleaned_data_path is missing in state")
    elif not Path(path).exists():
        issues.append(f"cleaned file does not exist: {path}")

    hint = (
        " ; ".join(issues)
        + ". Please verify save_dataset_from_store was called. "
        + f"Only complete the missing save step with output_path={path or 'the configured default path'}."
    )

    hints = dict(state.get("supervisor_hints") or {})
    hints[f"clean_{target}"] = hint
    internal_retry[target] = retry_count + 1
    print(f"[data_clean_agent] self_check fail (retry={retry_count + 1}/{_MAX_INTERNAL_RETRY})  {target}  {issues}")
    return {"supervisor_hints": hints, "clean_internal_retry": internal_retry}


def _route_after_self_check(state: GlobalState) -> str:
    target = state.get("cleaning_target", "train")
    hint = (state.get("supervisor_hints") or {}).get(f"clean_{target}", "")
    retry = (state.get("clean_internal_retry") or {}).get(target, 0)
    if hint and retry <= _MAX_INTERNAL_RETRY:
        return "data_clean_agent"
    return "timer_end"


def _show_graph_if_needed(agent_graph, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts" / "graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        mmd_path = output_dir / "data_clean_agent_graph.mmd"
        mmd_path.write_text(agent_graph.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"Mermaid 源码已保存: {mmd_path}")
        image_path = output_dir / "data_clean_agent_graph.png"
        image_bytes = agent_graph.get_graph().draw_mermaid_png()
        image_path.write_bytes(image_bytes)
        print(f"Graph 图片已保存: {image_path}")
        try:
            from IPython.display import Image, display

            display(Image(filename=str(image_path)))
        except Exception:
            pass
    except Exception as exc:
        print(f"图渲染失败: {exc}")


def _pretty_print_message(message) -> None:
    try:
        message.pretty_print()
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        text = message.pretty_repr()
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        print(safe_text)


def build_data_clean_agent(show_graph: bool = False):
    graph = StateGraph(GlobalState)
    graph.add_node("timer_start", _timer_start_clean)
    graph.add_node("data_clean_agent", _data_clean_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.add_node("self_check", _clean_self_check)
    graph.add_node("timer_end", _timer_end_clean)
    graph.set_entry_point("timer_start")
    graph.add_edge("timer_start", "data_clean_agent")
    graph.add_conditional_edges(
        "data_clean_agent", _route_after_agent, {"tools": "tools", "self_check": "self_check"}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "data_clean_agent")
    graph.add_conditional_edges(
        "self_check",
        _route_after_self_check,
        {"data_clean_agent": "data_clean_agent", "timer_end": "timer_end"},
    )
    graph.add_edge("timer_end", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


data_clean_agent = build_data_clean_agent()


if __name__ == "__main__":
    data_clean_agent = build_data_clean_agent(show_graph=True)
    user_input = input("Please enter request: ")
    events = data_clean_agent.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "data_path": str(PROJECT_ROOT / "datasets" / "Financials.csv"),
            "description": "Sample financial dataset with sales and date fields.",
            "cleaning_plan": {
                "summary": "Example: parse numeric-like text and convert dates.",
                "column_name_actions": [],
                "type_conversions": [
                    {"column": "Year", "target_type": "numeric"},
                    {"column": "Date", "target_type": "datetime"},
                ],
                "missing_value_actions": [],
                "outlier_actions": [],
                "drop_columns": [],
                "notes": [],
            },
            "cleaned_data_path": str(PROJECT_ROOT / "artifacts" / "Financials_cleaned.csv"),
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])

