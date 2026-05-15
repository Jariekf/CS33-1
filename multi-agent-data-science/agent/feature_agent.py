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
from tools.dataclean_tools.load_dataset_to_store import load_dataset_to_store
from tools.feature_tools.encode_categorical import encode_categorical
from tools.feature_tools.encode_target_column import encode_target_column
from tools.feature_tools.save_featured_dataset import save_featured_dataset
from tools.feature_tools.scale_features import scale_features

tools = [
    load_dataset_to_store,
    encode_target_column,
    scale_features,
    encode_categorical,
    save_featured_dataset,
]
llm_with_tools = llm_for("feature").bind_tools(tools)

_FEATURE_TOOL_NAMES = {
    "load_dataset_to_store",
    "encode_target_column",
    "scale_features",
    "encode_categorical",
    "save_featured_dataset",
}
_MAX_NO_TOOL_ROUNDS = 2


def _get_current_model(state: GlobalState) -> str | None:
    candidate_models = state.get("candidate_models") or []
    feature_dataset_ids = state.get("feature_dataset_ids") or {}
    remaining = [m for m in candidate_models if m not in feature_dataset_ids]
    return remaining[0] if remaining else None


def _trim_messages_to_current_round(messages: list[Any]) -> list[Any]:
    """裁剪历史 messages，只保留本轮 feature 子图相关的工具调用/响应链。

    避免上游 (EDA / data_clean) 阶段输出的长篇 Markdown 报告污染上下文，
    导致 LLM 走神变成「复读机」。
    """
    if not messages:
        return []

    last_feature_tool_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            if msg.name in _FEATURE_TOOL_NAMES:
                last_feature_tool_idx = i
            break

    if last_feature_tool_idx is None:
        return []

    start = last_feature_tool_idx
    for j in range(last_feature_tool_idx - 1, -1, -1):
        msg = messages[j]
        if isinstance(msg, ToolMessage) and msg.name in _FEATURE_TOOL_NAMES:
            start = j
            continue
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            names = {tc.get("name") for tc in msg.tool_calls}
            if names & _FEATURE_TOOL_NAMES:
                start = j
                continue
        break

    return messages[start:]


def _feature_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    training_data_path = state.get("training_data_path")
    testing_data_path = state.get("testing_data_path")
    current_model = _get_current_model(state)
    feature_engineering_plan = state.get("feature_engineering_plan") or {}
    task_type = (state.get("task_type") or "unknown").lower()
    target_column = state.get("target_column") or ""
    candidate_models = state.get("candidate_models") or []
    feature_dataset_ids = state.get("feature_dataset_ids") or {}

    # ── 空转检测：若上一轮 LLM 没调用工具，但当前模型仍未保存特征，累计计数 ──
    no_tool_rounds = dict(state.get("feature_no_tool_rounds") or {})
    if current_model and messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and not getattr(last_msg, "tool_calls", None):
            no_tool_rounds[current_model] = no_tool_rounds.get(current_model, 0) + 1

    # ── 达到上限：强制把当前模型标 skipped，避免无限循环 ──────────────────────
    if current_model and no_tool_rounds.get(current_model, 0) >= _MAX_NO_TOOL_ROUNDS:
        ids = dict(feature_dataset_ids)
        ids[current_model] = "skipped"
        print(
            f"[feature_agent] {current_model} 空转 "
            f"{no_tool_rounds[current_model]} 次未调用工具，强制标记 skipped"
        )
        return {
            "feature_dataset_ids": ids,
            "feature_no_tool_rounds": {},
            "messages": [
                AIMessage(
                    content=(
                        f"[forced skip] 模型 {current_model} 连续 "
                        f"{no_tool_rounds[current_model]} 轮未调用工具，已标记为 skipped。"
                    )
                )
            ],
        }

    model_plan = feature_engineering_plan.get(current_model, {}) if current_model else {}
    plan_text = json.dumps(model_plan, ensure_ascii=False, indent=2) if model_plan else "无特定计划，按需处理。"
    run_dir = state.get("run_dir") or ""
    output_dir = run_dir + "/features" if run_dir else ""

    has_test = bool(testing_data_path)
    test_load_step = (
        f"   然后调用 load_dataset_to_store 将测试集（{testing_data_path}）加载到内存，获取 test_id；\n"
        if has_test else ""
    )
    test_id_arg = "test_dataset_id=test_id（从步骤 1b 获取）" if has_test else "test_dataset_id 不传（无测试集）"
    test_save_step = (
        f"5. 调用 save_featured_dataset(dataset_id=test_id, model_name=当前模型名, "
        f"output_dir=\"{output_dir}\", split=\"test\") 保存预处理后的测试集。\n"
        if has_test else ""
    )

    supervisor_hint = (state.get("supervisor_hints") or {}).get(f"feature:{current_model}", "").strip()
    hint_line = (
        f"[Supervisor 反馈] 上一轮检查未通过：{supervisor_hint}\n请仅针对此问题修复，不要重新执行已完成的步骤。\n"
        if supervisor_hint else ""
    )

    system_content = (
        f"你是特征工程 Agent，当前任务是为模型 [{current_model}] 同步处理训练集和测试集特征。\n"
        f"训练集路径：{training_data_path or '未提供'}。\n"
        f"测试集路径：{testing_data_path or '未提供（跳过测试集步骤）'}。\n"
        f"任务类型：{task_type}。目标列：{target_column or '未提供'}。\n"
        f"全部候选模型：{candidate_models}，已完成：{list(feature_dataset_ids.keys())}。\n"
        f"本次运行工作目录：{run_dir or '未设置'}。\n"
        + hint_line
        + f"\n当前模型 [{current_model}] 的特征工程计划：\n{plan_text}\n"
        "工作流程（按顺序执行）：\n"
        f"1. 调用 load_dataset_to_store 将训练集（{training_data_path}）加载到内存，获取 train_id；\n"
        f"{test_load_step}"
        "2. 调用 encode_target_column，传入：\n"
        f"   dataset_id=train_id, target_column={target_column}, model_name={current_model},\n"
        f"   task_type={task_type}, run_dir=\"{run_dir}\", {test_id_arg}；\n"
        "   - XGBoost、XGBoostRegressor 需要 0/1 整数标签，工具自动编码并同步处理测试集；\n"
        "   - 其他模型返回 status=skipped，测试集目标列不变；\n"
        "3. 根据上方计划，按需调用 scale_features 和 encode_categorical，均传入：\n"
        f"   dataset_id=train_id, run_dir=\"{run_dir}\", {test_id_arg}；\n"
        "   - 目标列不参与特征缩放和类别编码；\n"
        "   - 树模型（XGBoost、RandomForest）通常不需要 scale_features；\n"
        "   - 线性模型（LogisticRegression、LinearRegression、Ridge）通常需要 scale_features；\n"
        "   - scale_features 和 encode_categorical 均在训练集上 fit，同步 transform 测试集；\n"
        f"4. 调用 save_featured_dataset(dataset_id=train_id, model_name=当前模型名, "
        f"output_dir=\"{output_dir}\", split=\"train\") 保存预处理后的训练集；\n"
        f"{test_save_step}"
        "重要约束：你必须通过 tool_call 调用工具完成上述步骤，禁止只输出文字描述假装已完成。\n"
        "完成后停止，不要处理其他模型。\n"
        "最终文字回复只汇报当前模型的特征工程处理步骤与结果（编码方式、缩放情况、保存路径），"
        "不要描述模型训练评估或其他候选模型的处理细节，并且不要说与数据清洗和EDA相关内容，只需要汇报特征工程处理步骤与结果。"
    )

    # ── 关键修复：只把本轮 feature 工具调用链送给 LLM ─────────────────────────
    # 这样可避免上游 EDA/data_clean 阶段输出的长篇 Markdown 干扰当前轮决策
    trimmed_history = _trim_messages_to_current_round(messages)
    if trimmed_history:
        llm_input = [SystemMessage(content=system_content), *trimmed_history]
    else:
        llm_input = [
            SystemMessage(content=system_content),
            HumanMessage(
                content=(
                    f"请开始处理模型 [{current_model}] 的特征工程，"
                    "严格按上述系统提示词第 1~5 步逐个调用工具，禁止只输出文字描述。"
                )
            ),
        ]

    response = llm_with_tools.invoke(llm_input)
    return {
        "messages": [response],
        "feature_no_tool_rounds": no_tool_rounds,
    }


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage) or not last_msg.name:
        return {}

    handlers = {
        "load_dataset_to_store": _handle_load,
        "encode_target_column": _handle_target_encode,
        "scale_features": _handle_feature_step,
        "encode_categorical": _handle_feature_step,
        "save_featured_dataset": _handle_save_featured,
    }

    result = {}
    handler = handlers.get(last_msg.name)
    if handler:
        result = handler(state, last_msg)

    # LLM may batch multiple tool_calls in one AIMessage (e.g. encode + save_train + save_test).
    # ToolNode adds one ToolMessage per call; only the last is processed above.
    # Scan the rest of the batch specifically for save_featured_dataset so that
    # feature_dataset_ids / testing_feature_dataset_ids are always written.
    messages = state["messages"]
    for msg in reversed(messages[:-1]):
        if not isinstance(msg, ToolMessage):
            break
        if msg.name == "save_featured_dataset":
            save_result = _handle_save_featured(state, msg)
            for k, v in save_result.items():
                if k not in result:
                    result[k] = v
                elif isinstance(v, dict) and isinstance(result.get(k), dict):
                    result[k] = {**result[k], **v}

    return result


def _parse_payload(tool_message: ToolMessage) -> dict[str, Any]:
    try:
        return json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        return {"error": {"type": "JSONDecodeError", "message": "工具输出不是合法 JSON"}}


def _handle_load(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "feature_load_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "feature_load", **payload["error"]}],
        }
    return {
        "dataset_id": payload["dataset_id"],
        "current_stage": "feature_dataset_loaded",
    }


def _handle_target_encode(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "target_encode_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "target_encode", **payload["error"]}],
        }

    current_model = _get_current_model(state)
    existing = dict(state.get("feature_preprocessors") or {})
    model_prep = dict(existing.get(current_model) or {})

    if payload.get("status") == "encoded":
        model_prep["target_encoder_path"] = payload.get("encoder_path", "")
        model_prep["target_original_classes"] = payload.get("original_classes", [])

    existing[current_model] = model_prep
    return {
        "feature_preprocessors": existing,
        "current_stage": "target_encode_done",
    }


def _handle_feature_step(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "feature_step_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "feature_step", "tool": tool_message.name, **payload["error"]}],
        }

    current_model = _get_current_model(state)
    existing = dict(state.get("feature_preprocessors") or {})
    model_prep = dict(existing.get(current_model) or {})

    if tool_message.name == "scale_features":
        model_prep["scaler_path"] = payload.get("scaler_path", "")
        model_prep["scaler_method"] = payload.get("method", "")
        model_prep["scaled_columns"] = payload.get("scaled_columns", [])
    elif tool_message.name == "encode_categorical":
        model_prep["encoder_paths"] = payload.get("encoder_paths", {})
        model_prep["encoding_method"] = payload.get("method", "")
        model_prep["encoded_columns"] = payload.get("encoded_columns", [])
        model_prep["new_columns"] = payload.get("new_columns", [])

    existing[current_model] = model_prep
    return {
        "feature_preprocessors": existing,
        "current_stage": "feature_step_done",
    }


def _handle_save_featured(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "feature_save_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "feature_save", **payload["error"]}],
        }
    model_name = payload["model_name"]
    output_path = payload["output_path"]
    split = payload.get("split", "train")

    if split == "test":
        existing = dict(state.get("testing_feature_dataset_ids") or {})
        existing[model_name] = output_path
        return {
            "testing_feature_dataset_ids": existing,
            "current_stage": "test_feature_saved",
        }
    else:
        existing = dict(state.get("feature_dataset_ids") or {})
        existing[model_name] = output_path
        return {
            "feature_dataset_ids": existing,
            "current_stage": "feature_saved",
            "feature_no_tool_rounds": {},
        }


def _timer_start_feature(state: GlobalState) -> GlobalState:
    model = _get_current_model(state) or "unknown"
    key = f"feature_{model}"
    timing = dict(state.get("timing") or {})
    timing[key] = {"start": time.time()}
    timing["_pending_feature_key"] = key  # timer_end reads this directly
    return {"timing": timing}


def _timer_end_feature(state: GlobalState) -> GlobalState:
    timing = dict(state.get("timing") or {})
    key = timing.pop("_pending_feature_key", None)
    if key is None:
        # fallback: should not normally happen
        feature_ids = state.get("feature_dataset_ids") or {}
        model = next(reversed(feature_ids), "unknown")
        key = f"feature_{model}"
    entry = dict(timing.get(key) or {})
    now = time.time()
    entry["end"] = now
    elapsed = now - entry.get("start", now)
    print(f"[timer] {key:<28} {elapsed:6.1f}s")
    timing[key] = entry
    return {"timing": timing}


def _route_after_feature(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    feature_ids = state.get("feature_dataset_ids") or {}
    model = next(reversed(feature_ids), "unknown")
    save_agent_report("feature_agent", last_msg, model=model, run_dir=state.get("run_dir") or "")
    return "timer_end"


def _show_graph_if_needed(agent_graph, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts" / "graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        mmd_path = output_dir / "feature_agent_graph.mmd"
        mmd_path.write_text(agent_graph.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"Mermaid 源码已保存: {mmd_path}")
        image_path = output_dir / "feature_agent_graph.png"
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


def build_feature_agent(show_graph: bool = False):
    graph = StateGraph(GlobalState)
    graph.add_node("timer_start", _timer_start_feature)
    graph.add_node("feature_agent", _feature_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.add_node("timer_end", _timer_end_feature)
    graph.set_entry_point("timer_start")
    graph.add_edge("timer_start", "feature_agent")
    graph.add_conditional_edges(
        "feature_agent", _route_after_feature, {"tools": "tools", "timer_end": "timer_end"}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "feature_agent")
    graph.add_edge("timer_end", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


feature_agent = build_feature_agent()


if __name__ == "__main__":
    feature_agent = build_feature_agent(show_graph=True)
    events = feature_agent.stream(
        {
            "messages": [HumanMessage(content="开始特征工程")],
            "training_data_path": str(PROJECT_ROOT / "artifacts" / "workspace" / "Financials_cleaned_train.csv"),
            "task_type": "regression",
            "target_column": "Profit",
            "candidate_models": ["XGBoost", "RandomForest", "Ridge"],
            "feature_dataset_ids": {},
            "feature_engineering_plan": {
                "XGBoost": {"scaling": False, "encoding": "label", "notes": "树模型无需缩放"},
                "RandomForest": {"scaling": False, "encoding": "label", "notes": "树模型无需缩放"},
                "Ridge": {"scaling": True, "encoding": "onehot", "notes": "线性模型需标准化"},
            },
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
        if event.get("current_stage"):
            print(f"\n[stage] {event['current_stage']}")
        if event.get("feature_dataset_ids"):
            print(f"[feature_ids] {event['feature_dataset_ids']}")
