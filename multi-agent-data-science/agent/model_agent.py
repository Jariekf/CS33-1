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

from core.global_state import GlobalState
from core.my_llm import llm_deepseek
from core.report_writer import save_agent_report
from tools.model_tools.evaluate_model import evaluate_model
from tools.model_tools.train_model import train_model

tools = [train_model, evaluate_model]
llm_with_tools = llm_deepseek.bind_tools(tools)


def _get_current_model(state: GlobalState) -> str | None:
    candidate_models = state.get("candidate_models") or []
    model_results = state.get("model_results") or {}
    remaining = [m for m in candidate_models if m not in model_results]
    return remaining[0] if remaining else None


def _model_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    current_model = _get_current_model(state)
    task_type = (state.get("task_type") or "unknown").lower()
    target_column = state.get("target_column") or ""
    testing_data_path = state.get("testing_data_path") or ""
    feature_dataset_ids = state.get("feature_dataset_ids") or {}
    feature_preprocessors = state.get("feature_preprocessors") or {}
    candidate_models = state.get("candidate_models") or []
    model_results = state.get("model_results") or {}

    featured_train_path = feature_dataset_ids.get(current_model, "")
    preprocessor = feature_preprocessors.get(current_model, {})
    preprocessor_text = json.dumps(preprocessor, ensure_ascii=False, indent=2) if preprocessor else "无（该模型不需要预处理）"

    run_dir = state.get("run_dir") or ""

    system_content = (
        f"你是模型训练与评估 Agent，当前任务是训练并评估模型 [{current_model}]。\n"
        f"任务类型：{task_type}。目标列：{target_column or '未提供'}。\n"
        f"全部候选模型：{candidate_models}，已完成：{list(model_results.keys())}。\n"
        f"本次运行工作目录：{run_dir or '未设置'}。\n"
        f"\n当前模型 [{current_model}] 的特征工程后训练集路径：{featured_train_path or '未提供'}。\n"
        f"测试集路径：{testing_data_path or '未提供'}。\n"
        f"\n当前模型的预处理参数（评估测试集时需传入）：\n{preprocessor_text}\n"
        "工作流程（按顺序执行）：\n"
        f"1. 调用 train_model 在特征工程后的训练集上训练模型，"
        f"output_dir 传入：{run_dir + '/models' if run_dir else ''}，获取 model_path 和 feature_columns_path；\n"
        "2. 调用 evaluate_model 在测试集上评估，传入 model_path、feature_columns_path、"
        "以及上方的预处理参数（scaler_path、scaled_columns、encoder_paths、encoding_method、encoded_columns）；\n"
        "   - 若预处理参数中含 target_encoder_path，必须传给 evaluate_model 的 target_encoder_path 字段，\n"
        "     这样测试集目标列才能用相同 encoder 转换为 0/1，确保 y_test 与模型预测值类型一致；\n"
        "   - 无预处理参数时对应字段留空。\n"
        "完成后停止，不要处理其他模型。"
    )

    messages = [SystemMessage(content=system_content), *messages]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage) or not last_msg.name:
        return {}

    handlers = {
        "train_model": _handle_train,
        "evaluate_model": _handle_evaluate,
    }
    handler = handlers.get(last_msg.name)
    return handler(state, last_msg) if handler else {}


def _parse_payload(tool_message: ToolMessage) -> dict[str, Any]:
    try:
        return json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        return {"error": {"type": "JSONDecodeError", "message": "工具输出不是合法 JSON"}}


def _handle_train(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "model_train_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "model_train", **payload["error"]}],
        }
    return {"current_stage": "model_trained"}


def _handle_evaluate(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "model_eval_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "model_eval", **payload["error"]}],
        }
    existing = dict(state.get("model_results") or {})
    existing[payload["model_name"]] = {
        "model_path": payload["model_path"],
        "metrics": payload["metrics"],
    }
    return {
        "model_results": existing,
        "current_stage": "model_evaluated",
    }


def _route_after_model(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    # last key in model_results is the model just evaluated
    model_results = state.get("model_results") or {}
    model = next(reversed(model_results), "unknown")
    save_agent_report("model_agent", last_msg, model=model, run_dir=state.get("run_dir") or "")
    return END


def _show_graph_if_needed(agent_graph, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "model_agent_graph.png"
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


def build_model_agent(show_graph: bool = False):
    graph = StateGraph(GlobalState)
    graph.add_node("model_agent", _model_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.set_entry_point("model_agent")
    graph.add_conditional_edges(
        "model_agent", _route_after_model, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "model_agent")
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


model_agent = build_model_agent()
