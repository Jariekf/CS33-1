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
from tools.model_tools.tune_and_select_model import tune_and_select_model

tools = [tune_and_select_model, train_model, evaluate_model]
llm_with_tools = llm_deepseek.bind_tools(tools)


def _get_current_model(state: GlobalState) -> str | None:
    candidate_models = state.get("candidate_models") or []
    model_results = state.get("model_results") or {}
    remaining = [m for m in candidate_models if m not in model_results]
    return remaining[0] if remaining else None


def _model_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    task_type = (state.get("task_type") or "unknown").lower()
    target_column = state.get("target_column") or ""
    testing_data_path = state.get("testing_data_path") or ""
    feature_dataset_ids = state.get("feature_dataset_ids") or {}
    feature_preprocessors = state.get("feature_preprocessors") or {}
    candidate_models = state.get("candidate_models") or []
    model_results = state.get("model_results") or {}
    best_model = state.get("best_model") or {}
    current_stage = state.get("current_stage") or ""

    chosen_model = best_model.get("model_name") or ""
    chosen_preprocessor = feature_preprocessors.get(chosen_model, {})
    preprocessor_text = (
        json.dumps(chosen_preprocessor, ensure_ascii=False, indent=2)
        if chosen_preprocessor
        else "无（该模型不需要预处理）"
    )

    run_dir = state.get("run_dir") or ""
    output_dir = f"{run_dir}/models" if run_dir else ""

    system_content = (
        "你是模型训练与评估 Agent。请严格遵循传统数据科学工作流："
        "先在训练集上做候选模型调参和选择，再用全量训练集重训最佳模型，最后仅一次使用测试集做最终评估。\n"
        f"任务类型：{task_type}。目标列：{target_column or '未提供'}。\n"
        f"全部候选模型：{candidate_models}。\n"
        f"当前阶段：{current_stage or '未开始'}。已有 model_results 键：{list(model_results.keys())}。\n"
        f"本次运行工作目录：{run_dir or '未设置'}。\n"
        f"\n各候选模型对应的特征训练集路径：{json.dumps(feature_dataset_ids, ensure_ascii=False)}。\n"
        f"测试集路径：{testing_data_path or '未提供'}。\n"
        f"\n已选最佳模型：{chosen_model or '尚未选择'}，其预处理参数（最终评估时使用）：\n{preprocessor_text}\n"
        "按以下顺序执行，不要跳步，不要重复：\n"
        "1) 若还未完成模型选择（current_stage 不是 model_selected / model_trained / model_evaluated），"
        "先调用 tune_and_select_model："
        "传入 feature_dataset_ids、candidate_models、task_type、target_column，"
        "并使用 cv_folds=5、max_iter_per_model=12；\n"
        "2) 若已选出 best_model 且尚未训练最终模型（current_stage 不是 model_trained / model_evaluated），"
        "调用 train_model："
        "featured_train_path 使用 feature_dataset_ids[best_model_name]，"
        "model_name 使用 best_model_name，"
        "model_params 使用 best_model_params，"
        f"output_dir 传入 {output_dir!r}；\n"
        "3) 若已训练最终模型且尚未最终评估（current_stage 不是 model_evaluated），调用 evaluate_model："
        "传入 model_path、feature_columns_path、test_data_path、model_name、task_type、target_column，"
        "并传入最佳模型对应的预处理参数（scaler_path、scaled_columns、encoder_paths、encoding_method、encoded_columns）；"
        "若预处理参数中含 target_encoder_path，必须传给 evaluate_model 的 target_encoder_path；\n"
        "4) 当 current_stage 为 model_evaluated 时，不再调用工具，直接给出一句完成说明。"
    )

    messages = [SystemMessage(content=system_content), *messages]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage) or not last_msg.name:
        return {}

    handlers = {
        "tune_and_select_model": _handle_tune_and_select,
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
    best_model = dict(state.get("best_model") or {})
    best_model.update(
        {
            "model_name": payload.get("model_name", best_model.get("model_name", "")),
            "model_path": payload.get("model_path", ""),
            "feature_columns_path": payload.get("feature_columns_path", ""),
            "model_params": payload.get("model_params", best_model.get("model_params", {})),
        }
    )
    existing = dict(state.get("model_results") or {})
    name = payload.get("model_name", "")
    if name and name in existing:
        merged = dict(existing[name])
        merged["model_path"] = payload.get("model_path", "")
        merged["feature_columns_path"] = payload.get("feature_columns_path", "")
        merged["model_params"] = payload.get("model_params", {})
        merged["train_score"] = payload.get("train_score")
        existing[name] = merged
    return {
        "best_model": best_model,
        "model_results": existing,
        "current_stage": "model_trained",
    }


def _handle_evaluate(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "model_eval_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "model_eval", **payload["error"]}],
        }
    existing = dict(state.get("model_results") or {})
    name = payload.get("model_name", "unknown")
    model_entry = dict(existing.get(name) or {})
    # keep selection metrics for compare node; store final test metrics separately
    model_entry["model_path"] = payload.get("model_path", model_entry.get("model_path", ""))
    model_entry["final_test_metrics"] = payload.get("metrics", {})
    existing[name] = model_entry
    best_model = dict(state.get("best_model") or {})
    if best_model.get("model_name") == name or not best_model.get("model_name"):
        best_model["model_name"] = name
        best_model["model_path"] = payload.get("model_path", "")
        best_model["final_test_metrics"] = payload.get("metrics", {})
    return {
        "best_model": best_model,
        "model_results": existing,
        "current_stage": "model_evaluated",
    }


def _handle_tune_and_select(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "model_select_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "model_select", **payload["error"]}],
        }

    selection_metric = payload.get("selection_metric", "")
    per_model_summary = payload.get("per_model_summary") or {}
    existing = dict(state.get("model_results") or {})
    for model_name, summary in per_model_summary.items():
        cv_best = float(summary.get("cv_best_score", -1))
        metrics = (
            {"f1_weighted": cv_best}
            if selection_metric == "f1_weighted"
            else {"r2": cv_best}
        )
        merged = dict(existing.get(model_name) or {})
        merged["metrics"] = metrics
        merged["selection_metric"] = selection_metric
        merged["cv_best_score"] = cv_best
        merged["model_params"] = summary.get("best_params", {})
        merged["selection_summary"] = summary
        existing[model_name] = merged

    best_name = payload.get("best_model_name", "")
    best_model = {
        "model_name": best_name,
        "model_params": payload.get("best_model_params", {}),
        "selection_metric": selection_metric,
        "selection_score": payload.get("best_cv_score"),
    }
    return {
        "best_model": best_model,
        "model_results": existing,
        "current_stage": "model_selected",
    }


def _route_after_model(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    # last key in model_results is the model just evaluated
    model = (state.get("best_model") or {}).get("model_name", "unknown")
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
