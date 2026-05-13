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
from core.report_writer import save_agent_report, save_state_snapshot
from core.timer_utils import make_timer_start
from tools.model_tools.evaluate_model import evaluate_model
from tools.model_tools.train_model import train_model
from tools.model_tools.tune_and_select_model import tune_and_select_model

tools = [tune_and_select_model, train_model, evaluate_model]
llm_with_tools = llm_for("model").bind_tools(tools)


def _get_current_model(state: GlobalState) -> str | None:
    candidate_models = state.get("candidate_models") or []
    model_results = state.get("model_results") or {}
    remaining = [m for m in candidate_models if m not in model_results]
    return remaining[0] if remaining else None


def _model_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    task_type = (state.get("task_type") or "unknown").lower()
    target_column = state.get("target_column") or ""
    feature_dataset_ids = state.get("feature_dataset_ids") or {}
    testing_feature_dataset_ids = state.get("testing_feature_dataset_ids") or {}
    candidate_models = state.get("candidate_models") or []
    model_results = state.get("model_results") or {}
    best_model = state.get("best_model") or {}
    current_stage = state.get("current_stage") or ""

    # EDA Agent 决定的指标
    primary_metric = state.get("primary_metric") or ""
    evaluation_metrics = state.get("evaluation_metrics") or []
    metrics_rationale = state.get("metrics_rationale") or ""

    chosen_model = best_model.get("model_name") or ""

    run_dir = state.get("run_dir") or ""
    output_dir = f"{run_dir}/models" if run_dir else ""

    system_content = (
        "你是模型训练与评估 Agent。请严格遵循传统数据科学工作流：\n"
        "先在训练集上做 CV 调参与模型选择，分析结果后再用全量训练集重训最佳模型，最后仅一次使用测试集做最终评估。\n\n"
        "【任务信息】\n"
        f"任务类型：{task_type}。目标列：{target_column or '未提供'}。\n"
        f"候选模型：{candidate_models}。\n"
        f"当前阶段：{current_stage or '未开始'}。已有 model_results 键：{list(model_results.keys())}。\n"
        f"本次运行工作目录：{run_dir or '未设置'}。\n"
        f"各候选模型预处理后训练集路径：{json.dumps(feature_dataset_ids, ensure_ascii=False)}。\n"
        f"各候选模型预处理后测试集路径：{json.dumps(testing_feature_dataset_ids, ensure_ascii=False)}。\n"
        f"已选最佳模型：{chosen_model or '尚未选择'}。\n\n"
        "【EDA Agent 已确定的评估指标（直接使用，无需重新判断）】\n"
        f"primary_metric（CV 调优指标）：{primary_metric or '未设置，请使用任务默认值'}。\n"
        f"evaluation_metrics（综合评估指标列表）：{evaluation_metrics or '未设置，请使用任务默认指标'}。\n"
        f"metrics 选择理由：{metrics_rationale or '未提供'}。\n\n"
        "【执行步骤，按顺序完成，不跳步，不重复】\n\n"
        "步骤 1 — 若 current_stage 不是 model_selected / model_trained / model_evaluated，\n"
        "调用 tune_and_select_model，需自主制定以下 HPO 策略：\n"
        f"  · scoring【强制】= '{primary_metric or ''}' —— 这是 EDA Agent 已经决定的 primary_metric，"
        "直接传入，不得替换为其他指标，不得留空。\n"
        "  · cv_strategy：特征含时序列 → timeseries；否则 → auto\n"
        "  · cv_folds：训练集行数 < 1000 → 3；1000~20000 → 5；> 20000 → 10\n"
        "  · search_strategy 选择原则：\n"
        "    - 参数空间总组合数 ≤ 30（如 Ridge/Lasso/LogisticRegression）→ grid\n"
        "    - 希望最大化搜索质量、iter 预算 ≤ 20 → bayesian（GP 代理模型，最样本高效）\n"
        "    - 参数空间大（组合数 > 100）且 iter 预算充裕 → halving（逐轮淘汰）\n"
        "    - 其余均衡场景 → random\n"
        "    - 多个候选模型混合类型时，以参数空间最大的模型为准决定策略\n"
        "  · max_iter_per_model：参数维度 ≤ 2 → 12；维度 3~4 → 20；维度 ≥ 5 → 30\n"
        "    （search_strategy=grid 时忽略此参数）\n"
        "  · extra_param_overrides（可选）：按模型名注入/覆盖特定超参，其余参数保持默认搜索空间。\n"
        "    只在有明确数据证据时才传，不确定则留 None。常见场景：\n"
        "    - 不平衡分类（EDA 报告显示类别比例 > 5:1）→\n"
        '      {"LogisticRegression": {"class_weight": ["balanced", null]},\n'
        '       "RandomForest": {"class_weight": ["balanced", null]},\n'
        '       "SVM": {"class_weight": ["balanced", null]}}\n'
        "      （仅支持 class_weight 的模型：LogisticRegression/RandomForest/SVM/LightGBM）\n"
        "    - 训练集 > 50000 行，扩充树模型 n_estimators 上限 →\n"
        '      {"RandomForest": {"n_estimators": [200, 300, 500]},\n'
        '       "XGBoost": {"n_estimators": [200, 300, 500]}}\n'
        "    - 特征维度 > 100，RandomForest 加大 max_features 范围 →\n"
        '      {"RandomForest": {"max_features": ["sqrt", "log2", 0.3, 0.5]}}\n'
        "  · feature_dataset_ids、candidate_models、task_type、target_column 从上方任务信息传入\n\n"
        "步骤 2 — 收到 tune_and_select_model 结果后，在【同一次回复】中完成分析并调用 train_model\n"
        "  （不要先输出纯文字分析再单独调用工具，分析作为 tool_call 前的思考文本随工具调用一起返回）：\n"
        "  分析要点（简短即可）：\n"
        "  · 对比各模型 cv_best_score 和 cv_score_std，判断最佳模型优势是否显著\n"
        "  · 若最优与次优分数差 < 0.01，需说明选择依据\n"
        "  · 若某模型 cv_score_std > 0.03，需指出 CV 不稳定\n"
        "  · 给出最终采用哪个模型的结论，然后立即调用 train_model：\n"
        "  featured_train_path = feature_dataset_ids[best_model_name]，\n"
        "  model_name = best_model_name，model_params = best_model_params，\n"
        f"  output_dir = {output_dir!r}。\n\n"
        "步骤 3 — 若 current_stage 为 model_trained，调用 evaluate_model：\n"
        "  featured_test_data_path = testing_feature_dataset_ids[best_model_name]，\n"
        "  model_path、feature_columns_path 来自 train_model 返回值，\n"
        "  model_name、task_type、target_column 从上方任务信息传入，\n"
        f"  evaluation_metrics={evaluation_metrics}，primary_metric='{primary_metric}'。\n\n"
        "步骤 4 — 当 current_stage 为 model_evaluated 时，不再调用工具，\n"
        "  输出简短总结：最终模型名称、cv_best_score vs final_test_metrics 中各指标对比、\n"
        "  以及对泛化能力的简要判断（CV 分与 test 分差距是否合理）。\n"
        "  最终文字回复严格限于上述模型结果分析，不要延伸至业务建议或下一步操作说明。"
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
    if name:
        merged = dict(existing.get(name) or {})
        merged["model_path"] = payload.get("model_path", "")
        merged["feature_columns_path"] = payload.get("feature_columns_path", "")
        merged["model_params"] = payload.get("model_params", {})
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
    test_metrics = payload.get("metrics", {})
    confusion_matrix = payload.get("confusion_matrix")
    model_entry = dict(existing.get(name) or {})
    model_entry["model_path"] = payload.get("model_path", model_entry.get("model_path", ""))
    # overwrite the cv-score snapshot in "metrics" with definitive test results so that
    # business_translator_agent always reads the most authoritative score per model.
    # cv data is preserved separately in cv_best_score and selection_summary fields.
    model_entry["metrics"] = test_metrics
    model_entry["final_test_metrics"] = test_metrics
    if confusion_matrix is not None:
        model_entry["confusion_matrix"] = confusion_matrix
    existing[name] = model_entry
    best_model = dict(state.get("best_model") or {})
    if best_model.get("model_name") == name or not best_model.get("model_name"):
        best_model["model_name"] = name
        best_model["model_path"] = payload.get("model_path", "")
        best_model["metrics"] = test_metrics
        best_model["final_test_metrics"] = test_metrics
        if confusion_matrix is not None:
            best_model["confusion_matrix"] = confusion_matrix
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
        merged = dict(existing.get(model_name) or {})
        merged["metrics"] = {selection_metric: cv_best}
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
    return "finalize"


def _model_finalize_node(state: GlobalState) -> GlobalState:
    """Pipeline 收尾：存 model_report、补全 best_model.task_type、保存 state 快照。"""
    last_msg = state["messages"][-1]
    model_report_text = str(last_msg.content) if hasattr(last_msg, "content") else ""

    best_model = dict(state.get("best_model") or {})
    task_type = (state.get("task_type") or "regression").lower()
    best_model["task_type"] = task_type

    model_name = best_model.get("model_name", "unknown")
    run_dir = state.get("run_dir") or ""

    save_agent_report("model_agent", last_msg, model=model_name, run_dir=run_dir)

    timing = dict(state.get("timing") or {})
    entry = dict(timing.get("model_agent") or {})
    now = time.time()
    entry["end"] = now
    elapsed = now - entry.get("start", now)
    print(f"[timer] {'model_agent':<28} {elapsed:6.1f}s")
    timing["model_agent"] = entry

    updated: GlobalState = {
        "model_report": model_report_text,
        "best_model": best_model,
        "current_stage": "model_done",
        "timing": timing,
    }
    save_state_snapshot({**state, **updated})
    return updated


def _show_graph_if_needed(agent_graph, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts" / "graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        mmd_path = output_dir / "model_agent_graph.mmd"
        mmd_path.write_text(agent_graph.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"Mermaid 源码已保存: {mmd_path}")
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
    graph.add_node("timer_start", make_timer_start("model_agent"))
    graph.add_node("model_agent", _model_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.add_node("finalize", _model_finalize_node)
    graph.set_entry_point("timer_start")
    graph.add_edge("timer_start", "model_agent")
    graph.add_conditional_edges(
        "model_agent", _route_after_model, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "model_agent")
    graph.add_edge("finalize", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


model_agent = build_model_agent()
