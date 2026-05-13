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
from core.my_llm import llm_for
from core.report_writer import save_agent_report
from core.timer_utils import make_timer_end, make_timer_start
from tools.eda_tools.compute_correlation_matrix import compute_correlation_matrix
from tools.eda_tools.generate_eda_charts import generate_eda_charts
from tools.eda_tools.profile_training_data import profile_training_data
from tools.eda_tools.select_candidate_models_and_metrics import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_POOL,
    REGRESSION_METRICS,
    REGRESSION_POOL,
    select_candidate_models_and_metrics,
)

tools = [
    profile_training_data,
    compute_correlation_matrix,
    generate_eda_charts,
    select_candidate_models_and_metrics,
]
llm_with_tools = llm_for("eda").bind_tools(tools)


def _merge_eda_report(state: GlobalState, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    existing = dict(state.get("eda_report") or {})
    steps = list(existing.get("steps", []))
    steps.append({"tool": tool_name, "result": payload})
    existing["steps"] = steps
    if tool_name == "generate_eda_charts" and isinstance(payload.get("charts"), list):
        existing["charts"] = payload["charts"]
        existing["chart_output_dir"] = payload.get("output_dir", "")
    return existing


def _eda_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    training_data_path = state.get("training_data_path")
    task_type = (state.get("task_type") or "unknown").lower()
    target_column = state.get("target_column") or ""
    description = state.get("description") or ""
    testing_data_path = state.get("testing_data_path") or ""

    pool = CLASSIFICATION_POOL if task_type == "classification" else REGRESSION_POOL
    pool_str = ", ".join(pool)
    metrics_allowed = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
    metrics_str = ", ".join(sorted(metrics_allowed))

    run_dir = state.get("run_dir") or ""

    supervisor_hint = (state.get("supervisor_hints") or {}).get("eda", "").strip()
    hint_line = (
        f"[Supervisor 反馈] 上一轮检查未通过：{supervisor_hint}\n"
        "请确保依次完成所有 4 个工具调用后再输出最终回复。\n"
        if supervisor_hint else ""
    )

    system_content = (
        "你是 EDA（探索性数据分析）Agent，负责分析训练集、制定特征工程计划、选出三个候选模型，"
        "并决定本次任务的评估指标。\n"
        f"训练集路径：{training_data_path or '未提供'}。\n"
        f"数据集描述：{description or '未提供'}。\n"
        f"任务类型：{task_type}。\n"
        f"目标列：{target_column or '未提供'}。\n"
        f"本次运行工作目录：{run_dir or '未设置'}。\n"
        + hint_line
        + "工作流程（按顺序执行）：\n"
        "1. 调用 profile_training_data 获取训练集数据画像；\n"
        "2. 调用 compute_correlation_matrix 计算特征相关矩阵（传入目标列名）；\n"
        "3. 调用 generate_eda_charts 生成完整 EDA 图表："
        f"file_path 传入训练集路径，target_column 传入目标列，task_type 传入 `{task_type}`，"
        f"test_file_path 传入测试集路径 `{testing_data_path}`，"
        f"output_dir 传入：{run_dir + '/plots' if run_dir else ''}（为空则使用默认路径）；\n"
        "4. 综合以上分析，调用 select_candidate_models_and_metrics 提交最终决策，需同时确定：\n"
        f"   【模型选择】候选模型池（{task_type}）：{pool_str}，恰好选 3 个，名称须精确匹配；\n"
        f"   【评估指标】可选指标：{metrics_str}。\n"
        "   需确定两项指标决策（须使用上方列出的合法指标名，不得使用其他字符串）：\n"
        "   · primary_metric（1 个）：CV 参数调优的优化目标，根据以下原则选择：\n"
        "     【分类】\n"
        "       类别均衡、整体准确率为主 → f1_weighted\n"
        "       存在少数类、需监控类别不平衡风险 → f1_macro\n"
        "         （macro 对每个类等权，少数类表现差时 CV 分数立即下降，暴露模型不均匀问题）\n"
        "       严重不平衡 → roc_auc（需 predict_proba）或 recall\n"
        "       多分类（类别数 > 2）→ 优先 f1_macro 或 f1_weighted；\n"
        "         roc_auc 会自动升级为 ovr_weighted，但若候选模型含 SVM 则改选 f1_macro\n"
        "       概率校准场景（CTR、点击率、风险评分）→ log_loss\n"
        "         （注：所有候选模型须支持 predict_proba，SVM 默认不支持，应从候选列表去除）\n"
        "     【回归】默认 → r2(全局拟合优度，直观)；存在明显异常值 → mae(对离群点更鲁棒)；\n"
        "             目标值偏态显著（profile 中 skewness > 1，如房价/销量/流量）→ rmsle(对偏态更鲁棒)；\n"
        "             业务关心百分比偏差且目标列无零值 → mape(业务可读性强，「偏离了百分之几」，适合销量/财务预测场景)；\n"
        "             极少用 max_error 调参，仅精度苛刻场景（如零件公差）考虑\n"
        "   · evaluation_metrics（2~4 个列表）：最终 test set 综合报告的指标，"
        "     primary_metric 必须包含在内，可加 1~2 个互补指标；\n"
        "     分类强烈建议：同时包含 f1_macro 和 f1_weighted，前者暴露少数类风险，后者反映整体分布下的表现；\n"
        "     回归建议：偏态目标值场景可加 max_error 观察最坏情况\n"
        "   · metrics_rationale：结合数据画像和业务目标，说明选择上述指标的原因。\n"
        "完成上述 4 个工具调用后，输出最终文字回复，内容严格限定为以下四项：\n"
        "  (1) 数据画像关键发现（缺失值、异常值、分布特征）；\n"
        "  (2) 相关矩阵分析结论（列出前几名相关特征，注明方向，指出疑似泄漏特征但不要描述删除操作——删除由 Feature Agent 执行）；\n"
        "  (3) 候选模型选择理由；\n"
        "  (4) 评估指标决策理由。\n"
        "【严禁事项】：\n"
        "  - 不得虚构或预测特征工程的执行结果（如「已进行 one-hot 编码」「已删除 duration」）；\n"
        "  - 不得虚构或预测模型训练、调参、CV 分数、测试集评估等任何后续阶段的结果；\n"
        "  - 不得编写业务建议、最终模型选择结论或完整流水线总结；\n"
        "  - EDA 报告只覆盖你实际执行的 4 个工具调用的真实输出，不得延伸至其他 Agent 的职责范围。"
    )

    messages = [SystemMessage(content=system_content), *messages]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage) or not last_msg.name:
        return {}

    handlers = {
        "profile_training_data": _handle_eda_step,
        "compute_correlation_matrix": _handle_eda_step,
        "generate_eda_charts": _handle_eda_step,
        "select_candidate_models_and_metrics": _handle_select_models,
    }
    handler = handlers.get(last_msg.name)
    return handler(state, last_msg) if handler else {}


def _parse_tool_payload(tool_message: ToolMessage) -> dict[str, Any]:
    try:
        return json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        return {"error": {"type": "JSONDecodeError", "message": "工具输出不是合法 JSON"}}


def _handle_eda_step(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    report = _merge_eda_report(state, tool_message.name, payload)
    if "error" in payload:
        return {
            "eda_report": report,
            "current_stage": "eda_step_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "eda", "tool": tool_message.name, **payload["error"]}],
        }
    update: GlobalState = {
        "eda_report": report,
        "current_stage": "eda_step_done",
    }
    if tool_message.name == "generate_eda_charts" and isinstance(payload.get("charts"), list):
        update["eda_charts"] = payload["charts"]
    return update


def _handle_select_models(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    report = _merge_eda_report(state, tool_message.name, payload)
    if "error" in payload:
        return {
            "eda_report": report,
            "current_stage": "eda_model_selection_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "eda_model_selection", **payload["error"]}],
        }
    return {
        "eda_report": report,
        "candidate_models": payload["selected_models"],
        "feature_engineering_plan": payload["feature_engineering_plan"],
        "model_selection_plan": payload["model_selection_plan"],
        "primary_metric": payload.get("primary_metric", ""),
        "evaluation_metrics": payload.get("evaluation_metrics", []),
        "metrics_rationale": payload.get("metrics_rationale", ""),
        "current_stage": "eda_done",
        "next_action": "run_feature_engineering",
    }


def _route_after_eda(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    save_agent_report("eda_agent", last_msg, run_dir=state.get("run_dir") or "")
    return "timer_end"


def _show_graph_if_needed(agent_graph, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts" / "graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        mmd_path = output_dir / "eda_agent_graph.mmd"
        mmd_path.write_text(agent_graph.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"Mermaid 源码已保存: {mmd_path}")
        image_path = output_dir / "eda_agent_graph.png"
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


def build_eda_agent(show_graph: bool = False):
    graph = StateGraph(GlobalState)
    graph.add_node("timer_start", make_timer_start("eda_agent"))
    graph.add_node("eda_agent", _eda_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.add_node("timer_end", make_timer_end("eda_agent"))
    graph.set_entry_point("timer_start")
    graph.add_edge("timer_start", "eda_agent")
    graph.add_conditional_edges(
        "eda_agent", _route_after_eda, {"tools": "tools", "timer_end": "timer_end"}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "eda_agent")
    graph.add_edge("timer_end", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


eda_agent = build_eda_agent()


if __name__ == "__main__":
    eda_agent = build_eda_agent(show_graph=True)
    user_input = input("请输入：")
    events = eda_agent.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "training_data_path": str(PROJECT_ROOT / "artifacts" / "workspace" / "Financials_cleaned_train.csv"),
            "task_type": "regression",
            "target_column": "Profit",
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
        if event.get("current_stage"):
            print(f"\n[stage] {event['current_stage']}")
        if event.get("candidate_models"):
            print(f"[models] {event['candidate_models']}")
