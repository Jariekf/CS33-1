"""Split Agent：先推断任务类型，再切分数据集。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
from tools.agentmanager_tools.infer_task import infer_task
from tools.split_tools.split_dataset_tool import split_dataset

tools = [infer_task, split_dataset]
llm_with_tools = llm_for("split").bind_tools(tools)


def _split_agent_node(state: GlobalState) -> GlobalState:
    data_path = state.get("data_path", "")
    description = state.get("description", "")
    user_request = state.get("user_request", "")
    run_dir = state.get("run_dir", "")

    system_content = (
        "你是一个数据集划分 Agent，负责推断建模任务并切分数据集。\n"
        f"数据集路径：{data_path or '未提供'}。\n"
        f"数据集描述：{description or '未提供'}。\n"
        f"用户问题：{user_request or '未提供'}。\n"
        f"本次运行目录：{run_dir or '未提供'}。\n"
        "工作流（严格按顺序执行）：\n"
        "1. 调用 infer_task，根据数据集描述和用户问题判断任务类型（classification/regression）和目标列；\n"
        "2. 调用 split_dataset，传入上一步推断的 task_type 和 target_column，"
        f"data_path='{data_path}', run_dir='{run_dir}'，将数据集切分为训练集和测试集。\n"
        "完成以上两步后停止，不要再调用任何工具。\n"
        "最终文字回复只说明任务类型推断结果和数据集切分情况（训练集/测试集路径与行数），"
        "不要提及后续数据清洗、EDA 或建模步骤。"
    )

    messages = [SystemMessage(content=system_content), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage):
        return {}
    if last_msg.name == "infer_task":
        return _update_task_state(last_msg)
    if last_msg.name == "split_dataset":
        return _update_split_state(state, last_msg)
    return {}


def _update_task_state(msg: ToolMessage) -> GlobalState:
    try:
        result = json.loads(str(msg.content))
    except json.JSONDecodeError as exc:
        return {
            "current_stage": "task_inference_failed",
            "errors": [{"stage": "task_inference", "type": exc.__class__.__name__, "message": str(exc)}],
        }
    return {
        "task_type": result.get("task_type", "unknown"),
        "target_column": result.get("target_column", ""),
        "current_stage": "task_inferred",
    }


def _update_split_state(state: GlobalState, msg: ToolMessage) -> GlobalState:
    try:
        report = json.loads(str(msg.content))
    except json.JSONDecodeError as exc:
        return {
            "current_stage": "split_failed",
            "errors": (state.get("errors") or []) + [{"stage": "split", "type": exc.__class__.__name__, "message": str(exc)}],
        }
    if "error" in report:
        return {
            "current_stage": "split_failed",
            "errors": (state.get("errors") or []) + [{"stage": "split", **report}],
            "split_report": report,
        }
    return {
        "training_data_path": report.get("train_path", ""),
        "testing_data_path": report.get("test_path", ""),
        "split_report": report,
        "current_stage": "split_done",
    }


def _route_after_split_agent(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    save_agent_report("split_agent", last_msg)
    return "timer_end"


def build_split_agent(show_graph: bool = False):
    graph = StateGraph(GlobalState)
    graph.add_node("timer_start", make_timer_start("split_agent"))
    graph.add_node("split_agent", _split_agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.add_node("timer_end", make_timer_end("split_agent"))
    graph.set_entry_point("timer_start")
    graph.add_edge("timer_start", "split_agent")
    graph.add_conditional_edges(
        "split_agent", _route_after_split_agent, {"tools": "tools", "timer_end": "timer_end"}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "split_agent")
    graph.add_edge("timer_end", END)
    compiled_graph = graph.compile()
    if show_graph:
        _show_graph(compiled_graph)
    return compiled_graph


def _show_graph(compiled) -> None:
    out = PROJECT_ROOT / "artifacts" / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    mmd_path = out / "split_agent_graph.mmd"
    mmd_path.write_text(compiled.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Split Agent Mermaid 源码已保存: {mmd_path}")
    try:
        path = out / "split_agent_graph.png"
        path.write_bytes(compiled.get_graph().draw_mermaid_png())
        print(f"Split Agent Graph 已保存: {path}")
    except Exception as exc:
        print(f"图渲染失败: {exc}")


if __name__ == "__main__":
    agent = build_split_agent(show_graph=True)
    user_input = input("请输入：")
    events = agent.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "data_path": str(PROJECT_ROOT / "datasets" / "bank-full.csv"),
            "description": "Bank marketing dataset, classify if client subscribes a term deposit (target: y).",
            "run_dir": str(PROJECT_ROOT / "artifacts" / "workspace" / "test"),
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
