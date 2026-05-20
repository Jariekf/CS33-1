"""Split Agent: First determine the task type, then split the dataset."""

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

    supervisor_hint = (state.get("supervisor_hints") or {}).get("split", "").strip()
    hint_line = (
        f"[Supervisor Feedback] Failed the previous round of checks: {supervisor_hint}\nPlease fix only this issue; do not repeat steps that have already been completed.\n"
        if supervisor_hint else ""
    )

    system_content = (
        "You are a dataset splitting agent responsible for inferring modeling tasks and splitting the dataset.\n"
       f"Dataset path: {data_path or 'Not provided'}.\n"
       f"Dataset description: {description or 'Not provided'}.\n"
       f"User query: {user_request or 'Not provided'}.\n"
       f"Current working directory: {run_dir or 'Not provided'}.\n"
        + hint_line
        + "Workflow (executed strictly in sequence):\n"
        "1. Call `infer_task` to determine the task type (classification/regression) and target column based on the dataset description and the user's query;\n"
        "2. Call `split_dataset`, passing in the `task_type` and `target_column` inferred in the previous step,"
        f"data_path='{data_path}', run_dir='{run_dir}', split the dataset into training and testing sets.\n"
        "Once you have completed the above two steps, stop there; do not call any further tools. \n"
        "The final text response should only describe the inferred task type and how the dataset has been split (paths and number of rows for the training and test sets),"
        "and should not mention subsequent data cleaning, EDA, or modeling steps."
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
    print(f"Split Agent Mermaid The source code has been saved: {mmd_path}")
    try:
        path = out / "split_agent_graph.png"
        path.write_bytes(compiled.get_graph().draw_mermaid_png())
        print(f"Split Agent Graph Saved: {path}")
    except Exception as exc:
        print(f"Image rendering failed: {exc}")


if __name__ == "__main__":
    agent = build_split_agent(show_graph=True)
    user_input = input("Please enter：")
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
