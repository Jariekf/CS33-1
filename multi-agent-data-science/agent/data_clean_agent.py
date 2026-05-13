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

    plan_text = (
        json.dumps(cleaning_plan, ensure_ascii=False, indent=2, default=str)
        if cleaning_plan
        else "未提供（请根据画像自行决定清洗步骤）。"
    )

    if run_dir:
        default_save_path = f"{run_dir}/clean/{prefix}.csv"
    else:
        default_save_path = f"artifacts/clean/{prefix}.csv"

    system_parts = [
        f"你是数据清洗执行 Agent（ReAct），当前处理：{target} 数据集。",
        f"源文件路径：{source_path or '未提供'}。",
        f"当前内存 dataset_id：{dataset_id or '尚未加载'}。",
        f"数据集描述：{description or '未提供'}。",
        f"用户诉求：{user_request or '未提供'}。",
        f"本次运行工作目录：{run_dir or '未设置'}。所有输出文件应保存至此目录下。",
        "若尚未加载数据，应优先调用 load_dataset_to_store(file_path=...)；路径缺失时使用上面的源文件路径。",
        "后续按清洗计划依次调用 clean_data_types、clean_missing_values、clean_outliers、clean_drop_columns、clean_duplicates；"
        "所有清洗工具的 dataset_id 必须与 load 返回的 dataset_id 一致。",
        "计划中若有 drop_columns，必须调用 clean_drop_columns(dataset_id=..., columns=...) 执行删列。",
        f"完成后必须调用 save_dataset_from_store 写出 CSV，output_path 使用：{default_save_path}。",
        f"\n--- cleaning_plan ({target}) ---\n",
        plan_text,
        "\n完成全部清洗步骤并保存后，最终文字回复只汇报本次数据清洗的执行结果（各步骤处理了哪些列、行数变化等），"
        "不要提及 EDA、特征工程或建模等后续环节。",
    ]

    messages = [SystemMessage(content="".join(system_parts)), *messages]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage) or not last_msg.name:
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
    handler = handlers.get(last_msg.name)
    if handler is None:
        return {}
    return handler(state, last_msg)


def _parse_tool_payload(tool_message: ToolMessage) -> dict[str, Any]:
    try:
        return json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        return {
            "error": {
                "type": "JSONDecodeError",
                "message": "工具输出不是合法 JSON",
            }
        }


def _handle_load_dataset(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    report_key, report = _merge_cleaning_report(state, tool_message.name, payload)
    if "error" in payload:
        return {
            report_key: report,
            "current_stage": "cleaning_load_failed",
            "errors": (state.get("errors") or []) + [{"stage": "cleaning_load", **payload["error"]}],
        }
    return {
        "dataset_id": payload["dataset_id"],
        report_key: report,
        "current_stage": "dataset_loaded",
    }


def _handle_save_dataset(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    report_key, report = _merge_cleaning_report(state, tool_message.name, payload)
    target = state.get("cleaning_target", "train")
    if "error" in payload:
        return {
            report_key: report,
            "current_stage": "cleaning_save_failed",
            "errors": (state.get("errors") or []) + [{"stage": "cleaning_save", **payload["error"]}],
        }
    output_path = payload["output_path"]
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
    report_key, report = _merge_cleaning_report(state, tool_message.name, payload)
    if "error" in payload:
        return {
            report_key: report,
            "current_stage": "cleaning_step_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "cleaning_step", "tool": tool_message.name, **payload["error"]}],
        }
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
    save_agent_report("data_clean_agent", last_msg, run_dir=state.get("run_dir") or "")
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
    graph.add_node("timer_end", _timer_end_clean)
    graph.set_entry_point("timer_start")
    graph.add_edge("timer_start", "data_clean_agent")
    graph.add_conditional_edges(
        "data_clean_agent", _route_after_agent, {"tools": "tools", "timer_end": "timer_end"}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "data_clean_agent")
    graph.add_edge("timer_end", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


data_clean_agent = build_data_clean_agent()


if __name__ == "__main__":
    data_clean_agent = build_data_clean_agent(show_graph=True)
    user_input = input("请输入：")
    events = data_clean_agent.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "data_path": str(PROJECT_ROOT / "datasets" / "Financials.csv"),
            "description": "这是一个金融行业销售数据集，包含销售、折扣、利润、国家、产品和时间字段。",
            "cleaning_plan": {
                "summary": "示例：先做数值列货币字符串转数值，再将 Date 转为日期（可按画像扩展）。",
                "column_name_actions": [],
                "type_conversions": [
                    {"column": "Year", "target_type": "numeric"},
                    {"column": "Date", "target_type": "datetime"},
                ],
                "missing_value_actions": [],
                "outlier_actions": [],
                "drop_columns": [],
                "notes": "",
            },
            "cleaned_data_path": str(PROJECT_ROOT / "artifacts" / "Financials_cleaned.csv"),
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
