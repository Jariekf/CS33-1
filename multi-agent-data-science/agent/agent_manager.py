from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from core.global_state import GlobalState
from core.my_llm import llm_for

from tools.agentmanager_tools.create_cleaning_plan import create_cleaning_plan
from tools.agentmanager_tools.infer_task import infer_task
from tools.agentmanager_tools.manager_rap import build_manager_rap_context
from tools.agentmanager_tools.profile_dataset import profile_dataset
from core.report_writer import append_manager_rap_midlayer_to_report, save_agent_report
from core.timer_utils import make_timer_end, make_timer_start


tools = [profile_dataset, infer_task, build_manager_rap_context, create_cleaning_plan]
llm_with_tools = llm_for("agent_manager").bind_tools(tools)


def _has_complete_task_meta(state: GlobalState) -> bool:
    """已由 split_agent 或 CLI 注入时，跳过 infer_task，避免重复推断。"""
    tt = (state.get("task_type") or "").strip().lower()
    tc = (state.get("target_column") or "").strip()
    return tt in {"classification", "regression"} and bool(tc)


def _manager_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    training_data_path = state.get("training_data_path") or state.get("data_path") or ""
    description = state.get("description", "")
    user_request = state.get("user_request", "")
    testing_data_path = state.get("testing_data_path") or ""
    has_meta = _has_complete_task_meta(state)
    has_rap_context = bool((state.get("manager_rap_context") or "").strip())
    rap_context = state.get("manager_rap_context") or ""
    rap_summary = state.get("user_requirement_summary") or ""

    rap_instruction = (
        "\nManager RAP advisory context（用于清洗计划参考，不得覆盖数据画像中的事实）：\n"
        f"需求摘要：{rap_summary or '尚未生成'}\n"
        f"RAP建议：{rap_context or '尚未生成'}\n"
    )

    if has_meta:
        rap_step = (
            "2. 调用 build_manager_rap_context(user_request, description, task_type, target_column) 生成 Manager RAP context；\n"
            if not has_rap_context
            else "2. 当前 state 已有 Manager RAP context，不要重复调用 build_manager_rap_context；\n"
        )
        workflow = (
            "工作流要求（严格按顺序执行）：\n"
            f"1. 调用 profile_dataset(file_path='{training_data_path}') 获取训练集数据画像；\n"
            f"{rap_step}"
            "3. 调用 create_cleaning_plan(target='train', ...) 制定训练集清洗计划；\n"
            "4. 调用 create_cleaning_plan(target='test', ...) 制定测试集清洗计划"
            "   （测试集通常与训练集步骤相同，但不做基于标签分布的操作）。\n"
            "说明：当前 state 已包含有效的 task_type 与 target_column（来自 split_agent 或用户指定），"
            "不要调用 infer_task。\n"
            "制定清洗计划时把 Manager RAP context 作为背景知识参考，但以实际数据画像为准。\n"
            "完成以上步骤后停止，生成报告。"
        )
    else:
        rap_step = (
            "3. 调用 build_manager_rap_context(user_request, description, task_type, target_column) 生成 Manager RAP context；\n"
            if not has_rap_context
            else "3. 当前 state 已有 Manager RAP context，不要重复调用 build_manager_rap_context；\n"
        )
        workflow = (
            "工作流要求（严格按顺序执行）：\n"
            f"1. 调用 profile_dataset(file_path='{training_data_path}') 获取训练集数据画像；\n"
            "2. 调用 infer_task，结合画像结果、数据集描述与用户问题判断"
            " task_type（classification/regression）与 target_column（列名须与实际 CSV 列一致）；\n"
            f"{rap_step}"
            "4. 调用 create_cleaning_plan(target='train', ...) 制定训练集清洗计划；\n"
            "5. 调用 create_cleaning_plan(target='test', ...) 制定测试集清洗计划"
            "   （测试集步骤与 train 对齐，不写基于标签分布的特殊操作）。\n"
            "制定清洗计划时把 Manager RAP context 作为背景知识参考，但以实际数据画像为准。\n"
            "完成以上步骤后停止，生成报告。"
        )

    messages = [
        SystemMessage(
            content=(
                "你是一个数据分析 Agent。数据集已完成路径约定：训练集与测试集已分开或即将使用下列路径。\n"
                f"训练集路径：{training_data_path or '未提供'}。\n"
                f"测试集路径：{testing_data_path or '未提供'}。\n"
                f"数据集描述：{description or '未提供'}。\n"
                f"用户的问题：{user_request or '未提供'}。\n"
                + rap_instruction
                + workflow
            )
        ),
        *messages,
    ]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage):
        return {}
    if last_msg.name == "profile_dataset":
        return _update_profile_state(state, last_msg)
    if last_msg.name == "infer_task":
        return _update_task_state(last_msg)
    if last_msg.name == "build_manager_rap_context":
        upd = _update_manager_rap_state(last_msg)
        if upd.get("current_stage") == "manager_rap_done":
            append_manager_rap_midlayer_to_report({**state, **upd})
        return upd
    if last_msg.name == "create_cleaning_plan":
        return _update_cleaning_plan_state(last_msg)
    return {}


def _update_profile_state(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    try:
        profile = json.loads(str(tool_message.content))
    except json.JSONDecodeError as exc:
        return {
            "current_stage": "profile_failed",
            "errors": [
                {
                    "stage": "profile",
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
            ],
        }

    if "error" in profile:
        return {
            "current_stage": "profile_failed",
            "errors": [
                {
                    "stage": "profile",
                    **profile["error"],
                }
            ],
        }

    dataset_info = profile.get("dataset", {})
    dataset_path = dataset_info.get("file_path")
    dataset_name = Path(dataset_path).name if dataset_path else state.get("dataset_name")
    return {
        "dataset_profile": profile,
        "dataset_name": dataset_name,
        "current_stage": "profile_done",
    }


def _update_task_state(tool_message: ToolMessage) -> GlobalState:
    try:
        task_result = json.loads(str(tool_message.content))
    except json.JSONDecodeError as exc:
        return {
            "current_stage": "task_inference_failed",
            "errors": [
                {
                    "stage": "task_inference",
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
            ],
        }

    return {
        "task_type": task_result.get("task_type", "unknown"),
        "target_column": task_result.get("target_column", ""),
        "current_stage": "task_inferred",
    }


def _update_manager_rap_state(tool_message: ToolMessage) -> GlobalState:
    try:
        rap_result = json.loads(str(tool_message.content))
    except json.JSONDecodeError as exc:
        return {
            "current_stage": "manager_rap_failed",
            "errors": [
                {
                    "stage": "manager_rap",
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
            ],
        }

    return {
        "user_requirement_summary": rap_result.get("user_requirement_summary", ""),
        "rap_retrieval_slots": rap_result.get("rap_retrieval_slots", {}),
        "manager_rap_context": rap_result.get("manager_rap_context", ""),
        "manager_rap_meta": rap_result.get("manager_rap_meta", {}),
        "current_stage": "manager_rap_done",
    }


def _update_cleaning_plan_state(tool_message: ToolMessage) -> GlobalState:
    try:
        plan = json.loads(str(tool_message.content))
    except json.JSONDecodeError as exc:
        return {
            "current_stage": "cleaning_plan_failed",
            "errors": [{"stage": "cleaning_plan", "type": exc.__class__.__name__, "message": str(exc)}],
        }

    target = plan.get("target", "train")
    if target == "test":
        return {
            "test_cleaning_plan": plan,
            "cleaning_plan": plan,  # keep legacy field pointing to last plan
            "current_stage": "cleaning_plans_created",
            "next_action": "run_data_cleaning",
        }
    return {
        "train_cleaning_plan": plan,
        "cleaning_plan": plan,
        "current_stage": "train_cleaning_plan_created",
    }


def _route_after_manager(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    save_agent_report("agent_manager", last_msg, run_dir=state.get("run_dir") or "")
    return "timer_end"


def _show_graph_if_needed(agent_graph, show_graph: bool) -> None:
    if not show_graph:
        return
    output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = output_dir / "agent_manager_graph.mmd"
    mmd_path.write_text(agent_graph.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Agent Manager Mermaid 源码已保存: {mmd_path}")
    try:
        image_path = output_dir / "agent_manager_graph.png"
        image_path.write_bytes(agent_graph.get_graph().draw_mermaid_png())
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


def build_agent_manager(show_graph: bool = False):
    graph = StateGraph(GlobalState)
    graph.add_node("timer_start", make_timer_start("agent_manager"))
    graph.add_node("agent_manager", _manager_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.add_node("timer_end", make_timer_end("agent_manager"))
    graph.set_entry_point("timer_start")
    graph.add_edge("timer_start", "agent_manager")
    graph.add_conditional_edges(
        "agent_manager", _route_after_manager, {"tools": "tools", "timer_end": "timer_end"}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "agent_manager")
    graph.add_edge("timer_end", END)
    compiled_graph = graph.compile()
    _show_graph_if_needed(compiled_graph, show_graph=show_graph)
    return compiled_graph




agent_manager = build_agent_manager()



if __name__ == "__main__":
    agent_manager = build_agent_manager(show_graph=True)
    user_input = input("请输入：")
    events = agent_manager.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "data_path": str(PROJECT_ROOT / "datasets" / "Financials.csv"),
            "description": "这是一个金融行业销售数据集，包含销售、折扣、利润、国家、产品和时间字段。",
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
