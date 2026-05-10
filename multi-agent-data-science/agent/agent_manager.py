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
from core.my_llm import llm_deepseek

from tools.agentmanager_tools.create_cleaning_plan import create_cleaning_plan
from tools.agentmanager_tools.infer_task import infer_task
from tools.agentmanager_tools.profile_dataset import profile_dataset
from core.report_writer import save_agent_report


tools = [profile_dataset, infer_task, create_cleaning_plan]
llm_with_tools = llm_deepseek.bind_tools(tools)


def _manager_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    data_path = state.get("data_path")
    description = state.get("description")
    user_request = state.get("user_request")
    if data_path or description or user_request:
        messages = [
            SystemMessage(
                content=(
                    "你是一个数据分析 Agent。"
                    f"当前默认数据集路径是：{data_path or '未提供'}。"
                    f"数据集描述是：{description or '未提供'}。"
                    f"用户的问题是：{user_request or '未提供'}。"
                    "当用户说“这个数据集”但没有提供路径时，请使用这个默认路径调用工具。"
                    "工作流要求：先调用 profile_dataset 获取数据画像；"
                    "然后调用 infer_task 判断任务类型和目标列；"
                    "最后调用 create_cleaning_plan 记录数据清洗计划。"
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


def _update_cleaning_plan_state(tool_message: ToolMessage) -> GlobalState:
    try:
        cleaning_plan = json.loads(str(tool_message.content))
    except json.JSONDecodeError as exc:
        return {
            "current_stage": "cleaning_plan_failed",
            "errors": [
                {
                    "stage": "cleaning_plan",
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
            ],
        }

    return {
        "cleaning_plan": cleaning_plan,
        "current_stage": "cleaning_plan_created",
        "next_action": "run_data_cleaning",
    }


def _route_after_manager(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    save_agent_report("agent_manager", last_msg, run_dir=state.get("run_dir") or "")
    return END


def _show_graph_if_needed(agent_graph, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = Path(__file__).resolve().parents[1] / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "agent_manager_graph.png"

        image_bytes = agent_graph.get_graph().draw_mermaid_png()
        image_path.write_bytes(image_bytes)
        print(f"Graph 图片已保存: {image_path}")

        # 在 notebook 环境下额外尝试直接渲染
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
    graph.add_node("agent_manager", _manager_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.set_entry_point("agent_manager")
    graph.add_conditional_edges(
        "agent_manager", _route_after_manager, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "agent_manager")
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





