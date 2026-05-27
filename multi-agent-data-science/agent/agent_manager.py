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
    """When tasks have already been injected by Agent_Manager_split or the CLI, skip `infer_task` to avoid duplicate inference."""
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
        "\nManager RAP advisory context (used as a reference for cleansing plans; must not override facts in the data profile):\n"
        f"Requirements Summary:{rap_summary or 'Not yet generated'}\n"
        f"RAP Recommendations:{rap_context or 'Not yet generated'}\n"
    )
    cleaning_plan_contract = (
        "\ncreate_cleaning_plan contract (these fields must be passed explicitly):\n"
        "target, summary, column_name_actions, type_conversions, missing_value_actions, "
        "outlier_actions, drop_columns, notes.\n"
        "If a field has no action, pass an empty list [], but do not omit the parameter.\n"
        "Example: create_cleaning_plan(target='train', summary='...', column_name_actions=[], "
        "type_conversions=[], missing_value_actions=[], outlier_actions=[], drop_columns=[], notes=[])\n"
    )

    if has_meta:
        rap_step = (
            "2. Call build_manager_rap_context(user_request, description, task_type, target_column) Generate Manager RAP context;\n"
            if not has_rap_context
            else "2. If the current state already contains a Manager RAP context, do not call `build_manager_rap_context` again;\n"
        )
        workflow = (
            "Workflow Requirements (to be executed strictly in order):\n"
            f"1. Call profile_dataset(file_path='{training_data_path}') Retrieve the training dataset profile;\n"
            f"{rap_step}"
            "3. Call create_cleaning_plan(target='train', ...) Develop a plan for cleaning the training set;\n"
            "4. Call create_cleaning_plan(target='test', ...) Develop a test dataset cleaning plan"
            "   (The process for the test set is typically the same as for the training set, except that no operations based on the label distribution are performed.)\n"
            "Note: The current state already contains valid `task_type` and `target_column` values (from `Agent_Manager_split` or specified by the user)"
            "Do not call `infer_task`.\n"
            "When developing a cleaning plan, use the Manager RAP context as a reference, but rely on the actual data profile.\n"
            "Once you have completed the steps above, stop the process and generate the report."
        )
    else:
        rap_step = (
            "3. Call build_manager_rap_context(user_request, description, task_type, target_column) generate Manager RAP context;\n"
            if not has_rap_context
            else "3. The current state already has a Manager RAP context; do not call `build_manager_rap_context` again;\n"
        )
        workflow = (
            "Workflow Requirements (to be executed strictly in order):\n"
            f"1. Call `profile_dataset(file_path='{training_data_path}')` to retrieve the training dataset profile;\n"
            "2. Call `infer_task` and use the profiling results, dataset description, and user query to make a determination"
            " task_type (classification/regression) and target_column (the column name must match the actual CSV column);\n"
            f"{rap_step}"
            "4. Call `create_cleaning_plan(target='train', ...)` to define a cleaning plan for the training set;\n"
            "5. Call `create_cleaning_plan(target='test', ...)` to define a cleaning plan for the test set"
            "  (The steps for the test set align with those for the training set; do not include any special operations based on the label distribution.)\n"
            "When developing a cleaning plan, use the Manager RAP context as a reference, but rely on the actual data profile.\n"
            "Once you have completed the steps above, stop the process and generate the report."
        )

    messages = [
        SystemMessage(
            content=(
                "You are a data analysis Agent. The dataset path convention has been established: the training set and test set have been separated or will be organized using the following paths.\n"
                f"Training data path: {training_data_path or 'Not provided'}.\n"
                f"Test set path: {testing_data_path or 'not provided'}.\n"
                f"Dataset description: {description or 'Not provided'}.\n"
                f"User's question: {user_request or 'Not provided'}.\n"
                + rap_instruction
                + workflow
                + cleaning_plan_contract
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


def _tool_content_preview(tool_message: ToolMessage, max_chars: int = 1000) -> str:
    raw = str(tool_message.content)
    if len(raw) <= max_chars:
        return raw
    return f"{raw[:max_chars]}... <truncated>"


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
                    "raw_content_preview": _tool_content_preview(tool_message),
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
    existing_name = (state.get("dataset_name") or "").strip()
    dataset_name = existing_name or (Path(dataset_path).stem if dataset_path else "dataset")
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
                    "raw_content_preview": _tool_content_preview(tool_message),
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
                    "raw_content_preview": _tool_content_preview(tool_message),
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
            "errors": [
                {
                    "stage": "cleaning_plan",
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "raw_content_preview": _tool_content_preview(tool_message),
                }
            ],
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
    print(f"Agent Manager Mermaid The source code has been saved: {mmd_path}")
    try:
        image_path = output_dir / "agent_manager_graph.png"
        image_path.write_bytes(agent_graph.get_graph().draw_mermaid_png())
        print(f"Graph Image saved: {image_path}")
        try:
            from IPython.display import Image, display
            display(Image(filename=str(image_path)))
        except Exception:
            pass
    except Exception as exc:
        print(f"Image rendering failed: {exc}")


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
    user_input = input("Please enter:")
    events = agent_manager.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "data_path": str(PROJECT_ROOT / "datasets" / "Financials.csv"),
            "description": "This is a financial industry sales dataset that includes fields for sales, discounts, profit, country, product, and time.",
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
