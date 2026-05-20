"""Partial demo subgraph: agent_manager → data_clean_agent (for quick validation of the cleaning subgraph; not the main repository graph)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from agent.agent_manager import _pretty_print_message, build_agent_manager
from agent.data_clean_agent import build_data_clean_agent
from core.global_state import GlobalState


def _route_after_agent_manager(state: GlobalState) -> str:
    """After the Manager subgraph: if train+test cleaning plans exist (or legacy next_action), enter Data Clean."""
    train_plan = state.get("train_cleaning_plan")
    test_plan = state.get("test_cleaning_plan")
    if (
        isinstance(train_plan, dict)
        and train_plan
        and isinstance(test_plan, dict)
        and test_plan
    ):
        return "data_clean_agent"
    plan = state.get("cleaning_plan")
    if (
        state.get("next_action") == "run_data_cleaning"
        and isinstance(plan, dict)
        and bool(plan)
    ):
        return "data_clean_agent"
    return END


def _show_pipeline_graph_if_needed(compiled, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "graphs" / "agentmanager_dataclean_pipeline.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = compiled.get_graph().draw_mermaid_png()
        image_path.write_bytes(image_bytes)
        print(f"Pipeline graph image saved: {image_path}")
        try:
            from IPython.display import Image, display

            display(Image(filename=str(image_path)))
        except Exception:
            pass
    except Exception as exc:
        print(f"Pipeline graph rendering failed: {exc}")


def build_agentmanager_dataclean_pipeline(show_graph: bool = False):
    manager_graph = build_agent_manager(show_graph=False)
    clean_graph = build_data_clean_agent(show_graph=False)

    graph = StateGraph(GlobalState)
    graph.add_node("agent_manager", manager_graph)
    graph.add_node("data_clean_agent", clean_graph)
    graph.set_entry_point("agent_manager")
    graph.add_conditional_edges(
        "agent_manager",
        _route_after_agent_manager,
        {"data_clean_agent": "data_clean_agent", END: END},
    )
    graph.add_edge("data_clean_agent", END)
    compiled = graph.compile()
    _show_pipeline_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


agentmanager_dataclean_pipeline = build_agentmanager_dataclean_pipeline()


if __name__ == "__main__":
    pipeline = build_agentmanager_dataclean_pipeline(show_graph=True)
    user_input = input("Enter input: ")
    events = pipeline.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "data_path": "C:/Projects/multi-agent-data-science/datasets/Financials.csv",
            "description": "This is a financial industry sales dataset containing fields for sales, discounts, profit, country, product, and time.",
            "cleaned_data_path": "C:/Projects/multi-agent-data-science/artifacts/Financials_cleaned.csv",
        },
        stream_mode="values",
        config={"recursion_limit": 120},
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
