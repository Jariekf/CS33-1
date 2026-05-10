"""集成：Agent Manager 与 Data Clean Agent 组合为父 LangGraph。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import END

from tests.agentmanager_dataclean_pipeline import (
    _route_after_agent_manager,
    build_agentmanager_dataclean_pipeline,
)


def test_pipeline_graph_contains_both_subgraphs():
    pipeline = build_agentmanager_dataclean_pipeline(show_graph=False)
    graph = pipeline.get_graph()
    assert "agent_manager" in graph.nodes
    assert "data_clean_agent" in graph.nodes

    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("__start__", "agent_manager") in edges
    assert ("data_clean_agent", "__end__") in edges


def test_route_enters_clean_when_plan_and_next_action():
    assert (
        _route_after_agent_manager(
            {
                "next_action": "run_data_cleaning",
                "cleaning_plan": {"summary": "x"},
            }
        )
        == "data_clean_agent"
    )


def test_route_ends_without_clean_when_no_plan():
    assert (
        _route_after_agent_manager(
            {
                "next_action": "run_data_cleaning",
                "cleaning_plan": {},
            }
        )
        == END
    )


def test_route_ends_when_next_action_not_cleaning():
    assert (
        _route_after_agent_manager(
            {
                "next_action": "",
                "cleaning_plan": {"summary": "x"},
            }
        )
        == END
    )
