"""
LangGraph workflow graph construction.
Defines nodes, edges, and conditional edges.
"""
from langgraph.graph import StateGraph, END

from core.logger import get_agent_logger
from core.state import GraphState
from agents.manager import manager_node
from agents.data_cleaning import data_cleaning_node
from agents.eda_analysis import eda_analysis_node
from agents.feature_engineering import feature_engineering_node
from agents.model_selection import model_selection_node
from agents.supervisor import supervisor_node, supervisor_router
from agents.business_translator import business_translator_node
from config import MAX_OUTER_RETRY

sys_logger = get_agent_logger("sys")

def _make_supervisor_routing(next_phase: str | None):
    """
    Factory: build the conditional-edge map for each phase's Supervisor.
    - "retry" → return to the current phase's worker node
    - "pass"  → advance to the next phase / end the pipeline
    - "error" → terminate the graph
    """
    def _router(state: GraphState) -> str:
        result = supervisor_router(state)
        sys_logger.info(f"Routing logic executed for phase transition to {next_phase}, result: {result}")
        return result
    return _router


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph workflow."""
    sys_logger.info("Building LangGraph workflow...")

    workflow = StateGraph(GraphState)

    # ==================== Register nodes ====================
    workflow.add_node("manager", manager_node)

    workflow.add_node("data_cleaning", data_cleaning_node)
    workflow.add_node("supervisor_dc", supervisor_node)

    workflow.add_node("eda_analysis", eda_analysis_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("supervisor_fe", supervisor_node)

    workflow.add_node("model_selection", model_selection_node)
    workflow.add_node("supervisor_ms", supervisor_node)

    workflow.add_node("business_translator", business_translator_node)

    # ==================== Define edges ====================

    # Entry: Manager → Data Cleaning
    workflow.set_entry_point("manager")
    workflow.add_edge("manager", "data_cleaning")

    # Data Cleaning → Supervisor(DC)
    workflow.add_edge("data_cleaning", "supervisor_dc")
    workflow.add_conditional_edges(
        "supervisor_dc",
        _make_supervisor_routing(next_phase="eda_analysis"),
        {
            "retry": "data_cleaning",
            "pass": "eda_analysis",
            "error": END,
        },
    )

    workflow.add_edge("eda_analysis", "feature_engineering")

    # Feature Engineering → Supervisor(FE)
    workflow.add_edge("feature_engineering", "supervisor_fe")
    workflow.add_conditional_edges(
        "supervisor_fe",
        _make_supervisor_routing(next_phase="model_selection"),
        {
            "retry": "feature_engineering",
            "pass": "model_selection",
            "error": END,
        },
    )

    # Model Selection → Supervisor(MS)
    workflow.add_edge("model_selection", "supervisor_ms")
    workflow.add_conditional_edges(
        "supervisor_ms",
        _make_supervisor_routing(next_phase="business_translator"),
        {
            "retry": "model_selection",
            "pass": "business_translator",
            "error": END,
        },
    )

    # Business Translator → END
    workflow.add_edge("business_translator", END)

    return workflow.compile()
