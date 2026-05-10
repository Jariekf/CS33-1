"""
LangGraph 工作流图构建
定义节点 (Nodes)、边 (Edges) 与条件路由 (Conditional Edges)。
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
    工厂函数：为每个阶段的 Supervisor 创建对应的条件路由映射。
    - "retry"  → 返回当前阶段的 Worker 节点
    - "pass"   → 推进到下一个阶段 / 结束流水线
    - "error"  → 终止图
    """
    def _router(state: GraphState) -> str:
        result = supervisor_router(state)
        sys_logger.info(f"Routing logic executed for phase transition to {next_phase}, result: {result}")
        return result
    return _router


def build_graph() -> StateGraph:
    """构建并返回编译后的 LangGraph 工作流图。"""
    sys_logger.info("Building LangGraph workflow...")

    workflow = StateGraph(GraphState)

    # ==================== 注册节点 ====================
    workflow.add_node("manager", manager_node)

    workflow.add_node("data_cleaning", data_cleaning_node)
    workflow.add_node("supervisor_dc", supervisor_node)

    workflow.add_node("eda_analysis", eda_analysis_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("supervisor_fe", supervisor_node)

    workflow.add_node("model_selection", model_selection_node)
    workflow.add_node("supervisor_ms", supervisor_node)

    workflow.add_node("business_translator", business_translator_node)

    # ==================== 定义边 ====================

    # 入口：Manager → Data Cleaning
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
