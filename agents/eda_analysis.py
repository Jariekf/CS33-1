"""
EDA Analysis node — generates static charts + markdown summary after data cleaning passes Supervisor.

Reads the cleaned train split from a fixed path (same contract as feature_engineering).
"""
from __future__ import annotations

from core.logger import get_agent_logger
from core.state import GraphState
from agents.data_cleaning import TRAIN_DATA_PATH
from tools.eda_charts import generate_eda_charts

logger = get_agent_logger("eda_analysis")


def eda_analysis_node(state: GraphState) -> dict:
    """Run deterministic EDA chart generation on the train CSV."""
    logger.info("EDA analysis node started.")
    result = generate_eda_charts(TRAIN_DATA_PATH)
    logger.info("EDA charts generated: %d file(s).", len(result.get("charts") or []))
    return {
        "current_phase": "eda_analysis",
        "eda_charts": result["charts"],
        "eda_summary": result["text_summary"],
    }
