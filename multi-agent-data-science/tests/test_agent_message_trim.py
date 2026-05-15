"""Unit tests for messages trimming in eda_agent and model_agent.

These tests guard the root-cause fix for the "EDA / Model agents leak business
recommendations" issue: the LangGraph ``add_messages`` reducer keeps the original
``user_request`` HumanMessage and the ``agent_manager`` RAP wrap-up AIMessage in
``state["messages"]`` for the entire pipeline. EDA and Model agents must
therefore trim that shared history down to *their own* tool-call chain before
invoking their LLM, otherwise upstream business framing pollutes the context
and the LLM hallucinates a "Business Recommendations" section.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.eda_agent import _trim_messages_to_current_round
from agent.model_agent import _trim_messages_to_modeling_round


def _ai_with_tool_call(name: str, content: str = "") -> AIMessage:
    """Build an AIMessage that carries a single tool_call to ``name``."""
    return AIMessage(
        content=content,
        tool_calls=[{"id": f"tc_{name}", "name": name, "args": {}}],
    )


def _tool_msg(name: str, content: str) -> ToolMessage:
    return ToolMessage(name=name, content=content, tool_call_id=f"tc_{name}")


def _has_user_request(msgs: list) -> bool:
    return any(
        isinstance(m, HumanMessage)
        and "actionable recommendations" in (m.content or "").lower()
        for m in msgs
    )


def _has_rap_wrapup(msgs: list) -> bool:
    return any(
        isinstance(m, AIMessage)
        and "manager rap" in (m.content or "").lower()
        for m in msgs
    )


def test_trim_to_modeling_round_drops_user_request_and_rap_wrapup() -> None:
    history = [
        HumanMessage(
            content=(
                "Please build a classifier and provide actionable recommendations "
                "to improve campaign effectiveness."
            )
        ),
        AIMessage(
            content=(
                "Manager RAP context:\n"
                "- Address class imbalance immediately.\n"
                "- Guard against data leakage."
            )
        ),
        _ai_with_tool_call("profile_dataset"),
        _tool_msg("profile_dataset", '{"rows": 36168}'),
        AIMessage(content="EDA wrap-up: 候选模型已选定。"),
        _ai_with_tool_call("tune_and_select_model"),
        _tool_msg("tune_and_select_model", '{"best_model_name": "RandomForest"}'),
        _ai_with_tool_call("train_model"),
        _tool_msg("train_model", '{"model_path": "models/rf.pkl"}'),
        _ai_with_tool_call("evaluate_model"),
        _tool_msg("evaluate_model", '{"metrics": {"f1_macro": 0.78}}'),
    ]

    trimmed = _trim_messages_to_modeling_round(history)

    assert trimmed, "Trim must keep the modeling tool chain"
    assert not _has_user_request(trimmed)
    assert not _has_rap_wrapup(trimmed)
    tool_names_seen = [m.name for m in trimmed if isinstance(m, ToolMessage)]
    assert tool_names_seen == ["tune_and_select_model", "train_model", "evaluate_model"]


def test_trim_to_modeling_round_returns_empty_before_first_tool_call() -> None:
    history = [
        HumanMessage(content="Predict y; provide actionable recommendations."),
        AIMessage(content="Manager RAP wrap-up text."),
        AIMessage(content="EDA agent technical summary."),
    ]
    assert _trim_messages_to_modeling_round(history) == []


def test_trim_to_modeling_round_ignores_non_modeling_tail() -> None:
    """If the most recent ToolMessage is not a modeling tool, trim returns []."""
    history = [
        _ai_with_tool_call("tune_and_select_model"),
        _tool_msg("tune_and_select_model", "{}"),
        _ai_with_tool_call("profile_dataset"),
        _tool_msg("profile_dataset", "{}"),
    ]
    assert _trim_messages_to_modeling_round(history) == []


def test_trim_eda_returns_empty_when_no_eda_tool_yet() -> None:
    """EDA first round (RAP context present, no EDA tool yet) must yield []."""
    history = [
        HumanMessage(
            content="Predict y and provide actionable recommendations for marketing."
        ),
        AIMessage(
            content=(
                "Manager RAP context: address class imbalance; "
                "guard against data leakage; preserve temporal ordering."
            )
        ),
        _ai_with_tool_call("profile_dataset"),
        _tool_msg("profile_dataset", '{"rows": 36168}'),
        AIMessage(content="agent_manager wrap-up: cleaning plan ready."),
    ]
    assert _trim_messages_to_current_round(history) == []


def test_trim_eda_keeps_only_eda_tool_chain() -> None:
    history = [
        HumanMessage(content="Predict y; actionable recommendations needed."),
        AIMessage(content="Manager RAP business strategy points."),
        _ai_with_tool_call("profile_training_data"),
        _tool_msg("profile_training_data", '{"rows": 36168}'),
        _ai_with_tool_call("compute_correlation_matrix"),
        _tool_msg("compute_correlation_matrix", '{"top": []}'),
        _ai_with_tool_call("generate_eda_charts"),
        _tool_msg("generate_eda_charts", '{"charts": []}'),
        _ai_with_tool_call("select_candidate_models_and_metrics"),
        _tool_msg("select_candidate_models_and_metrics", '{"selected_models": []}'),
    ]

    trimmed = _trim_messages_to_current_round(history)

    assert trimmed, "Trim must keep the EDA tool chain"
    assert not _has_user_request(trimmed)
    assert not _has_rap_wrapup(trimmed)
    tool_names_seen = [m.name for m in trimmed if isinstance(m, ToolMessage)]
    assert tool_names_seen == [
        "profile_training_data",
        "compute_correlation_matrix",
        "generate_eda_charts",
        "select_candidate_models_and_metrics",
    ]
