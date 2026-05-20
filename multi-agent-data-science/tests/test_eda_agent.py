"""EDA Agent structure, routing, tool, and state-update tests."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END

from agent.eda_agent import (
    _handle_eda_step,
    _handle_select_models,
    _tool_state_update_node,
    build_eda_agent,
)
from core.env_utils import DEEPSEEK_API_KEY
from tools.eda_tools.compute_correlation_matrix import compute_correlation_matrix
from tools.eda_tools.generate_eda_charts import generate_eda_charts
from tools.eda_tools.profile_training_data import profile_training_data
from tools.eda_tools.select_candidate_models_and_metrics import (
    CLASSIFICATION_POOL,
    REGRESSION_POOL,
    select_candidate_models_and_metrics,
)

TESTENV = Path(__file__).resolve().parent / "testenv"
TRAIN_CSV = TESTENV / "split" / "sample_train.csv"


# ── Graph structure ──────────────────────────────────────────────────────────────────

def test_eda_agent_graph_structure():
    agent = build_eda_agent(show_graph=False)
    graph = agent.get_graph()
    assert "eda_agent" in graph.nodes
    assert "tools" in graph.nodes
    assert "tool_state_update" in graph.nodes

    edges = {(e.source, e.target) for e in graph.edges}
    assert ("__start__", "timer_start") in edges
    assert ("timer_start", "eda_agent") in edges
    assert ("tools", "tool_state_update") in edges
    assert ("tool_state_update", "eda_agent") in edges


# ── select_candidate_models_and_metrics tool validation ────────────────────────────

_VALID_CLF_INVOKE = {
    "task_type": "classification",
    "selected_models": ["LogisticRegression", "RandomForest", "XGBoost"],
    "feature_engineering_plan": {
        "LogisticRegression": {"scaling": True, "encoding": "onehot"},
        "RandomForest": {"scaling": False, "encoding": "label"},
        "XGBoost": {"scaling": False, "encoding": "label"},
    },
    "model_selection_plan": "Selected based on dataset size and feature types",
    "primary_metric": "f1_weighted",
    "evaluation_metrics": ["accuracy", "f1_weighted", "roc_auc"],
    "metrics_rationale": "Classes are roughly balanced; f1_weighted is the primary metric, roc_auc aids ranking evaluation.",
}

_VALID_REG_INVOKE = {
    "task_type": "regression",
    "selected_models": ["LinearRegression", "Ridge", "XGBoostRegressor"],
    "feature_engineering_plan": {},
    "model_selection_plan": "Regression task with continuously distributed target column",
    "primary_metric": "r2",
    "evaluation_metrics": ["r2", "rmse", "mae"],
    "metrics_rationale": "Standard regression trio; r2 is the primary metric.",
}


def test_select_models_classification_valid():
    result = json.loads(
        select_candidate_models_and_metrics.invoke(_VALID_CLF_INVOKE)
    )
    assert result["selected_models"] == ["LogisticRegression", "RandomForest", "XGBoost"]
    assert result["primary_metric"] == "f1_weighted"
    assert result["evaluation_metrics"] == ["accuracy", "f1_weighted", "roc_auc"]
    assert "error" not in result


def test_select_models_regression_valid():
    result = json.loads(
        select_candidate_models_and_metrics.invoke(_VALID_REG_INVOKE)
    )
    assert result["selected_models"] == ["LinearRegression", "Ridge", "XGBoostRegressor"]
    assert result["primary_metric"] == "r2"
    assert "r2" in result["evaluation_metrics"]
    assert "error" not in result


def test_select_models_invalid_name_rejected():
    result = json.loads(
        select_candidate_models_and_metrics.invoke({
            **_VALID_CLF_INVOKE,
            "selected_models": ["LogisticRegression", "RandomForest", "xgboost"],
        })
    )
    assert "error" in result


def test_select_models_wrong_count_rejected():
    result = json.loads(
        select_candidate_models_and_metrics.invoke({
            **_VALID_REG_INVOKE,
            "selected_models": ["LinearRegression", "Ridge"],
        })
    )
    assert "error" in result


def test_select_models_cross_pool_rejected():
    result = json.loads(
        select_candidate_models_and_metrics.invoke({
            **_VALID_CLF_INVOKE,
            "selected_models": ["LogisticRegression", "RandomForest", "LinearRegression"],
        })
    )
    assert "error" in result


# ── metrics validation ────────────────────────────────────────────────────────────

def test_invalid_primary_metric_rejected():
    """primary_metric not in the whitelist should return error."""
    result = json.loads(
        select_candidate_models_and_metrics.invoke({
            **_VALID_CLF_INVOKE,
            "primary_metric": "neg_root_mean_squared_error",  # sklearn internal names not allowed
        })
    )
    assert "error" in result


def test_invalid_evaluation_metric_rejected():
    """evaluation_metrics containing invalid entries should return error."""
    result = json.loads(
        select_candidate_models_and_metrics.invoke({
            **_VALID_CLF_INVOKE,
            "evaluation_metrics": ["f1_weighted", "rmse"],  # rmse is regression-only
        })
    )
    assert "error" in result


def test_primary_metric_not_in_evaluation_metrics_rejected():
    """primary_metric not included in evaluation_metrics should return error."""
    result = json.loads(
        select_candidate_models_and_metrics.invoke({
            **_VALID_CLF_INVOKE,
            "primary_metric": "roc_auc",
            "evaluation_metrics": ["accuracy", "f1_weighted"],  # roc_auc missing
        })
    )
    assert "error" in result


def test_empty_evaluation_metrics_rejected():
    """evaluation_metrics as an empty list should return error."""
    result = json.loads(
        select_candidate_models_and_metrics.invoke({
            **_VALID_CLF_INVOKE,
            "evaluation_metrics": [],
        })
    )
    assert "error" in result


# ── tool_state_update state writes ──────────────────────────────────────────────

def test_handle_select_models_writes_state():
    payload = {
        "selected_models": ["XGBoost", "RandomForest", "LogisticRegression"],
        "feature_engineering_plan": {"XGBoost": {"scaling": False}},
        "model_selection_plan": "XGBoost performs well on tabular data",
        "primary_metric": "roc_auc",
        "evaluation_metrics": ["roc_auc", "f1_weighted", "accuracy"],
        "metrics_rationale": "Class imbalance present; roc_auc is prioritized.",
    }
    update = _tool_state_update_node({
        "messages": [
            ToolMessage(
                content=json.dumps(payload),
                name="select_candidate_models_and_metrics",
                tool_call_id="test-id",
            )
        ]
    })
    assert update["candidate_models"] == ["XGBoost", "RandomForest", "LogisticRegression"]
    assert update["feature_engineering_plan"] == payload["feature_engineering_plan"]
    assert update["model_selection_plan"] == payload["model_selection_plan"]
    assert update["primary_metric"] == "roc_auc"
    assert update["evaluation_metrics"] == ["roc_auc", "f1_weighted", "accuracy"]
    assert update["metrics_rationale"] == "Class imbalance present; roc_auc is prioritized."
    assert update["current_stage"] == "eda_done"
    assert update["next_action"] == "run_feature_engineering"


def test_handle_eda_step_merges_report():
    payload = {"dataset": {"rows": 100}, "missing_values": {}}
    existing_state = {"eda_report": {"steps": [{"tool": "prev", "result": {}}]}}
    update = _tool_state_update_node({
        **existing_state,
        "messages": [
            ToolMessage(
                content=json.dumps(payload),
                name="profile_training_data",
                tool_call_id="test-id",
            )
        ],
    })
    assert update["current_stage"] == "eda_step_done"
    assert len(update["eda_report"]["steps"]) == 2


def test_handle_generate_eda_charts_writes_chart_state():
    payload = {
        "output_dir": "artifacts/workspace/run/plots",
        "plot_count": 1,
        "charts": [
            {
                "name": "numeric_distributions.png",
                "path": "artifacts/workspace/run/plots/numeric_distributions.png",
                "rel_path": "plots/numeric_distributions.png",
                "desc": "Histograms for numeric features",
            }
        ],
    }
    update = _tool_state_update_node({
        "messages": [
            ToolMessage(
                content=json.dumps(payload),
                name="generate_eda_charts",
                tool_call_id="test-id",
            )
        ]
    })
    assert update["current_stage"] == "eda_step_done"
    assert update["eda_charts"] == payload["charts"]
    assert update["eda_report"]["charts"] == payload["charts"]


def test_handle_eda_step_records_error():
    payload = {"error": {"type": "FileNotFoundError", "message": "file not found"}}
    update = _tool_state_update_node({
        "messages": [
            ToolMessage(
                content=json.dumps(payload),
                name="profile_training_data",
                tool_call_id="test-id",
            )
        ]
    })
    assert update["current_stage"] == "eda_step_failed"
    assert update["errors"]


# ── Actual tool computation ─────────────────────────────────────────────────────────────

def test_profile_training_data_tool(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"x": range(50), "y": range(50)}).to_csv(csv_path, index=False)

    result = json.loads(profile_training_data.invoke({"file_path": str(csv_path)}))
    assert "dataset" in result
    assert result["dataset"]["rows"] == 50
    assert "outliers" in result


def test_compute_correlation_matrix_with_target(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({
        "a": range(30),
        "b": [i * 2 for i in range(30)],
        "target": range(30),
    }).to_csv(csv_path, index=False)

    result = json.loads(compute_correlation_matrix.invoke({
        "file_path": str(csv_path),
        "target_column": "target",
    }))
    assert "matrix" in result
    assert "target_correlation_ranking" in result
    assert "a" in result["target_correlation_ranking"]


def test_compute_correlation_matrix_no_numeric(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]}).to_csv(csv_path, index=False)

    result = json.loads(compute_correlation_matrix.invoke({"file_path": str(csv_path)}))
    assert "error" in result


def test_generate_eda_charts_writes_pngs_to_requested_output_dir(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({
        "age": [22, 35, 41, 58, 62, 29, 47, 51],
        "balance": [100, 250, 310, 125, 500, 75, 420, 390],
        "job": ["admin", "tech", "admin", "retired", "tech", "student", "admin", "retired"],
        "subscribed": ["no", "yes", "no", "yes", "yes", "no", "yes", "no"],
    }).to_csv(train_csv, index=False)
    pd.DataFrame({
        "age": [31, 44, 53, 27],
        "balance": [120, 330, 275, 90],
        "job": ["tech", "admin", "retired", "student"],
        "subscribed": ["no", "yes", "yes", "no"],
    }).to_csv(test_csv, index=False)

    plots_dir = tmp_path / "plots"
    result = json.loads(generate_eda_charts.invoke({
        "file_path": str(train_csv),
        "output_dir": str(plots_dir),
        "target_column": "subscribed",
        "test_file_path": str(test_csv),
    }))

    assert "error" not in result
    assert result["output_dir"] == str(plots_dir)
    assert result["plot_count"] >= 4
    chart_names = {chart["name"] for chart in result["charts"]}
    assert "numeric_distributions.png" in chart_names
    assert "categorical_distributions.png" in chart_names
    assert "corr_heatmap.png" in chart_names
    assert "class_balance.png" in chart_names
    for chart in result["charts"]:
        assert Path(chart["path"]).is_file()


def test_generate_eda_charts_uses_explicit_task_type_over_heuristic(tmp_path):
    csv_path = tmp_path / "regression_train.csv"
    pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5, 6, 7, 8],
        "feature_b": [2, 4, 6, 8, 10, 12, 14, 16],
        "score": [1, 2, 2, 3, 3, 4, 4, 5],
    }).to_csv(csv_path, index=False)

    result = json.loads(generate_eda_charts.invoke({
        "file_path": str(csv_path),
        "output_dir": str(tmp_path / "plots"),
        "target_column": "score",
        "task_type": "regression",
    }))

    assert "error" not in result
    chart_names = {chart["name"] for chart in result["charts"]}
    assert "target_distribution.png" in chart_names
    assert "numeric_vs_target_regression.png" in chart_names
    assert "class_balance.png" not in chart_names


# ── Candidate pool constants ──────────────────────────────────────────────────────────────

def test_candidate_pools_have_five_each():
    assert len(CLASSIFICATION_POOL) == 5
    assert len(REGRESSION_POOL) == 5


# ── LLM invoke tests (requires API Key)────────────────────────────────────────────

@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="DEEPSEEK_API_KEY must be configured")
def test_eda_agent_invoke(tmp_path):
    agent = build_eda_agent(show_graph=False)
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="Analyze the training set and select candidate models.")],
            "user_request": "Analyze the training set and select candidate models.",
            "training_data_path": str(TRAIN_CSV),
            "task_type": "classification",
            "target_column": "subscribed",
            "description": "Binary classification dataset: predict whether a customer subscribes to a bank term deposit.",
            "run_dir": str(tmp_path),
        },
        config={"recursion_limit": 60},
    )
    assert result.get("candidate_models"), "Expected candidate models to be selected"
    assert len(result["candidate_models"]) == 3, "Exactly 3 candidate models should be selected"
    assert result.get("feature_engineering_plan"), "Expected feature engineering plan to be output"
    for m in result["candidate_models"]:
        assert m in CLASSIFICATION_POOL, f"{m} is not in the classification model candidate pool"


# ── Manual integration run (not executed by pytest) ─────────────────────────────────────────────

def _run_eda_agent():
    from agent.eda_agent import _pretty_print_message

    agent = build_eda_agent(show_graph=True)
    user_input = input("Enter input: ")
    events = agent.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "training_data_path": str(TRAIN_CSV),
            "task_type": "classification",
            "target_column": "subscribed",
            "description": "Binary classification dataset: predict whether a customer subscribes to a bank term deposit.",
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
        if event.get("current_stage"):
            print(f"\n[stage] {event['current_stage']}")
        if event.get("candidate_models"):
            print(f"[models] {event['candidate_models']}")


if __name__ == "__main__":
    _run_eda_agent()
