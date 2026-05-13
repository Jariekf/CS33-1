"""Model Agent 结构、路由、工具和状态更新测试。"""

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

from agent.model_agent import (
    _get_current_model,
    _tool_state_update_node,
    build_model_agent,
)
from core.env_utils import DEEPSEEK_API_KEY
from tools.model_tools.evaluate_model import evaluate_model
from tools.model_tools.train_model import train_model
from tools.model_tools.tune_and_select_model import tune_and_select_model

TESTENV = Path(__file__).resolve().parent / "testenv"
FEATURE_DIR = TESTENV / "feature"
# 已数值化的特征工程后数据，model agent 可直接 train + evaluate
FEATURE_TRAIN = {m: FEATURE_DIR / f"{m}_train_featured.csv" for m in ["LogisticRegression", "RandomForest", "XGBoost"]}
FEATURE_TEST  = {m: FEATURE_DIR / f"{m}_test_featured.csv"  for m in ["LogisticRegression", "RandomForest", "XGBoost"]}


# ── 图结构 ──────────────────────────────────────────────────────────────────

def test_model_agent_graph_structure():
    agent = build_model_agent(show_graph=False)
    graph = agent.get_graph()
    assert "model_agent" in graph.nodes
    assert "tools" in graph.nodes
    assert "tool_state_update" in graph.nodes
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("__start__", "timer_start") in edges
    assert ("timer_start", "model_agent") in edges
    assert ("tools", "tool_state_update") in edges
    assert ("tool_state_update", "model_agent") in edges


# ── _get_current_model ───────────────────────────────────────────────────────

def test_get_current_model_for_eval():
    state = {
        "candidate_models": ["XGBoost", "RandomForest", "Ridge"],
        "model_results": {"XGBoost": {"metrics": {}}},
    }
    assert _get_current_model(state) == "RandomForest"


def test_get_current_model_returns_none_when_all_done():
    state = {
        "candidate_models": ["XGBoost", "Ridge"],
        "model_results": {"XGBoost": {}, "Ridge": {}},
    }
    assert _get_current_model(state) is None


# ── tool_state_update：evaluate_model ────────────────────────────────────────

def test_handle_evaluate_merges_model_results():
    payload = {
        "model_name": "Ridge",
        "model_path": "/artifacts/models/Ridge_model.pkl",
        "metrics": {"rmse": 1234.5, "mae": 900.0, "r2": 0.82},
    }
    state = {
        "model_results": {"XGBoost": {"model_path": "/a.pkl", "metrics": {"r2": 0.75}}},
        "messages": [
            ToolMessage(content=json.dumps(payload), name="evaluate_model", tool_call_id="t1")
        ],
    }
    update = _tool_state_update_node(state)
    assert "XGBoost" in update["model_results"]
    assert update["model_results"]["Ridge"]["metrics"]["r2"] == 0.82
    assert update["current_stage"] == "model_evaluated"


def test_handle_evaluate_error_recorded():
    payload = {"error": {"type": "FileNotFoundError", "message": "model not found"}}
    state = {
        "messages": [
            ToolMessage(content=json.dumps(payload), name="evaluate_model", tool_call_id="t1")
        ]
    }
    update = _tool_state_update_node(state)
    assert update["current_stage"] == "model_eval_failed"
    assert update["errors"]


# ── train_model 工具 ─────────────────────────────────────────────────────────

def test_train_model_regression(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"a": range(50), "b": range(50), "target": [i * 2.0 for i in range(50)]}).to_csv(csv_path, index=False)

    result = json.loads(train_model.invoke({
        "featured_train_path": str(csv_path),
        "model_name": "Ridge",
        "task_type": "regression",
        "target_column": "target",
        "output_dir": str(tmp_path),
    }))
    assert "error" not in result
    assert Path(result["model_path"]).exists()
    assert Path(result["feature_columns_path"]).exists()
    assert result["rows_trained"] > 0


def test_train_model_classification(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"a": range(40), "b": range(40), "label": [0] * 20 + [1] * 20}).to_csv(csv_path, index=False)

    result = json.loads(train_model.invoke({
        "featured_train_path": str(csv_path),
        "model_name": "LogisticRegression",
        "task_type": "classification",
        "target_column": "label",
        "output_dir": str(tmp_path),
    }))
    assert "error" not in result
    assert Path(result["model_path"]).exists()


def test_train_model_with_model_params(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"a": range(60), "b": range(60), "label": [0] * 30 + [1] * 30}).to_csv(csv_path, index=False)

    result = json.loads(train_model.invoke({
        "featured_train_path": str(csv_path),
        "model_name": "RandomForest",
        "task_type": "classification",
        "target_column": "label",
        "output_dir": str(tmp_path),
        "model_params": {"n_estimators": 50, "max_depth": 5},
    }))
    assert "error" not in result
    assert result["model_params"]["n_estimators"] == 50
    assert Path(result["model_path"]).exists()


def test_train_model_missing_target_column(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"a": range(10)}).to_csv(csv_path, index=False)

    result = json.loads(train_model.invoke({
        "featured_train_path": str(csv_path),
        "model_name": "Ridge",
        "task_type": "regression",
        "target_column": "missing_col",
        "output_dir": str(tmp_path),
    }))
    assert "error" in result


def test_tune_and_select_model_regression(tmp_path):
    ridge_csv = tmp_path / "ridge_train.csv"
    rf_csv = tmp_path / "rf_train.csv"
    pd.DataFrame(
        {"x1": range(120), "x2": range(120), "target": [i * 1.5 + 2 for i in range(120)]}
    ).to_csv(ridge_csv, index=False)
    pd.DataFrame(
        {"x1": range(120), "x2": range(120), "target": [i * 1.5 + 2 for i in range(120)]}
    ).to_csv(rf_csv, index=False)

    result = json.loads(
        tune_and_select_model.invoke(
            {
                "feature_dataset_ids": {
                    "Ridge": str(ridge_csv),
                    "RandomForestRegressor": str(rf_csv),
                },
                "candidate_models": ["Ridge", "RandomForestRegressor"],
                "task_type": "regression",
                "target_column": "target",
                "cv_folds": 3,
                "max_iter_per_model": 4,
            }
        )
    )
    assert "error" not in result
    assert result["best_model_name"] in {"Ridge", "RandomForestRegressor"}
    assert "per_model_summary" in result


# ── evaluate_model 工具 ──────────────────────────────────────────────────────

def test_evaluate_model_regression(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"a": range(80), "b": range(80), "target": [float(i) for i in range(80)]}).to_csv(train_csv, index=False)
    pd.DataFrame({"a": range(20), "b": range(20), "target": [float(i) for i in range(20)]}).to_csv(test_csv, index=False)

    train_result = json.loads(train_model.invoke({
        "featured_train_path": str(train_csv),
        "model_name": "Ridge",
        "task_type": "regression",
        "target_column": "target",
        "output_dir": str(tmp_path),
    }))

    eval_result = json.loads(evaluate_model.invoke({
        "model_path": train_result["model_path"],
        "feature_columns_path": train_result["feature_columns_path"],
        "test_data_path": str(test_csv),
        "model_name": "Ridge",
        "task_type": "regression",
        "target_column": "target",
    }))
    assert "error" not in eval_result
    assert "r2" in eval_result["metrics"]
    assert "rmse" in eval_result["metrics"]


def test_evaluate_model_classification(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"a": range(80), "b": range(80), "label": [0] * 40 + [1] * 40}).to_csv(train_csv, index=False)
    pd.DataFrame({"a": range(20), "b": range(20), "label": [0] * 10 + [1] * 10}).to_csv(test_csv, index=False)

    train_result = json.loads(train_model.invoke({
        "featured_train_path": str(train_csv),
        "model_name": "RandomForest",
        "task_type": "classification",
        "target_column": "label",
        "output_dir": str(tmp_path),
    }))

    eval_result = json.loads(evaluate_model.invoke({
        "model_path": train_result["model_path"],
        "feature_columns_path": train_result["feature_columns_path"],
        "test_data_path": str(test_csv),
        "model_name": "RandomForest",
        "task_type": "classification",
        "target_column": "label",
    }))
    assert "error" not in eval_result
    assert "accuracy" in eval_result["metrics"]
    assert "f1_weighted" in eval_result["metrics"]


# ── LLM invoke 测试（需要 API Key）────────────────────────────────────────────

@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="需要配置 DEEPSEEK_API_KEY")
def test_model_agent_invoke_single_model(tmp_path):
    """单模型训练+评估（只处理第一个模型）。testenv 特征数据已数值化，无需预处理参数。"""
    agent = build_model_agent(show_graph=False)
    # 用同一个已数值化的测试集，无需传入 preprocessors
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="请训练并评估模型")],
            "task_type": "classification",
            "target_column": "subscribed",
            "candidate_models": ["LogisticRegression", "RandomForest", "XGBoost"],
            "model_results": {},
            "feature_dataset_ids": {m: str(p) for m, p in FEATURE_TRAIN.items()},
            "feature_preprocessors": {},
            "testing_data_path": str(FEATURE_TEST["LogisticRegression"]),
            "run_dir": str(tmp_path),
        },
        config={"recursion_limit": 60},
    )
    model_results = result.get("model_results") or {}
    assert model_results, "应当至少完成一个模型的训练和评估"
    for model, info in model_results.items():
        assert Path(info["model_path"]).exists(), f"{model} 模型文件应已保存"
        assert "accuracy" in info["metrics"] or "f1_weighted" in info["metrics"]


# ── 外部循环路由 ─────────────────────────────────────────────────────────────

def test_full_pipeline_model_routing():
    from core.graph import _route_after_model, _route_after_feature

    assert _route_after_feature({
        "candidate_models": ["XGBoost", "Ridge"],
        "feature_dataset_ids": {"XGBoost": "/a.csv", "Ridge": "/b.csv"},
    }) == "model_agent"

    assert _route_after_model({
        "candidate_models": ["XGBoost", "Ridge"],
        "model_results": {"XGBoost": {}},
    }) == "model_agent"

    assert _route_after_model({
        "candidate_models": ["XGBoost", "Ridge"],
        "model_results": {"XGBoost": {}, "Ridge": {}},
        "best_model": {"model_name": "XGBoost"},
    }) == "business_translator"

    assert _route_after_model({
        "candidate_models": ["XGBoost", "Ridge"],
        "model_results": {"XGBoost": {}, "Ridge": {}},
    }) == END
