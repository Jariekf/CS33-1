"""Feature Agent 结构、路由、工具和状态更新测试。"""

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

from agent.feature_agent import (
    _get_current_model,
    _handle_save_featured,
    _tool_state_update_node,
    build_feature_agent,
)
from core.dataset_store import put_dataset
from core.env_utils import DEEPSEEK_API_KEY
from tools.feature_tools.encode_categorical import encode_categorical
from tools.feature_tools.scale_features import scale_features
from tools.feature_tools.save_featured_dataset import save_featured_dataset

TESTENV = Path(__file__).resolve().parent / "testenv"
TRAIN_CSV = TESTENV / "split" / "sample_train.csv"


# ── 图结构 ──────────────────────────────────────────────────────────────────

def test_feature_agent_graph_structure():
    agent = build_feature_agent(show_graph=False)
    graph = agent.get_graph()
    assert "feature_agent" in graph.nodes
    assert "tools" in graph.nodes
    assert "tool_state_update" in graph.nodes
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("__start__", "feature_agent") in edges
    assert ("tools", "tool_state_update") in edges
    assert ("tool_state_update", "feature_agent") in edges


# ── _get_current_model ───────────────────────────────────────────────────────

def test_get_current_model_returns_first_unprocessed():
    state = {
        "candidate_models": ["XGBoost", "RandomForest", "Ridge"],
        "feature_dataset_ids": {"XGBoost": "/some/path.csv"},
    }
    assert _get_current_model(state) == "RandomForest"


def test_get_current_model_returns_none_when_all_done():
    state = {
        "candidate_models": ["XGBoost", "RandomForest", "Ridge"],
        "feature_dataset_ids": {
            "XGBoost": "/a.csv",
            "RandomForest": "/b.csv",
            "Ridge": "/c.csv",
        },
    }
    assert _get_current_model(state) is None


def test_get_current_model_returns_first_when_none_done():
    state = {"candidate_models": ["XGBoost", "RandomForest", "Ridge"], "feature_dataset_ids": {}}
    assert _get_current_model(state) == "XGBoost"


# ── tool_state_update：save_featured_dataset ─────────────────────────────────

def test_handle_save_featured_merges_dict():
    payload = {"output_path": "/artifacts/features/Ridge_train_featured.csv", "model_name": "Ridge", "rows": 80, "cols": 10}
    state = {
        "feature_dataset_ids": {"XGBoost": "/a.csv"},
        "messages": [
            ToolMessage(content=json.dumps(payload), name="save_featured_dataset", tool_call_id="t1")
        ],
    }
    update = _tool_state_update_node(state)
    assert update["feature_dataset_ids"]["XGBoost"] == "/a.csv"
    assert update["feature_dataset_ids"]["Ridge"] == "/artifacts/features/Ridge_train_featured.csv"
    assert update["current_stage"] == "feature_saved"


def test_handle_save_featured_error_recorded():
    payload = {"error": {"type": "KeyError", "message": "dataset not found"}}
    state = {
        "messages": [
            ToolMessage(content=json.dumps(payload), name="save_featured_dataset", tool_call_id="t1")
        ]
    }
    update = _tool_state_update_node(state)
    assert update["current_stage"] == "feature_save_failed"
    assert update["errors"]


# ── scale_features 工具 ──────────────────────────────────────────────────────

def test_scale_features_standard(tmp_path):
    import uuid
    dataset_id = str(uuid.uuid4())
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0], "target": [0, 1, 0, 1]})
    put_dataset(df, dataset_id)

    result = json.loads(scale_features.invoke({"dataset_id": dataset_id, "columns": ["a", "b"], "method": "standard"}))
    assert result["scaled_columns"] == ["a", "b"]
    assert Path(result["scaler_path"]).exists()


def test_scale_features_skips_non_numeric(tmp_path):
    import uuid
    dataset_id = str(uuid.uuid4())
    df = pd.DataFrame({"text_col": ["x", "y", "z", "w"], "num": [1.0, 2.0, 3.0, 4.0]})
    put_dataset(df, dataset_id)

    result = json.loads(scale_features.invoke({"dataset_id": dataset_id, "columns": ["text_col", "num"], "method": "standard"}))
    assert "text_col" not in result["scaled_columns"]
    assert "num" in result["scaled_columns"]


# ── encode_categorical 工具 ──────────────────────────────────────────────────

def test_encode_categorical_label(tmp_path):
    import uuid
    dataset_id = str(uuid.uuid4())
    df = pd.DataFrame({"cat": ["a", "b", "a", "c"], "num": [1, 2, 3, 4]})
    put_dataset(df, dataset_id)

    result = json.loads(encode_categorical.invoke({"dataset_id": dataset_id, "columns": ["cat"], "method": "label"}))
    assert "cat" in result["encoded_columns"]
    assert Path(list(result["encoder_paths"].values())[0]).exists()


def test_encode_categorical_onehot():
    import uuid
    dataset_id = str(uuid.uuid4())
    df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3]})
    put_dataset(df, dataset_id)

    result = json.loads(encode_categorical.invoke({"dataset_id": dataset_id, "columns": ["cat"], "method": "onehot"}))
    assert "cat" in result["encoded_columns"]
    assert len(result["new_columns"]) == 2  # cat_a, cat_b


# ── save_featured_dataset 工具 ───────────────────────────────────────────────

def test_save_featured_dataset(tmp_path):
    import uuid
    dataset_id = str(uuid.uuid4())
    df = pd.DataFrame({"x": range(10), "y": range(10)})
    put_dataset(df, dataset_id)

    result = json.loads(save_featured_dataset.invoke({
        "dataset_id": dataset_id,
        "model_name": "TestModel",
        "output_dir": str(tmp_path),
    }))
    assert result["model_name"] == "TestModel"
    assert Path(result["output_path"]).exists()
    assert result["rows"] == 10


# ── 外部循环路由 ─────────────────────────────────────────────────────────────

def test_full_pipeline_feature_routing():
    from core.graph import _route_after_feature, _route_after_eda

    assert _route_after_eda({"candidate_models": ["XGBoost", "RF", "Ridge"]}) == "feature_agent"
    assert _route_after_eda({"candidate_models": []}) == END
    assert _route_after_eda({}) == END

    assert _route_after_feature({"candidate_models": ["XGBoost", "RF"], "feature_dataset_ids": {"XGBoost": "/a.csv"}}) == "feature_agent"
    assert _route_after_feature({"candidate_models": ["XGBoost"], "feature_dataset_ids": {"XGBoost": "/a.csv"}}) == "model_agent"


# ── LLM invoke 测试（需要 API Key）────────────────────────────────────────────

@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="需要配置 DEEPSEEK_API_KEY")
def test_feature_agent_invoke_single_model(tmp_path):
    """单模型特征工程（只处理第一个模型，验证流程可运行）。"""
    agent = build_feature_agent(show_graph=False)
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="开始特征工程")],
            "training_data_path": str(TRAIN_CSV),
            "task_type": "classification",
            "target_column": "subscribed",
            "candidate_models": ["LogisticRegression", "RandomForest", "XGBoost"],
            "feature_dataset_ids": {},
            "feature_engineering_plan": {
                "LogisticRegression": {"scaling": True, "encoding": "onehot", "notes": "线性模型需标准化"},
                "RandomForest": {"scaling": False, "encoding": "label", "notes": "树模型无需缩放"},
                "XGBoost": {"scaling": False, "encoding": "label", "notes": "树模型无需缩放"},
            },
            "run_dir": str(tmp_path),
        },
        config={"recursion_limit": 80},
    )
    feature_ids = result.get("feature_dataset_ids") or {}
    assert feature_ids, "应当至少完成一个模型的特征工程"
    for model, csv_path in feature_ids.items():
        assert Path(csv_path).exists(), f"{model} 的特征工程后 CSV 应已写入磁盘"


# ── 手动集成运行（pytest 不执行）─────────────────────────────────────────────

def _run_feature_agent():
    from agent.feature_agent import _pretty_print_message

    agent = build_feature_agent(show_graph=True)
    events = agent.stream(
        {
            "messages": [HumanMessage(content="开始特征工程")],
            "training_data_path": str(TRAIN_CSV),
            "task_type": "classification",
            "target_column": "subscribed",
            "candidate_models": ["LogisticRegression", "RandomForest", "XGBoost"],
            "feature_dataset_ids": {},
            "feature_engineering_plan": {
                "LogisticRegression": {"scaling": True, "encoding": "onehot", "notes": "线性模型需标准化"},
                "RandomForest": {"scaling": False, "encoding": "label", "notes": "树模型无需缩放"},
                "XGBoost": {"scaling": False, "encoding": "label", "notes": "树模型无需缩放"},
            },
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
        if event.get("current_stage"):
            print(f"\n[stage] {event['current_stage']}")
        if event.get("feature_dataset_ids"):
            print(f"[feature_ids] {event['feature_dataset_ids']}")


if __name__ == "__main__":
    _run_feature_agent()
