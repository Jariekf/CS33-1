"""集成测试：Agent Manager → Data Clean Agent → Split 完整流水线结构验证。"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from core.dataset_split import split_dataset_node
from core.env_utils import DEEPSEEK_API_KEY
from core.graph import (
    _route_after_agent_manager,
    _route_after_data_clean,
    build_full_pipeline,
)

TESTENV = Path(__file__).resolve().parent / "testenv"
RAW_CSV = TESTENV / "raw" / "sample.csv"
SAMPLE_DESCRIPTION = (
    "二分类数据集：预测客户是否订阅银行定期存款（subscribed: yes/no）。"
    "特征：age、job、education、balance。"
)


# ── 图结构 ──────────────────────────────────────────────────────────────────

def test_pipeline_graph_contains_all_nodes():
    pipeline = build_full_pipeline(show_graph=False)
    graph = pipeline.get_graph()
    for node in ["init_run", "agent_manager", "data_clean_agent", "split", "eda_agent", "feature_agent", "model_agent", "compare"]:
        assert node in graph.nodes

    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("__start__", "init_run") in edges
    assert ("init_run", "agent_manager") in edges
    assert ("split", "__end__") in edges


# ── 路由：Manager → Data Clean ───────────────────────────────────────────────

def test_route_manager_to_clean_when_plan_ready():
    assert (
        _route_after_agent_manager(
            {"next_action": "run_data_cleaning", "cleaning_plan": {"summary": "x"}}
        )
        == "data_clean_agent"
    )


def test_route_manager_ends_without_plan():
    assert (
        _route_after_agent_manager(
            {"next_action": "run_data_cleaning", "cleaning_plan": {}}
        )
        == END
    )


def test_route_manager_ends_without_next_action():
    assert (
        _route_after_agent_manager(
            {"next_action": "", "cleaning_plan": {"summary": "x"}}
        )
        == END
    )


# ── 路由：Data Clean → Split ─────────────────────────────────────────────────

def test_route_clean_to_split_when_passed():
    assert _route_after_data_clean({"cleaning_passed": True}) == "split"


def test_route_clean_ends_when_not_passed():
    assert _route_after_data_clean({"cleaning_passed": False}) == END


def test_route_clean_ends_when_key_missing():
    assert _route_after_data_clean({}) == END


# ── Split 节点（衔接清洗输出的路径写入）────────────────────────────────────────

def test_split_node_classification_writes_paths(tmp_path):
    cleaned = tmp_path / "cleaned.csv"
    pd.DataFrame({"x": range(50), "label": ["a"] * 25 + ["b"] * 25}).to_csv(cleaned, index=False)

    update = split_dataset_node(
        {
            "cleaned_data_path": str(cleaned),
            "task_type": "classification",
            "target_column": "label",
        }
    )

    assert update["current_stage"] == "split_done"
    assert Path(update["training_data_path"]).exists()
    assert Path(update["testing_data_path"]).exists()
    assert update["split_report"]["stratify_used"] is True


def test_split_node_regression_writes_paths(tmp_path):
    cleaned = tmp_path / "reg.csv"
    pd.DataFrame({"x": range(100), "y": range(100)}).to_csv(cleaned, index=False)

    update = split_dataset_node(
        {
            "cleaned_data_path": str(cleaned),
            "task_type": "regression",
        }
    )

    assert update["current_stage"] == "split_done"
    assert Path(update["training_data_path"]).exists()
    assert Path(update["testing_data_path"]).exists()
    assert update["split_report"]["stratify_used"] is False


def test_split_node_fallback_to_data_path_when_no_cleaned(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"a": range(20)}).to_csv(raw, index=False)

    update = split_dataset_node(
        {
            "data_path": str(raw),
            "task_type": "regression",
        }
    )

    assert update["current_stage"] == "split_done"
    assert Path(update["training_data_path"]).exists()


def test_split_node_fails_without_any_path():
    update = split_dataset_node({"task_type": "regression"})
    assert update["current_stage"] == "split_failed"
    assert update["errors"]


# ── 图结构（包含 init_run 节点）─────────────────────────────────────────────

def test_pipeline_graph_contains_init_run_node():
    pipeline = build_full_pipeline(show_graph=False)
    graph = pipeline.get_graph()
    assert "init_run" in graph.nodes
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("__start__", "init_run") in edges
    assert ("init_run", "agent_manager") in edges


# ── LLM invoke 集成测试（需要 API Key）────────────────────────────────────────

@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="需要配置 DEEPSEEK_API_KEY")
def test_full_pipeline_invoke_to_split(tmp_path):
    """集成测试：从原始数据到 split 完成（Manager → Clean → Split）。"""
    pipeline = build_full_pipeline(show_graph=False)
    user_input = "请分析数据集，执行清洗，并切分训练集和测试集。"
    result = pipeline.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "data_path": str(RAW_CSV),
            "description": SAMPLE_DESCRIPTION,
        },
        config={"recursion_limit": 120},
    )
    assert result.get("run_dir"), "init_run 应当生成 run_dir"
    assert Path(result["run_dir"]).exists()

    if result.get("cleaning_passed"):
        assert result.get("cleaned_data_path")
        assert Path(result["cleaned_data_path"]).exists()

    if result.get("current_stage") == "split_done":
        assert Path(result["training_data_path"]).exists()
        assert Path(result["testing_data_path"]).exists()


# ── 手动集成运行（pytest 不执行）─────────────────────────────────────────────

def _run_full_pipeline():
    from agent.agent_manager import _pretty_print_message

    pipeline = build_full_pipeline(show_graph=True)
    user_input = input("请输入：")
    events = pipeline.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "data_path": str(RAW_CSV),
            "description": SAMPLE_DESCRIPTION,
        },
        stream_mode="values",
        config={"recursion_limit": 120},
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
        if event.get("current_stage"):
            print(f"\n[stage] {event['current_stage']}")
        if event.get("training_data_path"):
            print(f"[split] train: {event['training_data_path']}")
            print(f"[split] test:  {event['testing_data_path']}")


if __name__ == "__main__":
    _run_full_pipeline()
