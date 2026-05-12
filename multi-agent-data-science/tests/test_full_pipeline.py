"""集成测试：主图结构、路由与 dataset_split 节点（工具层仍为 split_dataset_tool 使用）。"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from core.env_utils import DEEPSEEK_API_KEY
from core.graph import (
    _route_after_init,
    _route_after_model,
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
    for node in [
        "init_run",
        "split_agent",
        "agent_manager",
        "set_clean_train",
        "data_clean_train",
        "set_clean_test",
        "data_clean_test",
        "eda_agent",
        "feature_agent",
        "model_agent",
        "business_translator",
    ]:
        assert node in graph.nodes

    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("__start__", "init_run") in edges
    assert ("init_run", "agent_manager") in edges
    assert ("init_run", "split_agent") in edges
    assert ("split_agent", "agent_manager") in edges
    assert ("agent_manager", "set_clean_train") in edges
    assert ("data_clean_test", "eda_agent") in edges


def test_route_model_to_business_translator_when_best_model():
    assert _route_after_model({
        "candidate_models": ["X"],
        "model_results": {"X": {}},
        "best_model": {"model_name": "X"},
    }) == "business_translator"


def test_route_model_to_end_without_best_model():
    assert _route_after_model({
        "candidate_models": ["X"],
        "model_results": {"X": {}},
    }) == END


# ── 路由：init_run → split_agent / agent_manager ─────────────────────────────


def test_route_init_skips_split_when_presplit_paths():
    assert (
        _route_after_init(
            {"is_split": True, "training_data_path": "t.csv", "testing_data_path": "v.csv"}
        )
        == "agent_manager"
    )


def test_route_init_goes_split_agent_when_not_presplit():
    assert _route_after_init({"is_split": False}) == "split_agent"
    assert _route_after_init({}) == "split_agent"


# ── 图：start → init_run ─────────────────────────────────────────────────────


def test_pipeline_graph_contains_init_run_node():
    pipeline = build_full_pipeline(show_graph=False)
    graph = pipeline.get_graph()
    assert "init_run" in graph.nodes
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("__start__", "init_run") in edges
    assert ("init_run", "agent_manager") in edges
    assert ("init_run", "split_agent") in edges


# ── LLM invoke 集成测试（需要 API Key）────────────────────────────────────────


def _configured_deepseek_key() -> bool:
    k = (DEEPSEEK_API_KEY or "").strip()
    if not k or k == "your_deepseek_api_key":
        return False
    return not k.lower().startswith("your_")


@pytest.mark.skipif(
    not _configured_deepseek_key(),
    reason="需要配置有效的 DEEPSEEK_API_KEY（非 .env 示例占位符）",
)
def test_full_pipeline_invoke_progresses_from_raw_csv():
    """集成测试：从单 CSV 进入流水线，至少完成 init；若 LLM 走完则应有 split 路径等。"""
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

    tp = result.get("training_data_path") or ""
    sp = result.get("testing_data_path") or ""
    if tp and sp:
        assert Path(tp).exists()
        assert Path(sp).exists()

    if result.get("cleaning_passed"):
        cdp = result.get("cleaned_data_path") or result.get("train_cleaned_data_path")
        if cdp:
            assert Path(cdp).exists()


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
