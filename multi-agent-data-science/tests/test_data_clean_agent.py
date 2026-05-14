import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, ToolMessage

from agent.data_clean_agent import _tool_state_update_node, build_data_clean_agent
from core.env_utils import DEEPSEEK_API_KEY

TESTENV = Path(__file__).resolve().parent / "testenv"
RAW_CSV = TESTENV / "raw" / "sample.csv"

SAMPLE_CLEANING_PLAN = {
    "summary": "balance 列有2个缺失值，用中位数填充。",
    "column_name_actions": [],
    "type_conversions": [],
    "missing_value_actions": [{"column": "balance", "strategy": "median"}],
    "outlier_actions": [],
    "drop_columns": [],
    "notes": "",
}


# ── 图结构 ────────────────────────────────────────────────────────────────────

def test_build_data_clean_agent_graph():
    agent = build_data_clean_agent(show_graph=False)
    graph = agent.get_graph()

    assert "data_clean_agent" in graph.nodes
    assert "tools" in graph.nodes
    assert "tool_state_update" in graph.nodes

    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("__start__", "timer_start") in edges
    assert ("timer_start", "data_clean_agent") in edges
    assert ("tools", "tool_state_update") in edges
    assert ("tool_state_update", "data_clean_agent") in edges


# ── tool_state_update 状态写入 ────────────────────────────────────────────────

def test_load_tool_message_updates_dataset_id_and_report():
    payload = {
        "dataset_id": "ds_test_1",
        "source_path": "/tmp/raw.csv",
        "shape": {"rows": 10, "cols": 3},
        "columns": ["a", "b", "c"],
    }
    update = _tool_state_update_node(
        {
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    name="load_dataset_to_store",
                    tool_call_id="t1",
                )
            ]
        }
    )
    assert update["dataset_id"] == "ds_test_1"
    assert update["current_stage"] == "dataset_loaded"
    steps = update["cleaning_report"]["steps"]
    assert len(steps) == 1
    assert steps[0]["tool"] == "load_dataset_to_store"


def test_save_tool_message_updates_paths_and_passed_flag():
    load_payload = {
        "dataset_id": "ds_test_1",
        "source_path": "/tmp/raw.csv",
        "shape": {"rows": 5, "cols": 2},
        "columns": ["x", "y"],
    }
    prior = _tool_state_update_node(
        {
            "messages": [
                ToolMessage(
                    content=json.dumps(load_payload, ensure_ascii=False),
                    name="load_dataset_to_store",
                    tool_call_id="t0",
                )
            ]
        }
    )
    save_payload = {
        "dataset_id": "ds_test_1",
        "output_path": "/tmp/cleaned.csv",
        "shape": {"rows": 5, "cols": 2},
        "columns": ["x", "y"],
    }
    update = _tool_state_update_node(
        {
            "messages": [
                ToolMessage(
                    content=json.dumps(save_payload, ensure_ascii=False),
                    name="save_dataset_from_store",
                    tool_call_id="t2",
                )
            ],
            "cleaning_report": prior["cleaning_report"],
        }
    )
    assert update["cleaned_data_path"] == "/tmp/cleaned.csv"
    assert update["cleaning_passed"] is True
    assert update["current_stage"] == "cleaning_saved"
    assert len(update["cleaning_report"]["steps"]) == 2


def test_clean_step_error_records_stage_and_errors():
    payload = {"error": {"type": "KeyError", "message": "dataset_id 不存在: missing"}}
    update = _tool_state_update_node(
        {
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    name="clean_data_types",
                    tool_call_id="t3",
                )
            ]
        }
    )
    assert update["current_stage"] == "cleaning_step_failed"
    assert update["errors"][0]["type"] == "KeyError"
    assert update["cleaning_report"]["steps"][0]["result"] == payload


def test_clean_step_json_decode_error_keeps_raw_preview():
    update = _tool_state_update_node(
        {
            "cleaning_target": "train",
            "messages": [
                ToolMessage(
                    content="not-json",
                    name="clean_missing_values",
                    tool_call_id="t-json",
                )
            ],
        }
    )
    assert update["current_stage"] == "cleaning_step_failed"
    err = update["errors"][0]
    assert err["stage"] == "cleaning_step"
    assert err["tool"] == "clean_missing_values"
    assert err["type"] == "JSONDecodeError"
    assert "raw_content_preview" in err
    assert "not-json" in err["raw_content_preview"]


# ── LLM invoke 测试（需要 API Key）────────────────────────────────────────────

@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="需要配置 DEEPSEEK_API_KEY")
def test_data_clean_agent_invoke(tmp_path):
    agent = build_data_clean_agent(show_graph=False)
    cleaned_path = tmp_path / "sample_cleaned.csv"

    result = agent.invoke(
        {
            "messages": [HumanMessage(content="请按清洗计划执行内存清洗并保存结果。")],
            "user_request": "请按清洗计划执行内存清洗并保存结果。",
            "data_path": str(RAW_CSV),
            "description": "二分类数据集，balance 列有缺失值需要填充。",
            "task_type": "classification",
            "target_column": "subscribed",
            "cleaning_plan": SAMPLE_CLEANING_PLAN,
            "cleaned_data_path": str(cleaned_path),
            "run_dir": str(tmp_path),
            "current_stage": "cleaning_plan_created",
            "next_action": "run_data_cleaning",
        },
        config={"recursion_limit": 60},
    )

    assert result.get("dataset_id"), "应当加载数据集并记录 dataset_id"
    steps = (result.get("cleaning_report") or {}).get("steps") or []
    assert steps, "应当至少执行一步工具调用"

    if result.get("cleaning_passed") and result.get("cleaned_data_path"):
        out = Path(result["cleaned_data_path"])
        assert out.exists(), "清洗后的 CSV 应已写入磁盘"
        df = pd.read_csv(out)
        assert df["balance"].isna().sum() == 0, "balance 缺失值应已填充"


# ── 手动集成运行（pytest 不执行）─────────────────────────────────────────────

def _run_data_clean_agent():
    agent = build_data_clean_agent(show_graph=True)
    cleaned_path = PROJECT_ROOT / "artifacts" / "sample_cleaned.csv"
    events = agent.stream(
        {
            "messages": [HumanMessage(content="请按清洗计划执行内存清洗并保存结果。")],
            "user_request": "请按清洗计划执行内存清洗并保存结果。",
            "data_path": str(RAW_CSV),
            "description": "二分类数据集，balance 列有缺失值需要填充。",
            "task_type": "classification",
            "target_column": "subscribed",
            "cleaning_plan": SAMPLE_CLEANING_PLAN,
            "cleaned_data_path": str(cleaned_path),
            "current_stage": "cleaning_plan_created",
            "next_action": "run_data_cleaning",
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            print(event["messages"][-1].content[:200])
        if event.get("current_stage"):
            print(f"[stage] {event['current_stage']}")


if __name__ == "__main__":
    _run_data_clean_agent()
