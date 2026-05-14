import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from openai import APIConnectionError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.data_clean_agent import _clean_self_check, _tool_state_update_node, build_data_clean_agent
from core.env_utils import DEEPSEEK_API_KEY

TESTENV = Path(__file__).resolve().parent / "testenv"
RAW_CSV = TESTENV / "raw" / "sample.csv"

SAMPLE_CLEANING_PLAN = {
    "summary": "Fill missing balance with median.",
    "column_name_actions": [],
    "type_conversions": [],
    "missing_value_actions": [{"column": "balance", "strategy": "median"}],
    "outlier_actions": [],
    "drop_columns": [],
    "notes": "",
}


def test_build_data_clean_agent_graph():
    agent = build_data_clean_agent(show_graph=False)
    graph = agent.get_graph()

    assert "data_clean_agent" in graph.nodes
    assert "tools" in graph.nodes
    assert "tool_state_update" in graph.nodes
    assert "self_check" in graph.nodes

    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("__start__", "timer_start") in edges
    assert ("timer_start", "data_clean_agent") in edges
    assert ("tools", "tool_state_update") in edges
    assert ("tool_state_update", "data_clean_agent") in edges


def test_load_tool_message_updates_dataset_id_and_train_report():
    payload = {
        "dataset_id": "ds_test_1",
        "source_path": "/tmp/raw.csv",
        "shape": {"rows": 10, "cols": 3},
        "columns": ["a", "b", "c"],
    }
    update = _tool_state_update_node(
        {
            "cleaning_target": "train",
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    name="load_dataset_to_store",
                    tool_call_id="t1",
                )
            ],
        }
    )
    assert update["dataset_id"] == "ds_test_1"
    assert update["current_stage"] == "dataset_loaded"
    steps = update["train_cleaning_report"]["steps"]
    assert len(steps) == 1
    assert steps[0]["tool"] == "load_dataset_to_store"


def test_save_tool_message_updates_train_paths_and_passed_flag():
    load_payload = {
        "dataset_id": "ds_test_1",
        "source_path": "/tmp/raw.csv",
        "shape": {"rows": 5, "cols": 2},
        "columns": ["x", "y"],
    }
    prior = _tool_state_update_node(
        {
            "cleaning_target": "train",
            "messages": [
                ToolMessage(
                    content=json.dumps(load_payload, ensure_ascii=False),
                    name="load_dataset_to_store",
                    tool_call_id="t0",
                )
            ],
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
            "cleaning_target": "train",
            "messages": [
                ToolMessage(
                    content=json.dumps(save_payload, ensure_ascii=False),
                    name="save_dataset_from_store",
                    tool_call_id="t2",
                )
            ],
            "train_cleaning_report": prior["train_cleaning_report"],
        }
    )
    assert update["train_cleaned_data_path"] == "/tmp/cleaned.csv"
    assert update["training_data_path"] == "/tmp/cleaned.csv"
    assert update["cleaned_data_path"] == "/tmp/cleaned.csv"
    assert update["train_cleaning_passed"] is True
    assert update["cleaning_passed"] is True
    assert update["current_stage"] == "train_cleaning_saved"
    assert len(update["train_cleaning_report"]["steps"]) == 2


def test_save_tool_message_updates_test_paths_and_passed_flag():
    load_payload = {
        "dataset_id": "ds_test_2",
        "source_path": "/tmp/raw_test.csv",
        "shape": {"rows": 5, "cols": 2},
        "columns": ["x", "y"],
    }
    prior = _tool_state_update_node(
        {
            "cleaning_target": "test",
            "messages": [
                ToolMessage(
                    content=json.dumps(load_payload, ensure_ascii=False),
                    name="load_dataset_to_store",
                    tool_call_id="t0",
                )
            ],
        }
    )
    save_payload = {
        "dataset_id": "ds_test_2",
        "output_path": "/tmp/test_cleaned.csv",
        "shape": {"rows": 5, "cols": 2},
        "columns": ["x", "y"],
    }
    update = _tool_state_update_node(
        {
            "cleaning_target": "test",
            "messages": [
                ToolMessage(
                    content=json.dumps(save_payload, ensure_ascii=False),
                    name="save_dataset_from_store",
                    tool_call_id="t2",
                )
            ],
            "test_cleaning_report": prior["test_cleaning_report"],
        }
    )
    assert update["test_cleaned_data_path"] == "/tmp/test_cleaned.csv"
    assert update["testing_data_path"] == "/tmp/test_cleaned.csv"
    assert update["test_cleaning_passed"] is True
    assert update["cleaning_passed"] is True
    assert update["current_stage"] == "test_cleaning_saved"
    assert len(update["test_cleaning_report"]["steps"]) == 2


def test_clean_step_error_records_stage_and_errors():
    payload = {"error": {"type": "KeyError", "message": "dataset_id missing"}}
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
    assert update["train_cleaning_report"]["steps"][0]["result"] == payload


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


def test_trailing_tool_messages_record_save_step_when_save_not_last():
    prior = _tool_state_update_node(
        {
            "cleaning_target": "train",
            "messages": [
                ToolMessage(
                    content=json.dumps({"dataset_id": "ds_test_3"}, ensure_ascii=False),
                    name="load_dataset_to_store",
                    tool_call_id="t-load",
                )
            ],
        }
    )

    save_payload = {
        "dataset_id": "ds_test_3",
        "output_path": "/tmp/cleaned_batch.csv",
        "shape": {"rows": 3, "cols": 2},
        "columns": ["a", "b"],
    }
    clean_payload = {"dataset_id": "ds_test_3", "converted_columns": ["a"], "skipped_columns": []}
    update = _tool_state_update_node(
        {
            "cleaning_target": "train",
            "messages": [
                HumanMessage(content="run clean"),
                ToolMessage(
                    content=json.dumps(save_payload, ensure_ascii=False),
                    name="save_dataset_from_store",
                    tool_call_id="t-save",
                ),
                ToolMessage(
                    content=json.dumps(clean_payload, ensure_ascii=False),
                    name="clean_data_types",
                    tool_call_id="t-clean",
                ),
            ],
            "train_cleaning_report": prior["train_cleaning_report"],
        }
    )
    steps = update["train_cleaning_report"]["steps"]
    assert any(step.get("tool") == "save_dataset_from_store" for step in steps)
    assert update["train_cleaned_data_path"] == "/tmp/cleaned_batch.csv"
    assert update["train_cleaning_passed"] is True


def test_clean_self_check_recovers_missing_save_step_for_test_target():
    work_dir = PROJECT_ROOT / "artifacts" / "tmp_tests" / "clean_self_check"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "cleaned_test.csv"
    output_path.write_text("a\n1\n", encoding="utf-8")
    payload = {"dataset_id": "ds_test_4", "output_path": str(output_path)}

    update = _clean_self_check(
        {
            "cleaning_target": "test",
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    name="save_dataset_from_store",
                    tool_call_id="t-save",
                )
            ],
            "test_cleaning_passed": False,
            "test_cleaned_data_path": "",
            "run_dir": str(work_dir),
        }
    )

    assert update["test_cleaned_data_path"] == str(output_path)
    assert update["testing_data_path"] == str(output_path)
    assert update["test_cleaning_passed"] is True
    steps = (update.get("test_cleaning_report") or {}).get("steps") or []
    assert steps
    assert steps[-1]["tool"] == "save_dataset_from_store"


@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="Requires DEEPSEEK_API_KEY")
def test_data_clean_agent_invoke():
    agent = build_data_clean_agent(show_graph=False)
    work_dir = PROJECT_ROOT / "artifacts" / "tmp_tests" / "data_clean_invoke"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Please run cleaning and save.")],
                "user_request": "Please run cleaning and save.",
                "data_path": str(RAW_CSV),
                "description": "Binary dataset with missing values in balance.",
                "task_type": "classification",
                "target_column": "subscribed",
                "cleaning_plan": SAMPLE_CLEANING_PLAN,
                "run_dir": str(work_dir),
                "current_stage": "cleaning_plan_created",
                "next_action": "run_data_cleaning",
            },
            config={"recursion_limit": 60},
        )
    except APIConnectionError:
        pytest.skip("LLM endpoint unavailable in this environment")

    assert result.get("dataset_id"), "Expected a dataset_id after loading dataset"
    steps = (result.get("train_cleaning_report") or {}).get("steps") or []
    assert steps, "Expected at least one tool step in train_cleaning_report"

    if result.get("train_cleaning_passed") and result.get("train_cleaned_data_path"):
        out = Path(result["train_cleaned_data_path"])
        assert out.exists(), "Expected cleaned CSV to be saved"
        df = pd.read_csv(out)
        assert df["balance"].isna().sum() == 0, "Expected missing balance values to be filled"
