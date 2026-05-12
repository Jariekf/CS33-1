import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from agent.agent_manager import (
    _has_complete_task_meta,
    _pretty_print_message,
    _tool_state_update_node,
    agent_manager,
    build_agent_manager,
)
from core.env_utils import DEEPSEEK_API_KEY

TESTENV = Path(__file__).resolve().parent / "testenv"
RAW_CSV = TESTENV / "raw" / "sample.csv"
SAMPLE_DESCRIPTION = (
    "二分类数据集：预测客户是否订阅银行定期存款（subscribed: yes/no）。"
    "特征：age、job、education、balance。"
)


def test_has_complete_task_meta_true_when_both_set():
    assert _has_complete_task_meta({"task_type": "classification", "target_column": "y"})


def test_has_complete_task_meta_false_when_incomplete():
    assert not _has_complete_task_meta({"task_type": "", "target_column": "y"})
    assert not _has_complete_task_meta({"task_type": "classification", "target_column": ""})


def test_build_agent_manager_graph():
    agent_manager = build_agent_manager(show_graph=False)
    graph = agent_manager.get_graph()

    assert "agent_manager" in graph.nodes
    assert "tools" in graph.nodes
    assert "tool_state_update" in graph.nodes

    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("__start__", "agent_manager") in edges
    assert ("tools", "tool_state_update") in edges
    assert ("tool_state_update", "agent_manager") in edges


def test_profile_tool_message_updates_global_state():
    profile = {
        "dataset": {
            "file_path": "C:/Projects/multi-agent-data-science/datasets/Financials.csv",
            "rows": 700,
            "cols": 16,
            "columns": [" Product ", "Date", "Year"],
            "dtypes": {" Product ": "str", "Date": "str", "Year": "int64"},
        },
        "missing_values": {
            "columns": {
                " Product ": {"missing_count": 0, "missing_ratio": 0.0},
                "Date": {"missing_count": 0, "missing_ratio": 0.0},
                "Year": {"missing_count": 0, "missing_ratio": 0.0},
            }
        },
        "outliers": {
            "columns": {
                "Year": {
                    "outlier_count": 175,
                    "outlier_ratio": 0.25,
                }
            }
        },
        "type_issues": {
            "numeric_like_object_columns": [],
            "datetime_like_object_columns": ["Date"],
            "category_columns": {
                " Product ": {"unique_count": 6, "unique_ratio": 0.008571},
                "Date": {"unique_count": 16, "unique_ratio": 0.022857},
            },
            "high_cardinality_columns": [],
        },
        "drop_candidates": {
            "empty_columns": [],
            "constant_columns": [],
            "id_like_columns": [],
        }
    }
    state_update = _tool_state_update_node(
        {
            "messages": [
                ToolMessage(
                    content=json.dumps(profile),
                    name="profile_dataset",
                    tool_call_id="test-call",
                )
            ]
        }
    )

    assert state_update["dataset_profile"] == profile
    assert state_update["dataset_name"] == "Financials.csv"
    assert state_update["current_stage"] == "profile_done"


def test_infer_task_tool_message_updates_global_state():
    task_result = {
        "task_type": "regression",
        "target_column": " Profit ",
        "confidence": 0.9,
        "reason": "用户希望预测利润，利润是连续型业务指标。",
    }
    state_update = _tool_state_update_node(
        {
            "messages": [
                ToolMessage(
                    content=json.dumps(task_result, ensure_ascii=False),
                    name="infer_task",
                    tool_call_id="test-call",
                )
            ]
        }
    )

    assert state_update["task_type"] == "regression"
    assert state_update["target_column"] == " Profit "
    assert state_update["current_stage"] == "task_inferred"


def test_manager_rap_tool_message_updates_global_state(tmp_path):
    rap_result = {
        "user_requirement_summary": "Build a classifier for bank term deposit subscription.",
        "rap_retrieval_slots": {
            "downstream_task": "term deposit subscription prediction",
            "application_domain": "banking direct marketing",
            "area": "imbalanced tabular classification",
        },
        "manager_rap_context": "- Preserve campaign outcome and contact-duration signals.",
        "manager_rap_meta": {
            "rap_enabled": True,
            "fallback_used": False,
            "errors": [],
        },
    }

    state_update = _tool_state_update_node(
        {
            "run_dir": str(tmp_path),
            "messages": [
                ToolMessage(
                    content=json.dumps(rap_result, ensure_ascii=False),
                    name="build_manager_rap_context",
                    tool_call_id="test-call",
                )
            ],
        }
    )

    assert state_update["user_requirement_summary"] == rap_result["user_requirement_summary"]
    assert state_update["rap_retrieval_slots"] == rap_result["rap_retrieval_slots"]
    assert state_update["manager_rap_context"] == rap_result["manager_rap_context"]
    assert state_update["manager_rap_meta"] == rap_result["manager_rap_meta"]
    assert state_update["current_stage"] == "manager_rap_done"


def test_manager_rap_tool_appends_report_when_run_dir(tmp_path):
    rap_result = {
        "user_requirement_summary": "Summary line for preview.",
        "rap_retrieval_slots": {},
        "manager_rap_context": "Context line for preview.",
        "manager_rap_meta": {"rap_enabled": True, "errors": []},
    }
    _tool_state_update_node(
        {
            "run_dir": str(tmp_path),
            "messages": [
                ToolMessage(
                    content=json.dumps(rap_result, ensure_ascii=False),
                    name="build_manager_rap_context",
                    tool_call_id="test-call",
                )
            ],
        }
    )
    report_path = tmp_path / "reports" / "agent_manager_report.md"
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    assert "Manager RAP · 中层检测" in text
    assert "user_requirement_summary" in text
    assert "manager_rap_context" in text
    assert "Summary line for preview." in text
    assert "Context line for preview." in text


def test_append_manager_rap_midlayer_skips_when_no_rap_fields():
    from core.report_writer import append_manager_rap_midlayer_to_report

    assert append_manager_rap_midlayer_to_report({}) is None


def test_create_cleaning_plan_train_updates_state():
    cleaning_plan = {
        "target": "train",
        "summary": "清理列名空格，转换日期字段，并准备金额字段清洗。",
        "column_name_actions": [
            {
                "column": " Product ",
                "action": "strip",
                "reason": "列名前后存在空格。",
            }
        ],
        "type_conversions": [
            {
                "column": "Date",
                "target_type": "datetime",
                "reason": "Date 是日期字符串。",
            }
        ],
        "missing_value_actions": [],
        "outlier_actions": [],
        "drop_columns": [],
        "notes": ["Financials 数据集暂无缺失值。"],
    }
    state_update = _tool_state_update_node(
        {
            "messages": [
                ToolMessage(
                    content=json.dumps(cleaning_plan, ensure_ascii=False),
                    name="create_cleaning_plan",
                    tool_call_id="test-call",
                )
            ]
        }
    )

    assert state_update["train_cleaning_plan"] == cleaning_plan
    assert state_update["cleaning_plan"] == cleaning_plan
    assert state_update["current_stage"] == "train_cleaning_plan_created"


def test_create_cleaning_plan_test_updates_state():
    cleaning_plan = {
        "target": "test",
        "summary": "与 train 同构清洗。",
        "column_name_actions": [],
        "type_conversions": [],
        "missing_value_actions": [],
        "outlier_actions": [],
        "drop_columns": [],
        "notes": [],
    }
    state_update = _tool_state_update_node(
        {
            "messages": [
                ToolMessage(
                    content=json.dumps(cleaning_plan, ensure_ascii=False),
                    name="create_cleaning_plan",
                    tool_call_id="test-call",
                )
            ]
        }
    )

    assert state_update["test_cleaning_plan"] == cleaning_plan
    assert state_update["cleaning_plan"] == cleaning_plan
    assert state_update["current_stage"] == "cleaning_plans_created"
    assert state_update["next_action"] == "run_data_cleaning"


def _print_state_snapshot(state: dict):
    snapshot = {
        "current_stage": state.get("current_stage"),
        "next_action": state.get("next_action"),
        "dataset_name": state.get("dataset_name"),
        "task_type": state.get("task_type"),
        "target_column": state.get("target_column"),
        "has_dataset_profile": "dataset_profile" in state,
        "has_cleaning_plan": "cleaning_plan" in state,
    }
    if "dataset_profile" in state:
        dataset = state["dataset_profile"].get("dataset", {})
        snapshot["dataset_profile_summary"] = {
            "rows": dataset.get("rows"),
            "cols": dataset.get("cols"),
        }
    if "cleaning_plan" in state:
        snapshot["cleaning_plan_summary"] = state["cleaning_plan"].get("summary")

    print("\n[state snapshot]")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


# ── LLM invoke 测试（需要 API Key）────────────────────────────────────────────

@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="需要配置 DEEPSEEK_API_KEY")
def test_agent_manager_invoke(tmp_path):
    agent = build_agent_manager(show_graph=False)
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="请分析数据集并给出清洗计划")],
            "user_request": "请分析数据集并给出清洗计划",
            "training_data_path": str(RAW_CSV),
            "data_path": str(RAW_CSV),
            "testing_data_path": str(RAW_CSV),
            "description": SAMPLE_DESCRIPTION,
            "run_dir": str(tmp_path),
        },
        config={"recursion_limit": 60},
    )
    assert result.get("dataset_profile") is not None
    assert (
        result.get("train_cleaning_plan")
        or result.get("test_cleaning_plan")
        or result.get("cleaning_plan")
    ), "预期在完成 profile 后继续产出 train/test 清洗计划（若递归用尽可能失败，可调大 recursion_limit）"


# ── 手动集成运行（pytest 不执行）─────────────────────────────────────────────

def run_agent_manager_test():
    user_input = input("请输入：")
    events = agent_manager.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "data_path": str(RAW_CSV),
            "description": SAMPLE_DESCRIPTION,
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
        _print_state_snapshot(event)


if __name__ == "__main__":
    run_agent_manager_test()
