"""完整流水线：Agent Manager → Data Clean Agent → Split 节点。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from agent.agent_manager import _pretty_print_message, build_agent_manager
from agent.data_clean_agent import build_data_clean_agent
from agent.eda_agent import build_eda_agent
from agent.feature_agent import build_feature_agent, _get_current_model
from agent.model_agent import build_model_agent, _get_current_model as _get_current_model_for_eval
from core.compare_node import compare_models_node
from core.dataset_split import split_dataset_node
from core.global_state import GlobalState


def _init_run_node(state: GlobalState) -> GlobalState:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_dir = PROJECT_ROOT / "artifacts" / "workspace" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init_run] 本次运行目录：{run_dir}")
    return {"run_dir": str(run_dir)}


def _route_after_agent_manager(state: GlobalState) -> str:
    """Manager 结束后：若已生成清洗计划且约定下一步为清洗，则进入 Data Clean。"""
    plan = state.get("cleaning_plan")
    if (
        state.get("next_action") == "run_data_cleaning"
        and isinstance(plan, dict)
        and bool(plan)
    ):
        return "data_clean_agent"
    return END


def _route_after_data_clean(state: GlobalState) -> str:
    """Data Clean 结束后：清洗通过则进入 split，否则结束。"""
    if state.get("cleaning_passed"):
        return "split"
    return END


def _route_after_split(state: GlobalState) -> str:
    """Split 结束后：成功则进入 EDA，否则结束。"""
    if state.get("current_stage") == "split_done":
        return "eda_agent"
    return END


def _route_after_eda(state: GlobalState) -> str:
    """EDA 结束后：有候选模型则进入 Feature Agent，否则结束。"""
    if state.get("candidate_models"):
        return "feature_agent"
    return END


def _route_after_feature(state: GlobalState) -> str:
    """Feature Agent 结束后：还有未处理的模型则循环，否则进入 Model Agent。"""
    if _get_current_model(state) is not None:
        return "feature_agent"
    return "model_agent"


def _route_after_model(state: GlobalState) -> str:
    """Model Agent 结束后：还有未评估的模型则循环，否则进入 Compare。"""
    if _get_current_model_for_eval(state) is not None:
        return "model_agent"
    return "compare"


def _show_graph_if_needed(compiled, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "full_pipeline.png"
        image_bytes = compiled.get_graph().draw_mermaid_png()
        image_path.write_bytes(image_bytes)
        print(f"Pipeline Graph 图片已保存: {image_path}")
        try:
            from IPython.display import Image, display
            display(Image(filename=str(image_path)))
        except Exception:
            pass
    except Exception as exc:
        print(f"图渲染失败: {exc}")


def build_full_pipeline(show_graph: bool = False):
    manager_graph = build_agent_manager(show_graph=False)
    clean_graph = build_data_clean_agent(show_graph=False)
    eda_graph = build_eda_agent(show_graph=False)
    feature_graph = build_feature_agent(show_graph=False)
    model_graph = build_model_agent(show_graph=False)

    graph = StateGraph(GlobalState)
    graph.add_node("init_run", _init_run_node)
    graph.add_node("agent_manager", manager_graph)
    graph.add_node("data_clean_agent", clean_graph)
    graph.add_node("split", split_dataset_node)
    graph.add_node("eda_agent", eda_graph)
    graph.add_node("feature_agent", feature_graph)
    graph.add_node("model_agent", model_graph)
    graph.add_node("compare", compare_models_node)
    graph.set_entry_point("init_run")
    graph.add_edge("init_run", "agent_manager")
    graph.add_conditional_edges(
        "agent_manager",
        _route_after_agent_manager,
        {"data_clean_agent": "data_clean_agent", END: END},
    )
    graph.add_conditional_edges(
        "data_clean_agent",
        _route_after_data_clean,
        {"split": "split", END: END},
    )
    graph.add_conditional_edges(
        "split",
        _route_after_split,
        {"eda_agent": "eda_agent", END: END},
    )
    graph.add_conditional_edges(
        "eda_agent",
        _route_after_eda,
        {"feature_agent": "feature_agent", END: END},
    )
    graph.add_conditional_edges(
        "feature_agent",
        _route_after_feature,
        {"feature_agent": "feature_agent", "model_agent": "model_agent"},
    )
    graph.add_conditional_edges(
        "model_agent",
        _route_after_model,
        {"model_agent": "model_agent", "compare": "compare"},
    )
    graph.add_edge("compare", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


full_pipeline = build_full_pipeline()


if __name__ == "__main__":
    pipeline = build_full_pipeline(show_graph=True)
    user_input = input("请输入：")
    events = pipeline.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            # "data_path": str(PROJECT_ROOT / "datasets" / "Financials.csv"),
            "data_path": str(PROJECT_ROOT / "datasets" / "bank-full.csv"),
            # "description": "这是一个金融行业销售数据集，包含销售、折扣、利润、国家、产品和时间字段。",
            "description": "The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. There are four datasets: 1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014] 2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs. 3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs). 4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs). The smallest datasets are provided to test more computationally demanding machine learning algorithms(e.g., SVM). The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).",
            # "cleaned_data_path": str(PROJECT_ROOT / "artifacts" / "Financials_cleaned.csv"),
           # "cleaned_data_path": str(PROJECT_ROOT / "artifacts" / "bank-full_cleaned.csv"),
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
