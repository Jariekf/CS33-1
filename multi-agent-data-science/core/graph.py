"""Full pipeline: Agent Manager → Data Clean Agent → Split node."""

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
from agent.business_translator_agent import build_business_translator_agent
from agent.data_clean_agent import build_data_clean_agent
from agent.eda_agent import build_eda_agent
from agent.feature_agent import build_feature_agent, _get_current_model
from agent.model_agent import build_model_agent
from agent.split_agent import build_split_agent
from agent.supervisor import make_supervisor
from core.global_state import GlobalState
from core.report_writer import log_node, save_state_snapshot


def _init_run_node(state: GlobalState) -> GlobalState:
    import time
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_dir = PROJECT_ROOT / "artifacts" / "workspace" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    is_split = bool(
        (state.get("training_data_path") or "").strip()
        and (state.get("testing_data_path") or "").strip()
    )
    data_path = (state.get("data_path") or "").strip()
    training_data_path = (state.get("training_data_path") or "").strip()
    if data_path:
        dataset_name = Path(data_path).stem
    elif training_data_path:
        dataset_name = Path(training_data_path).stem
    else:
        dataset_name = "dataset"
    print(f"[init_run] run directory: {run_dir}  is_split={is_split}  dataset_name={dataset_name}")
    return {"run_dir": str(run_dir), "is_split": is_split, "dataset_name": dataset_name, "timing": {"pipeline_start": time.time()}}



def _route_after_init(state: GlobalState) -> str:
    """If train/test paths are already provided, skip splitting and go directly to agent_manager."""
    if state.get("is_split"):
        return "agent_manager"
    return "split_agent"


def _route_after_eda(state: GlobalState) -> str:
    """After EDA: if candidate models exist enter Feature Agent, otherwise end."""
    if state.get("candidate_models"):
        return "feature_agent"
    return END


def _route_after_feature(state: GlobalState) -> str:
    """After Feature Agent: loop if there are unprocessed models remaining, otherwise enter Model Agent."""
    if _get_current_model(state) is not None:
        return "feature_agent"
    return "model_agent"


def _route_after_model(state: GlobalState) -> str:
    """After Model Agent:
    The model_agent subgraph completes tune_and_select → train(best) → evaluate(best) → summary in a single run,
    so no per-model loop is needed at the main graph level (models not selected as best do not appear
    in model_results train/evaluate fields; looping would cause an infinite loop).
    Only check whether best_model has finished final evaluation to determine the terminal state."""
    best_model = state.get("best_model") or {}
    if best_model.get("model_name") and best_model.get("final_test_metrics"):
        log_node("finalize", success=True, detail=f"best={best_model.get('model_name')}")
        save_state_snapshot(state)
        return "business_translator"
    log_node("finalize", success=False, detail="best_model evaluation not completed")
    return END


def _show_graph_if_needed(compiled, show_graph: bool) -> None:
    if not show_graph:
        return
    output_dir = PROJECT_ROOT / "artifacts" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = output_dir / "full_pipeline.mmd"
    mmd_path.write_text(compiled.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Pipeline Mermaid source saved: {mmd_path}  (paste into https://mermaid.live to view)")
    try:
        image_path = output_dir / "full_pipeline.png"
        image_path.write_bytes(compiled.get_graph().draw_mermaid_png())
        print(f"Pipeline graph image saved: {image_path}")
        try:
            from IPython.display import Image, display
            display(Image(filename=str(image_path)))
        except Exception:
            pass
    except Exception as exc:
        print(f"Graph rendering failed: {exc}")


def _set_clean_train(state: GlobalState) -> GlobalState:
    """Enter the train cleaning subgraph: freeze the current train path as raw (for multi-round ReAct prompts and stable file naming)."""
    cur = (state.get("training_data_path") or "").strip()
    raw = (state.get("raw_training_data_path") or "").strip()
    out: GlobalState = {"cleaning_target": "train", "dataset_id": ""}
    if cur and not raw:
        out["raw_training_data_path"] = cur
    return out


def _set_clean_test(state: GlobalState) -> GlobalState:
    """Enter the test cleaning subgraph: freeze the current test path as raw."""
    cur = (state.get("testing_data_path") or "").strip()
    raw = (state.get("raw_testing_data_path") or "").strip()
    out: GlobalState = {"cleaning_target": "test", "dataset_id": ""}
    if cur and not raw:
        out["raw_testing_data_path"] = cur
    return out


def _reset_clean_train_internal_retry(state: GlobalState) -> GlobalState:
    """Reset data_clean_agent internal retry counter when external supervisor retries train."""
    internal_retry = dict(state.get("clean_internal_retry") or {})
    internal_retry["train"] = 0
    return {"clean_internal_retry": internal_retry}


def _reset_clean_test_internal_retry(state: GlobalState) -> GlobalState:
    """Reset data_clean_agent internal retry counter when external supervisor retries test."""
    internal_retry = dict(state.get("clean_internal_retry") or {})
    internal_retry["test"] = 0
    return {"clean_internal_retry": internal_retry}


def _prepare_feature_retry(state: GlobalState) -> GlobalState:
    """Remove the failed models from feature_dataset_ids so feature_agent re-processes them."""
    from agent.supervisor import _failed_feature_models
    failed = _failed_feature_models(state)
    feature_ids = dict(state.get("feature_dataset_ids") or {})
    test_ids = dict(state.get("testing_feature_dataset_ids") or {})
    retries = dict(state.get("supervisor_retry") or {})
    
    for model, _ in failed:
        if retries.get(f"feature:{model}", 0) <= _MAX_SUPERVISOR_RETRY:
            if model in feature_ids:
                del feature_ids[model]
            if model in test_ids:
                del test_ids[model]
                
    return {
        "feature_dataset_ids": feature_ids,
        "testing_feature_dataset_ids": test_ids,
    }


def _skip_current_feature_model(state: GlobalState) -> GlobalState:
    """Mark failed feature models as skipped so the pipeline can continue."""
    from agent.supervisor import _failed_feature_models
    failed = _failed_feature_models(state)
    feature_ids = dict(state.get("feature_dataset_ids") or {})
    hints = dict(state.get("supervisor_hints") or {})
    
    for model, _ in failed:
        feature_ids[model] = "skipped"
        hints[f"feature:{model}"] = ""
        log_node("supervisor_feature", success=False, detail=f"model={model} skipped after max retries")
        
    return {
        "feature_dataset_ids": feature_ids,
        "supervisor_hints": hints,
    }


_MAX_SUPERVISOR_RETRY = 2


def _route_supervisor_split(state: GlobalState) -> str:
    hint = (state.get("supervisor_hints") or {}).get("split", "")
    retry = (state.get("supervisor_retry") or {}).get("split", 0)
    if not hint:
        return "agent_manager"
    if retry <= _MAX_SUPERVISOR_RETRY:
        return "split_agent"
    return END


def _route_supervisor_clean_train(state: GlobalState) -> str:
    hint = (state.get("supervisor_hints") or {}).get("clean_train", "")
    retry = (state.get("supervisor_retry") or {}).get("clean_train", 0)
    if not hint:
        return "set_clean_test"
    if retry <= _MAX_SUPERVISOR_RETRY:
        return "reset_clean_train_retry"
    return END


def _route_supervisor_clean_test(state: GlobalState) -> str:
    hint = (state.get("supervisor_hints") or {}).get("clean_test", "")
    retry = (state.get("supervisor_retry") or {}).get("clean_test", 0)
    if not hint:
        return "eda_agent"
    if retry <= _MAX_SUPERVISOR_RETRY:
        return "reset_clean_test_retry"
    return END


def _route_supervisor_eda(state: GlobalState) -> str:
    hint = (state.get("supervisor_hints") or {}).get("eda", "")
    retry = (state.get("supervisor_retry") or {}).get("eda", 0)
    if not hint:
        return "feature_agent"
    if retry <= _MAX_SUPERVISOR_RETRY:
        return "eda_agent"
    return END


def _route_supervisor_feature(state: GlobalState) -> str:
    from agent.feature_agent import _get_current_model
    if _get_current_model(state) is not None:
        return "feature_agent"
        
    from agent.supervisor import _failed_feature_models
    failed = _failed_feature_models(state)
    if not failed:
        return "model_agent"
        
    retries = dict(state.get("supervisor_retry") or {})
    can_retry = False
    for model, _ in failed:
        if retries.get(f"feature:{model}", 0) <= _MAX_SUPERVISOR_RETRY:
            can_retry = True
            break
            
    if can_retry:
        return "prepare_feature_retry"
    return "skip_feature_model"


def build_full_pipeline(show_graph: bool = False):
    split_graph = build_split_agent(show_graph=False)
    manager_graph = build_agent_manager(show_graph=False)
    clean_train_graph = build_data_clean_agent(show_graph=False)
    clean_test_graph = build_data_clean_agent(show_graph=False)
    eda_graph = build_eda_agent(show_graph=False)
    feature_graph = build_feature_agent(show_graph=False)
    model_graph = build_model_agent(show_graph=False)
    business_translator_graph = build_business_translator_agent(show_graph=False)

    graph = StateGraph(GlobalState)
    graph.add_node("init_run", _init_run_node)
    graph.add_node("split_agent", split_graph)
    graph.add_node("supervisor_split", make_supervisor("split"))
    graph.add_node("agent_manager", manager_graph)
    graph.add_node("set_clean_train", _set_clean_train)
    graph.add_node("data_clean_train", clean_train_graph)
    graph.add_node("supervisor_clean_train", make_supervisor("clean_train"))
    graph.add_node("reset_clean_train_retry", _reset_clean_train_internal_retry)
    graph.add_node("set_clean_test", _set_clean_test)
    graph.add_node("data_clean_test", clean_test_graph)
    graph.add_node("supervisor_clean_test", make_supervisor("clean_test"))
    graph.add_node("reset_clean_test_retry", _reset_clean_test_internal_retry)
    graph.add_node("eda_agent", eda_graph)
    graph.add_node("supervisor_eda", make_supervisor("eda"))
    graph.add_node("feature_agent", feature_graph)
    graph.add_node("supervisor_feature", make_supervisor("feature"))
    graph.add_node("prepare_feature_retry", _prepare_feature_retry)
    graph.add_node("skip_feature_model", _skip_current_feature_model)
    graph.add_node("model_agent", model_graph)
    graph.add_node("business_translator", business_translator_graph)

    graph.set_entry_point("init_run")
    graph.add_conditional_edges(
        "init_run",
        _route_after_init,
        {"split_agent": "split_agent", "agent_manager": "agent_manager"},
    )
    # split → supervisor_split → (pass: agent_manager / retry: split_agent / exhaust: END)
    graph.add_edge("split_agent", "supervisor_split")
    graph.add_conditional_edges(
        "supervisor_split",
        _route_supervisor_split,
        {"agent_manager": "agent_manager", "split_agent": "split_agent", END: END},
    )

    graph.add_edge("agent_manager", "set_clean_train")
    graph.add_edge("set_clean_train", "data_clean_train")
    # data_clean_train → supervisor_clean_train → (pass: set_clean_test / retry: reset+data_clean_train / exhaust: END)
    graph.add_edge("data_clean_train", "supervisor_clean_train")
    graph.add_conditional_edges(
        "supervisor_clean_train",
        _route_supervisor_clean_train,
        {
            "set_clean_test": "set_clean_test",
            "reset_clean_train_retry": "reset_clean_train_retry",
            END: END,
        },
    )
    graph.add_edge("reset_clean_train_retry", "data_clean_train")

    graph.add_edge("set_clean_test", "data_clean_test")
    # data_clean_test → supervisor_clean_test → (pass: eda_agent / retry: reset+data_clean_test / exhaust: END)
    graph.add_edge("data_clean_test", "supervisor_clean_test")
    graph.add_conditional_edges(
        "supervisor_clean_test",
        _route_supervisor_clean_test,
        {
            "eda_agent": "eda_agent",
            "reset_clean_test_retry": "reset_clean_test_retry",
            END: END,
        },
    )
    graph.add_edge("reset_clean_test_retry", "data_clean_test")

    # eda_agent → supervisor_eda → (pass: feature_agent / retry: eda_agent / exhaust: END)
    graph.add_edge("eda_agent", "supervisor_eda")
    graph.add_conditional_edges(
        "supervisor_eda",
        _route_supervisor_eda,
        {"feature_agent": "feature_agent", "eda_agent": "eda_agent", END: END},
    )

    # feature_agent → supervisor_feature → (pass+more: feature_agent / pass+done: model_agent /
    #                                        retry: feature_agent / exhaust: skip_feature_model)
    graph.add_edge("feature_agent", "supervisor_feature")
    graph.add_conditional_edges(
        "supervisor_feature",
        _route_supervisor_feature,
        {
            "feature_agent": "feature_agent",
            "model_agent": "model_agent",
            "prepare_feature_retry": "prepare_feature_retry",
            "skip_feature_model": "skip_feature_model",
        },
    )
    # prepare_feature_retry removes the failed model entry, then feature_agent re-processes it
    graph.add_edge("prepare_feature_retry", "feature_agent")
    # After skipping, check again if more models remain
    graph.add_conditional_edges(
        "skip_feature_model",
        lambda s: "feature_agent" if _get_current_model(s) is not None else "model_agent",
        {"feature_agent": "feature_agent", "model_agent": "model_agent"},
    )

    graph.add_conditional_edges(
        "model_agent",
        _route_after_model,
        {"business_translator": "business_translator", END: END},
    )
    graph.add_edge("business_translator", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


full_pipeline = build_full_pipeline()


if __name__ == "__main__":
    pipeline = build_full_pipeline(show_graph=True)
    user_input = input("Enter input: ")
    events = pipeline.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            # "data_path": str(PROJECT_ROOT / "datasets" / "Financials.csv"),
            "data_path": str(PROJECT_ROOT / "datasets" / "bank-full.csv"),
            # "description": "This is a financial industry sales dataset containing fields for sales, discounts, profit, country, product, and time.",
            "description": "The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. There are four datasets: 1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014] 2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs. 3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs). 4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs). The smallest datasets are provided to test more computationally demanding machine learning algorithms(e.g., SVM). The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).",
            # "cleaned_data_path": str(PROJECT_ROOT / "artifacts" / "Financials_cleaned.csv"),
           # "cleaned_data_path": str(PROJECT_ROOT / "artifacts" / "bank-full_cleaned.csv"),
        },
        stream_mode="values",
        # EDA/Feature/Model involve multi-round tool calls + BAT subgraph; default limit of 120 easily triggers GraphRecursionError
        config={"recursion_limit": 280},
    )
    import time
    from main import _print_timing_summary

    pipeline_end = time.time()
    final_state: dict = {}
    _split_logged = False
    _last_stage: str = ""
    for event in events:
        final_state = event
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
        if (stage := event.get("current_stage")) and stage != _last_stage:
            _last_stage = stage
            print(f"\n[stage] {stage}")
        if not _split_logged and event.get("training_data_path"):
            _split_logged = True
            print(f"[split] train: {event['training_data_path']}")
            print(f"[split] test:  {event['testing_data_path']}")
        pipeline_end = time.time()

    _print_timing_summary(final_state.get("timing") or {}, pipeline_end)
