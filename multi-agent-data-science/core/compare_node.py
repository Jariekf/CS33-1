"""Compare Node：纯函数节点，从 model_results 中选出最优模型写入 best_model。"""

from __future__ import annotations

from core.global_state import GlobalState
from core.report_writer import log_node, save_state_snapshot


def compare_models_node(state: GlobalState) -> GlobalState:
    """LangGraph 节点：比较所有候选模型的评估指标，选出最优模型。

    分类任务以 f1_weighted 为主指标（越高越好）。
    回归任务以 r2 为主指标（越高越好）；r2 相同时再比较 rmse（越低越好）。
    """
    model_results = state.get("model_results") or {}
    task_type = (state.get("task_type") or "regression").lower()

    if not model_results:
        log_node("compare", success=False, detail="model_results is empty")
        return {
            "current_stage": "compare_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "compare", "type": "NoResults", "message": "model_results 为空，无法比较。"}],
        }

    best_name: str | None = None
    best_score: float = float("-inf")

    for model_name, result in model_results.items():
        metrics = result.get("metrics") or {}
        if task_type == "classification":
            score = float(metrics.get("f1_weighted", -1))
        else:
            score = float(metrics.get("r2", -1))

        if score > best_score:
            best_score = score
            best_name = model_name

    if best_name is None:
        log_node("compare", success=False, detail="no valid scores in model_results")
        return {
            "current_stage": "compare_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "compare", "type": "NoValidScore", "message": "所有模型均无有效评估指标。"}],
        }

    best_result = model_results[best_name]
    metric_key = "f1_weighted" if task_type == "classification" else "r2"
    log_node(
        "compare",
        success=True,
        detail=f"best={best_name}  {metric_key}={best_score:.4f}",
    )
    update = {
        "best_model": {
            "model_name": best_name,
            "model_path": best_result.get("model_path", ""),
            "metrics": best_result.get("metrics", {}),
            "task_type": task_type,
        },
        "current_stage": "compare_done",
    }
    save_state_snapshot({**state, **update})
    return update
