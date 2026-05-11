"""Finalize Node：pipeline 收尾节点，校验 best_model 并持久化最终 state 快照。

model 阶段已由 tune_and_select_model（CV 调参选型）+ train_model（全量重训）+
evaluate_model（test 一次评估）完整执行，best_model 在 model_agent 退出时即已确定。
本节点不再重新比较模型，只负责：
  1. 校验 best_model 是否存在且含 final_test_metrics；
  2. 记录 pipeline 日志；
  3. 保存最终 state 快照供报告和 business_translator 使用。
"""

from __future__ import annotations

from core.global_state import GlobalState
from core.report_writer import log_node, save_state_snapshot


def compare_models_node(state: GlobalState) -> GlobalState:
    """Pipeline 收尾节点（原 compare node）。

    保留原函数名以兼容 graph.py 中的节点注册与路由，但内部逻辑已简化：
    不再迭代比较候选模型，直接取 model_agent 写入的 best_model 做最终校验。
    """
    best_model = state.get("best_model") or {}
    best_name = best_model.get("model_name", "")

    if not best_name:
        log_node("finalize", success=False, detail="best_model not set after model_agent")
        return {
            "current_stage": "compare_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "finalize", "type": "NoBestModel",
                "message": "model_agent 未写入 best_model，无法进入下一阶段。"}],
        }

    task_type = (state.get("task_type") or "regression").lower()
    final_metrics = best_model.get("final_test_metrics") or {}

    # 优先使用 EDA Agent 确定的 primary_metric，回退到任务类型默认值
    primary_metric = state.get("primary_metric") or (
        "f1_weighted" if task_type == "classification" else "r2"
    )
    evaluation_metrics = state.get("evaluation_metrics") or [primary_metric]
    score = final_metrics.get(primary_metric, "N/A")

    # 生成综合指标摘要字符串
    metrics_summary = "  ".join(
        f"{m}={final_metrics.get(m, 'N/A')}" for m in evaluation_metrics
    )

    log_node(
        "finalize",
        success=True,
        detail=(
            f"best={best_name}  primary={primary_metric}={score}  "
            f"eval=[{metrics_summary}]  (selected by CV in tune_and_select_model)"
        ),
    )

    update = {
        "best_model": {
            **best_model,
            "task_type": task_type,
        },
        "current_stage": "compare_done",
    }
    save_state_snapshot({**state, **update})
    return update
