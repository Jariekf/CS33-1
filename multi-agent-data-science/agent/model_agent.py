from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

import time

from core.global_state import GlobalState
from core.my_llm import llm_for
from core.report_writer import save_agent_report, save_state_snapshot
from core.timer_utils import make_timer_start
from tools.model_tools.evaluate_model import evaluate_model
from tools.model_tools.explain_model import explain_model
from tools.model_tools.train_model import train_model
from tools.model_tools.tune_and_select_model import tune_and_select_model

tools = [tune_and_select_model, train_model, evaluate_model, explain_model]
llm_with_tools = llm_for("model").bind_tools(tools)


_MODELING_TOOL_NAMES = {
    "tune_and_select_model",
    "train_model",
    "evaluate_model",
    "explain_model",
}


def _trim_messages_to_modeling_round(messages: list[Any]) -> list[Any]:
    """
    Filter historical messages to retain only the tool call/response chains for the current modeling subgraph.

    This prevents business-oriented text generated during the upstream user_request / agent_manager RAP wrap-up / EDA / feature stages
    from contaminating the model_agent LLM's context, which could trigger unauthorized writing
    to sections classified as "business recommendations."
    """
    if not messages:
        return []

    last_modeling_tool_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            if msg.name in _MODELING_TOOL_NAMES:
                last_modeling_tool_idx = i
            break

    if last_modeling_tool_idx is None:
        return []

    start = last_modeling_tool_idx
    for j in range(last_modeling_tool_idx - 1, -1, -1):
        msg = messages[j]
        if isinstance(msg, ToolMessage) and msg.name in _MODELING_TOOL_NAMES:
            start = j
            continue
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            names = {tc.get("name") for tc in msg.tool_calls}
            if names & _MODELING_TOOL_NAMES:
                start = j
                continue
        break

    return messages[start:]


def _get_current_model(state: GlobalState) -> str | None:
    candidate_models = state.get("candidate_models") or []
    model_results = state.get("model_results") or {}
    remaining = [m for m in candidate_models if m not in model_results]
    return remaining[0] if remaining else None


def _model_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    task_type = (state.get("task_type") or "unknown").lower()
    target_column = state.get("target_column") or ""
    feature_dataset_ids = state.get("feature_dataset_ids") or {}
    testing_feature_dataset_ids = state.get("testing_feature_dataset_ids") or {}
    candidate_models = state.get("candidate_models") or []
    model_results = state.get("model_results") or {}
    best_model = state.get("best_model") or {}
    current_stage = state.get("current_stage") or ""

    # EDA Agent 决定的指标
    primary_metric = state.get("primary_metric") or ""
    evaluation_metrics = state.get("evaluation_metrics") or []
    metrics_rationale = state.get("metrics_rationale") or ""

    chosen_model = best_model.get("model_name") or ""

    run_dir = state.get("run_dir") or ""
    output_dir = f"{run_dir}/models" if run_dir else ""

    system_content = (
        "You are the Model Training and Evaluation Agent. Please strictly follow the traditional data science workflow: \n"
        "First, perform hyperparameter tuning and model selection on the training set. After analyzing the results, "
        "retrain the best model using the full training set, and finally evaluate it using the test set just once.\n\n"
        "[Task Information]\n"
        f"Task type: {task_type}. Target column: {target_column or 'Not provided'}.\n"
        f"Candidate models: {candidate_models}.\n"
        f"Current stage: {current_stage or 'Not started'}. Existing model_results key:{list(model_results.keys())}。\n"
        f"Working directory for this run: {run_dir or 'Not set'}}。\n"
        f"Paths to the preprocessed training sets for each candidate model:{json.dumps(feature_dataset_ids, ensure_ascii=False)}。\n"
        f"Paths to the test sets for each candidate model after preprocessing:{json.dumps(testing_feature_dataset_ids, ensure_ascii=False)}。\n"
        f"Selected best model:{chosen_model or 'Not selected yet'}。\n\n"
        "[EDA Agent: Predefined Evaluation Metrics (Use as is; no need to re-evaluate)]\n"
        f"primary_metric (CV tuning metric):{primary_metric or 'Not set, please use the task default'}。\n"
        f"evaluation_metrics (List of Comprehensive Evaluation Indicators):{evaluation_metrics or 'Not set, please use the default task metrics'}.\n"
        f"Reasons for choosing metrics: {metrics_rationale or 'Not provided'}。\n\n"
        "[Follow these steps in order; do not skip any steps or repeat any steps.]\n\n"
        "Step 1 — If current_stage is not model_selected / model_trained / model_evaluated，\n"
        "When calling `tune_and_select_model`, you must define the following HPO strategies yourself:\n"
        f"· scoring [Required] = '{primary_metric or ''}' — This is the primary_metric already determined by the EDA Agent,"
        "Pass directly; do not substitute with other metrics, and do not leave blank.\n"
        "  · cv_strategy: If features include time series → timeseries; otherwise → auto\n"
        "  · cv_folds: Number of rows in the training set < 1000 → 3; 1000–20,000 → 5; > 20,000 → 10\n"
        "  · search_strategy Selection Criteria:\n"
        "    - Total number of combinations in the parameter space ≤ 30 (e.g., Ridge, Lasso, LogisticRegression) → grid\n"
        "    - To maximize search quality, with an iter budget ≤ 20 → Bayesian (GP surrogate model, most sample-efficient)\n"
        "    - Large parameter space (number of combinations > 100) and ample iter budget → halving (elimination by rounds)\n"
        "    - Other equilibrium scenarios → random\n"
        "    - When combining multiple candidate models, the strategy is determined based on the model with the largest parameter space.\n"
        "  · max_iter_per_model: For models with ≤ 2 dimensions → 12; for models with 3–4 dimensions → 20; for models with ≥ 5 dimensions → 30\n"
        "    （Ignore this parameter when `search_strategy=grid`）\n"
        "  · extra_param_overrides (optional): Injects or overrides specific hyperparameters by model name; "
        "    all other parameters remain in the default search space.\n"
        "    Pass a value only when there is clear data evidence; otherwise, leave it as `None`. Common scenarios:\n"
        "    - Imbalanced classification (EDA report shows a class ratio > 5:1) →\n"
        '      {"LogisticRegression": {"class_weight": ["balanced", null]},\n'
        '       "RandomForest": {"class_weight": ["balanced", null]},\n'
        '       "SVM": {"class_weight": ["balanced", null]}}\n'
        "      （Models that support only `class_weight`: LogisticRegression, RandomForest, SVM, LightGBM\n"
        "    - Training set > 50,000 rows; increase the upper limit of `n_estimators` for the tree model →\n"
        '      {"RandomForest": {"n_estimators": [200, 300, 500]},\n'
        '       "XGBoost": {"n_estimators": [200, 300, 500]}}\n'
        "    - Number of feature dimensions > 100; RandomForest increases the max_features range →\n"
        '      {"RandomForest": {"max_features": ["sqrt", "log2", 0.3, 0.5]}}\n'
        "  · feature_dataset_ids、candidate_models、task_type、target_column Retrieved from the task information above\n\n"
        "  [Self-healing Strategy — If a tool fails, retry using the following priority order; do not abandon the entire modeling phase]\n\n"
        "  ① tune_and_select_model returns error (All candidate models failed)\n"
        "    → Retry 1: Keep the original parameters, but change `search_strategy=random` and `max_iter_per_model=8` to determine if the issue is related to HPO configuration.\n"
        "    → Retry 2: Halve the number of CV folds (minimum 3) to mitigate the issue of insufficient CV splits caused by a training set that is too small\n"
        "    → Retry 3: Downgrade scoring to the task defaults (classification → f1_weighted, regression → r2) to rule out scorer incompatibility\n"
        "    → Fallback option: Replace `candidate_models` with `baseline` (classification → LogisticRegression, regression → Ridge),\n"
        "              model_params={}, call train_model directly to skip hyperparameter tuning\n\n"
        "  ② tune_and_select_model returns skipped_models (some models failed)\n"
        "    → No need to retry; select `best_model_name` from the successful models and explain the reason for skipping this step in the analysis in Step 2.\n\n"
        "  ③ train_model returns an error\n"
        "    → Retry: model_params={} Retrain using default parameters (to rule out incompatibility issues with hyperparameter tuning results)\n"
        "    → Still failing: Replace `best_model_name` with the second-highest-scoring successful model from `cv_best_score` in `per_model_summary`.\n\n"
        "  ④ evaluate_model returns an error\n"
        "    → Check whether `target_encoder_path`, `scaler_path`, and `encoder_paths` match the values returned by `train_model`.\n"
        "    → When retrying, ensure that the parameters are aligned; if it still fails, log the error message, set `current_stage` to `model_evaluated`, and report the cause of the failure.\n\n"
        "  Each time you retry after a downgrade, you must include a single sentence before the `tool_call` to explain [which self-healing strategy was used and why.]\n"
        "  Attempt up to two retries; if exceeded, log the original error and continue the pipeline (without interrupting the entire system).\n\n"
        "Step 2 — After receiving the results from `tune_and_select_model`, complete the analysis and call `train_model` in the same reply.\n"
        "  (Do not first output the plain text analysis and then call the tool separately; instead, include the analysis as the input text preceding `tool_call` and return it along with the tool call):\n"
        "  Key points of the analysis (briefly):\n"
        "  · If `skipped_models` exists, specify the reason for skipping\n"
        "  · Compare the `cv_best_score` and `cv_score_std` of the successful model to determine whether the best model offers a significant advantage.\n"
        "  · If the difference between the highest and second-highest scores is less than 0.01, the basis for the selection must be explained.\n"
        "  · If a model's cv_score_std is greater than 0.03, it should be noted that the CV is unstable.\n"
        "  · Provide a conclusion on which model to use, and then immediately call `train_model`:\n"
        "  featured_train_path = feature_dataset_ids[best_model_name]，\n"
        "  model_name = best_model_name，model_params = best_model_params，\n"
        f"  output_dir = {output_dir!r}。\n\n"
        "Step 3 — If `current_stage` is `model_trained`, call `evaluate_model`:\n"
        "  featured_test_data_path = testing_feature_dataset_ids[best_model_name]，\n"
        "  model_path and feature_columns_path are returned by train_model, \n"
        "  model_name、task_type、target_column Based on the task information provided above,\n"
        f"  evaluation_metrics={evaluation_metrics}，primary_metric='{primary_metric}'。\n\n"
        "Step 3.5 — If `current_stage` is `model_evaluated`, call `explain_model` to perform SHAP analysis:\n"
        "  model_path comes from best_model.model_path,\n"
        "  featured_test_path = testing_feature_dataset_ids[best_model_name]，\n"
        "  target_column, model_name, and task_type are passed in from the task information above,\n"
        f" output_dir = '{run_dir + '/plots' if run_dir else ''}',\n"
        "  max_samples=500，top_n_features=20。\n"
        "  If `explain_model` returns an error, log the cause and proceed to step 4 (SHAP failure does not interrupt the pipeline).\n\n"
        "Step 4 — When `current_stage` is `model_explained`, the tool is no longer called,\n"
        "  Brief summary: Final model name, comparison of metrics between cv_best_score and final_test_metrics,\n"
        "  Assessment of generalization ability (whether the difference between the CV score and the test score is reasonable),\n"
        "  and the top 3 most important features in SHAP, along with their directions (if SHAP is successful).\n"
        "  The final text response is strictly limited to the analysis of the model results described above.\n"
        "[Prohibited Actions — Business recommendations are the responsibility of BAT; this agent provides technical conclusions only]:\n"
        "  - The following chapter titles (in both Chinese and English) are strictly prohibited:\n"
        "    · 中文：业务建议 / 业务影响 / 业务洞察 / 业务推荐 / 业务价值 / 业务意义 /\n"
        "      行动建议 / 推荐操作 / 策略建议 / 战略建议 / 下一步 / 下一步行动 / 下一步操作 / 后续行动\n"
        "    · English：Business Recommendation(s) / Business Insight(s) / Business Action(s) /\n"
        "      Business Implication(s) / Business Value / Recommendation(s) / Recommended Action(s) /\n"
        "      Next Step(s) / Action Item(s) / Strategic Implication(s) / Strategic Recommendation(s)\n"
        "  - Do not discuss business-related topics such as marketing strategies, ROI, customer screening, thresholds, and business trade-offs\n"
        "  - Content such as business recommendations, customer actions, and advertising strategies is handled by business_translator_agent in the final\n"
        "    All results are consolidated in `business_insight_report.md`; this agent reports only model metrics and generalization capabilities.\n"
        "  - If you output the above sections in violation of the rules, they will be automatically stripped by the `save_agent_report` filter before being written to disk."
    )

    trimmed_history = _trim_messages_to_modeling_round(messages)
    if trimmed_history:
        llm_input = [SystemMessage(content=system_content), *trimmed_history]
    else:
        llm_input = [
            SystemMessage(content=system_content),
            HumanMessage(
                content=(
                    "Please follow the system prompts to complete modeling steps 1 to 4."
                    "Do not repeat the upstream user_request / agent_manager / EDA / feature reports,"
                    "Do not generate business-related sections such as business suggestions, recommended actions, next steps, Recommendations, or Next Steps."
                )
            ),
        ]

    response = llm_with_tools.invoke(llm_input)
    return {"messages": [response]}


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage) or not last_msg.name:
        return {}

    handlers = {
        "tune_and_select_model": _handle_tune_and_select,
        "train_model": _handle_train,
        "evaluate_model": _handle_evaluate,
        "explain_model": _handle_explain,
    }
    handler = handlers.get(last_msg.name)
    return handler(state, last_msg) if handler else {}


def _parse_payload(tool_message: ToolMessage) -> dict[str, Any]:
    try:
        return json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        return {"error": {"type": "JSONDecodeError", "message": "The tool's output is not valid JSON"}}


def _handle_train(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "model_train_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "model_train", **payload["error"]}],
        }
    best_model = dict(state.get("best_model") or {})
    best_model.update(
        {
            "model_name": payload.get("model_name", best_model.get("model_name", "")),
            "model_path": payload.get("model_path", ""),
            "feature_columns_path": payload.get("feature_columns_path", ""),
            "model_params": payload.get("model_params", best_model.get("model_params", {})),
        }
    )
    existing = dict(state.get("model_results") or {})
    name = payload.get("model_name", "")
    if name:
        merged = dict(existing.get(name) or {})
        merged["model_path"] = payload.get("model_path", "")
        merged["feature_columns_path"] = payload.get("feature_columns_path", "")
        merged["model_params"] = payload.get("model_params", {})
        existing[name] = merged
    return {
        "best_model": best_model,
        "model_results": existing,
        "current_stage": "model_trained",
    }


def _handle_evaluate(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "model_eval_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "model_eval", **payload["error"]}],
        }
    existing = dict(state.get("model_results") or {})
    name = payload.get("model_name", "unknown")
    test_metrics = payload.get("metrics", {})
    confusion_matrix = payload.get("confusion_matrix")
    model_entry = dict(existing.get(name) or {})
    model_entry["model_path"] = payload.get("model_path", model_entry.get("model_path", ""))
    # overwrite the cv-score snapshot in "metrics" with definitive test results so that
    # business_translator_agent always reads the most authoritative score per model.
    # cv data is preserved separately in cv_best_score and selection_summary fields.
    model_entry["metrics"] = test_metrics
    model_entry["final_test_metrics"] = test_metrics
    if confusion_matrix is not None:
        model_entry["confusion_matrix"] = confusion_matrix
    existing[name] = model_entry
    best_model = dict(state.get("best_model") or {})
    if best_model.get("model_name") == name or not best_model.get("model_name"):
        best_model["model_name"] = name
        best_model["model_path"] = payload.get("model_path", "")
        best_model["metrics"] = test_metrics
        best_model["final_test_metrics"] = test_metrics
        if confusion_matrix is not None:
            best_model["confusion_matrix"] = confusion_matrix
        if payload.get("resource_metrics"):
            best_model["resource_metrics"] = payload["resource_metrics"]
        if payload.get("test_predictions_path"):
            best_model["test_predictions_path"] = payload["test_predictions_path"]
    return {
        "best_model": best_model,
        "model_results": existing,
        "current_stage": "model_evaluated",
    }


def _handle_explain(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    best_model = dict(state.get("best_model") or {})
    if "error" in payload:
        # SHAP failure does not interrupt the pipeline; errors are logged and processing continues
        best_model["shap_explanation"] = {"error": payload["error"]}
    else:
        best_model["shap_explanation"] = {
            "explainer_type": payload.get("explainer_type", ""),
            "samples_explained": payload.get("samples_explained"),
            "feature_importance": payload.get("feature_importance", []),
            "top_features": payload.get("top_features", []),
            "shap_summary_plot_path": payload.get("shap_summary_plot_path"),
        }
    return {
        "best_model": best_model,
        "current_stage": "model_explained",
    }


def _handle_tune_and_select(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "model_select_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "model_select", **payload["error"]}],
        }

    selection_metric = payload.get("selection_metric", "")
    per_model_summary = payload.get("per_model_summary") or {}
    existing = dict(state.get("model_results") or {})
    for model_name, summary in per_model_summary.items():
        cv_best = float(summary.get("cv_best_score", -1))
        merged = dict(existing.get(model_name) or {})
        merged["metrics"] = {selection_metric: cv_best}
        merged["selection_metric"] = selection_metric
        merged["cv_best_score"] = cv_best
        merged["model_params"] = summary.get("best_params", {})
        merged["selection_summary"] = summary
        existing[model_name] = merged

    best_name = payload.get("best_model_name", "")
    best_model = {
        "model_name": best_name,
        "model_params": payload.get("best_model_params", {}),
        "selection_metric": selection_metric,
        "selection_score": payload.get("best_cv_score"),
    }
    return {
        "best_model": best_model,
        "model_results": existing,
        "current_stage": "model_selected",
    }


def _route_after_model(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "finalize"


def _build_model_report(state: GlobalState, llm_summary: str) -> str:
    """Generate a complete model report in Markdown based on the state data, with the text from LLM Step 4 appended as supplementary information."""
    best_model = state.get("best_model") or {}
    model_results = state.get("model_results") or {}
    task_type = (state.get("task_type") or "unknown").lower()
    primary_metric = state.get("primary_metric") or ""
    evaluation_metrics = state.get("evaluation_metrics") or []

    model_name = best_model.get("model_name", "unknown")
    lines: list[str] = []

    # ── 1. CV Hyperparameter Tuning Results (All Candidate Models) ──────────────────────────────────────
    lines.append("### CV Hyperparameter Tuning Results (Training Set, Comparison of Candidate Models)\n")
    has_cv = False
    for name, entry in model_results.items():
        sel = entry.get("selection_summary") or {}
        status = sel.get("status", "success")
        cv_score = entry.get("cv_best_score")
        cv_std = sel.get("cv_score_std")
        best_params = entry.get("model_params") or {}
        search_strategy = sel.get("search_strategy", "")
        rows_trained = sel.get("rows_trained", "")

        if status == "failed":
            lines.append(f"**{name}** — ❌ Skip（{sel.get('error', 'Unknown error')}）\n")
        else:
            is_best = "✅ **Final choice**" if name == model_name else ""
            lines.append(f"**{name}** {is_best}")
            lines.append(f"- CV {primary_metric or 'Score'}：**{cv_score}**（std={cv_std}）")
            lines.append(f"- Search Strategy: {search_strategy}　Number of training samples: {rows_trained}")
            if best_params:
                params_str = "、".join(f"`{k}={v}`" for k, v in best_params.items())
                lines.append(f"- Optimal parameters: {params_str}")
            lines.append("")
            has_cv = True

    if not has_cv:
        lines.append("_(CV tuning results not recorded)_\n")

    # ── 2. Evaluation on the Final Test Set ────────────────────────────────────────────
    lines.append("### Final validation set evaluation\n")
    final_metrics = best_model.get("final_test_metrics") or {}
    if final_metrics:
        # Display the items specified by `evaluation_metrics` first, then display the remaining items
        ordered = list(evaluation_metrics) + [
            k for k in final_metrics if k not in evaluation_metrics and not k.endswith("_skipped_reason")
        ]
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for m in ordered:
            if m in final_metrics and not m.endswith("_skipped_reason"):
                val = final_metrics[m]
                star = " ⭐" if m == primary_metric else ""
                lines.append(f"| {m}{star} | {val} |")
        lines.append("")
    else:
        lines.append("_(Final evaluation data not recorded)_\n")

    # Confusion Matrix
    cm = best_model.get("confusion_matrix")
    if cm and task_type == "classification":
        lines.append("**Confusion matrix:**")
        lines.append("```")
        for row in cm:
            lines.append("  " + "  ".join(str(v) for v in row))
        lines.append("```\n")

    # ── 3. SHAP Feature Importance ──────────────────────────────────────────────────
    shap = best_model.get("shap_explanation") or {}
    if shap and "error" not in shap:
        lines.append("### SHAP Feature Importance (Global)\n")
        lines.append(f"Explainer：{shap.get('explainer_type', '')}　"
                     f"Sample size：{shap.get('samples_explained', '')}\n")
        fi = shap.get("feature_importance") or []
        if fi:
            lines.append("| Rank | Feature | mean\\|SHAP\\| | Direction |")
            lines.append("|---|---|---|---|")
            for item in fi[:10]:
                dir_icon = {"positive": "↑ positive", "negative": "↓ negative", "mixed": "↔ mixed"}.get(
                    item.get("direction", ""), item.get("direction", "")
                )
                lines.append(
                    f"| {item['rank']} | {item['feature']} "
                    f"| {item['mean_abs_shap']} | {dir_icon} |"
                )
            lines.append("")
        plot_path = shap.get("shap_summary_plot_path")
        if plot_path:
            lines.append(f"SHAP Chart：`{plot_path}`\n")
    elif shap.get("error"):
        lines.append(f"_SHAP parsing failed: {shap['error']}(Does not affect model metrics)_\n")

    # ── 4. Resource-aware metrics ─────────────────────────────────────────────────────
    resource = best_model.get("resource_metrics") or {}
    if resource:
        latency = resource.get("inference_latency_ms_per_sample")
        size_kb = resource.get("model_size_kb")
        parts = []
        if latency is not None:
            parts.append(f"Inference latency {latency} ms/sample")
        if size_kb is not None:
            parts.append(f"Model size {size_kb} KB")
        if parts:
            lines.append("### Resource Metrics\n")
            lines.append("　".join(parts) + "\n")

    # ── 5. Predict file paths ─────────────────────────────────────────────────────
    pred_path = best_model.get("test_predictions_path")
    if pred_path:
        lines.append(f"Predicted results: `{pred_path}`\n")

    # ── 6. Supplementary Analysis for LLM Step 4 ─────────────────────────────────────────────
    if llm_summary.strip():
        lines.append("### Summary of the Agent Model Analysis\n")
        lines.append(llm_summary.strip())

    return "\n".join(lines)


def _model_finalize_node(state: GlobalState) -> GlobalState:
    """Pipeline cleanup: Save `model_report`, complete `best_model.task_type`, and save a state snapshot."""
    last_msg = state["messages"][-1]
    llm_summary = str(last_msg.content) if hasattr(last_msg, "content") else ""

    best_model = dict(state.get("best_model") or {})
    task_type = (state.get("task_type") or "regression").lower()
    best_model["task_type"] = task_type

    model_name = best_model.get("model_name", "unknown")
    run_dir = state.get("run_dir") or ""

    # Generate a complete report from the state data; LLM Step 4 is included below as a supplement
    full_report = _build_model_report({**state, "best_model": best_model}, llm_summary)

    class _FakeMsg:
        """Allow `save_agent_report` to accept plain text."""
        content = full_report

    save_agent_report("model_agent", _FakeMsg(), model=model_name, run_dir=run_dir)
    model_report_text = full_report

    timing = dict(state.get("timing") or {})
    entry = dict(timing.get("model_agent") or {})
    now = time.time()
    entry["end"] = now
    elapsed = now - entry.get("start", now)
    print(f"[timer] {'model_agent':<28} {elapsed:6.1f}s")
    timing["model_agent"] = entry

    updated: GlobalState = {
        "model_report": full_report,
        "best_model": best_model,
        "current_stage": "model_done",
        "timing": timing,
    }
    save_state_snapshot({**state, **updated})
    return updated


def _show_graph_if_needed(agent_graph, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts" / "graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        mmd_path = output_dir / "model_agent_graph.mmd"
        mmd_path.write_text(agent_graph.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"The Mermaid source code has been saved:{mmd_path}")
        image_path = output_dir / "model_agent_graph.png"
        image_bytes = agent_graph.get_graph().draw_mermaid_png()
        image_path.write_bytes(image_bytes)
        print(f"Graph image saved: {image_path}")
        try:
            from IPython.display import Image, display
            display(Image(filename=str(image_path)))
        except Exception:
            pass
    except Exception as exc:
        print(f"Image rendering failed: {exc}")


def _pretty_print_message(message) -> None:
    try:
        message.pretty_print()
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        text = message.pretty_repr()
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        print(safe_text)


def build_model_agent(show_graph: bool = False):
    graph = StateGraph(GlobalState)
    graph.add_node("timer_start", make_timer_start("model_agent"))
    graph.add_node("model_agent", _model_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.add_node("finalize", _model_finalize_node)
    graph.set_entry_point("timer_start")
    graph.add_edge("timer_start", "model_agent")
    graph.add_conditional_edges(
        "model_agent", _route_after_model, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "model_agent")
    graph.add_edge("finalize", END)
    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


model_agent = build_model_agent()
