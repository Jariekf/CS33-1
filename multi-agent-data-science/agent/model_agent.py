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
    """裁剪历史 messages，只保留本轮 modeling 子图的工具调用/响应链。

    避免上游 user_request / agent_manager RAP wrap-up / EDA / feature 阶段
    输出的业务化文本污染 model_agent LLM 的上下文，进而诱发越权写入
    "业务建议" 类章节。
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
        "你是模型训练与评估 Agent。请严格遵循传统数据科学工作流：\n"
        "先在训练集上做 CV 调参与模型选择，分析结果后再用全量训练集重训最佳模型，最后仅一次使用测试集做最终评估。\n\n"
        "【任务信息】\n"
        f"任务类型：{task_type}。目标列：{target_column or '未提供'}。\n"
        f"候选模型：{candidate_models}。\n"
        f"当前阶段：{current_stage or '未开始'}。已有 model_results 键：{list(model_results.keys())}。\n"
        f"本次运行工作目录：{run_dir or '未设置'}。\n"
        f"各候选模型预处理后训练集路径：{json.dumps(feature_dataset_ids, ensure_ascii=False)}。\n"
        f"各候选模型预处理后测试集路径：{json.dumps(testing_feature_dataset_ids, ensure_ascii=False)}。\n"
        f"已选最佳模型：{chosen_model or '尚未选择'}。\n\n"
        "【EDA Agent 已确定的评估指标（直接使用，无需重新判断）】\n"
        f"primary_metric（CV 调优指标）：{primary_metric or '未设置，请使用任务默认值'}。\n"
        f"evaluation_metrics（综合评估指标列表）：{evaluation_metrics or '未设置，请使用任务默认指标'}。\n"
        f"metrics 选择理由：{metrics_rationale or '未提供'}。\n\n"
        "【执行步骤，按顺序完成，不跳步，不重复】\n\n"
        "步骤 1 — 若 current_stage 不是 model_selected / model_trained / model_evaluated，\n"
        "调用 tune_and_select_model，需自主制定以下 HPO 策略：\n"
        f"  · scoring【强制】= '{primary_metric or ''}' —— 这是 EDA Agent 已经决定的 primary_metric，"
        "直接传入，不得替换为其他指标，不得留空。\n"
        "  · cv_strategy：特征含时序列 → timeseries；否则 → auto\n"
        "  · cv_folds：训练集行数 < 1000 → 3；1000~20000 → 5；> 20000 → 10\n"
        "  · search_strategy 选择原则：\n"
        "    - 参数空间总组合数 ≤ 30（如 Ridge/Lasso/LogisticRegression）→ grid\n"
        "    - 希望最大化搜索质量、iter 预算 ≤ 20 → bayesian（GP 代理模型，最样本高效）\n"
        "    - 参数空间大（组合数 > 100）且 iter 预算充裕 → halving（逐轮淘汰）\n"
        "    - 其余均衡场景 → random\n"
        "    - 多个候选模型混合类型时，以参数空间最大的模型为准决定策略\n"
        "  · max_iter_per_model：参数维度 ≤ 2 → 12；维度 3~4 → 20；维度 ≥ 5 → 30\n"
        "    （search_strategy=grid 时忽略此参数）\n"
        "  · extra_param_overrides（可选）：按模型名注入/覆盖特定超参，其余参数保持默认搜索空间。\n"
        "    只在有明确数据证据时才传，不确定则留 None。常见场景：\n"
        "    - 不平衡分类（EDA 报告显示类别比例 > 5:1）→\n"
        '      {"LogisticRegression": {"class_weight": ["balanced", null]},\n'
        '       "RandomForest": {"class_weight": ["balanced", null]},\n'
        '       "SVM": {"class_weight": ["balanced", null]}}\n'
        "      （仅支持 class_weight 的模型：LogisticRegression/RandomForest/SVM/LightGBM）\n"
        "    - 训练集 > 50000 行，扩充树模型 n_estimators 上限 →\n"
        '      {"RandomForest": {"n_estimators": [200, 300, 500]},\n'
        '       "XGBoost": {"n_estimators": [200, 300, 500]}}\n'
        "    - 特征维度 > 100，RandomForest 加大 max_features 范围 →\n"
        '      {"RandomForest": {"max_features": ["sqrt", "log2", 0.3, 0.5]}}\n'
        "  · feature_dataset_ids、candidate_models、task_type、target_column 从上方任务信息传入\n\n"
        "【自愈策略 — 遇到工具失败时，按以下优先级降级重试，不得放弃整个建模阶段】\n\n"
        "  ① tune_and_select_model 返回 error（全部候选模型均失败）\n"
        "    → 重试 1：保持原参数，改 search_strategy=random、max_iter_per_model=8，排查是否为 HPO 配置问题\n"
        "    → 重试 2：cv_folds 减半（最低 3），缓解训练集过小导致的 CV split 不足\n"
        "    → 重试 3：scoring 降级为任务默认值（分类→f1_weighted，回归→r2），排除 scorer 不兼容\n"
        "    → 最终兜底：candidate_models 替换为 baseline（分类→LogisticRegression，回归→Ridge），\n"
        "               model_params={}，直接调用 train_model 跳过调参\n\n"
        "  ② tune_and_select_model 返回 skipped_models（部分模型失败）\n"
        "    → 无需重试，从成功模型中选 best_model_name，在步骤 2 分析中说明跳过原因\n\n"
        "  ③ train_model 返回 error\n"
        "    → 重试：model_params={} 以默认参数重训（排除调参结果不兼容问题）\n"
        "    → 仍失败：换 best_model_name 为 per_model_summary 中 cv_best_score 次高的成功模型\n\n"
        "  ④ evaluate_model 返回 error\n"
        "    → 检查 target_encoder_path / scaler_path / encoder_paths 是否与 train_model 返回值一致\n"
        "    → 重试时确保参数对齐；若仍失败，记录错误信息，current_stage 置为 model_evaluated 并报告失败原因\n\n"
        "  每次降级重试时，须在 tool_call 前用一句话说明「采用了哪条自愈策略及原因」。\n"
        "  最多重试 2 次；超过后记录 error 原文，继续 pipeline（不中断整个系统）。\n\n"
        "步骤 2 — 收到 tune_and_select_model 结果后，在【同一次回复】中完成分析并调用 train_model\n"
        "  （不要先输出纯文字分析再单独调用工具，分析作为 tool_call 前的思考文本随工具调用一起返回）：\n"
        "  分析要点（简短即可）：\n"
        "  · 若存在 skipped_models，说明跳过原因\n"
        "  · 对比成功模型的 cv_best_score 和 cv_score_std，判断最佳模型优势是否显著\n"
        "  · 若最优与次优分数差 < 0.01，需说明选择依据\n"
        "  · 若某模型 cv_score_std > 0.03，需指出 CV 不稳定\n"
        "  · 给出最终采用哪个模型的结论，然后立即调用 train_model：\n"
        "  featured_train_path = feature_dataset_ids[best_model_name]，\n"
        "  model_name = best_model_name，model_params = best_model_params，\n"
        f"  output_dir = {output_dir!r}。\n\n"
        "步骤 3 — 若 current_stage 为 model_trained，调用 evaluate_model：\n"
        "  featured_test_data_path = testing_feature_dataset_ids[best_model_name]，\n"
        "  model_path、feature_columns_path 来自 train_model 返回值，\n"
        "  model_name、task_type、target_column 从上方任务信息传入，\n"
        f"  evaluation_metrics={evaluation_metrics}，primary_metric='{primary_metric}'。\n\n"
        "步骤 3.5 — 若 current_stage 为 model_evaluated，调用 explain_model 进行 SHAP 分析：\n"
        "  model_path 来自 best_model.model_path，\n"
        "  featured_test_path = testing_feature_dataset_ids[best_model_name]，\n"
        "  target_column、model_name、task_type 从上方任务信息传入，\n"
        f"  output_dir = '{run_dir + '/plots' if run_dir else ''}',\n"
        "  max_samples=500，top_n_features=20。\n"
        "  若 explain_model 返回 error，记录原因后继续步骤 4（SHAP 失败不中断 pipeline）。\n\n"
        "步骤 4 — 当 current_stage 为 model_explained 时，不再调用工具，\n"
        "  输出简短总结：最终模型名称、cv_best_score vs final_test_metrics 各指标对比、\n"
        "  泛化能力判断（CV 分与 test 分差距是否合理）、\n"
        "  以及 SHAP top-3 最重要特征及其方向（若 SHAP 成功）。\n"
        "  最终文字回复严格限于上述模型结果分析。\n"
        "【严禁事项 — 业务建议归 BAT，本 agent 仅做技术结论】：\n"
        "  - 严禁出现以下任何章节标题（中英文）：\n"
        "    · 中文：业务建议 / 业务影响 / 业务洞察 / 业务推荐 / 业务价值 / 业务意义 /\n"
        "      行动建议 / 推荐操作 / 策略建议 / 战略建议 / 下一步 / 下一步行动 / 下一步操作 / 后续行动\n"
        "    · 英文：Business Recommendation(s) / Business Insight(s) / Business Action(s) /\n"
        "      Business Implication(s) / Business Value / Recommendation(s) / Recommended Action(s) /\n"
        "      Next Step(s) / Action Item(s) / Strategic Implication(s) / Strategic Recommendation(s)\n"
        "  - 不得讨论营销策略、ROI、客户筛选、阈值与业务取舍等业务层面话题；\n"
        "  - 业务建议、客户行动、投放策略等内容由 business_translator_agent 在最终\n"
        "    business_insight_report.md 中统一产出；本 agent 仅汇报模型指标与泛化能力。\n"
        "  - 若你违规输出上述章节，落盘前会被 save_agent_report 的过滤器自动剥离。"
    )

    trimmed_history = _trim_messages_to_modeling_round(messages)
    if trimmed_history:
        llm_input = [SystemMessage(content=system_content), *trimmed_history]
    else:
        llm_input = [
            SystemMessage(content=system_content),
            HumanMessage(
                content=(
                    "请按系统提示词执行 modeling 步骤 1~4。"
                    "禁止复读上游 user_request / agent_manager / EDA / feature 报告，"
                    "禁止生成业务建议、推荐操作、下一步、Recommendations、Next Steps 等业务类章节。"
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
        return {"error": {"type": "JSONDecodeError", "message": "工具输出不是合法 JSON"}}


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
        # SHAP 失败不中断 pipeline，记录错误继续
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
    """从 state 数据构建完整的模型报告 Markdown，LLM 步骤 4 文字作为补充追加。"""
    best_model = state.get("best_model") or {}
    model_results = state.get("model_results") or {}
    task_type = (state.get("task_type") or "unknown").lower()
    primary_metric = state.get("primary_metric") or ""
    evaluation_metrics = state.get("evaluation_metrics") or []

    model_name = best_model.get("model_name", "unknown")
    lines: list[str] = []

    # ── 1. CV 调参结果（所有候选模型）────────────────────────────────────────
    lines.append("### CV 调参结果（训练集，候选模型对比）\n")
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
            lines.append(f"**{name}** — ❌ 跳过（{sel.get('error', '未知错误')}）\n")
        else:
            is_best = "✅ **最终选择**" if name == model_name else ""
            lines.append(f"**{name}** {is_best}")
            lines.append(f"- CV {primary_metric or '得分'}：**{cv_score}**（std={cv_std}）")
            lines.append(f"- 搜索策略：{search_strategy}　训练样本数：{rows_trained}")
            if best_params:
                params_str = "、".join(f"`{k}={v}`" for k, v in best_params.items())
                lines.append(f"- 最优参数：{params_str}")
            lines.append("")
            has_cv = True

    if not has_cv:
        lines.append("_（CV 调参结果未记录）_\n")

    # ── 2. 最终测试集评估 ───────────────────────────────────────────────────
    lines.append("### 最终测试集评估\n")
    final_metrics = best_model.get("final_test_metrics") or {}
    if final_metrics:
        # 优先展示 evaluation_metrics 指定的，再展示剩余的
        ordered = list(evaluation_metrics) + [
            k for k in final_metrics if k not in evaluation_metrics and not k.endswith("_skipped_reason")
        ]
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        for m in ordered:
            if m in final_metrics and not m.endswith("_skipped_reason"):
                val = final_metrics[m]
                star = " ⭐" if m == primary_metric else ""
                lines.append(f"| {m}{star} | {val} |")
        lines.append("")
    else:
        lines.append("_（最终评估数据未记录）_\n")

    # 混淆矩阵
    cm = best_model.get("confusion_matrix")
    if cm and task_type == "classification":
        lines.append("**混淆矩阵：**")
        lines.append("```")
        for row in cm:
            lines.append("  " + "  ".join(str(v) for v in row))
        lines.append("```\n")

    # ── 3. SHAP 特征重要性 ──────────────────────────────────────────────────
    shap = best_model.get("shap_explanation") or {}
    if shap and "error" not in shap:
        lines.append("### SHAP 特征重要性（全局）\n")
        lines.append(f"Explainer：{shap.get('explainer_type', '')}　"
                     f"解释样本数：{shap.get('samples_explained', '')}\n")
        fi = shap.get("feature_importance") or []
        if fi:
            lines.append("| 排名 | 特征 | mean\\|SHAP\\| | 方向 |")
            lines.append("|---|---|---|---|")
            for item in fi[:10]:
                dir_icon = {"positive": "↑ 正向", "negative": "↓ 负向", "mixed": "↔ 混合"}.get(
                    item.get("direction", ""), item.get("direction", "")
                )
                lines.append(
                    f"| {item['rank']} | {item['feature']} "
                    f"| {item['mean_abs_shap']} | {dir_icon} |"
                )
            lines.append("")
        plot_path = shap.get("shap_summary_plot_path")
        if plot_path:
            lines.append(f"SHAP 图表：`{plot_path}`\n")
    elif shap.get("error"):
        lines.append(f"_SHAP 解析失败：{shap['error']}（不影响模型指标）_\n")

    # ── 4. 资源感知指标 ─────────────────────────────────────────────────────
    resource = best_model.get("resource_metrics") or {}
    if resource:
        latency = resource.get("inference_latency_ms_per_sample")
        size_kb = resource.get("model_size_kb")
        parts = []
        if latency is not None:
            parts.append(f"推理延迟 {latency}ms/样本")
        if size_kb is not None:
            parts.append(f"模型大小 {size_kb}KB")
        if parts:
            lines.append("### 资源指标\n")
            lines.append("　".join(parts) + "\n")

    # ── 5. 预测文件路径 ─────────────────────────────────────────────────────
    pred_path = best_model.get("test_predictions_path")
    if pred_path:
        lines.append(f"预测值落盘：`{pred_path}`\n")

    # ── 6. LLM 步骤 4 补充分析 ─────────────────────────────────────────────
    if llm_summary.strip():
        lines.append("### 模型 Agent 分析总结\n")
        lines.append(llm_summary.strip())

    return "\n".join(lines)


def _model_finalize_node(state: GlobalState) -> GlobalState:
    """Pipeline 收尾：存 model_report、补全 best_model.task_type、保存 state 快照。"""
    last_msg = state["messages"][-1]
    llm_summary = str(last_msg.content) if hasattr(last_msg, "content") else ""

    best_model = dict(state.get("best_model") or {})
    task_type = (state.get("task_type") or "regression").lower()
    best_model["task_type"] = task_type

    model_name = best_model.get("model_name", "unknown")
    run_dir = state.get("run_dir") or ""

    # 从 state 数据构建完整报告，LLM 步骤 4 作为补充附后
    full_report = _build_model_report({**state, "best_model": best_model}, llm_summary)

    class _FakeMsg:
        """让 save_agent_report 能接受纯文本。"""
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
        print(f"Mermaid 源码已保存: {mmd_path}")
        image_path = output_dir / "model_agent_graph.png"
        image_bytes = agent_graph.get_graph().draw_mermaid_png()
        image_path.write_bytes(image_bytes)
        print(f"Graph 图片已保存: {image_path}")
        try:
            from IPython.display import Image, display
            display(Image(filename=str(image_path)))
        except Exception:
            pass
    except Exception as exc:
        print(f"图渲染失败: {exc}")


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
