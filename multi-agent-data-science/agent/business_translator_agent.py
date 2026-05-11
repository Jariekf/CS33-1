"""Business Action Translator — LangGraph Tool-Loop + McKinsey RAG + arXiv → Markdown 业务报告."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# 单次 BAT 子图内「LLM→Tools→更新」循环上限，防止模型反复 tool_calls 耗尽全图 recursion_limit
BAT_MAX_TOOL_ROUNDS = int(os.getenv("BAT_MAX_TOOL_ROUNDS", "15"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from core.global_state import GlobalState
from core.my_llm import call_llm, llm_deepseek
from core.report_writer import save_agent_report
from tools.business_tools.bat_tools import bat_tools
from tools.business_tools.file_reader import read_file

logger = logging.getLogger(__name__)

STYLE_SKELETONS_PATH = PROJECT_ROOT / "data" / "mckinsey_kb" / "style_skeletons.md"

tools = bat_tools
llm_with_tools = llm_deepseek.bind_tools(tools)


def _extract_top_features(state: GlobalState, limit: int = 10) -> list[str]:
    """From EDA correlation ranking, feature_sets, or dataset columns."""
    eda = state.get("eda_report") or {}
    steps = eda.get("steps") or []
    for step in reversed(steps):
        if step.get("tool") != "compute_correlation_matrix":
            continue
        res = step.get("result") or {}
        if not isinstance(res, dict):
            continue
        ranking = res.get("target_correlation_ranking")
        if isinstance(ranking, dict) and ranking:
            sorted_feats = sorted(
                ranking.items(),
                key=lambda x: abs(float(x[1])),
                reverse=True,
            )
            return [str(f[0]) for f in sorted_feats[:limit]]

    best_name = (state.get("best_model") or {}).get("model_name")
    fs_map = state.get("feature_sets") or {}
    if best_name and isinstance(fs_map.get(best_name), list):
        return list(fs_map[best_name])[:limit]

    prof = state.get("dataset_profile") or {}
    ds = prof.get("dataset") if isinstance(prof, dict) else {}
    cols = (ds or {}).get("columns") if isinstance(ds, dict) else []
    if isinstance(cols, list) and cols:
        tgt = state.get("target_column") or ""
        return [c for c in cols if c != tgt][:limit]

    return []


def _build_model_validation_summary(state: GlobalState) -> str:
    best = state.get("best_model") or {}
    results = state.get("model_results") or {}
    lines: list[str] = []
    if best:
        lines.append(
            f"**Best model:** {best.get('model_name', 'n/a')}  "
            f"task_type={best.get('task_type', 'n/a')}"
        )
        m = best.get("metrics") or {}
        if m:
            lines.append("**Best model metrics (test):** " + json.dumps(m, ensure_ascii=False))
    if results:
        lines.append("**All candidate results:**")
        for name, r in results.items():
            lines.append(
                f"- {name}: metrics={json.dumps(r.get('metrics', {}), ensure_ascii=False)}"
            )
    return "\n".join(lines) if lines else "No model validation summary available."


def _build_feature_and_predictions_section(state: GlobalState) -> tuple[str, str]:
    top_features = _extract_top_features(state)
    if top_features:
        feat_block = (
            "Features prioritized by EDA correlation with target (when available) / pipeline context:\n"
            + ", ".join(top_features)
        )
    else:
        feat_block = (
            "(No numeric correlation ranking found in eda_report; do not invent importance scores — "
            "ground narrative in EDA steps and model metrics below.)"
        )

    best = state.get("best_model") or {}
    bm_name = best.get("model_name", "")
    fep = state.get("feature_engineering_plan") or {}
    extra = ""
    if bm_name and isinstance(fep.get(bm_name), dict):
        extra = (
            f"\n\nFeature engineering notes for **{bm_name}**: "
            + json.dumps(fep[bm_name], ensure_ascii=False)
        )

    test_path = state.get("testing_data_path") or ""
    model_path = best.get("model_path") or ""
    pred_block = (
        f"- Holdout test CSV path: `{test_path or 'not provided'}`\n"
        f"- Best model artifact path: `{model_path or 'not provided'}`\n"
        "- Row-level test predictions CSV is **not** exported by this pipeline; do not claim exact counts "
        "unless derived from the stated metrics / EDA."
    )
    return feat_block + extra, pred_block


def _format_mckinsey_bucket(label: str, items: list[dict]) -> str:
    if not items:
        return f"### {label}\n(no matching context retrieved.)\n"
    blocks: list[str] = [f"### {label}"]
    for i, item in enumerate(items, start=1):
        meta = item.get("metadata") or {}
        title = meta.get("title", "Untitled")
        section = meta.get("section_path") or ""
        year = meta.get("year")
        head = f"[{i}] {title}"
        if year:
            head += f" ({year})"
        if section:
            head += f"  -  {section}"
        body = (item.get("content") or "").strip()
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks) + "\n"


def _format_eda_placeholder(eda_report: Any) -> str:
    if not eda_report:
        return "(not provided — upstream did not export structured EDA summary.)"
    try:
        return json.dumps(eda_report, ensure_ascii=False, indent=2)[:12000]
    except Exception:
        return str(eda_report)[:12000]


def _resolve_chart_rel_paths(state: GlobalState) -> list[dict[str, str]]:
    """Return chart metadata with Markdown-friendly relative paths from ``reports/`` to ``plots/``."""
    raw_charts: list[dict[str, Any]] = list(state.get("eda_charts") or [])
    if not raw_charts:
        eda = state.get("eda_report") or {}
        raw_charts = list(eda.get("charts") or [])
    if not raw_charts:
        return []

    run_dir = state.get("run_dir") or ""
    eda_report = state.get("eda_report") or {}
    chart_output_dir = eda_report.get("chart_output_dir", "")

    resolved: list[dict[str, str]] = []
    for chart in raw_charts:
        name = chart.get("name", "")
        desc = chart.get("desc", "")
        if not name:
            continue

        if run_dir:
            rel = f"../plots/{name}"
        elif chart_output_dir:
            rel = f"{chart_output_dir}/{name}"
        else:
            abs_path = chart.get("path", "")
            rel = abs_path if abs_path else name

        resolved.append({"name": name, "rel_path": rel, "desc": desc})
    return resolved


def _build_eda_charts_prompt_block(charts: list[dict[str, str]]) -> str:
    """Build a prompt section listing available EDA charts for the LLM."""
    if not charts:
        return "(no EDA charts available from upstream pipeline.)"
    lines = [
        "The following EDA charts have been generated and saved. "
        "Embed them in the report using standard Markdown image syntax.",
        "",
        "| Chart file | Markdown embed | Description |",
        "|---|---|---|",
    ]
    for c in charts:
        embed = f'![{c["desc"]}]({c["rel_path"]})'
        lines.append(f'| {c["name"]} | `{embed}` | {c["desc"]} |')
    return "\n".join(lines)


def _inject_remaining_charts(report: str, charts: list[dict[str, str]]) -> str:
    """Append unreferenced charts as an appendix section at the end of the report."""
    if not charts:
        return report
    unreferenced = [c for c in charts if c["name"] not in report]
    if not unreferenced:
        return report
    appendix_lines = [
        "",
        "## Appendix: Additional EDA Visualizations",
        "",
    ]
    for c in unreferenced:
        appendix_lines.append(f"### {c['desc']}")
        appendix_lines.append("")
        appendix_lines.append(f'![{c["desc"]}]({c["rel_path"]})')
        appendix_lines.append("")
    return report.rstrip() + "\n" + "\n".join(appendix_lines)


def _build_charts_fallback_section(charts: list[dict[str, str]]) -> str:
    """Build a standalone charts section for the fallback template."""
    if not charts:
        return ""
    lines = ["## EDA Visualizations", ""]
    for c in charts:
        lines.append(f'![{c["desc"]}]({c["rel_path"]})')
        lines.append("")
    return "\n".join(lines)


def _default_rag_buckets(state: GlobalState) -> dict[str, list[dict[str, Any]]]:
    """Fallback when the agent never called RAG tools."""
    from tools.business_tools.bat_tools import retrieve_mckinsey_context_impl

    goal = (state.get("user_request") or "business analytics").strip()
    top = _extract_top_features(state)
    if top:
        queries = [f"{goal} " + " ".join(top[:3]) + " strategy"]
    else:
        queries = [f"{goal} business analytics strategy"]
    return retrieve_mckinsey_context_impl(queries)


def _draft_business_report_markdown(state: GlobalState) -> str:
    user_goal = state.get("user_request") or ""
    dataset_description = (state.get("description") or "").strip()
    eda_block = _format_eda_placeholder(state.get("eda_report"))
    model_summary = _build_model_validation_summary(state)
    feat_block, pred_block = _build_feature_and_predictions_section(state)

    charts = _resolve_chart_rel_paths(state)
    charts_prompt_block = _build_eda_charts_prompt_block(charts)

    buckets = state.get("bat_rag_buckets") or {}
    if not buckets or sum(len(v) for v in buckets.values()) == 0:
        buckets = _default_rag_buckets(state)

    retrieved_arxiv = state.get("bat_arxiv_papers") or []
    style_skeletons = (state.get("bat_style_skeletons") or "").strip()
    if not style_skeletons:
        try:
            style_skeletons = read_file(str(STYLE_SKELETONS_PATH)).strip()
        except Exception as e:
            logger.warning("Style skeletons unavailable (%s); proceeding without.", e)
            style_skeletons = ""

    arxiv_text = (
        "\n".join(
            f"- {d.get('title', 'N/A')}: {str(d.get('summary', ''))[:150]}..."
            for d in retrieved_arxiv
        )
        if retrieved_arxiv
        else "(no arxiv papers retrieved.)"
    )

    desc_text = dataset_description.strip() if dataset_description else ""
    dataset_description_block = (
        desc_text
        if desc_text
        else "(not provided; base the report on the user goal and model outputs only.)"
    )

    mck_block = _format_mckinsey_bucket("McKinsey Reports", buckets.get("mckinsey", []))
    eq_block = _format_mckinsey_bucket(
        "Equity Research & Strategy", buckets.get("equity_research", [])
    )
    tb_block = _format_mckinsey_bucket("Management Textbook", buckets.get("textbook", []))
    skeletons_block = (
        style_skeletons or "(no style skeletons available; rely on McKinsey style rules below.)"
    )

    sys_prompt = (
        "You are an elite Data Strategy Consultant writing in McKinsey consulting style. "
        "Apply the Pyramid Principle (top-down, answer-first), open with SCQA, "
        "structure findings into 3-5 MECE themes, and close with concrete sequenced actions. "
        "CRITICAL: be concise; ground every claim in model evidence or retrieved external context. "
        "Output MUST be plain Markdown in English (no YAML front matter). "
        "When EDA charts are available, embed them in relevant sections using standard Markdown "
        "image syntax `![description](path)` to strengthen evidence presentation."
    )

    user_prompt = f"""
Produce a **Business Strategy & Insight Report** as Markdown only (English).

# User Goal
{user_goal}

# Dataset Description (business context)
{dataset_description_block}

# EDA / profiling context (may be JSON from upstream agents)
{eda_block}

# EDA Visualization Charts
{charts_prompt_block}

# Model Validation Summary
{model_summary}

# Top Features & Importance
{feat_block}

# Test-Set / Artifacts
{pred_block}

# External Context - McKinsey Knowledge Base (primary)
{mck_block}
{eq_block}
{tb_block}

# External Context - Arxiv (fallback, may be empty)
{arxiv_text}

# Reference Skeletons (structural anchors only - imitate the *shape*, not the content)
{skeletons_block}

# Required Output Structure (in this exact order)
1. **Executive Summary** — bottom-line answer + three bullet takeaways.
2. **Situation & Complication** — SCQA opening; if a target distribution or class balance chart is available, embed it here.
3. **Key Findings** — 3 to 5 MECE themes; claim → evidence → implication. After each theme's evidence paragraph, embed the most relevant EDA chart(s) from the list above (e.g. correlation heatmap for correlation themes, feature importance for model themes, distribution charts for data profile themes). Use each chart at most once.
4. **Strategic Implications** — business meaning; cite external sources as [1], [2] where applicable.
5. **Recommended Actions** — sequenced 90-day priorities; concrete owners/metrics/timeframe where possible.

Style rules:
- Match the tone of the McKinsey Reports section where relevant.
- Do NOT invent statistics absent from the model summary or retrieved context.
- Embed EDA chart images inline where they support the narrative. Use the exact Markdown syntax from the chart table above. Each chart should be used at most once across the entire report.
"""

    reply = call_llm(sys_prompt, user_prompt, agent_name="business_translator")
    if reply:
        return _inject_remaining_charts(reply, charts)

    fallback_charts = _build_charts_fallback_section(charts)
    return f"""# Business Insight Analysis Report

## Executive Summary
User goal: {user_goal}

## Model Validation
{model_summary}

{fallback_charts}## External Context
{mck_block}
{eq_block}
{tb_block}

Arxiv fallback:
{arxiv_text}

*(LLM returned empty content; check API / quota / STUB_MODE.)*
"""


def _parse_tool_payload(tool_message: ToolMessage) -> dict[str, Any]:
    try:
        return json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        return {"error": {"type": "JSONDecodeError", "message": "工具输出不是合法 JSON"}}


def _merge_arxiv(existing: list[dict[str, Any]], new_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for group in (existing, new_papers):
        for p in group:
            t = str(p.get("title", ""))
            key = t or str(id(p))
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out[:50]


def _handle_retrieve_mckinsey(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "bat_rag_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "bat_rag", **payload["error"]}],
        }
    buckets = payload.get("buckets")
    if not isinstance(buckets, dict):
        return {
            "current_stage": "bat_rag_failed",
            "errors": (state.get("errors") or [])
            + [
                {
                    "stage": "bat_rag",
                    "type": "InvalidPayload",
                    "message": "retrieve_mckinsey_knowledge 返回缺少 buckets",
                }
            ],
        }
    return {
        "bat_rag_buckets": buckets,
        "current_stage": "bat_rag_retrieved",
    }


def _handle_search_arxiv(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "bat_arxiv_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "bat_arxiv", **payload["error"]}],
        }
    papers = payload.get("papers") or []
    if not isinstance(papers, list):
        papers = []
    merged = _merge_arxiv(list(state.get("bat_arxiv_papers") or []), papers)
    return {
        "bat_arxiv_papers": merged,
        "current_stage": "bat_arxiv_retrieved",
    }


def _handle_load_skeletons(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    payload = _parse_tool_payload(tool_message)
    if "error" in payload:
        return {
            "current_stage": "bat_skeleton_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "bat_skeletons", **payload["error"]}],
        }
    sk = payload.get("skeletons")
    if not isinstance(sk, str):
        sk = ""
    return {
        "bat_style_skeletons": sk,
        "current_stage": "bat_skeletons_loaded",
    }


def _handle_generate_report(state: GlobalState, tool_message: ToolMessage) -> GlobalState:
    _ = tool_message
    try:
        report = _draft_business_report_markdown(state)
    except Exception as e:
        return {
            "current_stage": "bat_report_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "bat_report", "type": e.__class__.__name__, "message": str(e)}],
        }

    run_dir = state.get("run_dir") or ""
    if run_dir:
        reports_dir = Path(run_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / "business_report.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"[business_translator] report saved → {out_path}")

    return {
        "business_report": report,
        "current_stage": "business_report_done",
    }


def _business_translator_node(state: GlobalState) -> GlobalState:
    messages = state["messages"]
    user_goal = state.get("user_request") or ""
    run_dir = state.get("run_dir") or ""
    task_type = (state.get("task_type") or "unknown").lower()
    target_column = state.get("target_column") or ""
    best = state.get("best_model") or {}
    best_name = best.get("model_name", "n/a")
    top_features = _extract_top_features(state)
    top_feat_str = ", ".join(top_features[:12]) if top_features else "(暂无 — 将仅用目标与指标检索)"

    system_content = (
        "你是业务分析翻译（Business Translator）Agent：把上游数据分析与模型结果转化为 **英文** McKinsey 风格的业务战略报告。\n"
        f"用户目标：{user_goal or '未提供'}\n"
        f"任务类型：{task_type}；目标列：{target_column or '未提供'}\n"
        f"当前最优模型：{best_name}\n"
        f"本次运行目录：{run_dir or '未设置'}\n"
        f"用于检索的优先特征（来自 EDA 相关排名 / 元数据）：{top_feat_str}\n\n"
        "工作流程（按顺序，可重复检索）：\n"
        "1. 调用 **retrieve_mckinsey_knowledge**，传入 2–4 条英文检索查询（结合用户目标与上述特征）。\n"
        "2. 若 McKinsey 知识库命中偏少，可调用 **search_arxiv_papers** 补充学术文献（1–3 条查询）。\n"
        "3. 调用 **load_report_style_skeletons** 加载写作结构参考（可选但推荐）。\n"
        "4. 调用 **generate_business_report** 生成最终 Markdown；该步骤会汇总 GlobalState 中的指标、EDA、RAG 结果并落盘到 "
        f"{run_dir + '/reports/business_report.md' if run_dir else 'run_dir/reports/business_report.md'}。\n"
        "**务必控制轮次**：完成检索与可选 arXiv / skeleton 后，立即调用 **generate_business_report**，"
        "不要重复检索；若无更多信息，也应直接调用 generate_business_report。\n"
        "除上述工具外不要编造外部数据来源。"
    )

    messages = [SystemMessage(content=system_content), *messages]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tool_state_update_node(state: GlobalState) -> GlobalState:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage) or not last_msg.name:
        return {}

    handlers = {
        "retrieve_mckinsey_knowledge": _handle_retrieve_mckinsey,
        "search_arxiv_papers": _handle_search_arxiv,
        "load_report_style_skeletons": _handle_load_skeletons,
        "generate_business_report": _handle_generate_report,
    }
    handler = handlers.get(last_msg.name)
    if not handler:
        return {}
    out = handler(state, last_msg)
    if not out:
        return {}
    merged = dict(out)
    merged["bat_tool_rounds"] = (state.get("bat_tool_rounds") or 0) + 1
    return merged


def _save_business_artifacts(state: GlobalState) -> None:
    report = state.get("business_report") or ""
    run_dir = state.get("run_dir") or ""
    if run_dir and report:
        reports_dir = Path(run_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / "business_report.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"[business_translator] report saved → {out_path}")
    if report:
        save_agent_report(
            "business_translator",
            AIMessage(content=report),
            run_dir=run_dir,
        )


def _route_after_business_translator(state: GlobalState) -> str:
    last_msg = state["messages"][-1]
    rounds = state.get("bat_tool_rounds") or 0
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        if rounds >= BAT_MAX_TOOL_ROUNDS:
            logger.warning(
                "BAT exceeded BAT_MAX_TOOL_ROUNDS=%s (current=%s); forcing bat_finalize.",
                BAT_MAX_TOOL_ROUNDS,
                rounds,
            )
            return "bat_finalize"
        return "tools"
    if state.get("business_report"):
        _save_business_artifacts(state)
        return END
    return "bat_finalize"


def _bat_finalize_node(state: GlobalState) -> GlobalState:
    """LLM 未调用 generate_business_report 时自动起草并落盘。"""
    if state.get("business_report"):
        return {}
    try:
        report = _draft_business_report_markdown(state)
    except Exception as e:
        err_report = f"# Report generation failed\n\n{e!s}"
        return {
            "business_report": err_report,
            "current_stage": "business_report_failed",
            "errors": (state.get("errors") or [])
            + [{"stage": "bat_finalize", "type": e.__class__.__name__, "message": str(e)}],
        }

    run_dir = state.get("run_dir") or ""
    if run_dir:
        reports_dir = Path(run_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "business_report.md").write_text(report, encoding="utf-8")
        print(f"[business_translator] auto-finalize report saved → {reports_dir / 'business_report.md'}")

    save_agent_report(
        "business_translator",
        AIMessage(content=report),
        run_dir=run_dir,
    )
    return {
        "business_report": report,
        "current_stage": "business_report_done",
        "messages": [
            AIMessage(
                content="(auto-finalize) Business report drafted from pipeline state and BAT tool context."
            )
        ],
    }


def _show_graph_if_needed(compiled: Any, show_graph: bool) -> None:
    if not show_graph:
        return
    try:
        output_dir = PROJECT_ROOT / "artifacts" / "graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        mmd_path = output_dir / "business_translator_graph.mmd"
        mmd_path.write_text(compiled.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"Mermaid 源码已保存: {mmd_path}")
        image_path = output_dir / "business_translator_graph.png"
        image_bytes = compiled.get_graph().draw_mermaid_png()
        image_path.write_bytes(image_bytes)
        print(f"Graph 图片已保存: {image_path}")
        try:
            from IPython.display import Image, display

            display(Image(filename=str(image_path)))
        except Exception:
            pass
    except Exception as exc:
        print(f"图渲染失败: {exc}")


def _pretty_print_message(message: Any) -> None:
    try:
        message.pretty_print()
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        text = message.pretty_repr()
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        print(safe_text)


def build_business_translator_agent(show_graph: bool = False):
    graph = StateGraph(GlobalState)
    graph.add_node("business_translator", _business_translator_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("tool_state_update", _tool_state_update_node)
    graph.add_node("bat_finalize", _bat_finalize_node)

    graph.set_entry_point("business_translator")
    graph.add_conditional_edges(
        "business_translator",
        _route_after_business_translator,
        {"tools": "tools", "bat_finalize": "bat_finalize", END: END},
    )
    graph.add_edge("tools", "tool_state_update")
    graph.add_edge("tool_state_update", "business_translator")
    graph.add_edge("bat_finalize", END)

    compiled = graph.compile()
    _show_graph_if_needed(compiled, show_graph=show_graph)
    return compiled


business_translator = build_business_translator_agent()


if __name__ == "__main__":
    agent = build_business_translator_agent(show_graph=True)
    user_input = input("请输入：")
    demo_run = PROJECT_ROOT / "artifacts" / "workspace" / "bat_standalone_demo"
    demo_run.mkdir(parents=True, exist_ok=True)
    events = agent.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_request": user_input,
            "description": "Demo dataset description.",
            "task_type": "classification",
            "target_column": "y",
            "testing_data_path": str(PROJECT_ROOT / "datasets" / "bank-full.csv"),
            "run_dir": str(demo_run),
            "best_model": {
                "model_name": "XGBoost",
                "metrics": {"accuracy": 0.9, "f1_weighted": 0.89},
                "task_type": "classification",
                "model_path": "",
            },
            "model_results": {
                "XGBoost": {"metrics": {"f1_weighted": 0.89}},
                "RandomForest": {"metrics": {"f1_weighted": 0.72}},
            },
            "eda_report": {"steps": []},
        },
        stream_mode="values",
        config={"recursion_limit": 80},
    )
    for event in events:
        if "messages" in event:
            _pretty_print_message(event["messages"][-1])
        if event.get("current_stage"):
            print(f"\n[stage] {event['current_stage']}")
        if event.get("business_report"):
            print(f"\n[report chars] {len(event['business_report'])}")
