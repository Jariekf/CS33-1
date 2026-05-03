"""
多智能体数据分析系统 - 主入口
基于 LangGraph 构建的全流水线自动化数据分析与商业洞察系统。
"""
import json
import os
from datetime import datetime

from config import WORKSPACE_DIR
from tools.eda_charts import purge_eda_figs_dir
from core.graph import build_graph
from core.manager_prompt_eval import write_manager_prompt_eval_report
from core.state import GraphState
from core.logger import get_agent_logger

sys_logger = get_agent_logger("sys")


# ---------------------------------------------------------------------------
# Transient artifacts produced by each pipeline run.
#
# These files are regenerated every run. If a previous run crashed or a later
# stage failed, a stale copy can still linger and silently get picked up by
# downstream consumers (e.g. business_translator reading an old
# feature_importance.json written 40 minutes ago). Cleaning them up at the
# start of each run guarantees that any file present after invoke() actually
# came from THIS run.
#
# We intentionally do NOT touch:
#   - data_cleaning_plan_latest.json / data_cleaning_plan_<stamp>.json
#     (reused by scripts/run_data_cleaning_only.py --plan-json ...)
#   - manager_prompt_eval_* / manager_eval/   (historical evaluation log)
#   - supervisor_raw_reply_*.txt              (debug breadcrumbs)
# ---------------------------------------------------------------------------
_TRANSIENT_ARTIFACTS: tuple[str, ...] = (
    # Per-stage CSV outputs (agents read these by fixed path)
    "cleaned_data.csv",
    "fe_output.csv",
    # Feature importance consumed by business_translator
    "feature_importance.json",
    # Supervisor detailed reports written per-phase
    "data_cleaning_report.md",
    "feature_engineering_report.md",
    "model_selection_report.md",
    # Final business report (Quarto source)
    "business_insight_report.qmd",
)


def _purge_stale_transient_artifacts() -> None:
    """Delete per-run outputs from any previous run before the graph starts.

    Never raises: cleanup is best-effort. A missing file is the nominal case
    on a fresh checkout, and a permission error should not block the pipeline.
    """
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    purge_eda_figs_dir(WORKSPACE_DIR)
    # Legacy filename before Quarto switch
    for legacy in ("business_insight_report.md",):
        try:
            os.remove(os.path.join(WORKSPACE_DIR, legacy))
        except FileNotFoundError:
            pass
        except OSError as e:
            sys_logger.warning("Failed to remove legacy artifact %s: %s", legacy, e)

    removed: list[str] = []
    for name in _TRANSIENT_ARTIFACTS:
        path = os.path.join(WORKSPACE_DIR, name)
        try:
            os.remove(path)
            removed.append(name)
        except FileNotFoundError:
            continue
        except OSError as e:
            sys_logger.warning("Failed to remove stale artifact %s: %s", path, e)
    if removed:
        sys_logger.info(
            "Purged %d stale transient artifact(s) before run: %s",
            len(removed),
            ", ".join(removed),
        )


def _write_data_cleaning_plan_snapshot(
    final_state: dict, run_started_at: datetime
) -> None:
    """
    Persist Manager's data_cleaning_plan after a full run (for reuse with
    scripts/run_data_cleaning_only.py --plan-json workspace/data_cleaning_plan_latest.json).
    """
    gp = final_state.get("global_plan") or {}
    raw = gp.get("data_cleaning_plan") if isinstance(gp, dict) else None
    plan: dict = raw if isinstance(raw, dict) else {}

    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    latest_path = os.path.join(WORKSPACE_DIR, "data_cleaning_plan_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    stamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    hist_path = os.path.join(WORKSPACE_DIR, f"data_cleaning_plan_{stamp}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    sys_logger.info(
        "Saved data_cleaning_plan snapshots: %s and %s",
        latest_path,
        hist_path,
    )


def run(
    data_path: str,
    user_goal: str,
    manager_cleaning_hints: str = "",
    dataset_description: str = "",
) -> str:
    """
    启动多智能体分析流水线。

    Args:
        data_path: 数据集文件路径
        user_goal: 用户的自然语言业务目标
        manager_cleaning_hints: 可选；清洗规则/contract 等文字，写入检索摘要并与规划对齐
        dataset_description: 可选；数据集权威说明（README / 数据字典 / 论文等文本），
            由 ``tools.description_loader.load_dataset_description`` 从文件/目录加载后传入。
            所有 Agent prompt 都会注入此文本作为语义锚。

    Returns:
        最终的商业洞察分析报告文本
    """
    run_started_at = datetime.now()
    # 构建 LangGraph 工作流
    sys_logger.info(f"Main execution started with data_path: '{data_path}', goal: '{user_goal}'")
    _purge_stale_transient_artifacts()
    app = build_graph()

    # 初始化输入状态
    initial_state: GraphState = {
        "data_path": data_path,
        "user_goal": user_goal,
        "manager_cleaning_hints": manager_cleaning_hints,
        "dataset_description": dataset_description,
        "global_plan": {},
        "current_phase": "",
        "current_data_path": data_path,
        "current_code": "",
        "current_execution_log": "",
        "current_error": "",
        "retry_count": 0,
        "current_review": None,
        "outer_retry_total": 0,
        "memory": [],
        "eda_summary": "",
        "eda_charts": [],
        "final_business_report": "",
    }

    # 执行工作流
    sys_logger.info("Starting graph invocation...")
    final_state = app.invoke(initial_state)
    sys_logger.info("Graph invocation finished successfully.")

    try:
        _write_data_cleaning_plan_snapshot(final_state, run_started_at)
    except Exception as e:
        sys_logger.warning("data_cleaning_plan snapshot skipped: %s", e)

    try:
        write_manager_prompt_eval_report(
            initial_state=initial_state,
            final_state=final_state,
            run_started_at=run_started_at,
        )
    except Exception as e:
        sys_logger.warning("Manager prompt eval report generation skipped due to error: %s", e)

    return final_state.get("final_business_report", "No report generated.")


if __name__ == "__main__":
    import argparse
    import sys

    from tools.description_loader import load_dataset_description

    parser = argparse.ArgumentParser(
        description="Run the multi-agent data-analysis pipeline.",
    )
    parser.add_argument(
        "data_path",
        help="Path to the input CSV dataset, e.g. ./data/train.csv",
    )
    parser.add_argument(
        "user_goal",
        help='Business goal in natural language, e.g. "Predict customer churn and provide retention strategies"',
    )
    parser.add_argument(
        "--desc",
        "-d",
        default="",
        metavar="PATH",
        help="Path to dataset description file OR directory "
             "(README.md / data_dictionary.xlsx / docs/ / paper.pdf ...). "
             "Content is injected into every agent as authoritative context.",
    )
    parser.add_argument(
        "--hints",
        default="",
        help="Optional narrow cleaning hints text (prefer --desc for general context).",
    )
    args = parser.parse_args()

    try:
        dataset_description = load_dataset_description(args.desc)
    except (FileNotFoundError, ValueError) as e:
        print(f"[!] Failed to load --desc: {e}", file=sys.stderr)
        sys.exit(1)

    if args.desc and not dataset_description:
        sys_logger.warning(
            "Dataset description path '%s' loaded empty content; proceeding without it.",
            args.desc,
        )

    report = run(
        args.data_path,
        args.user_goal,
        manager_cleaning_hints=args.hints,
        dataset_description=dataset_description,
    )

    # 【修复】先落盘保存，防止终端打印遇到 Unicode 报错导致没保存上
    import os
    os.makedirs("./workspace", exist_ok=True)
    report_path = "./workspace/business_insight_report.qmd"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[+] Final report has been generated and saved to: {report_path}")

    import shutil
    import subprocess

    quarto_bin = shutil.which("quarto")
    if quarto_bin:
        try:
            repo_root = os.path.dirname(os.path.abspath(__file__))
            proc = subprocess.run(
                [quarto_bin, "render", report_path, "--to", "html"],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            if proc.returncode == 0:
                print("[+] Quarto: rendered HTML from business_insight_report.qmd")
            else:
                sys_logger.warning(
                    "Quarto render failed (exit %s): %s",
                    proc.returncode,
                    (proc.stderr or proc.stdout or "")[:2000],
                )
        except OSError as e:
            sys_logger.warning("Quarto render raised: %s", e)
    else:
        sys_logger.info("Quarto CLI not found on PATH; skip HTML render. Install from https://quarto.org")

    print("\n" + "=" * 60)
    print("Final Business Analysis Report")
    print("=" * 60)
    try:
        print(report)
    except UnicodeEncodeError:
        print("[!] The report contains characters that cannot be displayed in this terminal. Please open the file in workspace directly.")
