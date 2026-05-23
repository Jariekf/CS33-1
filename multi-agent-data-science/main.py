"""Entry point: run the full multi-agent data science pipeline.

Usage (single CSV, train/test split happens inside the graph):
  python main.py --data datasets/bank-full.csv --description "..." --request "..."
  or: python main.py --data ... --description-file datasets/desc.txt --request "..."

Usage (pre-split train / test provided, skips split_agent; recommended to also set target column and task type):
  python main.py --train datasets/train.csv --test datasets/test.csv \\
                 --task-type classification --target y \\
                 --description "..." --request "..."
  or use --description-file instead of --description
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage

from agent.agent_manager import _pretty_print_message
from core.graph import build_full_pipeline
from core.my_llm import TOKEN_COUNTER
from core.report_writer import save_state_snapshot

_DIVIDER = "=" * 60
_COL = 28


def _print_timing_summary(timing: dict, pipeline_end: float) -> None:
    pipeline_start = timing.get("pipeline_start")
    rows = [
        (k, v)
        for k, v in timing.items()
        if k != "pipeline_start" and isinstance(v, dict) and "start" in v and "end" in v
    ]
    if not rows and pipeline_start is None:
        return
    sep = "─" * (_COL + 12)
    print(f"\n{sep}")
    print(f"  {'Agent':<{_COL}} {'Time':>6}")
    print(sep)
    for key, entry in rows:
        elapsed = entry["end"] - entry["start"]
        print(f"  {key:<{_COL}} {elapsed:6.1f}s")
    if pipeline_start is not None:
        total = pipeline_end - pipeline_start
        print(sep)
        print(f"  {'total (wall clock)':<{_COL}} {total:6.1f}s")
    print(sep)


def run_pipeline(
    data_path: str,
    description: str,
    user_request: str,
    show_graph: bool = False,
    recursion_limit: int = 200,
    *,
    training_data_path: str = "",
    testing_data_path: str = "",
    target_column: str = "",
    task_type: str = "",
    description_file_path: str = "",
) -> dict:
    """Run the full multi-agent data science pipeline and return the final state.

    Args:
        data_path:           Path to the raw dataset CSV.
        description:         Dataset description text (or read from .txt via CLI `--description-file`); injected into state for agents.
        user_request:        User instruction describing the task objective.
        show_graph:          If True, write Mermaid and PNG to artifacts/graphs/ (full_pipeline.mmd / full_pipeline.png).
        recursion_limit:     LangGraph maximum recursion steps; increase for longer pipelines.
        training_data_path:  If provided together with testing_data_path, init_run sets is_split=True and skips split_agent.
        testing_data_path:   See above.
        target_column:       Optional. When non-empty together with a valid task_type, written to state so Agent Manager skips infer_task.
        task_type:           classification or regression; must be paired with target_column.

    Returns:
        The complete GlobalState dict at pipeline completion.
    """
    TOKEN_COUNTER.reset()
    pipeline = build_full_pipeline(show_graph=show_graph)

    print(f"\n{_DIVIDER}")
    print("Multi-Agent Data Science Pipeline — Starting")
    train_t = (training_data_path or "").strip()
    test_t = (testing_data_path or "").strip()
    if train_t and test_t:
        print("Mode: pre-split (train / test provided, skipping split_agent)")
        print(f"train CSV: {train_t}")
        print(f"test CSV : {test_t}")
    else:
        print(f"Mode: single file -> split_agent will split internally")
        print(f"Dataset: {data_path}")

    tc = (target_column or "").strip()
    tt = (task_type or "").strip().lower()
    if tc and tt in {"classification", "regression"}:
        print(f"Task metadata (skipping infer_task in Agent Manager): task_type={tt}  target_column={tc}")
    print(_DIVIDER)

    final_state: dict = {}
    _split_logged = False
    _last_stage: str = ""

    initial: dict = {
        "messages": [HumanMessage(content=user_request)],
        "user_request": user_request,
        "data_path": data_path,
        "description": description,
        "description_file_path": description_file_path,
    }
    if train_t and test_t:
        initial["training_data_path"] = train_t
        initial["testing_data_path"] = test_t
    if tc and tt in {"classification", "regression"}:
        initial["target_column"] = tc
        initial["task_type"] = tt

    for event in pipeline.stream(
        initial,
        stream_mode="values",
        config={"recursion_limit": recursion_limit},
    ):
        final_state = event

        if "messages" in event:
            _pretty_print_message(event["messages"][-1])

        if (stage := event.get("current_stage")) and stage != _last_stage:
            _last_stage = stage
            print(f"\n[stage] {stage}")

        if not _split_logged and event.get("training_data_path"):
            _split_logged = True
            print(f"[split] train -> {event['training_data_path']}")
            print(f"[split] test  -> {event['testing_data_path']}")

        if event.get("best_model"):
            bm = event["best_model"]
            print(f"\n[best_model] {bm.get('model_name')}  metrics={bm.get('metrics')}")

    pipeline_end = time.time()

    # Final snapshot includes business_translator timing; older snapshots
    # (e.g. model_agent finalize) omit BAT and break benchmark phase totals.
    if final_state.get("run_dir"):
        save_state_snapshot(final_state)

    print(f"\n{_DIVIDER}")
    print("Pipeline complete")
    if run_dir := final_state.get("run_dir"):
        print(f"Output directory: {run_dir}")
        if final_state.get("business_report"):
            br_path = Path(run_dir) / "reports" / "business_insight_report.md"
            print(f"Business insight report: {br_path}")
    if bm := final_state.get("best_model"):
        print(f"Best model: {bm.get('model_name')}")
        print(f"Evaluation metrics: {bm.get('metrics')}")

    _print_timing_summary(final_state.get("timing") or {}, pipeline_end)

    # ── Save token usage ──────────────────────────────────────────────────
    if run_dir := final_state.get("run_dir"):
        token_data = {
            "prompt":     TOKEN_COUNTER.prompt,
            "completion": TOKEN_COUNTER.completion,
            "total":      TOKEN_COUNTER.total,
        }
        token_path = Path(run_dir) / "reports" / "token_usage.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
        print(f"[token_usage] prompt={TOKEN_COUNTER.prompt}  "
              f"completion={TOKEN_COUNTER.completion}  "
              f"total={TOKEN_COUNTER.total}")

    print(_DIVIDER)
    return final_state


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Data Science Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data source (choose one):
  - Single CSV: --data <path>  (split_agent splits internally)
  - Pre-split:  --train <path> --test <path>  (skips split_agent)

Dataset description (choose one):
  - --description "TEXT"
  - --description-file <path/to/file.txt>  (UTF-8)

Examples:
  python main.py --data datasets/bank-full.csv \\
                 --description "Binary classification task predicting whether a client subscribes to a term deposit (target column: y)" \\
                 --request "Run the full pipeline: clean data, split dataset, feature engineering, train models, and select the best classifier"

  python main.py --train datasets/train.csv --test datasets/test.csv \\
                 --task-type regression --target SalePrice \\
                 --description-file datasets/dataset_desc.txt \\
                 --request "Run the full pipeline and train models"

  python main.py --data datasets/housing.csv \\
                 --description "Regression task predicting house prices (target column: price)" \\
                 --request "Analyze the data and train regression models, selecting the one with the lowest RMSE" \\
                 --show-graph --recursion-limit 300
""",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--data",
        default="",
        metavar="PATH",
        help="Single-file CSV: path to the raw dataset; split_agent will split it into train/test",
    )
    src.add_argument(
        "--train",
        default="",
        metavar="PATH",
        help="Used with --test: path to the training CSV (pre-split mode, skips split_agent)",
    )
    parser.add_argument(
        "--test",
        default="",
        metavar="PATH",
        help="Used with --train: path to the test / hold-out CSV",
    )
    desc = parser.add_mutually_exclusive_group(required=True)
    desc.add_argument(
        "--description",
        default=None,
        metavar="TEXT",
        help="Dataset description (plain text); mutually exclusive with --description-file",
    )
    desc.add_argument(
        "--description-file",
        default=None,
        dest="description_file",
        metavar="PATH",
        help="Dataset description TXT (UTF-8); mutually exclusive with --description",
    )
    parser.add_argument(
        "--request",
        required=True,
        metavar="TEXT",
        help="User instruction describing the task objective (e.g. 'Run the full pipeline and select the best model')",
    )
    parser.add_argument(
        "--target",
        default="",
        metavar="COLUMN",
        help="Target column name; when provided together with --task-type, written to state so Agent Manager skips infer_task",
    )
    parser.add_argument(
        "--task-type",
        default="",
        dest="task_type",
        metavar="classification|regression",
        help="Used with --target: classification or regression",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Export pipeline graph to artifacts/graphs/full_pipeline.mmd and full_pipeline.png (paste into mermaid.live to view)",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=200,
        metavar="N",
        help="LangGraph maximum recursion steps (default: 200)",
    )
    # ── Optional: run benchmark evaluation automatically after the pipeline ──
    eval_grp = parser.add_argument_group(
        "benchmark (optional)",
        "Supply all three to auto-record metrics into experiment_records after the run.",
    )
    eval_grp.add_argument(
        "--competition",
        default="",
        metavar="NAME",
        choices=[
            "", "bank_churn", "house_prices",
            "obesity_risks",
            "classification_with_an_academic_success_dataset",
            "student_study",
        ],
        help="Competition key for benchmark evaluation",
    )
    eval_grp.add_argument(
        "--eval-test",
        default="",
        dest="eval_test",
        metavar="PATH",
        help="Split test CSV with true labels. Optional when using --data: "
             "split_agent's test split is used automatically.",
    )
    eval_grp.add_argument(
        "--system-name",
        default="MAC-ADS",
        dest="system_name",
        metavar="NAME",
        help="System name written to experiment record (default: MAC-ADS)",
    )
    return parser


def _cli_task_meta_or_exit(args: argparse.Namespace) -> tuple[str, str]:
    tc = (getattr(args, "target", "") or "").strip()
    tt = (getattr(args, "task_type", "") or "").strip().lower()
    if not tc and not tt:
        return "", ""
    if not tc or not tt:
        print(
            "[!] --target and --task-type must both be provided or both omitted.",
            file=sys.stderr,
        )
        sys.exit(2)
    if tt not in {"classification", "regression"}:
        print(
            f"[!] --task-type must be classification or regression, got: {args.task_type!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    return tc, tt


def _description_from_cli_or_exit(args: argparse.Namespace) -> str:
    """CLI: --description and --description-file are mutually exclusive, enforced by argparse."""
    path = getattr(args, "description_file", None)
    if path:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            print(f"[!] --description-file is not a valid file: {p}", file=sys.stderr)
            sys.exit(2)
        if p.suffix.lower() != ".txt":
            print(
                "[!] --description-file currently only supports .txt files.",
                file=sys.stderr,
            )
            sys.exit(2)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[!] Cannot read description file: {exc}", file=sys.stderr)
            sys.exit(2)
        stripped = text.strip()
        if not stripped:
            print("[!] Description file is empty or contains only whitespace.", file=sys.stderr)
            sys.exit(2)
        return stripped
    raw = args.description
    if raw is None:
        print("[!] Must provide --description or --description-file.", file=sys.stderr)
        sys.exit(2)
    return str(raw).strip()


def _run_benchmark_eval(final_state: dict, args: argparse.Namespace) -> None:
    """Auto-call benchmark evaluate() when --competition is set."""
    competition = (getattr(args, "competition",  "") or "").strip()
    system_name = (getattr(args, "system_name",  "MAC-ADS") or "MAC-ADS").strip()
    desc_file   = (getattr(args, "description_file", "") or "").strip()
    run_dir     = (final_state.get("run_dir") or "").strip()

    if not (competition and run_dir):
        return  # --competition not supplied — skip silently

    # Import evaluate() from scripts/benchmark_eval_system.py
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        from benchmark_eval_system import evaluate  # type: ignore[import]
    except ImportError as exc:
        print(f"[benchmark] Could not import benchmark_eval_system: {exc}", file=sys.stderr)
        return

    print(f"\n[benchmark] Running evaluate()  competition={competition}")
    try:
        evaluate(
            run_dir=run_dir,
            competition=competition,
            description_file=desc_file,
            system_name=system_name,
            system_type="multi-agent DS pipeline",
        )
    except Exception as exc:
        print(f"[benchmark] evaluate() failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    tc_cli, tt_cli = _cli_task_meta_or_exit(args)
    desc_text = _description_from_cli_or_exit(args)
    desc_file_path = str(Path(args.description_file).expanduser().resolve()) if getattr(args, "description_file", None) else ""
    train_p = (args.train or "").strip()
    test_p = (args.test or "").strip()
    data_p = (args.data or "").strip()

    if train_p or test_p:
        if not train_p or not test_p:
            print(
                "[!] Pre-split mode requires both --train and --test.",
                file=sys.stderr,
            )
            sys.exit(2)
        p_train = Path(train_p).expanduser().resolve()
        p_test = Path(test_p).expanduser().resolve()
        for label, p in (("--train", p_train), ("--test", p_test)):
            if not p.is_file():
                print(f"[!] {label} is not a valid file: {p}", file=sys.stderr)
                sys.exit(2)
        data_path_eff = str(p_train)
        final = run_pipeline(
            data_path=data_path_eff,
            description=desc_text,
            user_request=args.request,
            show_graph=args.show_graph,
            recursion_limit=args.recursion_limit,
            training_data_path=str(p_train),
            testing_data_path=str(p_test),
            target_column=tc_cli,
            task_type=tt_cli,
            description_file_path=desc_file_path,
        )
    else:
        if not data_p:
            print("[!] Please provide --data, or both --train and --test.", file=sys.stderr)
            sys.exit(2)
        p_data = Path(data_p).expanduser().resolve()
        if not p_data.is_file():
            print(f"[!] --data is not a valid file: {p_data}", file=sys.stderr)
            sys.exit(2)
        final = run_pipeline(
            data_path=str(p_data),
            description=desc_text,
            user_request=args.request,
            show_graph=args.show_graph,
            recursion_limit=args.recursion_limit,
            target_column=tc_cli,
            task_type=tt_cli,
            description_file_path=desc_file_path,
        )

    _run_benchmark_eval(final, args)
