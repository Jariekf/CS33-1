"""项目启动入口：运行完整多智能体数据科学流水线。

用法（单 CSV，图内再划分 train/test）：
  python main.py --data datasets/bank-full.csv --description "..." --request "..."
  或：python main.py --data ... --description-file datasets/desc.txt --request "..."

用法（已提供 train / test，跳过 split_agent；建议显式给定目标列与任务类型）：
  python main.py --train datasets/train.csv --test datasets/test.csv \\
                 --task-type classification --target y \\
                 --description "..." --request "..."
  或使用 --description-file 代替 --description
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage

from agent.agent_manager import _pretty_print_message
from core.graph import build_full_pipeline

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
) -> dict:
    """运行完整多智能体数据科学流水线并返回最终 state。

    Args:
        data_path:       原始数据集路径（CSV）。
        description:     数据集描述正文（或由 CLI `--description-file` 从 .txt 读入）；注入 state 供 Agent 使用。
        user_request:    用户指令，描述本次任务目标。
        show_graph:      为 True 时将 Mermaid 与 PNG 写入 artifacts/graphs/（full_pipeline.mmd / full_pipeline.png）。
        recursion_limit: LangGraph 最大递归步数，流程较长时可适当调大。
        training_data_path: 若与 testing_data_path 同时非空，init_run 将 is_split=True，跳过 split_agent。
        testing_data_path: 同上。
        target_column:   可选。与 task_type 同时为非空且合法时写入 state，Agent Manager 将跳过 infer_task。
        task_type:       classification 或 regression；须与 target_column 配对使用。

    Returns:
        流水线结束时的完整 GlobalState dict。
    """
    pipeline = build_full_pipeline(show_graph=show_graph)

    print(f"\n{_DIVIDER}")
    print("多智能体数据科学流水线 — 启动")
    train_t = (training_data_path or "").strip()
    test_t = (testing_data_path or "").strip()
    if train_t and test_t:
        print("模式：预切分（train / test 已给定，不经过 split_agent）")
        print(f"train CSV：{train_t}")
        print(f"test CSV ：{test_t}")
    else:
        print(f"模式：单文件 → split_agent 内划分")
        print(f"数据集：{data_path}")

    tc = (target_column or "").strip()
    tt = (task_type or "").strip().lower()
    if tc and tt in {"classification", "regression"}:
        print(f"任务元数据（跳过 Manager 内 infer_task）：task_type={tt}  target_column={tc}")
    print(_DIVIDER)

    final_state: dict = {}

    initial: dict = {
        "messages": [HumanMessage(content=user_request)],
        "user_request": user_request,
        "data_path": data_path,
        "description": description,
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

        if stage := event.get("current_stage"):
            print(f"\n[stage] {stage}")

        if event.get("training_data_path"):
            print(f"[split] train → {event['training_data_path']}")
            print(f"[split] test  → {event['testing_data_path']}")

        if event.get("best_model"):
            bm = event["best_model"]
            print(f"\n[best_model] {bm.get('model_name')}  metrics={bm.get('metrics')}")

    pipeline_end = time.time()

    print(f"\n{_DIVIDER}")
    print("流水线运行完毕")
    if run_dir := final_state.get("run_dir"):
        print(f"输出目录：{run_dir}")
        if final_state.get("business_report"):
            br_path = Path(run_dir) / "reports" / "business_report.md"
            print(f"业务洞察报告：{br_path}")
    if bm := final_state.get("best_model"):
        print(f"最优模型：{bm.get('model_name')}")
        print(f"评估指标：{bm.get('metrics')}")

    _print_timing_summary(final_state.get("timing") or {}, pipeline_end)
    print(_DIVIDER)

    return final_state


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="多智能体数据科学流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据来源（二选一）：
  • 单 CSV：--data <path>（流水线内 split_agent 再划分）
  • 已切分：--train <path> --test <path>（跳过 split_agent）

数据集描述（二选一）：
  • --description "TEXT"
  • --description-file <path/to/file.txt>（UTF-8）

示例：
  python main.py --data datasets/bank-full.csv \\
                 --description "二分类任务，预测客户是否订阅银行定期存款（目标列 y）" \\
                 --request "请执行全流程分析，清洗数据、切分数据集、特征工程并训练模型，选出最优分类模型"

  python main.py --train datasets/train.csv --test datasets/test.csv \\
                 --task-type regression --target SalePrice \\
                 --description-file datasets/dataset_desc.txt \\
                 --request "请全流程分析并训练模型"

  python main.py --data datasets/housing.csv \\
                 --description "回归任务，预测房价（目标列 price）" \\
                 --request "请分析数据并训练回归模型，选出 RMSE 最低的模型" \\
                 --show-graph --recursion-limit 300
""",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--data",
        default="",
        metavar="PATH",
        help="单文件 CSV：原始数据集路径，将由 split_agent 切分为 train/test",
    )
    src.add_argument(
        "--train",
        default="",
        metavar="PATH",
        help="与 --test 成对使用：训练集 CSV 路径（预切分模式，跳过 split_agent）",
    )
    parser.add_argument(
        "--test",
        default="",
        metavar="PATH",
        help="与 --train 成对使用：测试集 / hold-out CSV 路径",
    )
    desc = parser.add_mutually_exclusive_group(required=True)
    desc.add_argument(
        "--description",
        default=None,
        metavar="TEXT",
        help="数据集描述（纯文本）；与 --description-file 二选一",
    )
    desc.add_argument(
        "--description-file",
        default=None,
        dest="description_file",
        metavar="PATH",
        help="数据集描述 TXT（UTF-8）；与 --description 二选一",
    )
    parser.add_argument(
        "--request",
        required=True,
        metavar="TEXT",
        help="用户指令，描述本次任务目标（例如：请执行全流程分析并选出最优模型）",
    )
    parser.add_argument(
        "--target",
        default="",
        metavar="COLUMN",
        help="目标列名；与 --task-type 同时提供时写入 state，Agent Manager 跳过 infer_task",
    )
    parser.add_argument(
        "--task-type",
        default="",
        dest="task_type",
        metavar="classification|regression",
        help="与 --target 成对使用：classification 或 regression",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="导出流水线图：artifacts/graphs/full_pipeline.mmd 与 full_pipeline.png（可粘贴 mermaid.live 查看）",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=200,
        metavar="N",
        help="LangGraph 最大递归步数（默认 200）",
    )
    return parser


def _cli_task_meta_or_exit(args: argparse.Namespace) -> tuple[str, str]:
    tc = (getattr(args, "target", "") or "").strip()
    tt = (getattr(args, "task_type", "") or "").strip().lower()
    if not tc and not tt:
        return "", ""
    if not tc or not tt:
        print(
            "[!] --target 与 --task-type 须同时提供或同时省略。",
            file=sys.stderr,
        )
        sys.exit(2)
    if tt not in {"classification", "regression"}:
        print(
            f"[!] --task-type 只能是 classification 或 regression，收到：{args.task_type!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    return tc, tt


def _description_from_cli_or_exit(args: argparse.Namespace) -> str:
    """CLI：--description 与 --description-file 二选一，由 mutually_exclusive_group 保证。"""
    path = getattr(args, "description_file", None)
    if path:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            print(f"[!] --description-file 不是有效文件：{p}", file=sys.stderr)
            sys.exit(2)
        if p.suffix.lower() != ".txt":
            print(
                "[!] --description-file 目前仅支持 .txt 后缀。",
                file=sys.stderr,
            )
            sys.exit(2)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[!] 无法读取描述文件：{exc}", file=sys.stderr)
            sys.exit(2)
        stripped = text.strip()
        if not stripped:
            print("[!] 描述文件为空或仅空白。", file=sys.stderr)
            sys.exit(2)
        return stripped
    raw = args.description
    if raw is None:
        print("[!] 必须提供 --description 或 --description-file。", file=sys.stderr)
        sys.exit(2)
    return str(raw).strip()


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    tc_cli, tt_cli = _cli_task_meta_or_exit(args)
    desc_text = _description_from_cli_or_exit(args)
    train_p = (args.train or "").strip()
    test_p = (args.test or "").strip()
    data_p = (args.data or "").strip()

    if train_p or test_p:
        if not train_p or not test_p:
            print(
                "[!] 预切分模式须同时提供 --train 与 --test。",
                file=sys.stderr,
            )
            sys.exit(2)
        p_train = Path(train_p).expanduser().resolve()
        p_test = Path(test_p).expanduser().resolve()
        for label, p in (("--train", p_train), ("--test", p_test)):
            if not p.is_file():
                print(f"[!] {label} 不是有效文件：{p}", file=sys.stderr)
                sys.exit(2)
        data_path_eff = str(p_train)
        run_pipeline(
            data_path=data_path_eff,
            description=desc_text,
            user_request=args.request,
            show_graph=args.show_graph,
            recursion_limit=args.recursion_limit,
            training_data_path=str(p_train),
            testing_data_path=str(p_test),
            target_column=tc_cli,
            task_type=tt_cli,
        )
    else:
        if not data_p:
            print("[!] 请提供 --data，或同时提供 --train 与 --test。", file=sys.stderr)
            sys.exit(2)
        p_data = Path(data_p).expanduser().resolve()
        if not p_data.is_file():
            print(f"[!] --data 不是有效文件：{p_data}", file=sys.stderr)
            sys.exit(2)
        run_pipeline(
            data_path=str(p_data),
            description=desc_text,
            user_request=args.request,
            show_graph=args.show_graph,
            recursion_limit=args.recursion_limit,
            target_column=tc_cli,
            task_type=tt_cli,
        )
