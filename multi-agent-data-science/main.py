"""项目启动入口：运行完整多智能体数据科学流水线。

用法：
  python main.py --data datasets/bank-full.csv --description "二分类任务，预测客户是否订阅定期存款"
  python main.py --data datasets/my_data.csv --description "回归任务，预测房价" --show-graph
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage

from agent.agent_manager import _pretty_print_message
from core.graph import build_full_pipeline

_DIVIDER = "=" * 60


def run_pipeline(
    data_path: str,
    description: str,
    user_request: str,
    show_graph: bool = False,
    recursion_limit: int = 200,
) -> dict:
    """运行完整多智能体数据科学流水线并返回最终 state。

    Args:
        data_path:       原始数据集路径（CSV）。
        description:     数据集描述，帮助 Agent 理解任务背景和目标列。
        user_request:    用户指令，描述本次任务目标。
        show_graph:      是否将流水线结构图保存为 PNG（artifacts/full_pipeline.png）。
        recursion_limit: LangGraph 最大递归步数，流程较长时可适当调大。

    Returns:
        流水线结束时的完整 GlobalState dict。
    """
    pipeline = build_full_pipeline(show_graph=show_graph)

    print(f"\n{_DIVIDER}")
    print("多智能体数据科学流水线 — 启动")
    print(f"数据集：{data_path}")
    print(_DIVIDER)

    final_state: dict = {}

    for event in pipeline.stream(
        {
            "messages": [HumanMessage(content=user_request)],
            "user_request": user_request,
            "data_path": data_path,
            "description": description,
        },
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

    print(f"\n{_DIVIDER}")
    print("流水线运行完毕")
    if run_dir := final_state.get("run_dir"):
        print(f"输出目录：{run_dir}")
    if bm := final_state.get("best_model"):
        print(f"最优模型：{bm.get('model_name')}")
        print(f"评估指标：{bm.get('metrics')}")
    print(_DIVIDER)

    return final_state


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="多智能体数据科学流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py --data datasets/bank-full.csv \\
                 --description "二分类任务，预测客户是否订阅银行定期存款（目标列 y）" \\
                 --request "请执行全流程分析，清洗数据、切分数据集、特征工程并训练模型，选出最优分类模型"

  python main.py --data datasets/housing.csv \\
                 --description "回归任务，预测房价（目标列 price）" \\
                 --request "请分析数据并训练回归模型，选出 RMSE 最低的模型" \\
                 --show-graph --recursion-limit 300
""",
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="原始数据集路径（CSV 文件）",
    )
    parser.add_argument(
        "--description",
        required=True,
        metavar="TEXT",
        help="数据集描述，包含任务类型、目标列名等背景信息",
    )
    parser.add_argument(
        "--request",
        required=True,
        metavar="TEXT",
        help="用户指令，描述本次任务目标（例如：请执行全流程分析并选出最优模型）",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="将流水线结构图保存为 artifacts/full_pipeline.png",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=200,
        metavar="N",
        help="LangGraph 最大递归步数（默认 200）",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_pipeline(
        data_path=args.data,
        description=args.description,
        user_request=args.request,
        show_graph=args.show_graph,
        recursion_limit=args.recursion_limit,
    )
