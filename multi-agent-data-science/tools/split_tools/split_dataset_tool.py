import json
from pathlib import Path

from langchain_core.tools import tool

from core.dataset_split import perform_train_test_file_split
from tools.tool_trace import trace_tool_call


@tool
def split_dataset(
    data_path: str,
    task_type: str,
    target_column: str,
    run_dir: str,
) -> str:
    """将数据集划分为训练集和测试集。分类任务使用分层抽样，回归任务随机划分。

    Args:
        data_path: 数据集 CSV 文件路径。
        task_type: 任务类型，"classification" 或 "regression"。
        target_column: 目标列名称（分类任务必须提供）。
        run_dir: 本次运行输出目录，用于存放 train/test 文件。
    """
    source = Path(data_path).expanduser().resolve()
    if not source.is_file():
        result = {"error": f"File not found: {data_path}"}
        return json.dumps(result, ensure_ascii=False)

    if run_dir:
        out_dir = Path(run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = source.parent / "workspace"
        out_dir.mkdir(parents=True, exist_ok=True)

    train_out = out_dir / f"{source.stem}_train.csv"
    test_out = out_dir / f"{source.stem}_test.csv"

    task_type_norm = (task_type or "unknown").strip().lower()
    if task_type_norm not in {"classification", "regression"}:
        task_type_norm = "regression"

    try:
        report = perform_train_test_file_split(
            source_path=source,
            train_out=train_out,
            test_out=test_out,
            task_type=task_type_norm,
            target_column=target_column or None,
        )
    except Exception as exc:
        result = {"error": str(exc)}
        return json.dumps(result, ensure_ascii=False)

    output = json.dumps(report, ensure_ascii=False, indent=2)
    trace_tool_call("split_dataset", {"data_path": data_path, "task_type": task_type, "target_column": target_column, "run_dir": run_dir}, output)
    return output
