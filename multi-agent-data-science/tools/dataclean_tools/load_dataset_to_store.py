import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from core.dataset_store import put_dataset
from tools.tool_trace import trace_tool_call


@tool
def load_dataset_to_store(file_path: str, dataset_id: str = "") -> str:
    """读取 CSV 数据集到内存 dataset_store，并返回 dataset_id。"""
    tool_inputs = {"file_path": file_path, "dataset_id": dataset_id}
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"路径不是文件: {path}")

        df = pd.read_csv(path, sep=None, engine="python")
        final_dataset_id = put_dataset(df, dataset_id=dataset_id or None)
        report = {
            "dataset_id": final_dataset_id,
            "source_path": str(path),
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": list(df.columns),
        }
    except Exception as exc:
        report = {"error": {"type": exc.__class__.__name__, "message": str(exc)}}

    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("load_dataset_to_store", tool_inputs, output)
    return output
