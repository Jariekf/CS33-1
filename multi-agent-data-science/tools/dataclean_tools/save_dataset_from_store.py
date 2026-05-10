import json
from pathlib import Path

from langchain_core.tools import tool

from core.dataset_store import get_dataset
from tools.tool_trace import trace_tool_call


@tool
def save_dataset_from_store(dataset_id: str, output_path: str) -> str:
    """将内存 dataset_store 中的数据集保存为 CSV 文件。"""
    tool_inputs = {"dataset_id": dataset_id, "output_path": output_path}
    try:
        df = get_dataset(dataset_id)
        final_output_path = Path(output_path).expanduser().resolve()
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(final_output_path, index=False)
        report = {
            "dataset_id": dataset_id,
            "output_path": str(final_output_path),
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": list(df.columns),
        }
    except Exception as exc:
        report = {"error": {"type": exc.__class__.__name__, "message": str(exc)}}

    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("save_dataset_from_store", tool_inputs, output)
    return output
