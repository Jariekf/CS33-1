import json

from langchain_core.tools import tool

from core.dataset_store import get_dataset, update_dataset
from tools.tool_trace import trace_tool_call


@tool
def clean_drop_columns(dataset_id: str, columns: list[str], strict: bool = False) -> str:
    """Drop the specified columns. Missing columns are ignored by default; when strict=True, an error is raised for missing columns."""
    tool_inputs = {"dataset_id": dataset_id, "columns": columns, "strict": strict}
    try:
        df = get_dataset(dataset_id).copy()
        requested_columns = list(dict.fromkeys(columns or []))
        dropped_columns = [col for col in requested_columns if col in df.columns]
        missing_columns = [col for col in requested_columns if col not in df.columns]

        if strict and missing_columns:
            raise KeyError(f"columns_not_found: {missing_columns}")

        if dropped_columns:
            df = df.drop(columns=dropped_columns)

        update_dataset(dataset_id, df)
        report = {
            "dataset_id": dataset_id,
            "requested_columns": requested_columns,
            "dropped_columns": dropped_columns,
            "missing_columns": missing_columns,
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": list(df.columns),
        }
    except Exception as exc:
        report = {"error": {"type": exc.__class__.__name__, "message": str(exc)}}

    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("clean_drop_columns", tool_inputs, output)
    return output
