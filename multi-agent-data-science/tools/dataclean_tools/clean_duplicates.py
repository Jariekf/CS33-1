import json

from langchain_core.tools import tool

from core.dataset_store import get_dataset, update_dataset
from tools.tool_trace import trace_tool_call


@tool
def clean_duplicates(
    dataset_id: str,
    subset: list[str] | None = None,
    keep: str = "first",
) -> str:
    """Handle duplicate data, supporting full-row deduplication or deduplication by subset."""
    tool_inputs = {"dataset_id": dataset_id, "subset": subset, "keep": keep}
    try:
        df = get_dataset(dataset_id).copy()
        before_rows = len(df)
        final_subset = subset or None
        df = df.drop_duplicates(subset=final_subset, keep=keep)
        dropped_rows = before_rows - len(df)
        update_dataset(dataset_id, df)
        report = {
            "dataset_id": dataset_id,
            "subset": final_subset,
            "keep": keep,
            "dropped_rows": dropped_rows,
            "rows": int(len(df)),
        }
    except Exception as exc:
        report = {"error": {"type": exc.__class__.__name__, "message": str(exc)}}

    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("clean_duplicates", tool_inputs, output)
    return output
