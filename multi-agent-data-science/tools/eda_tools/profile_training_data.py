import json

from langchain_core.tools import tool

from tools.agentmanager_tools.profile_dataset import (
    _build_dataset_summary,
    _build_sample_rows,
    _load_dataset,
    _profile_drop_candidates,
    _profile_missing_values,
    _profile_outliers,
    _profile_type_issues,
)
from tools.tool_trace import trace_tool_call


@tool
def profile_training_data(file_path: str, n_rows: int = 5) -> str:
    """为训练集生成数据画像，包含缺失值、异常值、类型问题和删除候选列统计。

    参数：
      file_path (str)：训练集 CSV 文件路径（即 state 中的 training_data_path）。
      n_rows (int)：返回的样例行数，默认 5。

    返回 JSON，包含 dataset、missing_values、outliers、type_issues、drop_candidates、sample_rows。
    """
    tool_inputs = {"file_path": file_path, "n_rows": n_rows}
    try:
        df = _load_dataset(file_path)
        profile = {
            "dataset": _build_dataset_summary(df, file_path),
            "missing_values": _profile_missing_values(df),
            "outliers": _profile_outliers(df),
            "type_issues": _profile_type_issues(df),
            "drop_candidates": _profile_drop_candidates(df),
            "sample_rows": _build_sample_rows(df, n_rows),
        }
        result = json.dumps(profile, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        result = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("profile_training_data", tool_inputs, result)
    return result
