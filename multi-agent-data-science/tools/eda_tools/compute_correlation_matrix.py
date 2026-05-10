import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


@tool
def compute_correlation_matrix(file_path: str, target_column: str = "") -> str:
    """计算训练集数值列的 Pearson 相关矩阵。

    参数：
      file_path (str)：训练集 CSV 文件路径。
      target_column (str)：目标列名。若提供且存在于数值列中，则额外输出
        各特征与目标列的相关系数绝对值排名（从高到低），帮助判断特征重要性。

    返回 JSON，包含：
      columns：参与计算的列名列表。
      matrix：完整相关矩阵（dict of dict）。
      target_correlation_ranking：各特征与目标列相关度排名（仅当 target_column 有效时存在）。
    """
    tool_inputs = {"file_path": file_path, "target_column": target_column}
    try:
        df = pd.read_csv(file_path)
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            raise ValueError("训练集中没有数值列，无法计算相关矩阵。")

        corr = numeric_df.corr().round(4)
        result: dict[str, Any] = {
            "columns": list(corr.columns),
            "matrix": corr.to_dict(),
        }

        if target_column and target_column in corr.columns:
            ranking = (
                corr[target_column]
                .drop(target_column)
                .abs()
                .sort_values(ascending=False)
                .round(4)
                .to_dict()
            )
            result["target_correlation_ranking"] = ranking

        output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("compute_correlation_matrix", tool_inputs, output)
    return output
