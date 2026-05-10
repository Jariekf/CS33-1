import json
from typing import Any

from langchain_core.tools import tool

from core.dataset_store import get_dataset, update_dataset
from tools.tool_trace import trace_tool_call


@tool
def clean_missing_values(
    dataset_id: str,
    actions: list[dict[str, Any]],
    fallback_strategy: str = "fill_zero",
) -> str:
    """处理缺失值，支持 fill 和 drop_rows，并对未覆盖列自动兜底。

    参数：
      dataset_id (str)：内存中的数据集 ID。

      actions (list)：针对特定列的缺失值处理规则，每条包含：
        - column   (str)：列名。
        - strategy (str)："fill"（填充固定值）或 "drop_rows"（删除含缺失的行）。
        - value    (any)：strategy 为 "fill" 时的填充值（数值列常用 0 或均值）。

      fallback_strategy (str)：actions 处理完后，对仍有缺失值的列自动兜底。
        - "fill_zero"   （默认）：数值列填 0，文本列填空字符串。
        - "fill_median" ：数值列填中位数，文本列填众数。
        - "none"        ：不做兜底，保留剩余 NaN。
        建议始终使用默认值 "fill_zero"，防止类型转换产生的 NaN 漏网。
    """
    tool_inputs = {"dataset_id": dataset_id, "actions": actions, "fallback_strategy": fallback_strategy}
    filled_columns = []
    dropped_rows = 0
    skipped_columns = []
    fallback_filled: dict[str, Any] = {}
    try:
        df = get_dataset(dataset_id).copy()

        for action in actions:
            column = action.get("column")
            strategy = action.get("strategy")
            if column not in df.columns:
                skipped_columns.append({"column": column, "reason": "column_not_found"})
                continue
            if strategy == "fill":
                df[column] = df[column].fillna(action.get("value"))
                filled_columns.append(column)
            elif strategy == "drop_rows":
                before_rows = len(df)
                df = df.dropna(subset=[column])
                dropped_rows += before_rows - len(df)
            else:
                skipped_columns.append(
                    {"column": column, "reason": f"unsupported_strategy:{strategy}"}
                )

        if fallback_strategy != "none":
            for col in df.columns[df.isna().any()]:
                if fallback_strategy == "fill_zero":
                    fill_val = 0 if df[col].dtype.kind in "iufcb" else ""
                else:  # fill_median
                    if df[col].dtype.kind in "iufcb":
                        fill_val = df[col].median()
                    else:
                        mode = df[col].mode()
                        fill_val = mode.iloc[0] if not mode.empty else ""
                df[col] = df[col].fillna(fill_val)
                fallback_filled[col] = fill_val

        update_dataset(dataset_id, df)
        report = {
            "dataset_id": dataset_id,
            "filled_columns": filled_columns,
            "dropped_rows": dropped_rows,
            "skipped_columns": skipped_columns,
            "fallback_filled": fallback_filled,
        }
    except Exception as exc:
        report = {"error": {"type": exc.__class__.__name__, "message": str(exc)}}

    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("clean_missing_values", tool_inputs, output)
    return output
