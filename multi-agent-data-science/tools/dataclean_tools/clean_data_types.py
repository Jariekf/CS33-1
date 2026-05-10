import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.dataset_store import get_dataset, update_dataset
from tools.tool_trace import trace_tool_call


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    # 独立的 "-" 表示无数据，转为 NaN；不能在上面统一替换，否则会把负号也抹掉
    cleaned = cleaned.replace({"-": None, "": None, "nan": None, "None": None})
    return pd.to_numeric(cleaned, errors="coerce")


@tool
def clean_data_types(dataset_id: str, conversions: list[dict[str, Any]]) -> str:
    """修正数据类型，支持 numeric 和 datetime。"""
    tool_inputs = {"dataset_id": dataset_id, "conversions": conversions}
    converted_columns = []
    skipped_columns = []
    try:
        df = get_dataset(dataset_id).copy()
        for conversion in conversions:
            column = conversion.get("column")
            target_type = conversion.get("target_type")
            if column not in df.columns:
                skipped_columns.append({"column": column, "reason": "column_not_found"})
                continue
            if target_type == "numeric":
                df[column] = _clean_numeric_series(df[column])
            elif target_type == "datetime":
                df[column] = pd.to_datetime(df[column], errors="coerce")
            else:
                skipped_columns.append(
                    {"column": column, "reason": f"unsupported_target_type:{target_type}"}
                )
                continue
            converted_columns.append(column)
        update_dataset(dataset_id, df)
        report = {
            "dataset_id": dataset_id,
            "converted_columns": converted_columns,
            "skipped_columns": skipped_columns,
        }
    except Exception as exc:
        report = {"error": {"type": exc.__class__.__name__, "message": str(exc)}}

    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    trace_tool_call("clean_data_types", tool_inputs, output)
    return output
