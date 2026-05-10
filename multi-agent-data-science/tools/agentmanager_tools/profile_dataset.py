import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


def _load_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
    raise ValueError(f"不支持的文件类型: {suffix}。支持: {supported}")


def _build_dataset_summary(df: pd.DataFrame, file_path: str) -> dict[str, Any]:
    return {
        "file_path": str(Path(file_path).expanduser().resolve()),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [str(col) for col in df.columns],
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
    }


def _profile_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    total_cells = int(df.shape[0] * df.shape[1])
    missing_counts = df.isna().sum()
    columns = {
        str(col): {
            "missing_count": int(count),
            "missing_ratio": round(float(count / len(df)), 6) if len(df) else 0.0,
        }
        for col, count in missing_counts.items()
    }

    high_missing_columns = [
        str(col)
        for col, count in missing_counts.items()
        if len(df) and float(count / len(df)) >= 0.3
    ]

    return {
        "total_missing_cells": int(missing_counts.sum()),
        "total_missing_ratio": round(float(missing_counts.sum() / total_cells), 6)
        if total_cells
        else 0.0,
        "high_missing_columns": high_missing_columns,
        "columns": columns,
    }


def _profile_outliers(df: pd.DataFrame) -> dict[str, Any]:
    numeric_df = df.select_dtypes(include="number")
    columns: dict[str, Any] = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(mask.sum())

        columns[str(col)] = {
            "outlier_count": outlier_count,
            "outlier_ratio": round(float(outlier_count / len(df)), 6) if len(df) else 0.0,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    return {
        "method": "iqr",
        "columns": columns,
    }


def _profile_type_issues(df: pd.DataFrame) -> dict[str, Any]:
    numeric_like_object_columns: list[str] = []
    datetime_like_object_columns: list[str] = []
    category_columns: dict[str, Any] = {}
    datetime_keywords = ("date", "time", "datetime", "created", "updated", "timestamp")

    object_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in object_columns:
        series = df[col].dropna()
        if series.empty:
            continue

        numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
        if numeric_ratio >= 0.6:
            numeric_like_object_columns.append(str(col))

        col_name = str(col).lower()
        if any(keyword in col_name for keyword in datetime_keywords):
            datetime_ratio = pd.to_datetime(series, errors="coerce").notna().mean()
            if datetime_ratio >= 0.6:
                datetime_like_object_columns.append(str(col))

        unique_count = int(series.nunique(dropna=True))
        category_columns[str(col)] = {
            "unique_count": unique_count,
            "unique_ratio": round(float(unique_count / len(df)), 6) if len(df) else 0.0,
        }

    high_cardinality_columns = [
        col
        for col, info in category_columns.items()
        if info["unique_ratio"] >= 0.5 and info["unique_count"] > 20
    ]

    return {
        "numeric_like_object_columns": numeric_like_object_columns,
        "datetime_like_object_columns": datetime_like_object_columns,
        "category_columns": category_columns,
        "high_cardinality_columns": high_cardinality_columns,
    }


def _profile_drop_candidates(df: pd.DataFrame) -> dict[str, Any]:
    empty_columns = [str(col) for col in df.columns if df[col].isna().all()]
    constant_columns = [
        str(col) for col in df.columns if df[col].nunique(dropna=False) <= 1
    ]

    id_keywords = ("id", "uuid", "key")
    id_like_columns = []
    for col in df.columns:
        col_name = str(col).lower()
        unique_ratio = df[col].nunique(dropna=True) / len(df) if len(df) else 0.0
        if unique_ratio >= 0.9 and any(keyword in col_name for keyword in id_keywords):
            id_like_columns.append(str(col))

    return {
        "empty_columns": empty_columns,
        "constant_columns": constant_columns,
        "id_like_columns": id_like_columns,
    }


def _build_sample_rows(df: pd.DataFrame, n_rows: int) -> list[dict[str, Any]]:
    head_rows = max(1, min(int(n_rows), 20))
    sample = df.head(head_rows).where(pd.notna(df.head(head_rows)), None)
    return sample.to_dict(orient="records")


@tool
def profile_dataset(file_path: str, n_rows: int = 5) -> str:
    """生成数据集质量画像，包含缺失值、异常值、类型问题和删除候选列。"""
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
        error = {
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
        }
        result = json.dumps(error, ensure_ascii=False, indent=2)

    trace_tool_call("profile_dataset", tool_inputs, result)
    return result
