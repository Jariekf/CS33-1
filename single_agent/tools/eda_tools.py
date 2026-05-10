"""Deterministic EDA tools with leakage-aware train-only target analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_eda_report(
    cleaned_df: pd.DataFrame,
    train_df: pd.DataFrame,
    target_col: str | None,
    task_type: str,
) -> dict[str, Any]:
    """Build an EDA report.

    Data quality summaries use the cleaned dataset. Any feature-target
    relationship analysis uses train_df only to avoid test-set leakage.
    """

    report: dict[str, Any] = {
        "stage": "eda_analysis",
        "status": "completed",
        "leakage_policy": {
            "data_quality_scope": "cleaned_data",
            "feature_target_analysis_scope": "train_data_only",
            "test_data_used_for_feature_target_analysis": False,
        },
        "dataset_overview": _dataset_overview(cleaned_df),
        "feature_quality": _feature_quality(cleaned_df, target_col),
        "target_analysis": _target_analysis(train_df, target_col, task_type),
        "feature_target_relationships": _feature_target_relationships(train_df, target_col, task_type),
    }
    report["feature_engineering_recommendations"] = _feature_engineering_recommendations(report, task_type)
    return report


def build_eda_summary(report: dict[str, Any]) -> str:
    """Create a compact markdown EDA summary for business reporting."""

    overview = report.get("dataset_overview", {})
    target = report.get("target_analysis", {})
    quality = report.get("feature_quality", {})
    relationships = report.get("feature_target_relationships", {})
    recommendations = report.get("feature_engineering_recommendations", [])
    lines = [
        "# EDA Summary",
        "",
        "## Dataset Overview",
        f"- Rows: {overview.get('rows')}",
        f"- Columns: {overview.get('columns')}",
        f"- Numeric features: {len(overview.get('numeric_columns', []))}",
        f"- Categorical features: {len(overview.get('categorical_columns', []))}",
        "",
        "## Target Analysis",
        f"- Target column: {target.get('target_column')}",
        f"- Task type: {target.get('task_type')}",
        f"- Summary: {target.get('summary')}",
        "",
        "## Feature Quality",
        f"- High-missing columns: {quality.get('high_missing_columns', [])}",
        f"- High-cardinality columns: {quality.get('high_cardinality_columns', [])}",
        f"- ID-like columns: {quality.get('id_like_columns', [])}",
        "",
        "## Top Feature Relationships",
    ]
    for item in relationships.get("top_numeric_relationships", [])[:5]:
        lines.append(f"- {item['feature']}: {item['score_name']}={item['score']}")
    for item in relationships.get("top_categorical_relationships", [])[:5]:
        lines.append(f"- {item['feature']}: {item['score_name']}={item['score']}")
    if not relationships.get("top_numeric_relationships") and not relationships.get("top_categorical_relationships"):
        lines.append("- No feature-target relationships were available.")
    lines.extend(["", "## Feature Engineering Recommendations"])
    lines.extend([f"- {item}" for item in recommendations] or ["- No additional recommendations."])
    return "\n".join(lines)


def _dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize the cleaned dataset without using target relationships."""

    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(exclude="number").columns)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_ratio": df.isna().mean().round(4).to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _feature_quality(df: pd.DataFrame, target_col: str | None) -> dict[str, Any]:
    """Detect feature quality risks from cleaned data."""

    feature_cols = [col for col in df.columns if col != target_col]
    high_missing = [
        col for col in feature_cols
        if float(df[col].isna().mean()) >= 0.3
    ]
    high_cardinality = [
        col for col in feature_cols
        if df[col].dtype == "object" and df[col].nunique(dropna=True) > 20
    ]
    id_like = [
        col for col in feature_cols
        if _is_identifier_like_column(col) or df[col].nunique(dropna=True) / max(len(df), 1) >= 0.8
    ]
    constant = [
        col for col in feature_cols
        if df[col].nunique(dropna=False) <= 1
    ]
    return {
        "high_missing_columns": high_missing,
        "high_cardinality_columns": high_cardinality,
        "id_like_columns": id_like,
        "constant_columns": constant,
    }


def _target_analysis(train_df: pd.DataFrame, target_col: str | None, task_type: str) -> dict[str, Any]:
    """Analyze target distribution using train data only."""

    if not target_col or target_col not in train_df.columns:
        return {"target_column": target_col, "task_type": task_type, "summary": "target column unavailable"}
    target = train_df[target_col]
    if task_type == "classification":
        distribution = target.value_counts(dropna=False).to_dict()
        total = max(len(target), 1)
        majority_ratio = float(target.value_counts(dropna=False).iloc[0] / total) if len(target) else 0.0
        return {
            "target_column": target_col,
            "task_type": task_type,
            "class_distribution": distribution,
            "majority_class_ratio": round(majority_ratio, 4),
            "summary": f"{len(distribution)} classes; majority ratio {majority_ratio:.2f}",
        }
    numeric = pd.to_numeric(target, errors="coerce")
    return {
        "target_column": target_col,
        "task_type": task_type,
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "skew": float(numeric.skew()),
        "summary": f"mean={numeric.mean():.3f}, median={numeric.median():.3f}, skew={numeric.skew():.3f}",
    }


def _feature_target_relationships(train_df: pd.DataFrame, target_col: str | None, task_type: str) -> dict[str, Any]:
    """Rank simple feature-target relationships using train data only."""

    if not target_col or target_col not in train_df.columns:
        return {"top_numeric_relationships": [], "top_categorical_relationships": []}
    feature_cols = [col for col in train_df.columns if col != target_col]
    numeric_items: list[dict[str, Any]] = []
    categorical_items: list[dict[str, Any]] = []
    if task_type == "classification":
        target_codes = train_df[target_col].astype("category").cat.codes
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                corr = train_df[col].corr(target_codes)
                if pd.notna(corr):
                    numeric_items.append({"feature": col, "score_name": "abs_corr_with_target_code", "score": round(abs(float(corr)), 4)})
            else:
                score = _classification_categorical_spread(train_df, col, target_col)
                if score is not None:
                    categorical_items.append({"feature": col, "score_name": "target_rate_spread", "score": round(score, 4)})
    else:
        target_numeric = pd.to_numeric(train_df[target_col], errors="coerce")
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                corr = train_df[col].corr(target_numeric)
                if pd.notna(corr):
                    numeric_items.append({"feature": col, "score_name": "abs_corr", "score": round(abs(float(corr)), 4)})
            else:
                score = _regression_categorical_mean_spread(train_df, col, target_col)
                if score is not None:
                    categorical_items.append({"feature": col, "score_name": "target_mean_spread", "score": round(score, 4)})
    return {
        "top_numeric_relationships": sorted(numeric_items, key=lambda item: item["score"], reverse=True)[:10],
        "top_categorical_relationships": sorted(categorical_items, key=lambda item: item["score"], reverse=True)[:10],
    }


def _feature_engineering_recommendations(report: dict[str, Any], task_type: str) -> list[str]:
    """Generate deterministic feature engineering recommendations from EDA."""

    quality = report.get("feature_quality", {})
    target = report.get("target_analysis", {})
    recommendations: list[str] = []
    if quality.get("high_missing_columns"):
        recommendations.append("Create missing indicators and impute high-missingness columns using train-fitted values.")
    if quality.get("high_cardinality_columns"):
        recommendations.append("Use frequency encoding or drop high-cardinality categorical columns instead of wide one-hot encoding.")
    if quality.get("id_like_columns"):
        recommendations.append("Drop ID-like columns from model features to reduce memorization risk.")
    if quality.get("constant_columns"):
        recommendations.append("Remove constant or near-constant features before modelling.")
    if task_type == "classification" and target.get("majority_class_ratio", 0) >= 0.8:
        recommendations.append("Use imbalance-aware evaluation and consider class weights or resampling.")
    if task_type == "regression" and abs(float(target.get("skew", 0) or 0)) >= 1.0:
        recommendations.append("Consider robust metrics or transformations for highly skewed regression targets.")
    return recommendations


def _classification_categorical_spread(df: pd.DataFrame, col: str, target_col: str) -> float | None:
    """Compute spread in positive-class rate across categories for binary targets."""

    if df[target_col].nunique(dropna=True) != 2:
        return None
    target_codes = df[target_col].astype("category").cat.codes
    grouped = target_codes.groupby(df[col]).mean()
    if grouped.empty:
        return None
    return float(grouped.max() - grouped.min())


def _regression_categorical_mean_spread(df: pd.DataFrame, col: str, target_col: str) -> float | None:
    """Compute normalized target mean spread across categories."""

    target = pd.to_numeric(df[target_col], errors="coerce")
    grouped = target.groupby(df[col]).mean()
    if grouped.empty:
        return None
    denom = target.std()
    if pd.isna(denom) or denom == 0:
        return None
    return float((grouped.max() - grouped.min()) / denom)


def _is_identifier_like_column(col: str) -> bool:
    """Return whether a feature name looks like an identifier."""

    lowered = str(col).lower()
    return any(token in lowered for token in ("id", "code", "zip", "postal", "phone"))
