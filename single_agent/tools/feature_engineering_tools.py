"""Feature policy and feature transformation tools."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler


HIGH_CARDINALITY_THRESHOLD = 20
ID_UNIQUENESS_RATIO = 0.8


def recommend_feature_engineering_policy(
    task_type: str,
    candidate_models: list[dict[str, Any]],
    data_profile: dict[str, Any],
) -> dict[str, Any]:
    """Recommend skip/light/standard/task-specific feature engineering."""

    dtypes = data_profile.get("dtypes", {})
    has_object = any(dtype == "object" for dtype in dtypes.values())
    has_datetime = any("datetime" in dtype for dtype in dtypes.values())
    model_ids = [model.get("model_id", "") for model in candidate_models]
    scaling_required = any(model.get("preprocessing", {}).get("numeric_scaling") for model in candidate_models)
    encoding_required = any(
        model.get("preprocessing", {}).get("requires_all_features_numeric") for model in candidate_models
    )
    if task_type in {"forecasting", "time_series"}:
        mode = "task_specific"
        reason = "Time-series tasks need calendar, lag, or rolling features when time columns are available."
    elif not has_object and not has_datetime and not scaling_required:
        mode = "skip"
        reason = "Cleaned numeric data satisfies candidate model preprocessing requirements."
    elif encoding_required and not scaling_required:
        mode = "light"
        reason = "Candidate models require numeric encoded features but do not require numeric scaling."
    else:
        mode = "standard"
        reason = "At least one candidate model requires encoded numeric features and numeric scaling."
    return {
        "mode": mode,
        "reason": reason,
        "task_type": task_type,
        "candidate_model_ids": model_ids,
        "candidate_model_requirements": {
            "encoding_required": encoding_required,
            "scaling_required": scaling_required,
        },
    }


def create_missing_indicators(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add missing-value indicator columns based on training columns."""

    train_out, test_out = train_df.copy(), test_df.copy()
    for col in train_df.columns:
        if col not in test_df.columns:
            continue
        if train_df[col].isna().any() or test_df[col].isna().any():
            indicator = f"{col}_was_missing"
            train_out[indicator] = train_df[col].isna().astype(int)
            test_out[indicator] = test_df[col].isna().astype(int)
    return train_out, test_out


def profile_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Profile train/test feature columns before model-specific transformations."""

    feature_cols = [col for col in train_df.columns if col != target_col]
    profile: dict[str, Any] = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "feature_count": len(feature_cols),
        "numeric_features": [],
        "categorical_features": [],
        "datetime_like_features": [],
        "boolean_like_features": [],
        "id_like_features": [],
        "high_cardinality_features": [],
        "missing_ratio_train": train_df[feature_cols].isna().mean().round(4).to_dict() if feature_cols else {},
        "missing_ratio_test": test_df[feature_cols].isna().mean().round(4).to_dict() if feature_cols else {},
    }
    row_count = max(len(train_df), 1)
    for col in feature_cols:
        series = train_df[col]
        unique_count = int(series.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(series):
            profile["numeric_features"].append(col)
        elif _is_boolean_like_series(series):
            profile["boolean_like_features"].append(col)
        else:
            profile["categorical_features"].append(col)
        if _is_datetime_like_column(col, series):
            profile["datetime_like_features"].append(col)
        if _is_identifier_like_column(col) or unique_count / row_count >= ID_UNIQUENESS_RATIO:
            profile["id_like_features"].append(col)
        elif unique_count > HIGH_CARDINALITY_THRESHOLD:
            profile["high_cardinality_features"].append(col)
    return profile


def create_missing_indicators_with_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Add missing indicators and report which columns received indicators."""

    before_cols = set(train_df.columns)
    feature_cols = [col for col in train_df.columns if col != target_col]
    train_out, test_out = train_df.copy(), test_df.copy()
    source_columns: list[str] = []
    for col in feature_cols:
        if train_out[col].isna().any() or test_out[col].isna().any():
            indicator = f"{col}_was_missing"
            train_out[indicator] = train_out[col].isna().astype(int)
            test_out[indicator] = test_out[col].isna().astype(int)
            source_columns.append(col)
    return (
        train_out,
        test_out,
        {
            "source_columns": source_columns,
            "created_columns": [col for col in train_out.columns if col not in before_cols],
        },
    )


def create_datetime_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create simple calendar features from parseable datetime columns."""

    train_out, test_out = train_df.copy(), test_df.copy()
    for col in list(train_df.columns):
        if col == target_col:
            continue
        name_hint = col.lower()
        may_be_datetime = (
            pd.api.types.is_datetime64_any_dtype(train_df[col])
            or train_df[col].dtype == "object"
            and any(token in name_hint for token in ["date", "time", "day", "month", "year"])
        )
        if not may_be_datetime:
            continue
        train_dt = _parse_datetime_safely(train_df[col])
        test_dt = _parse_datetime_safely(test_df[col])
        if train_dt.notna().mean() < 0.8:
            continue
        for part, values in {
            "year": (train_dt.dt.year, test_dt.dt.year),
            "month": (train_dt.dt.month, test_dt.dt.month),
            "dayofweek": (train_dt.dt.dayofweek, test_dt.dt.dayofweek),
        }.items():
            train_out[f"{col}_{part}"] = values[0]
            test_out[f"{col}_{part}"] = values[1]
        train_out = train_out.drop(columns=[col])
        test_out = test_out.drop(columns=[col])
    return train_out, test_out


def create_datetime_features_with_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create calendar features from datetime-like columns and report details."""

    train_out, test_out = train_df.copy(), test_df.copy()
    transformed: dict[str, list[str]] = {}
    for col in list(train_df.columns):
        if col == target_col or not _is_datetime_like_column(col, train_df[col]):
            continue
        train_dt = _parse_datetime_safely(train_df[col])
        test_dt = _parse_datetime_safely(test_df[col])
        if train_dt.notna().mean() < 0.8:
            continue
        created: list[str] = []
        for part, values in {
            "year": (train_dt.dt.year, test_dt.dt.year),
            "month": (train_dt.dt.month, test_dt.dt.month),
            "dayofweek": (train_dt.dt.dayofweek, test_dt.dt.dayofweek),
        }.items():
            new_col = f"{col}_{part}"
            train_out[new_col] = values[0]
            test_out[new_col] = values[1]
            created.append(new_col)
        train_out = train_out.drop(columns=[col])
        test_out = test_out.drop(columns=[col])
        transformed[col] = created
    return (
        train_out,
        test_out,
        {
            "transformed_columns": list(transformed.keys()),
            "created_columns": transformed,
        },
    )


def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical feature columns and align train/test schemas."""

    train_out, test_out, _ = encode_categorical_features_with_report(train_df, test_df, target_col=target_col)
    return train_out, test_out


def encode_categorical_features_with_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
    max_one_hot_cardinality: int = HIGH_CARDINALITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Encode categorical columns with one-hot, frequency encoding, or ID-like dropping."""

    train_out, test_out = train_df.copy(), test_df.copy()
    feature_cols = [col for col in train_out.columns if col != target_col]
    cat_cols = [col for col in feature_cols if train_out[col].dtype == "object"]
    one_hot_cols: list[str] = []
    frequency_cols: list[str] = []
    dropped_cols: list[str] = []
    for col in cat_cols:
        unique_count = int(train_out[col].nunique(dropna=True))
        uniqueness_ratio = unique_count / max(len(train_out), 1)
        if _is_identifier_like_column(col) or uniqueness_ratio >= ID_UNIQUENESS_RATIO:
            train_out = train_out.drop(columns=[col])
            test_out = test_out.drop(columns=[col])
            dropped_cols.append(col)
        elif unique_count > max_one_hot_cardinality:
            frequencies = train_out[col].value_counts(normalize=True, dropna=False)
            encoded_col = f"{col}_frequency"
            train_out[encoded_col] = train_out[col].map(frequencies).fillna(0.0)
            test_out[encoded_col] = test_out[col].map(frequencies).fillna(0.0)
            train_out = train_out.drop(columns=[col])
            test_out = test_out.drop(columns=[col])
            frequency_cols.append(col)
        else:
            one_hot_cols.append(col)
    if one_hot_cols:
        train_out = pd.get_dummies(train_out, columns=one_hot_cols, dummy_na=False, dtype=int)
        test_out = pd.get_dummies(test_out, columns=one_hot_cols, dummy_na=False, dtype=int)
        train_out, test_out = train_out.align(test_out, join="left", axis=1, fill_value=0)
    else:
        train_out, test_out = train_out.align(test_out, join="left", axis=1, fill_value=0)
    return (
        train_out,
        test_out,
        {
            "one_hot_encoded_columns": one_hot_cols,
            "frequency_encoded_columns": frequency_cols,
            "dropped_identifier_like_columns": dropped_cols,
            "output_column_count": int(len(train_out.columns)),
        },
    )


def apply_feature_policy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    policy: dict[str, Any],
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the selected feature engineering policy."""

    mode = policy.get("mode", "light")
    steps: list[str] = []
    train_out, test_out = train_df.copy(), test_df.copy()
    if mode == "skip":
        return train_out, test_out, {"mode": mode, "steps": ["pass_through"]}
    train_out, test_out = create_missing_indicators(train_out, test_out)
    steps.append("create_missing_indicators")
    train_out, test_out = create_datetime_features(train_out, test_out, target_col=target_col)
    steps.append("create_datetime_features")
    train_out, test_out = encode_categorical_features(train_out, test_out, target_col=target_col)
    steps.append("encode_categorical_features")
    return train_out, test_out, {"mode": mode, "steps": steps}


def apply_model_feature_policy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_plan: dict[str, Any],
    target_col: str | None = None,
    policy: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply preprocessing required by one concrete candidate model."""

    policy = policy or {}
    preprocessing = model_plan.get("preprocessing", {})
    steps: list[str] = []
    step_reports: dict[str, Any] = {}
    train_out, test_out = train_df.copy(), test_df.copy()
    input_profile = profile_features(train_out, test_out, target_col=target_col)
    train_out, test_out, report = create_missing_indicators_with_report(train_out, test_out, target_col=target_col)
    steps.append("create_missing_indicators")
    step_reports["create_missing_indicators"] = report
    train_out, test_out, report = impute_missing_features_with_report(
        train_out,
        test_out,
        numeric_strategy=preprocessing.get("numeric_imputation", "median"),
        categorical_strategy=preprocessing.get("categorical_imputation", "mode"),
        target_col=target_col,
    )
    steps.append("impute_missing_features")
    step_reports["impute_missing_features"] = report
    train_out, test_out, report = convert_boolean_like_features(train_out, test_out, target_col=target_col)
    steps.append("convert_boolean_like_features")
    step_reports["convert_boolean_like_features"] = report
    train_out, test_out, report = create_datetime_features_with_report(train_out, test_out, target_col=target_col)
    steps.append("create_datetime_features")
    step_reports["create_datetime_features"] = report
    if preprocessing.get("categorical_encoding") == "one_hot":
        train_out, test_out, report = encode_categorical_features_with_report(train_out, test_out, target_col=target_col)
        steps.append("encode_categorical_features")
        step_reports["encode_categorical_features"] = report
    if preprocessing.get("numeric_outlier_clipping", True):
        train_out, test_out, report = clip_numeric_outliers_with_report(train_out, test_out, target_col=target_col)
        steps.append("clip_numeric_outliers")
        step_reports["clip_numeric_outliers"] = report
    if preprocessing.get("remove_constant_features", True):
        train_out, test_out, report = remove_low_information_features_with_report(train_out, test_out, target_col=target_col)
        steps.append("remove_low_information_features")
        step_reports["remove_low_information_features"] = report
    correlation_threshold = preprocessing.get("correlation_threshold")
    if correlation_threshold:
        train_out, test_out, report = remove_highly_correlated_features_with_report(
            train_out,
            test_out,
            target_col=target_col,
            threshold=float(correlation_threshold),
        )
        steps.append("remove_highly_correlated_features")
        step_reports["remove_highly_correlated_features"] = report
    if preprocessing.get("drop_target_leakage_features", True):
        train_out, test_out, report = drop_target_leakage_features_with_report(
            train_out,
            test_out,
            target_col=target_col,
        )
        steps.append("drop_target_leakage_features")
        step_reports["drop_target_leakage_features"] = report
    train_out, test_out, report = align_and_validate_feature_matrices(train_out, test_out, target_col=target_col)
    steps.append("align_and_validate_feature_matrices")
    step_reports["align_and_validate_feature_matrices"] = report
    if preprocessing.get("numeric_scaling"):
        train_out, test_out, report = scale_numeric_features_with_report(train_out, test_out, target_col=target_col)
        steps.append("scale_numeric_features")
        step_reports["scale_numeric_features"] = report
    output_profile = profile_features(train_out, test_out, target_col=target_col)
    eda_actions = _map_eda_recommendations_to_actions(policy.get("eda_recommendations", []), steps)
    return (
        train_out,
        test_out,
        {
            "model_id": model_plan.get("model_id"),
            "steps": steps,
            "step_reports": step_reports,
            "input_profile": input_profile,
            "output_profile": output_profile,
            "preprocessing": preprocessing,
            "eda_recommendations": policy.get("eda_recommendations", []),
            "eda_policy_actions": eda_actions,
        },
    )


def impute_missing_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute feature missing values using train-fitted fill values."""

    train_out, test_out = train_df.copy(), test_df.copy()
    feature_cols = [col for col in train_out.columns if col != target_col]
    for col in feature_cols:
        if not train_out[col].isna().any() and not test_out[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(train_out[col]):
            fill_value = train_out[col].median() if numeric_strategy == "median" else train_out[col].mean()
            if pd.isna(fill_value):
                fill_value = 0
            train_out[col] = pd.to_numeric(train_out[col], errors="coerce").fillna(fill_value)
            test_out[col] = pd.to_numeric(test_out[col], errors="coerce").fillna(fill_value)
        else:
            mode = train_out[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty and categorical_strategy == "mode" else "Unknown"
            train_out[col] = train_out[col].astype("object").fillna(fill_value)
            test_out[col] = test_out[col].astype("object").fillna(fill_value)
    return train_out, test_out


def impute_missing_features_with_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Impute feature missing values and report train-fitted fill values."""

    train_out, test_out = impute_missing_features(
        train_df,
        test_df,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
        target_col=target_col,
    )
    report: dict[str, Any] = {
        "numeric_strategy": numeric_strategy,
        "categorical_strategy": categorical_strategy,
        "imputed_columns": [],
        "fill_values": {},
    }
    for col in [col for col in train_df.columns if col != target_col]:
        if not train_df[col].isna().any() and not test_df[col].isna().any():
            continue
        report["imputed_columns"].append(col)
        if pd.api.types.is_numeric_dtype(train_df[col]):
            value = train_df[col].median() if numeric_strategy == "median" else train_df[col].mean()
            if pd.isna(value):
                value = 0
            value = float(value)
        else:
            mode = train_df[col].mode(dropna=True)
            value = mode.iloc[0] if not mode.empty and categorical_strategy == "mode" else "Unknown"
        report["fill_values"][col] = value
    return train_out, test_out, report


def convert_boolean_like_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Convert boolean-like feature strings to 0/1 using train-observed labels."""

    train_out, test_out = train_df.copy(), test_df.copy()
    converted: list[str] = []
    for col in [col for col in train_out.columns if col != target_col]:
        if not _is_boolean_like_series(train_out[col]):
            continue
        train_out[col] = _map_boolean_like_series(train_out[col])
        test_out[col] = _map_boolean_like_series(test_out[col])
        converted.append(col)
    return train_out, test_out, {"converted_columns": converted}


def scale_numeric_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale numeric feature columns using train-fitted standard scaling."""

    train_out, test_out = train_df.copy(), test_df.copy()
    feature_cols = [col for col in train_out.columns if col != target_col]
    numeric_cols = [
        col for col in feature_cols
        if pd.api.types.is_numeric_dtype(train_out[col])
    ]
    if not numeric_cols:
        return train_out, test_out
    scaler = StandardScaler()
    train_out[numeric_cols] = scaler.fit_transform(train_out[numeric_cols])
    test_out[numeric_cols] = scaler.transform(test_out[numeric_cols])
    return train_out, test_out


def clip_numeric_outliers_with_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Clip non-binary numeric features using train-fitted quantile bounds."""

    train_out, test_out = train_df.copy(), test_df.copy()
    clipped: dict[str, dict[str, float]] = {}
    for col in [col for col in train_out.columns if col != target_col]:
        if not pd.api.types.is_numeric_dtype(train_out[col]) or _is_binary_numeric_series(train_out[col]):
            continue
        lower = train_out[col].quantile(lower_quantile)
        upper = train_out[col].quantile(upper_quantile)
        if pd.isna(lower) or pd.isna(upper) or lower >= upper:
            continue
        train_changed = int(((train_out[col] < lower) | (train_out[col] > upper)).sum())
        test_changed = int(((test_out[col] < lower) | (test_out[col] > upper)).sum())
        if train_changed == 0 and test_changed == 0:
            continue
        train_out[col] = train_out[col].clip(lower=lower, upper=upper)
        test_out[col] = test_out[col].clip(lower=lower, upper=upper)
        clipped[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "train_values_clipped": train_changed,
            "test_values_clipped": test_changed,
        }
    return train_out, test_out, {"clipped_columns": clipped}


def remove_low_information_features_with_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Remove constant or all-missing feature columns based on train data."""

    train_out, test_out = train_df.copy(), test_df.copy()
    removed: list[str] = []
    for col in [col for col in train_out.columns if col != target_col]:
        if train_out[col].nunique(dropna=False) <= 1:
            removed.append(col)
    if removed:
        train_out = train_out.drop(columns=removed)
        test_out = test_out.drop(columns=[col for col in removed if col in test_out.columns])
    return train_out, test_out, {"removed_columns": removed}


def remove_highly_correlated_features_with_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
    threshold: float = 0.98,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Remove highly correlated numeric features using train correlations."""

    train_out, test_out = train_df.copy(), test_df.copy()
    numeric_cols = [
        col for col in train_out.columns
        if col != target_col and pd.api.types.is_numeric_dtype(train_out[col])
    ]
    if len(numeric_cols) < 2:
        return train_out, test_out, {"removed_columns": [], "threshold": threshold}
    corr = train_out[numeric_cols].corr().abs()
    removed: list[str] = []
    for i, col in enumerate(numeric_cols):
        if col in removed:
            continue
        for other in numeric_cols[i + 1:]:
            if other in removed:
                continue
            value = corr.loc[col, other]
            if pd.notna(value) and value >= threshold:
                removed.append(other)
    if removed:
        train_out = train_out.drop(columns=removed)
        test_out = test_out.drop(columns=[col for col in removed if col in test_out.columns])
    return train_out, test_out, {"removed_columns": removed, "threshold": threshold}


def drop_target_leakage_features_with_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Drop feature columns that exactly duplicate the target in train data."""

    train_out, test_out = train_df.copy(), test_df.copy()
    removed: list[str] = []
    warnings: list[str] = []
    if not target_col or target_col not in train_out.columns:
        return train_out, test_out, {"removed_columns": removed, "warnings": ["target column not available for leakage checks"]}
    target = train_out[target_col]
    for col in [col for col in train_out.columns if col != target_col]:
        if train_out[col].astype(str).equals(target.astype(str)):
            removed.append(col)
        elif target_col.lower() in col.lower():
            warnings.append(f"Column name may reference target: {col}")
    if removed:
        train_out = train_out.drop(columns=removed)
        test_out = test_out.drop(columns=[col for col in removed if col in test_out.columns])
    return train_out, test_out, {"removed_columns": removed, "warnings": warnings}


def align_and_validate_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Align train/test columns and report model-readiness checks."""

    train_out, test_out = train_df.copy(), test_df.copy()
    train_out, test_out = train_out.align(test_out, join="left", axis=1, fill_value=0)
    if target_col and target_col in train_df.columns:
        train_out[target_col] = train_df[target_col].values
    if target_col and target_col in test_df.columns:
        test_out[target_col] = test_df[target_col].values
    feature_cols = [col for col in train_out.columns if col != target_col]
    non_numeric = [
        col for col in feature_cols
        if not pd.api.types.is_numeric_dtype(train_out[col])
    ]
    missing_train = int(train_out[feature_cols].isna().sum().sum()) if feature_cols else 0
    missing_test = int(test_out[feature_cols].isna().sum().sum()) if feature_cols else 0
    return (
        train_out,
        test_out,
        {
            "train_test_columns_match": list(train_out.columns) == list(test_out.columns),
            "all_features_numeric": not non_numeric,
            "non_numeric_features": non_numeric,
            "missing_feature_values_train": missing_train,
            "missing_feature_values_test": missing_test,
            "feature_count": len(feature_cols),
        },
    )


def scale_numeric_features_with_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Scale numeric features and report scaled columns."""

    train_out, test_out = scale_numeric_features(train_df, test_df, target_col=target_col)
    numeric_cols = [
        col for col in train_df.columns
        if col != target_col and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    return train_out, test_out, {"scaled_columns": numeric_cols}


def _is_datetime_like_column(col: str, series: pd.Series) -> bool:
    """Return whether a feature column appears to contain datetime-like values."""

    name_hint = str(col).lower()
    if _is_month_name_series(series):
        return False
    return (
        pd.api.types.is_datetime64_any_dtype(series)
        or series.dtype == "object"
        and any(token in name_hint for token in ["date", "time", "timestamp"])
    )


def _is_identifier_like_column(col: str) -> bool:
    """Return whether a feature name looks like an identifier."""

    lowered = str(col).lower()
    return any(token in lowered for token in ("id", "code", "zip", "postal", "phone"))


def _is_boolean_like_series(series: pd.Series) -> bool:
    """Return whether non-null object values look boolean-like."""

    if not pd.api.types.is_object_dtype(series):
        return False
    values = {str(value).strip().lower() for value in series.dropna().unique()}
    if not values or len(values) > 2:
        return False
    allowed = {"yes", "no", "true", "false", "y", "n", "1", "0"}
    return values.issubset(allowed)


def _is_binary_numeric_series(series: pd.Series) -> bool:
    """Return whether a numeric feature contains only binary values."""

    values = set(series.dropna().unique())
    return bool(values) and values.issubset({0, 1, 0.0, 1.0})


def _map_boolean_like_series(series: pd.Series) -> pd.Series:
    """Map common boolean-like strings to numeric 0/1 values."""

    mapping = {
        "yes": 1,
        "true": 1,
        "y": 1,
        "1": 1,
        "no": 0,
        "false": 0,
        "n": 0,
        "0": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)


def _parse_datetime_safely(series: pd.Series) -> pd.Series:
    """Parse datetimes while suppressing pandas format-inference warnings."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")


def _is_month_name_series(series: pd.Series) -> bool:
    """Return whether a series appears to contain standalone month names."""

    month_names = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    }
    values = {str(value).strip().lower() for value in series.dropna().unique()}
    return bool(values) and values.issubset(month_names)


def _map_eda_recommendations_to_actions(recommendations: list[str], steps: list[str]) -> dict[str, Any]:
    """Map EDA recommendations to feature pipeline actions for auditability."""

    applied: list[dict[str, str]] = []
    deferred: list[dict[str, str]] = []
    for recommendation in recommendations:
        lowered = recommendation.lower()
        if "missing" in lowered and {"create_missing_indicators", "impute_missing_features"}.issubset(steps):
            applied.append({"recommendation": recommendation, "action": "created missing indicators and train-fitted imputations"})
        elif "high-cardinality" in lowered and "encode_categorical_features" in steps:
            applied.append({"recommendation": recommendation, "action": "used frequency encoding for high-cardinality categorical features"})
        elif "id-like" in lowered and "encode_categorical_features" in steps:
            applied.append({"recommendation": recommendation, "action": "dropped identifier-like categorical features during encoding"})
        elif "constant" in lowered and "remove_low_information_features" in steps:
            applied.append({"recommendation": recommendation, "action": "removed constant and near-constant features using train data"})
        elif "imbalance" in lowered:
            deferred.append({"recommendation": recommendation, "reason": "handled during model selection metric/tuning rather than feature engineering"})
        elif "skewed regression targets" in lowered:
            deferred.append({"recommendation": recommendation, "reason": "target transformation requires inverse-transform evaluation support"})
        else:
            deferred.append({"recommendation": recommendation, "reason": "no deterministic feature action matched this recommendation"})
    return {"applied": applied, "deferred": deferred}
