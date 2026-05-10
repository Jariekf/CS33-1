"""Deterministic model selection and evaluation tools."""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def infer_task_type(user_goal: str, target_series: pd.Series | None = None) -> str:
    """Infer a simple supervised task type."""

    text = user_goal.lower()
    if any(word in text for word in ["classify", "classification", "churn", "fraud", "yes/no"]):
        return "classification"
    if any(word in text for word in ["forecast", "time series", "predict sales over time"]):
        return "time_series"
    if target_series is not None:
        unique_count = target_series.nunique(dropna=True)
        if target_series.dtype == "object" or unique_count <= 20:
            return "classification"
    return "regression"


def select_metric(task_type: str) -> str:
    """Select a default metric for the inferred task type."""

    if task_type == "classification":
        return "f1_weighted"
    return "r2"


def candidate_models(task_type: str) -> dict[str, Any]:
    """Return a compact candidate model set."""

    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest_classifier": RandomForestClassifier(n_estimators=200, random_state=42),
            "gradient_boosting_classifier": GradientBoostingClassifier(random_state=42),
        }
    return {
        "ridge_regression": Ridge(),
        "random_forest_regressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
    }


def evaluate_predictions(task_type: str, y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    """Compute final test metrics."""

    if task_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        }
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_and_select_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """Train candidate models and select the best by cross-validation."""

    X_train = train_df.drop(columns=[target_col]).select_dtypes(include="number").fillna(0)
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col]).select_dtypes(include="number").fillna(0)
    y_test = test_df[target_col]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    models = candidate_models(task_type)
    scoring = select_metric(task_type)
    scores: dict[str, float] = {}
    best_name, best_score, best_model = "", float("-inf"), None
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring)
        mean_score = float(cv_scores.mean())
        scores[name] = mean_score
        if mean_score > best_score:
            best_name, best_score, best_model = name, mean_score, model
    if best_model is None:
        raise ValueError("No model could be selected")
    best_model.fit(X_train, y_train)
    test_metrics = evaluate_predictions(task_type, y_test, best_model.predict(X_test))
    report = {
        "task_type": task_type,
        "target_column": target_col,
        "candidate_model_ids": list(models.keys()),
        "metric": scoring,
        "cv_scores": scores,
        "best_model": best_name,
        "best_cv_score": best_score,
        "test_metrics": test_metrics,
    }
    return best_model, report, _feature_importance(best_model, list(X_train.columns))


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    task_type: str,
    model_id: str,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """Train one concrete candidate model and evaluate it."""

    models = candidate_models(task_type)
    if model_id not in models:
        raise ValueError(f"Unsupported model_id for {task_type}: {model_id}")
    X_train = train_df.drop(columns=[target_col]).select_dtypes(include="number").fillna(0)
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col]).select_dtypes(include="number").fillna(0)
    y_test = test_df[target_col]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    model = models[model_id]
    scoring = select_metric(task_type)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring)
    mean_score = float(cv_scores.mean())
    model.fit(X_train, y_train)
    test_metrics = evaluate_predictions(task_type, y_test, model.predict(X_test))
    report = {
        "task_type": task_type,
        "target_column": target_col,
        "model_id": model_id,
        "metric": scoring,
        "cv_score": mean_score,
        "cv_scores": [float(score) for score in cv_scores],
        "test_metrics": test_metrics,
    }
    return model, report, _feature_importance(model, list(X_train.columns))


def _feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Extract model feature importance or coefficients."""

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0] if getattr(model.coef_, "ndim", 1) > 1 else model.coef_)
    else:
        values = [0.0] * len(feature_names)
    pairs = zip(feature_names, [float(v) for v in values])
    return dict(sorted(pairs, key=lambda item: item[1], reverse=True))
