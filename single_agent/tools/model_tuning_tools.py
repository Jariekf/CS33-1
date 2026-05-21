"""LLM-assisted, deterministic hyperparameter tuning tools."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score

from core.llm import call_llm
from tools.model_selection_tools import (
    candidate_models,
    evaluate_predictions,
    select_metric,
    _feature_importance,
)


SYSTEM_PROMPT = """You are a machine-learning tuning planner.
Suggest a compact, safe hyperparameter search plan for the given model and dataset profile.
Return only valid JSON. Do not invent parameter names."""


ALLOWED_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    },
    "random_forest_classifier": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", None],
        "class_weight": [None, "balanced"],
    },
    "gradient_boosting_classifier": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.03, 0.1],
        "max_depth": [2, 3],
        "subsample": [0.8, 1.0],
    },
    "ridge_regression": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
    },
    "random_forest_regressor": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", None],
    },
    "gradient_boosting_regressor": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.03, 0.1],
        "max_depth": [2, 3],
        "subsample": [0.8, 1.0],
    },
}


def tune_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    task_type: str,
    model_id: str,
    model_plan: dict[str, Any] | None = None,
    tuning_budget: str = "small",
    use_llm: bool = True,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """Tune one candidate model, evaluate the best estimator, and return reports."""

    models = candidate_models(task_type)
    if model_id not in models:
        raise ValueError(f"Unsupported model_id for {task_type}: {model_id}")
    X_train = train_df.drop(columns=[target_col]).select_dtypes(include="number").fillna(0)
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col]).select_dtypes(include="number").fillna(0)
    y_test = test_df[target_col]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    metric = select_metric(task_type)
    cv = _safe_cv_splits(task_type, y_train)
    if cv < 2:
        model = clone(models[model_id])
        model.fit(X_train, y_train)
        test_metrics = evaluate_predictions(task_type, y_test, model.predict(X_test))
        report = {
            "task_type": task_type,
            "target_column": target_col,
            "model_id": model_id,
            "metric": metric,
            "cv_score": _primary_metric_value(test_metrics, metric),
            "cv_scores": [],
            "test_metrics": test_metrics,
            "tuning": {
                "status": "skipped",
                "reason": "Not enough target samples per class or rows for cross-validation.",
                "cv": cv,
            },
        }
        return model, report, _feature_importance(model, list(X_train.columns))

    baseline_report = _baseline_cv_report(models[model_id], X_train, y_train, metric, cv)
    tuning_plan = llm_tuning_plan(
        model_id=model_id,
        task_type=task_type,
        metric=metric,
        X_train=X_train,
        y_train=y_train,
        model_plan=model_plan or {},
        baseline_report=baseline_report,
        tuning_budget=tuning_budget,
        use_llm=use_llm,
    )
    search = GridSearchCV(
        estimator=clone(models[model_id]),
        param_grid=tuning_plan["param_grid"],
        scoring=tuning_plan["metric"],
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    try:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        tuned_cv_score = float(search.best_score_)
        best_params = _jsonable_params(search.best_params_)
        status = "completed"
        failure_reason = ""
    except Exception as exc:
        best_model = clone(models[model_id])
        best_model.fit(X_train, y_train)
        tuned_cv_score = baseline_report["mean_cv_score"]
        best_params = {}
        status = "fallback_to_baseline"
        failure_reason = str(exc)
    test_metrics = evaluate_predictions(task_type, y_test, best_model.predict(X_test))
    report = {
        "task_type": task_type,
        "target_column": target_col,
        "model_id": model_id,
        "metric": metric,
        "cv_score": tuned_cv_score,
        "cv_scores": [],
        "test_metrics": test_metrics,
        "tuning": {
            "status": status,
            "failure_reason": failure_reason,
            "baseline": baseline_report,
            "plan": tuning_plan,
            "n_jobs": 1,
            "best_params": best_params,
            "best_cv_score": tuned_cv_score,
            "explanation": tuning_plan.get("explanation", ""),
        },
    }
    return best_model, report, _feature_importance(best_model, list(X_train.columns))


def llm_tuning_plan(
    model_id: str,
    task_type: str,
    metric: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_plan: dict[str, Any],
    baseline_report: dict[str, Any],
    tuning_budget: str = "small",
    use_llm: bool = True,
) -> dict[str, Any]:
    """Return an LLM-suggested tuning plan with deterministic fallback."""

    fallback = default_tuning_plan(model_id, task_type, metric, tuning_budget)
    if not use_llm:
        return fallback
    profile = {
        "rows": int(len(X_train)),
        "features": int(len(X_train.columns)),
        "numeric_features": list(X_train.columns),
        "target_unique_values": int(y_train.nunique(dropna=True)),
        "target_distribution": y_train.value_counts(dropna=False).head(20).to_dict(),
        "model_plan": model_plan,
        "baseline_report": baseline_report,
        "allowed_param_grid": ALLOWED_PARAM_GRIDS.get(model_id, {}),
        "default_metric": metric,
        "budget": tuning_budget,
    }
    try:
        raw = call_llm(SYSTEM_PROMPT, _build_tuning_prompt(model_id, task_type, profile))
        payload = _extract_json_object(raw)
        return _sanitize_tuning_plan(payload, fallback, model_id, metric)
    except Exception as exc:
        fallback["source"] = "deterministic_fallback"
        fallback["llm_error"] = str(exc)
        return fallback


def default_tuning_plan(
    model_id: str,
    task_type: str,
    metric: str | None = None,
    tuning_budget: str = "small",
) -> dict[str, Any]:
    """Return a compact deterministic hyperparameter search plan."""

    return {
        "source": "deterministic",
        "model_id": model_id,
        "task_type": task_type,
        "metric": metric or select_metric(task_type),
        "cv": 3,
        "budget": tuning_budget,
        "param_grid": _small_param_grid(model_id),
        "explanation": "Compact deterministic search over stable sklearn hyperparameters.",
    }


def _small_param_grid(model_id: str) -> dict[str, list[Any]]:
    """Return a small grid from the allowed model search space."""

    grids = {
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
        "random_forest_classifier": {
            "n_estimators": [100],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 2],
            "class_weight": [None, "balanced"],
        },
        "gradient_boosting_classifier": {
            "n_estimators": [50, 100],
            "learning_rate": [0.03, 0.1],
            "max_depth": [2, 3],
        },
        "ridge_regression": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "random_forest_regressor": {
            "n_estimators": [100],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 2],
        },
        "gradient_boosting_regressor": {
            "n_estimators": [50, 100],
            "learning_rate": [0.03, 0.1],
            "max_depth": [2, 3],
        },
    }
    return grids.get(model_id, ALLOWED_PARAM_GRIDS.get(model_id, {}))


def _baseline_cv_report(model: Any, X_train: pd.DataFrame, y_train: pd.Series, metric: str, cv: int) -> dict[str, Any]:
    """Evaluate the default estimator before tuning."""

    scores = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring=metric)
    return {
        "metric": metric,
        "cv": cv,
        "scores": [float(score) for score in scores],
        "mean_cv_score": float(scores.mean()),
    }


def _safe_cv_splits(task_type: str, y_train: pd.Series) -> int:
    """Choose a safe CV split count for the available target values."""

    if len(y_train) < 2:
        return 0
    if task_type == "classification":
        counts = y_train.value_counts(dropna=False)
        return min(3, int(counts.min())) if int(counts.min()) >= 2 else 0
    return min(3, len(y_train))


def _primary_metric_value(metrics: dict[str, float], metric: str) -> float:
    """Read the primary metric from a metric dictionary."""

    if metric in metrics:
        return float(metrics[metric])
    return float(next(iter(metrics.values()))) if metrics else 0.0


def _build_tuning_prompt(model_id: str, task_type: str, profile: dict[str, Any]) -> str:
    """Build a compact tuning-plan prompt."""

    return f"""# Model
{model_id}

# Task Type
{task_type}

# Dataset And Allowed Search Space
```json
{json.dumps(profile, ensure_ascii=False, indent=2, default=str)}
```

# Required JSON
{{
  "metric": "{profile["default_metric"]}",
  "param_grid": {{}},
  "explanation": "short tuning rationale"
}}

Rules:
- Choose only parameter names and values from allowed_param_grid.
- Keep the grid compact for the requested budget.
- If baseline_report indicates weak performance, broaden one or two high-impact parameters.
- Do not include unsupported parameters or values.
"""


def _sanitize_tuning_plan(
    payload: dict[str, Any],
    fallback: dict[str, Any],
    model_id: str,
    metric: str,
) -> dict[str, Any]:
    """Validate and constrain an LLM tuning plan to allowed parameters."""

    allowed = ALLOWED_PARAM_GRIDS.get(model_id, {})
    raw_grid = payload.get("param_grid", {})
    if not isinstance(raw_grid, dict):
        return fallback
    grid: dict[str, list[Any]] = {}
    for key, values in raw_grid.items():
        if key not in allowed:
            continue
        if not isinstance(values, list):
            values = [values]
        selected = [value for value in values if value in allowed[key]]
        if selected:
            grid[key] = selected[:3]
    if not grid:
        return fallback
    return {
        "source": "llm",
        "model_id": model_id,
        "task_type": fallback["task_type"],
        "metric": metric if payload.get("metric") != metric else payload.get("metric", metric),
        "cv": fallback["cv"],
        "budget": fallback["budget"],
        "param_grid": grid,
        "explanation": str(payload.get("explanation", "")),
    }


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert sklearn params to JSON-friendly values."""

    return {key: value for key, value in params.items()}


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from an LLM response."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain a JSON object")
    payload = json.loads(stripped[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload
