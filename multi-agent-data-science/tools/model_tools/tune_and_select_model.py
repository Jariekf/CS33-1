import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from sklearn.model_selection import RandomizedSearchCV

from tools.model_tools.train_model import _get_model
from tools.tool_trace import trace_tool_call


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _get_scoring(task_type: str) -> str:
    return "f1_weighted" if task_type == "classification" else "r2"


def _get_param_space(model_name: str) -> dict[str, list[Any]]:
    spaces: dict[str, dict[str, list[Any]]] = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs", "liblinear"],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
        "SVM": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "LinearRegression": {},
        "Ridge": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "RandomForestRegressor": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "XGBoostRegressor": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
        "SVR": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "epsilon": [0.01, 0.1, 0.2],
        },
    }
    return spaces.get(model_name, {})


@tool
def tune_and_select_model(
    feature_dataset_ids: dict[str, str],
    candidate_models: list[str],
    task_type: str,
    target_column: str,
    cv_folds: int = 5,
    max_iter_per_model: int = 12,
    random_state: int = 42,
) -> str:
    """对候选模型执行 CV 调参与选择，返回最佳模型及每个模型的最优参数。

    注意：
      - 该工具只使用训练集（feature_dataset_ids），不会使用 test set。
      - 不会训练/保存最终模型文件；最终重训应交给 train_model。
    """
    tool_inputs = {
        "candidate_models": candidate_models,
        "task_type": task_type,
        "target_column": target_column,
        "cv_folds": cv_folds,
        "max_iter_per_model": max_iter_per_model,
        "random_state": random_state,
    }
    try:
        if not candidate_models:
            raise ValueError("candidate_models 不能为空。")

        scoring = _get_scoring(task_type)
        summary: dict[str, Any] = {}
        best_model_name = ""
        best_score = float("-inf")

        for model_name in candidate_models:
            featured_train_path = feature_dataset_ids.get(model_name, "")
            if not featured_train_path:
                raise ValueError(f"模型 {model_name} 缺少 feature_dataset_ids 路径。")

            df = pd.read_csv(featured_train_path)
            if target_column not in df.columns:
                raise ValueError(f"目标列 '{target_column}' 不在 {model_name} 训练集中。")

            feature_cols = [c for c in df.columns if c != target_column]
            X = df[feature_cols].select_dtypes(include="number")
            y = df[target_column]

            estimator = _get_model(model_name)
            param_space = _get_param_space(model_name)

            if not param_space:
                cv_score = round(float(estimator.fit(X, y).score(X, y)), 4)
                summary[model_name] = {
                    "best_params": {},
                    "cv_best_score": cv_score,
                    "scoring": scoring,
                    "rows_trained": int(len(df)),
                    "search_strategy": "none",
                }
            else:
                search_iter = min(max_iter_per_model, max(1, len(param_space) * 3))
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_space,
                    n_iter=search_iter,
                    scoring=scoring,
                    cv=cv_folds,
                    random_state=random_state,
                    n_jobs=-1,
                )
                search.fit(X, y)
                summary[model_name] = {
                    "best_params": _to_builtin(search.best_params_),
                    "cv_best_score": round(float(search.best_score_), 4),
                    "scoring": scoring,
                    "rows_trained": int(len(df)),
                    "search_strategy": "random_search",
                }

            score = float(summary[model_name]["cv_best_score"])
            if score > best_score:
                best_score = score
                best_model_name = model_name

        if not best_model_name:
            raise ValueError("无法从候选模型中选出最佳模型。")

        result = {
            "best_model_name": best_model_name,
            "best_model_params": summary[best_model_name]["best_params"],
            "selection_metric": scoring,
            "best_cv_score": round(best_score, 4),
            "per_model_summary": summary,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("tune_and_select_model", tool_inputs, output)
    return output
