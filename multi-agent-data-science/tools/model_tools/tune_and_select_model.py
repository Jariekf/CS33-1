import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)

from tools.model_tools.train_model import _get_model
from tools.tool_trace import trace_tool_call

# ── 白名单 ────────────────────────────────────────────────────────────────────
_ALLOWED_SCORING = {
    "classification": {
        "f1_weighted",
        "f1_macro",
        "roc_auc",
        "accuracy",
        "recall",
        "precision",
    },
    "regression": {
        "r2",
        "neg_root_mean_squared_error",
        "neg_mean_absolute_error",
    },
}

# 人类可读名 → sklearn scoring 字符串的映射（仅需要转换的条目）
_SCORING_ALIAS: dict[str, str] = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}

_DEFAULT_SCORING = {"classification": "f1_weighted", "regression": "r2"}

_ALLOWED_CV_STRATEGY = {"auto", "stratified", "kfold", "timeseries"}


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


def _resolve_scoring(scoring: str, task_type: str) -> str:
    """验证并返回 sklearn 兼容的 scoring 字符串。

    - 接受人类可读名（rmse、mae）并映射到 sklearn 的 neg_* 前缀形式
    - 无效或空字符串时回退到任务默认值
    """
    allowed = _ALLOWED_SCORING.get(task_type, set())
    # 先做别名映射（rmse → neg_root_mean_squared_error 等）
    sklearn_name = _SCORING_ALIAS.get(scoring, scoring)
    # 检查原始名或 sklearn 名是否在白名单内
    if scoring and (scoring in allowed or sklearn_name in allowed):
        return sklearn_name
    return _DEFAULT_SCORING.get(task_type, "r2")


def _build_cv(cv_strategy: str, task_type: str, cv_folds: int) -> Any:
    """根据策略和任务类型构建 sklearn CV 对象。

    策略说明：
      auto       - 分类用 StratifiedKFold，回归用 KFold（默认）
      stratified - StratifiedKFold，适合类别不平衡的分类任务
      kfold      - 普通 KFold，适合回归或类别均衡的分类任务
      timeseries - TimeSeriesSplit，数据有时序关系时必须使用，避免数据泄漏
    """
    strategy = cv_strategy.lower() if cv_strategy else "auto"
    if strategy not in _ALLOWED_CV_STRATEGY:
        strategy = "auto"

    if strategy == "timeseries":
        return TimeSeriesSplit(n_splits=cv_folds)
    if strategy == "stratified" or (strategy == "auto" and task_type == "classification"):
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return KFold(n_splits=cv_folds, shuffle=True, random_state=42)


def _get_param_space(model_name: str) -> dict[str, list[Any]]:
    _TREE_ENSEMBLE = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    _GRADIENT_BOOSTING = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    _LIGHTGBM = {
        "n_estimators": [100, 200, 300],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_samples": [20, 50, 100],
    }

    spaces: dict[str, dict[str, list[Any]]] = {
        # ── Classification ────────────────────────────────────────────────────
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs", "liblinear"],
        },
        "RandomForest": _TREE_ENSEMBLE,
        "XGBoost": _GRADIENT_BOOSTING,
        "LightGBM": _LIGHTGBM,
        "GradientBoosting": _GRADIENT_BOOSTING,
        "ExtraTrees": _TREE_ENSEMBLE,
        "AdaBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.5, 1.0, 2.0],
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
        "GaussianNB": {
            "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3],
        },
        "MLPClassifier": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100), (200, 100)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        },

        # ── Regression ────────────────────────────────────────────────────────
        "LinearRegression": {},
        "Ridge": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "Lasso": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        },
        "ElasticNet": {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "HuberRegressor": {
            "epsilon": [1.1, 1.35, 1.5, 2.0],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
        },
        "RandomForestRegressor": _TREE_ENSEMBLE,
        "XGBoostRegressor": _GRADIENT_BOOSTING,
        "LightGBMRegressor": _LIGHTGBM,
        "GradientBoostingRegressor": _GRADIENT_BOOSTING,
        "ExtraTreesRegressor": _TREE_ENSEMBLE,
        "SVR": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "epsilon": [0.01, 0.1, 0.2],
        },
        "MLPRegressor": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100), (200, 100)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
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
    scoring: str = "",
    cv_strategy: str = "auto",
) -> str:
    """对候选模型执行 CV 调参与选择，返回最佳模型及每个模型的最优参数。

    该工具只使用训练集（feature_dataset_ids），不会接触 test set。
    不保存模型文件；最终重训由 train_model 完成。

    参数：
      feature_dataset_ids (dict)：各候选模型对应的特征工程后训练集路径。
      candidate_models (list)：候选模型名称列表。
      task_type (str)："classification" 或 "regression"。
      target_column (str)：目标列名。
      cv_folds (int)：交叉验证折数，默认 5。
      max_iter_per_model (int)：每个模型 RandomizedSearchCV 的最大迭代次数，默认 12。
      random_state (int)：随机种子，默认 42。
      scoring (str)：评分指标，留空时使用任务默认值。
        分类可选：f1_weighted（默认）、f1_macro、roc_auc、accuracy、recall、precision
        回归可选：r2（默认）、neg_root_mean_squared_error、neg_mean_absolute_error
        建议：类别严重不平衡时选 roc_auc；关注少数类召回时选 recall；
             回归对异常值敏感时选 neg_mean_absolute_error。
      cv_strategy (str)：交叉验证策略，默认 "auto"。
        auto       - 分类用 StratifiedKFold，回归用 KFold
        stratified - StratifiedKFold，类别不平衡时推荐
        kfold      - 普通 KFold
        timeseries - TimeSeriesSplit，数据存在时序关系时必须使用，否则会产生数据泄漏

    返回 JSON，包含：
      best_model_name、best_model_params、selection_metric、cv_strategy_used、
      best_cv_score、per_model_summary（每个模型的 best_params / cv_best_score / cv_score_std）
    """
    resolved_scoring = _resolve_scoring(scoring, task_type)
    cv = _build_cv(cv_strategy, task_type, cv_folds)
    cv_strategy_used = type(cv).__name__

    tool_inputs = {
        "candidate_models": candidate_models,
        "task_type": task_type,
        "target_column": target_column,
        "cv_folds": cv_folds,
        "max_iter_per_model": max_iter_per_model,
        "random_state": random_state,
        "scoring": resolved_scoring,
        "cv_strategy": cv_strategy_used,
    }
    try:
        if not candidate_models:
            raise ValueError("candidate_models 不能为空。")

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
                raw_scores = cross_val_score(estimator, X, y, cv=cv, scoring=resolved_scoring)
                cv_score = round(float(raw_scores.mean()), 4)
                cv_std = round(float(raw_scores.std()), 4)
                summary[model_name] = {
                    "best_params": {},
                    "cv_best_score": cv_score,
                    "cv_score_std": cv_std,
                    "scoring": resolved_scoring,
                    "rows_trained": int(len(df)),
                    "search_strategy": "cv_only",
                }
            else:
                total_combinations = 1
                for v in param_space.values():
                    total_combinations *= len(v)
                search_iter = min(max_iter_per_model, total_combinations)
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_space,
                    n_iter=search_iter,
                    scoring=resolved_scoring,
                    cv=cv,
                    random_state=random_state,
                    n_jobs=-1,
                )
                search.fit(X, y)
                best_idx = search.best_index_
                cv_std = round(float(search.cv_results_["std_test_score"][best_idx]), 4)
                summary[model_name] = {
                    "best_params": _to_builtin(search.best_params_),
                    "cv_best_score": round(float(search.best_score_), 4),
                    "cv_score_std": cv_std,
                    "scoring": resolved_scoring,
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
            "selection_metric": resolved_scoring,
            "cv_strategy_used": cv_strategy_used,
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
