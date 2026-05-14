import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from sklearn.experimental import enable_halving_search_cv  # noqa: F401

from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)

from tools.model_tools.train_model import _get_model
from tools.tool_trace import trace_tool_call

# ── 白名单（human-readable display names）────────────────────────────────────
_ALLOWED_SCORING = {
    "classification": {
        "f1_weighted",          # weighted avg → reflects overall perf on actual distribution
        "f1_macro",             # macro avg → equal weight per class, highlights minority class
        "roc_auc",              # binary; auto-upgraded to ovr_weighted for multi-class
        "accuracy",
        "recall",               # weighted; auto-upgraded to recall_weighted for multi-class
        "precision",            # weighted; auto-upgraded to precision_weighted for multi-class
        "log_loss",             # continuous probability calibration metric (CTR, risk, etc.)
        # multi-class compatible variants (auto-upgraded or passed directly)
        "roc_auc_ovr_weighted",
        "roc_auc_ovr",
        "recall_weighted",
        "precision_weighted",
    },
    "regression": {
        "r2",
        "rmse",   "neg_root_mean_squared_error",   # same metric, two accepted spellings
        "mae",    "neg_mean_absolute_error",
        "mape",   "neg_mean_absolute_percentage_error",
        "rmsle",  # custom scorer (no sklearn built-in string)
        "max_error",
    },
}

# display name → sklearn scoring string（仅需转换的条目；display name == sklearn name 的无需列出）
_SCORING_ALIAS: dict[str, str] = {
    "rmse":     "neg_root_mean_squared_error",
    "mae":      "neg_mean_absolute_error",
    "mape":     "neg_mean_absolute_percentage_error",
    "log_loss": "neg_log_loss",    # lower log-loss = better → negate for CV maximisation
    # rmsle is handled via make_scorer, not an alias
}

_DEFAULT_SCORING = {"classification": "f1_weighted", "regression": "r2"}

_ALLOWED_CV_STRATEGY = {"auto", "stratified", "kfold", "timeseries"}

_ALLOWED_HPO_STRATEGY = {"random", "grid", "halving", "bayesian"}
# random   - RandomizedSearchCV：随机采样，均衡默认选项
# grid     - GridSearchCV：穷举全部组合，仅适合参数空间小的模型（总组合数 ≤ 30）
# halving  - HalvingRandomSearchCV：逐轮淘汰劣候选，大空间下比 random 快 2-3 倍
# bayesian - BayesSearchCV（scikit-optimize）：GP 代理模型指导搜索，有限 iter 下最高效

# sklearn 中仅适用于二分类的 scorer；多分类时需升级到对应的 multi-class 版本
_MULTICLASS_SCORER_UPGRADE: dict[str, str] = {
    "roc_auc": "roc_auc_ovr_weighted",
    "recall":  "recall_weighted",
    "precision": "precision_weighted",
}

# Scorer names that have no sklearn built-in string and require a custom scorer object
_CUSTOM_SCORER_NAMES: frozenset[str] = frozenset({"rmsle"})


def _to_sklearn_scorer(display_name: str) -> Any:
    """display name → 实际 sklearn scorer（字符串或 scorer 对象）。

    - 标准指标：通过 _SCORING_ALIAS 转换为 sklearn scoring 字符串。
    - rmsle：无内置字符串，构建 make_scorer；预测值 clip ≥ 0 防 log 域错误。
    - 其余 display name 与 sklearn name 相同（如 r2、f1_weighted）：原样返回。
    """
    if display_name == "rmsle":
        import numpy as np
        from sklearn.metrics import make_scorer, mean_squared_log_error

        def _rmsle(y_true: Any, y_pred: Any) -> float:
            return float(np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None))))

        return make_scorer(_rmsle, greater_is_better=False)
    return _SCORING_ALIAS.get(display_name, display_name)


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
    """验证并返回经过规范化的 display name（人类可读名）。

    始终返回 display name（如 "rmse"、"mape"、"roc_auc"），不返回 sklearn neg_* 字符串。
    调用 _to_sklearn_scorer(display_name) 获取实际用于 sklearn API 的 scorer。
    无效或空字符串时回退到任务默认值。
    """
    allowed = _ALLOWED_SCORING.get(task_type, set())
    sklearn_name = _SCORING_ALIAS.get(scoring, scoring)
    if scoring in _CUSTOM_SCORER_NAMES:
        return scoring
    if scoring and (scoring in allowed or sklearn_name in allowed):
        return scoring  # return display name, sklearn conversion deferred to _to_sklearn_scorer
    return _DEFAULT_SCORING.get(task_type, "r2")


def _adapt_scorer_for_multiclass(scoring: str, n_classes: int) -> str:
    """二分类专用 scorer 在多分类（>2 类）任务下会崩溃，自动升级到多分类兼容版本。

    升级映射：
      roc_auc   → roc_auc_ovr_weighted  （OvR + support 加权，sklearn 兼容）
      recall    → recall_weighted
      precision → precision_weighted

    注：升级仅解决多分类兼容性问题；对 class imbalance 的最佳应对是 StratifiedKFold
    + EDA agent 选取 f1_weighted / roc_auc_ovr_weighted 作为 primary_metric。
    """
    if n_classes <= 2:
        return scoring
    return _MULTICLASS_SCORER_UPGRADE.get(scoring, scoring)


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
    search_strategy: str = "random",
    extra_param_overrides: dict[str, dict] | None = None,
) -> str:
    """对候选模型执行 CV 超参数搜索与模型选择，返回最佳模型及每个模型的最优参数。

    该工具只使用训练集（feature_dataset_ids），不会接触 test set。
    不保存模型文件；最终重训由 train_model 完成。

    参数：
      feature_dataset_ids (dict)：各候选模型对应的特征工程后训练集路径。
      candidate_models (list)：候选模型名称列表。
      task_type (str)："classification" 或 "regression"。
      target_column (str)：目标列名。
      cv_folds (int)：交叉验证折数，默认 5。
        建议：训练集 < 1000 行 → 3 折；1000~20000 行 → 5 折；> 20000 行 → 可用 10 折获得更稳定估计。
      max_iter_per_model (int)：search_strategy="random"/"halving" 时每模型的采样次数上限，默认 12。
        search_strategy="grid" 时忽略此参数（穷举全部组合）。
        建议：参数维度 ≤ 2 → 8~12；维度 3~4 → 15~25；维度 ≥ 5 → 25~40。
      random_state (int)：随机种子，默认 42。
      scoring (str)：评分指标，留空时使用任务默认值。
        分类可选：f1_weighted（默认）、f1_macro、roc_auc、accuracy、recall、precision、log_loss
          f1_macro   - 等权 per-class 平均，对少数类更敏感，适合类别不平衡场景
          f1_weighted- 样本量加权平均，反映整体分布下的性能，为默认选项
          log_loss   - 连续概率校准指标（CTR 预估、风险评分等场景首选）；
                       需要模型支持 predict_proba，SVM 默认不支持，应避免同时选用
        回归可选：r2（默认）、rmse、mae、mape、rmsle、max_error
          技术约束：mape 的 y_true 含 0 时 CV 产生 inf（安全降级）；
            rmsle 用 make_scorer（无 sklearn 内置字符串）；
            max_error 的 cv_best_score 为负值（越接近 0 越好）。
        注：roc_auc、recall、precision 为二分类专用；目标列类别数 > 2 时自动升级为
        roc_auc_ovr_weighted / recall_weighted / precision_weighted。
      cv_strategy (str)：交叉验证策略，默认 "auto"。
        auto       - 分类用 StratifiedKFold，回归用 KFold
        stratified - StratifiedKFold，类别不平衡时推荐
        kfold      - 普通 KFold
        timeseries - TimeSeriesSplit，数据存在时序关系时必须使用，否则会产生数据泄漏
      search_strategy (str)：超参数搜索策略，默认 "random"。
        random   - RandomizedSearchCV：随机采样 max_iter_per_model 次。均衡默认选项。
        grid     - GridSearchCV：穷举全部参数组合。
                   建议：仅当模型参数空间总组合数 ≤ 30 时使用（如 Ridge、Lasso、LogisticRegression）。
        halving  - HalvingRandomSearchCV：逐轮淘汰表现差的候选，在相同时间内比 random 探索更多配置。
                   建议：参数空间大（组合数 > 100）或候选模型多时使用（如 XGBoost、RandomForest）。
        bayesian - BayesSearchCV（scikit-optimize）：高斯过程代理模型指导搜索，
                   在相同 iter 预算内通常比 random/halving 找到更优参数组合。
                   建议：max_iter_per_model 较小（≤ 20）且希望最大化搜索质量时使用。
                   所有 list 型参数自动转为 Categorical 离散空间，无需额外配置。

      extra_param_overrides (dict | None)：按模型名注入或覆盖搜索空间中的特定参数，默认 None。
        key 为模型名，value 为需要追加/覆盖的参数 dict（合并到 _get_param_space 的默认空间之上）。
        只传需要调整的参数，其余参数保持默认空间不变。

        常见注入场景：
          1. 不平衡分类（支持 class_weight 的模型）：
             {"LogisticRegression": {"class_weight": ["balanced", None]},
              "RandomForest":       {"class_weight": ["balanced", None]},
              "SVM":                {"class_weight": ["balanced", None]}}
          2. 大样本扩充树模型 n_estimators 上限：
             {"RandomForest":    {"n_estimators": [200, 300, 500]},
              "XGBoost":         {"n_estimators": [200, 300, 500]}}
          3. 高维特征调整 max_features：
             {"RandomForest": {"max_features": ["sqrt", "log2", 0.3, 0.5]}}

        注意：注入的参数名和类型必须与模型的 sklearn/xgboost 参数接口完全一致，
        否则会在 fit() 时引发 ValueError。不确定时不传，使用默认空间。

    返回 JSON，包含：
      best_model_name、best_model_params、
      selection_metric（实际使用的 scorer，多分类时可能已升级）、
      n_classes（目标列唯一类别数，供下游判断任务是二分类还是多分类）、
      cv_strategy_used、search_strategy_used、best_cv_score、
      per_model_summary（每个模型的结果，含 status 字段）：
        - status="success"：best_params / cv_best_score / cv_score_std / scoring / search_strategy
        - status="failed"：error（失败原因，含异常类型和消息）；该模型不参与 best 评选
      skipped_models（失败模型列表，供 model_agent 分析失败原因）
    """
    resolved_scoring = _resolve_scoring(scoring, task_type)
    cv = _build_cv(cv_strategy, task_type, cv_folds)
    cv_strategy_used = type(cv).__name__
    hpo_strategy = search_strategy.lower() if search_strategy else "random"
    if hpo_strategy not in _ALLOWED_HPO_STRATEGY:
        hpo_strategy = "random"

    tool_inputs = {
        "candidate_models": candidate_models,
        "task_type": task_type,
        "target_column": target_column,
        "cv_folds": cv_folds,
        "max_iter_per_model": max_iter_per_model,
        "random_state": random_state,
        "scoring": resolved_scoring,
        "cv_strategy": cv_strategy_used,
        "search_strategy": hpo_strategy,
        "extra_param_overrides": extra_param_overrides or {},
    }
    try:
        if not candidate_models:
            raise ValueError("candidate_models 不能为空。")

        # ── 多分类检测：用第一个可用数据集探测目标列的唯一类别数 ─────────────
        # 仅分类任务需要检测；回归任务 resolved_scoring 不含二分类专用 scorer
        n_classes = 2
        if task_type == "classification":
            for _m in candidate_models:
                _path = feature_dataset_ids.get(_m, "")
                if _path:
                    try:
                        n_classes = int(pd.read_csv(_path)[target_column].nunique())
                        break
                    except Exception:
                        pass

        # effective_scoring_name: human-readable display name stored in results
        # effective_sklearn_scorer: actual scorer string/object passed to sklearn
        effective_scoring_name = _adapt_scorer_for_multiclass(resolved_scoring, n_classes)
        effective_sklearn_scorer = _to_sklearn_scorer(effective_scoring_name)

        summary: dict[str, Any] = {}
        best_model_name = ""
        best_score = float("-inf")

        for model_name in candidate_models:
            # ── Per-model fallback：单个模型失败不影响其他模型 ──────────────────
            try:
                featured_train_path = feature_dataset_ids.get(model_name, "")
                if not featured_train_path:
                    raise ValueError(f"feature_dataset_ids 中缺少 '{model_name}' 的路径。")

                df = pd.read_csv(featured_train_path)
                if target_column not in df.columns:
                    raise ValueError(f"目标列 '{target_column}' 不在训练集中。")

                feature_cols = [c for c in df.columns if c != target_column]
                X = df[feature_cols]
                y = df[target_column]

                estimator = _get_model(model_name)
                param_space = dict(_get_param_space(model_name))  # copy to avoid mutating the default
                if extra_param_overrides and model_name in extra_param_overrides:
                    param_space.update(extra_param_overrides[model_name])

                if not param_space:
                    # No hyperparameters to tune: plain CV scoring
                    raw_scores = cross_val_score(estimator, X, y, cv=cv, scoring=effective_sklearn_scorer)
                    cv_score = round(float(raw_scores.mean()), 4)
                    cv_std = round(float(raw_scores.std()), 4)
                    summary[model_name] = {
                        "status": "success",
                        "best_params": {},
                        "cv_best_score": cv_score,
                        "cv_score_std": cv_std,
                        "scoring": effective_scoring_name,
                        "rows_trained": int(len(df)),
                        "search_strategy": "cv_only",
                    }
                else:
                    total_combinations = 1
                    for v in param_space.values():
                        total_combinations *= len(v)

                    if hpo_strategy == "grid":
                        search = GridSearchCV(
                            estimator=estimator,
                            param_grid=param_space,
                            scoring=effective_sklearn_scorer,
                            cv=cv,
                            n_jobs=-1,
                        )
                        actual_strategy = "grid"
                    elif hpo_strategy == "halving":
                        search = HalvingRandomSearchCV(
                            estimator=estimator,
                            param_distributions=param_space,
                            n_candidates=min(max_iter_per_model * 3, total_combinations),
                            scoring=effective_sklearn_scorer,
                            cv=cv,
                            factor=3,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                        actual_strategy = "halving"
                    elif hpo_strategy == "bayesian":
                        from skopt import BayesSearchCV
                        from skopt.space import Categorical as SkoptCategorical
                        # Convert discrete lists to skopt Categorical spaces.
                        # GP surrogate learns to avoid poor regions even in categorical spaces.
                        bayes_space = {k: SkoptCategorical(v) for k, v in param_space.items()}
                        search = BayesSearchCV(
                            estimator=estimator,
                            search_spaces=bayes_space,
                            n_iter=min(max_iter_per_model, total_combinations),
                            scoring=effective_sklearn_scorer,
                            cv=cv,
                            random_state=random_state,
                            n_jobs=1,  # BayesSearchCV is not thread-safe with n_jobs=-1
                        )
                        actual_strategy = "bayesian"
                    else:
                        search = RandomizedSearchCV(
                            estimator=estimator,
                            param_distributions=param_space,
                            n_iter=min(max_iter_per_model, total_combinations),
                            scoring=effective_sklearn_scorer,
                            cv=cv,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                        actual_strategy = "random"

                    search.fit(X, y)
                    best_idx = search.best_index_
                    cv_std = round(float(search.cv_results_["std_test_score"][best_idx]), 4)
                    summary[model_name] = {
                        "status": "success",
                        "best_params": _to_builtin(search.best_params_),
                        "cv_best_score": round(float(search.best_score_), 4),
                        "cv_score_std": cv_std,
                        "scoring": effective_scoring_name,
                        "rows_trained": int(len(df)),
                        "search_strategy": actual_strategy,
                    }

                score = float(summary[model_name]["cv_best_score"])
                if score > best_score:
                    best_score = score
                    best_model_name = model_name

            except Exception as model_exc:
                # 记录失败原因，继续处理下一个候选模型
                summary[model_name] = {
                    "status": "failed",
                    "error": f"{type(model_exc).__name__}: {model_exc}",
                    "cv_best_score": None,
                }

        # 若全部模型均失败，整体报错
        successful = [m for m, v in summary.items() if v.get("status") == "success"]
        if not successful:
            failed_reasons = {m: v["error"] for m, v in summary.items()}
            raise ValueError(f"所有候选模型均失败，无法选出最佳模型。失败原因：{failed_reasons}")

        if not best_model_name:
            raise ValueError("无法从候选模型中选出最佳模型。")

        skipped_models = [m for m, v in summary.items() if v.get("status") == "failed"]
        result = {
            "best_model_name": best_model_name,
            "best_model_params": summary[best_model_name]["best_params"],
            "selection_metric": effective_scoring_name,
            "n_classes": n_classes,
            "cv_strategy_used": cv_strategy_used,
            "search_strategy_used": hpo_strategy,
            "best_cv_score": round(best_score, 4),
            "per_model_summary": summary,
            "skipped_models": skipped_models,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("tune_and_select_model", tool_inputs, output)
    return output
