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

# -- Whitelist (human-readable display names) --
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

# display name -> sklearn scoring string (only entries that need conversion; entries where display name == sklearn name are omitted)
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
# random   - RandomizedSearchCV: random sampling; balanced default option
# grid     - GridSearchCV: exhaustive search over all combinations; suitable only for models with a small parameter space (total combinations <= 30)
# halving  - HalvingRandomSearchCV: progressively eliminate inferior candidates; 2-3x faster than random on large spaces
# bayesian - BayesSearchCV (scikit-optimize): GP surrogate model guides the search; most efficient under a limited iteration budget

# Scorers in sklearn that only apply to binary classification; must be upgraded to the corresponding multi-class version for multi-class tasks.
_MULTICLASS_SCORER_UPGRADE: dict[str, str] = {
    "roc_auc": "roc_auc_ovr_weighted",
    "recall":  "recall_weighted",
    "precision": "precision_weighted",
}

# Scorer names that have no sklearn built-in string and require a custom scorer object
_CUSTOM_SCORER_NAMES: frozenset[str] = frozenset({"rmsle"})


def _to_sklearn_scorer(display_name: str) -> Any:
    """display name -> actual sklearn scorer (string or scorer object).

    - Standard metrics: converted to a sklearn scoring string via _SCORING_ALIAS.
    - rmsle: no built-in string; build a make_scorer; clip predictions to >= 0 to prevent log-domain errors.
    - All other display names that match the sklearn name (e.g. r2, f1_weighted): returned as-is.
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
    """Validate and return the normalized display name (human-readable name).

    Always returns the display name (e.g. "rmse", "mape", "roc_auc"), not a sklearn neg_* string.
    Call _to_sklearn_scorer(display_name) to obtain the scorer actually passed to the sklearn API.
    Falls back to the task default when the value is invalid or an empty string.
    """
    allowed = _ALLOWED_SCORING.get(task_type, set())
    sklearn_name = _SCORING_ALIAS.get(scoring, scoring)
    if scoring in _CUSTOM_SCORER_NAMES:
        return scoring
    if scoring and (scoring in allowed or sklearn_name in allowed):
        return scoring  # return display name, sklearn conversion deferred to _to_sklearn_scorer
    return _DEFAULT_SCORING.get(task_type, "r2")


def _adapt_scorer_for_multiclass(scoring: str, n_classes: int) -> str:
    """Binary-classification-only scorers crash on multi-class (> 2 classes) tasks; automatically upgrade to the multi-class compatible version.

    Upgrade mappings:
      roc_auc   -> roc_auc_ovr_weighted  (OvR + support-weighted; sklearn-compatible)
      recall    -> recall_weighted
      precision -> precision_weighted

    Note: the upgrade only resolves multi-class compatibility; the best response to class imbalance
    is StratifiedKFold + EDA agent selecting f1_weighted / roc_auc_ovr_weighted as primary_metric.
    """
    if n_classes <= 2:
        return scoring
    return _MULTICLASS_SCORER_UPGRADE.get(scoring, scoring)


def _build_cv(cv_strategy: str, task_type: str, cv_folds: int) -> Any:
    """Build a sklearn CV object based on strategy and task type.

    Strategy descriptions:
      auto       - StratifiedKFold for classification, KFold for regression (default)
      stratified - StratifiedKFold; recommended for class-imbalanced classification tasks
      kfold      - plain KFold; suitable for regression or class-balanced classification tasks
      timeseries - TimeSeriesSplit; must be used when data has a temporal ordering to avoid data leakage
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
    """Perform CV hyperparameter search and model selection for candidate models, returning the best model and optimal parameters for each.

    This tool only uses the training set (feature_dataset_ids) and never touches the test set.
    It does not save model files; the final retraining is done by train_model.

    Args:
      feature_dataset_ids (dict): paths to the feature-engineered training set for each candidate model.
      candidate_models (list): list of candidate model names.
      task_type (str): "classification" or "regression".
      target_column (str): target column name.
      cv_folds (int): number of cross-validation folds; default 5.
        Recommendation: training set < 1000 rows -> 3 folds; 1000-20000 rows -> 5 folds; > 20000 rows -> 10 folds for a more stable estimate.
      max_iter_per_model (int): upper limit on the number of samples per model when search_strategy="random"/"halving"; default 12.
        Ignored when search_strategy="grid" (exhaustive search).
        Recommendation: parameter dimensions <= 2 -> 8-12; dimensions 3-4 -> 15-25; dimensions >= 5 -> 25-40.
      random_state (int): random seed; default 42.
      scoring (str): scoring metric; uses the task default when empty.
        Classification options: f1_weighted (default), f1_macro, roc_auc, accuracy, recall, precision, log_loss
          f1_macro    - equal-weight per-class average; more sensitive to minority classes; suitable for class-imbalanced scenarios
          f1_weighted - sample-count-weighted average; reflects performance on the actual distribution; the default option
          log_loss    - continuous probability calibration metric (preferred for CTR estimation, risk scoring, etc.);
                        requires models to support predict_proba; SVM does not support it by default and should not be used together
        Regression options: r2 (default), rmse, mae, mape, rmsle, max_error
          Technical constraints: mape produces inf during CV when y_true contains 0 (safely degraded);
            rmsle uses make_scorer (no built-in sklearn string);
            cv_best_score for max_error is negative (closer to 0 is better).
        Note: roc_auc, recall, precision are binary-classification-only; when the target column has > 2 classes they are automatically upgraded to
        roc_auc_ovr_weighted / recall_weighted / precision_weighted.
      cv_strategy (str): cross-validation strategy; default "auto".
        auto       - StratifiedKFold for classification, KFold for regression
        stratified - StratifiedKFold; recommended for class-imbalanced tasks
        kfold      - plain KFold
        timeseries - TimeSeriesSplit; must be used when data has a temporal ordering, otherwise data leakage will occur
      search_strategy (str): hyperparameter search strategy; default "random".
        random   - RandomizedSearchCV: random sampling for max_iter_per_model iterations. Balanced default option.
        grid     - GridSearchCV: exhaustive search over all parameter combinations.
                   Recommended: use only when the total number of combinations in the model's parameter space is <= 30 (e.g. Ridge, Lasso, LogisticRegression).
        halving  - HalvingRandomSearchCV: progressively eliminate poor-performing candidates; explores more configurations than random in the same time.
                   Recommended: use when the parameter space is large (combinations > 100) or there are many candidate models (e.g. XGBoost, RandomForest).
        bayesian - BayesSearchCV (scikit-optimize): Gaussian-process surrogate model guides the search;
                   typically finds better parameter combinations than random/halving with the same iteration budget.
                   Recommended: use when max_iter_per_model is small (<= 20) and you want to maximize search quality.
                   All list-type parameters are automatically converted to Categorical discrete spaces with no additional configuration.

      extra_param_overrides (dict | None): inject or override specific parameters in the search space by model name; default None.
        key is the model name, value is a dict of parameters to append/override (merged on top of the default space from _get_param_space).
        Pass only the parameters that need adjustment; all other parameters remain in the default space.

        Common injection scenarios:
          1. Imbalanced classification (for models that support class_weight):
             {"LogisticRegression": {"class_weight": ["balanced", None]},
              "RandomForest":       {"class_weight": ["balanced", None]},
              "SVM":                {"class_weight": ["balanced", None]}}
          2. Expand n_estimators upper bound for large-sample tree models:
             {"RandomForest":    {"n_estimators": [200, 300, 500]},
              "XGBoost":         {"n_estimators": [200, 300, 500]}}
          3. Adjust max_features for high-dimensional features:
             {"RandomForest": {"max_features": ["sqrt", "log2", 0.3, 0.5]}}

        Note: injected parameter names and types must exactly match the model's sklearn/xgboost parameter interface,
        otherwise a ValueError will be raised at fit() time. When uncertain, omit and use the default space.

    Returns JSON containing:
      best_model_name, best_model_params,
      selection_metric (the scorer actually used; may have been upgraded for multi-class),
      n_classes (number of unique target classes, for downstream determination of binary vs. multi-class),
      cv_strategy_used, search_strategy_used, best_cv_score,
      per_model_summary (results for each model, including a status field):
        - status="success": best_params / cv_best_score / cv_score_std / scoring / search_strategy
        - status="failed": error (reason for failure, including exception type and message); the model is excluded from best selection
      skipped_models (list of failed models, for model_agent to analyze failure reasons)
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
            raise ValueError("candidate_models must not be empty.")

        # -- Multi-class detection: probe the number of unique classes in the target column using the first available dataset --
        # Only needed for classification tasks; resolved_scoring for regression tasks does not contain binary-only scorers.
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
            print(f"Starting tuning for model: {model_name} (search strategy: {hpo_strategy})...", flush=True)
            # -- Per-model fallback: a failure in one model does not affect the others --
            try:
                featured_train_path = feature_dataset_ids.get(model_name, "")
                if not featured_train_path:
                    raise ValueError(f"Path for '{model_name}' is missing in feature_dataset_ids.")

                df = pd.read_csv(featured_train_path)
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in training set.")

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
                            verbose=1,
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
                            verbose=1,
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
                            verbose=1,
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
                            verbose=1,
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
                # Record the failure reason and continue processing the next candidate model.
                summary[model_name] = {
                    "status": "failed",
                    "error": f"{type(model_exc).__name__}: {model_exc}",
                    "cv_best_score": None,
                }

        # If all models failed, raise an overall error.
        successful = [m for m, v in summary.items() if v.get("status") == "success"]
        if not successful:
            failed_reasons = {m: v["error"] for m, v in summary.items()}
            raise ValueError(f"All candidate models failed; unable to select the best model. Failure reasons: {failed_reasons}")

        if not best_model_name:
            raise ValueError("Unable to select the best model from the candidates.")

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
