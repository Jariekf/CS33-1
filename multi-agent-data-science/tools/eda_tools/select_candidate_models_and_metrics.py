import json
from typing import Any

from langchain_core.tools import tool

from tools.tool_trace import trace_tool_call

# -- Metrics whitelist --
# primary_metric / evaluation_metrics both use human-readable names (rmse/mae rather than neg_* prefix)
# tune_and_select_model internally maps rmse -> neg_root_mean_squared_error
CLASSIFICATION_METRICS = {"accuracy", "f1_macro", "f1_weighted", "roc_auc", "precision", "recall", "log_loss"}
REGRESSION_METRICS = {"r2", "rmse", "mae", "mape", "rmsle", "max_error"}

CLASSIFICATION_POOL = [
    "LogisticRegression",    # linear baseline; strong interpretability
    "RandomForest",          # ensemble trees; stable and noise-resistant
    "XGBoost",               # gradient boosting; competition-grade performance
    "LightGBM",              # faster gradient boosting; preferred for large datasets
    "GradientBoosting",      # sklearn native gradient boosting; no extra dependencies
    "ExtraTrees",            # extremely randomized trees; fast with low variance
    "AdaBoost",              # adaptive boosting; effective for imbalanced data
    "SVM",                   # kernel method; performs well on small-to-medium datasets
    "KNN",                   # instance-based learning; no parametric assumption
    "GaussianNB",            # naive Bayes; extremely fast probabilistic baseline
    "MLPClassifier",         # multilayer perceptron; advantageous when many numeric features or large sample size
]

REGRESSION_POOL = [
    "LinearRegression",          # linear baseline
    "Ridge",                     # L2 regularization; guards against multicollinearity
    "Lasso",                     # L1 regularization; built-in feature selection
    "ElasticNet",                # L1+L2; balances sparsity and stability
    "HuberRegressor",            # robust regression; insensitive to outliers
    "RandomForestRegressor",     # ensemble trees
    "XGBoostRegressor",          # gradient boosting
    "LightGBMRegressor",         # faster gradient boosting
    "GradientBoostingRegressor", # sklearn native gradient boosting
    "ExtraTreesRegressor",       # extremely randomized trees
    "SVR",                       # support vector regression
    "MLPRegressor",              # multilayer perceptron; optional for complex nonlinear relationships
]


@tool
def select_candidate_models_and_metrics(
    task_type: str,
    selected_models: list[str],
    feature_engineering_plan: dict[str, Any],
    model_selection_plan: str,
    primary_metric: str,
    evaluation_metrics: list[str],
    metrics_rationale: str,
) -> str:
    """Submit the three candidate models selected after EDA analysis, the feature engineering plan, the model selection rationale, and the evaluation metric decisions.

    Args:
      task_type (str): "classification" or "regression".

      selected_models (list[str]): exactly 3 model names selected from the candidate pool; case-sensitive.
        Classification pool: LogisticRegression, RandomForest, XGBoost, LightGBM, GradientBoosting,
                    ExtraTrees, AdaBoost, SVM, KNN, GaussianNB, MLPClassifier
        Regression pool: LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor,
                    RandomForestRegressor, XGBoostRegressor, LightGBMRegressor,
                    GradientBoostingRegressor, ExtraTreesRegressor, SVR, MLPRegressor
        Invalid examples: xgboost, random_forest, XGboost (all will be rejected).

      feature_engineering_plan (dict): feature engineering plan for each model; key is the model name, value is a plan dict.
        Example:
        {
          "XGBoost":            {"scaling": false, "encoding": "label",  "notes": "tree model; no scaling needed"},
          "RandomForest":       {"scaling": false, "encoding": "label",  "notes": "ensemble trees; no scaling needed"},
          "LogisticRegression": {"scaling": true,  "encoding": "onehot", "notes": "linear model requires standardization"}
        }

      model_selection_plan (str): rationale for choosing these three models, explained in terms of data characteristics, task type, and target column.

      primary_metric (str): the single optimization metric used for CV hyperparameter tuning (determines the criterion for "best parameters").
        Classification options: accuracy, f1_macro, f1_weighted, roc_auc, precision, recall, log_loss
        Regression options: r2, rmse, mae, mape, rmsle, max_error
        Selection guidance (classification):
          - Class-balanced; overall accuracy is the main concern -> f1_weighted
          - Minority classes exist; need to monitor class-imbalance risk -> f1_macro
            (macro weights each class equally; a drop in minority-class performance is immediately reflected in CV)
          - Severely imbalanced classes; focus on minority-class recall -> roc_auc or recall
          - Multi-class (number of classes > 2) -> f1_macro or f1_weighted;
            roc_auc will be automatically upgraded to roc_auc_ovr_weighted,
            but if the candidate models include SVM (does not support predict_proba by default), switch to f1_macro
          - Business goal is probability calibration (CTR, click rate, risk scoring, etc.) -> log_loss
            Note: log_loss requires all candidate models to support predict_proba;
            SVM does not support it by default and should be removed from the candidates or another primary_metric should be chosen
        Selection guidance (regression):
          - Default -> r2 (global goodness-of-fit; intuitive)
          - Significant outliers present -> mae (more robust to outliers than rmse)
          - Target distribution is markedly skewed (skewness > 1, e.g. house prices, sales volume, traffic) -> rmsle
            Note: rmsle requires y_true >= 0, and penalizes under-prediction more than over-prediction
          - Business cares about "percentage deviation"; target has no zero values -> mape
            Note: mape requires y_true != 0; when zeros are present the tool automatically degrades to None
          - Rarely use max_error as primary_metric; suitable for scenarios where precision requirements are strict
            and worst-case errors are not allowed (e.g. part-tolerance prediction)

      evaluation_metrics (list[str]): list of metrics reported during the final test-set comprehensive evaluation (2-4 items).
        Classification options: accuracy, f1_macro, f1_weighted, roc_auc, precision, recall, log_loss
        Regression options: r2, rmse, mae, mape, rmsle, max_error
        Recommendation: primary_metric must be included in this list; optionally add 1-2 complementary metrics.
        Classification strongly recommended: include both f1_macro and f1_weighted; the former reflects minority-class risk, the latter reflects overall distribution performance.
        Example (balanced classification): ["f1_weighted", "f1_macro", "accuracy"]
        Example (imbalanced classification): ["roc_auc", "f1_macro", "f1_weighted"]
        Example (probability calibration scenario): ["log_loss", "f1_weighted", "roc_auc"]
        Example (regression, skewed target): ["rmsle", "mae", "r2"]
        Example (regression, worst-case focus): ["rmse", "mae", "max_error"]

      metrics_rationale (str): rationale for choosing the above metrics, explained in terms of task type, data characteristics, and business goals.
    """
    tool_inputs = {
        "task_type": task_type,
        "selected_models": selected_models,
        "feature_engineering_plan": feature_engineering_plan,
        "model_selection_plan": model_selection_plan,
        "primary_metric": primary_metric,
        "evaluation_metrics": evaluation_metrics,
        "metrics_rationale": metrics_rationale,
    }
    try:
        # -- Model name validation --
        pool = CLASSIFICATION_POOL if task_type == "classification" else REGRESSION_POOL
        invalid_models = [m for m in selected_models if m not in pool]
        if invalid_models:
            raise ValueError(f"Models not in candidate pool: {invalid_models}. Candidate pool: {pool}")
        if len(selected_models) != 3:
            raise ValueError(f"Exactly 3 models must be selected; received {len(selected_models)}.")

        # -- Metrics validation --
        allowed = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
        if primary_metric not in allowed:
            raise ValueError(
                f"primary_metric '{primary_metric}' is invalid. {task_type} options: {sorted(allowed)}"
            )
        invalid_metrics = [m for m in evaluation_metrics if m not in allowed]
        if invalid_metrics:
            raise ValueError(
                f"evaluation_metrics contains invalid items: {invalid_metrics}. {task_type} options: {sorted(allowed)}"
            )
        if not evaluation_metrics:
            raise ValueError("evaluation_metrics must not be an empty list.")
        if primary_metric not in evaluation_metrics:
            raise ValueError(
                f"primary_metric '{primary_metric}' must be included in evaluation_metrics."
            )

        result = {
            "selected_models": selected_models,
            "feature_engineering_plan": feature_engineering_plan,
            "model_selection_plan": model_selection_plan,
            "primary_metric": primary_metric,
            "evaluation_metrics": evaluation_metrics,
            "metrics_rationale": metrics_rationale,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        output = json.dumps(
            {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
            ensure_ascii=False,
        )

    trace_tool_call("select_candidate_models_and_metrics", tool_inputs, output)
    return output
