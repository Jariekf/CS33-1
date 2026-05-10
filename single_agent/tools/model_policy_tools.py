"""Model policy tools for task-specific candidate model plans."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MODEL_POLICIES: dict[str, list[dict[str, Any]]] = {
    "classification": [
        {
            "model_id": "logistic_regression",
            "task_type": "classification",
            "estimator": "LogisticRegression",
            "preprocessing": {
                "categorical_encoding": "one_hot",
                "numeric_imputation": "median",
                "categorical_imputation": "mode",
                "numeric_scaling": True,
                "numeric_outlier_clipping": True,
                "remove_constant_features": True,
                "correlation_threshold": 0.98,
                "drop_target_leakage_features": True,
                "requires_all_features_numeric": True,
                "allows_missing_feature_values": False,
            },
            "validation": {
                "selection_metric": "f1_weighted",
                "required_checks": [
                    "all_features_numeric",
                    "no_missing_feature_values",
                    "train_test_columns_match",
                    "target_column_present",
                ],
            },
        },
        {
            "model_id": "random_forest_classifier",
            "task_type": "classification",
            "estimator": "RandomForestClassifier",
            "preprocessing": {
                "categorical_encoding": "one_hot",
                "numeric_imputation": "median",
                "categorical_imputation": "mode",
                "numeric_scaling": False,
                "numeric_outlier_clipping": True,
                "remove_constant_features": True,
                "correlation_threshold": None,
                "drop_target_leakage_features": True,
                "requires_all_features_numeric": True,
                "allows_missing_feature_values": False,
            },
            "validation": {
                "selection_metric": "f1_weighted",
                "required_checks": [
                    "all_features_numeric",
                    "no_missing_feature_values",
                    "train_test_columns_match",
                    "target_column_present",
                ],
            },
        },
        {
            "model_id": "gradient_boosting_classifier",
            "task_type": "classification",
            "estimator": "GradientBoostingClassifier",
            "preprocessing": {
                "categorical_encoding": "one_hot",
                "numeric_imputation": "median",
                "categorical_imputation": "mode",
                "numeric_scaling": False,
                "numeric_outlier_clipping": True,
                "remove_constant_features": True,
                "correlation_threshold": None,
                "drop_target_leakage_features": True,
                "requires_all_features_numeric": True,
                "allows_missing_feature_values": False,
            },
            "validation": {
                "selection_metric": "f1_weighted",
                "required_checks": [
                    "all_features_numeric",
                    "no_missing_feature_values",
                    "train_test_columns_match",
                    "target_column_present",
                ],
            },
        },
    ],
    "regression": [
        {
            "model_id": "ridge_regression",
            "task_type": "regression",
            "estimator": "Ridge",
            "preprocessing": {
                "categorical_encoding": "one_hot",
                "numeric_imputation": "median",
                "categorical_imputation": "mode",
                "numeric_scaling": True,
                "numeric_outlier_clipping": True,
                "remove_constant_features": True,
                "correlation_threshold": 0.98,
                "drop_target_leakage_features": True,
                "requires_all_features_numeric": True,
                "allows_missing_feature_values": False,
            },
            "validation": {
                "selection_metric": "r2",
                "required_checks": [
                    "all_features_numeric",
                    "no_missing_feature_values",
                    "train_test_columns_match",
                    "target_column_present",
                ],
            },
        },
        {
            "model_id": "random_forest_regressor",
            "task_type": "regression",
            "estimator": "RandomForestRegressor",
            "preprocessing": {
                "categorical_encoding": "one_hot",
                "numeric_imputation": "median",
                "categorical_imputation": "mode",
                "numeric_scaling": False,
                "numeric_outlier_clipping": True,
                "remove_constant_features": True,
                "correlation_threshold": None,
                "drop_target_leakage_features": True,
                "requires_all_features_numeric": True,
                "allows_missing_feature_values": False,
            },
            "validation": {
                "selection_metric": "r2",
                "required_checks": [
                    "all_features_numeric",
                    "no_missing_feature_values",
                    "train_test_columns_match",
                    "target_column_present",
                ],
            },
        },
        {
            "model_id": "gradient_boosting_regressor",
            "task_type": "regression",
            "estimator": "GradientBoostingRegressor",
            "preprocessing": {
                "categorical_encoding": "one_hot",
                "numeric_imputation": "median",
                "categorical_imputation": "mode",
                "numeric_scaling": False,
                "numeric_outlier_clipping": True,
                "remove_constant_features": True,
                "correlation_threshold": None,
                "drop_target_leakage_features": True,
                "requires_all_features_numeric": True,
                "allows_missing_feature_values": False,
            },
            "validation": {
                "selection_metric": "r2",
                "required_checks": [
                    "all_features_numeric",
                    "no_missing_feature_values",
                    "train_test_columns_match",
                    "target_column_present",
                ],
            },
        },
    ],
}


def candidate_model_plans(task_type: str) -> list[dict[str, Any]]:
    """Return concrete candidate model plans for a supported task type."""

    return deepcopy(MODEL_POLICIES.get(task_type, []))


def candidate_model_ids(task_type: str) -> list[str]:
    """Return concrete candidate model IDs for a supported task type."""

    return [plan["model_id"] for plan in candidate_model_plans(task_type)]
