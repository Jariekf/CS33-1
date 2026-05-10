"""Tool registries used by worker agents.

Keeping registries explicit makes tool selection auditable and prevents agents
from calling arbitrary functions.
"""

from __future__ import annotations

from tools import (
    data_cleaning_tools,
    data_intake_tools,
    eda_tools,
    feature_engineering_tools,
    model_policy_tools,
    model_selection_tools,
    model_tuning_tools,
    target_inference_tools,
    validation_tools,
)


DATA_INTAKE_TOOLS = {
    "dataset_intake": data_intake_tools.dataset_intake,
    "read_preview": data_intake_tools.read_preview,
    "build_preview_metadata": data_intake_tools.build_preview_metadata,
    "detect_encoding": data_intake_tools.detect_encoding,
    "detect_delimiter": data_intake_tools.detect_delimiter,
}


DATA_CLEANING_TOOLS = {
    "basic_clean_dataframe": data_cleaning_tools.basic_clean_dataframe,
    "decide_duplicate_removal": data_cleaning_tools.decide_duplicate_removal,
    "drop_duplicate_rows": data_cleaning_tools.drop_duplicate_rows,
    "drop_empty_columns": data_cleaning_tools.drop_empty_columns,
    "impute_missing_values": data_cleaning_tools.impute_missing_values,
    "numeric_like_conversion_report": data_cleaning_tools.numeric_like_conversion_report,
    "parse_numeric_like_columns": data_cleaning_tools.parse_numeric_like_columns,
    "profile_dataframe": data_cleaning_tools.profile_dataframe,
    "split_cleaned_data": data_cleaning_tools.split_cleaned_data,
}

EDA_TOOLS = {
    "build_eda_report": eda_tools.build_eda_report,
    "build_eda_summary": eda_tools.build_eda_summary,
}

FEATURE_ENGINEERING_TOOLS = {
    "recommend_feature_engineering_policy": feature_engineering_tools.recommend_feature_engineering_policy,
    "apply_feature_policy": feature_engineering_tools.apply_feature_policy,
    "apply_model_feature_policy": feature_engineering_tools.apply_model_feature_policy,
    "align_and_validate_feature_matrices": feature_engineering_tools.align_and_validate_feature_matrices,
    "clip_numeric_outliers_with_report": feature_engineering_tools.clip_numeric_outliers_with_report,
    "convert_boolean_like_features": feature_engineering_tools.convert_boolean_like_features,
    "create_datetime_features_with_report": feature_engineering_tools.create_datetime_features_with_report,
    "create_missing_indicators_with_report": feature_engineering_tools.create_missing_indicators_with_report,
    "drop_target_leakage_features_with_report": feature_engineering_tools.drop_target_leakage_features_with_report,
    "encode_categorical_features_with_report": feature_engineering_tools.encode_categorical_features_with_report,
    "impute_missing_features": feature_engineering_tools.impute_missing_features,
    "impute_missing_features_with_report": feature_engineering_tools.impute_missing_features_with_report,
    "profile_features": feature_engineering_tools.profile_features,
    "remove_highly_correlated_features_with_report": feature_engineering_tools.remove_highly_correlated_features_with_report,
    "remove_low_information_features_with_report": feature_engineering_tools.remove_low_information_features_with_report,
    "scale_numeric_features": feature_engineering_tools.scale_numeric_features,
    "scale_numeric_features_with_report": feature_engineering_tools.scale_numeric_features_with_report,
}

MODEL_SELECTION_TOOLS = {
    "infer_task_type": model_selection_tools.infer_task_type,
    "train_and_evaluate_model": model_selection_tools.train_and_evaluate_model,
    "train_and_select_model": model_selection_tools.train_and_select_model,
}

MODEL_TUNING_TOOLS = {
    "default_tuning_plan": model_tuning_tools.default_tuning_plan,
    "llm_tuning_plan": model_tuning_tools.llm_tuning_plan,
    "tune_and_evaluate_model": model_tuning_tools.tune_and_evaluate_model,
}

MODEL_POLICY_TOOLS = {
    "candidate_model_plans": model_policy_tools.candidate_model_plans,
    "candidate_model_ids": model_policy_tools.candidate_model_ids,
}

TARGET_INFERENCE_TOOLS = {
    "infer_target_and_task": target_inference_tools.infer_target_and_task,
}

VALIDATION_TOOLS = {
    "validate_csv": validation_tools.validate_csv,
    "validate_json_object": validation_tools.validate_json_object,
    "validate_matching_columns": validation_tools.validate_matching_columns,
}
