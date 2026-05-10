"""Manager agent node.

The Manager defines task-level intent and stage contracts. It does not write
detailed worker instructions; workers make local plans from real artifacts.
"""

from __future__ import annotations

from core.state_schema import MANAGER_PLAN_PATH, PLAN_PATCH_LOG_PATH, WorkflowState
from tools.data_intake_tools import build_preview_metadata, dataset_intake, read_preview
from tools.data_cleaning_tools import normalize_column_names, numeric_like_conversion_report, parse_numeric_like_columns
from tools.model_policy_tools import candidate_model_plans
from tools.model_selection_tools import select_metric
from tools.target_inference_tools import infer_target_and_task
from tools.validation_tools import write_json

SUPPORTED_SUPERVISED_TASKS = {"classification", "regression"}


def manager_node(state: WorkflowState) -> dict:
    """Create an initial global plan from user goal and input data."""

    data_path = state["data_path"]
    test_data_path = state.get("test_data_path")
    input_mode = state.get("input_mode", "single_file_split")
    user_goal = state.get("user_goal", "")
    data_description = state.get("data_description", "")
    intake = dataset_intake(data_path)
    read_options = intake["data_reading_options"]
    raw_preview = normalize_column_names(read_preview(data_path, read_options))
    test_intake = dataset_intake(test_data_path) if test_data_path else None
    initial_inference = infer_target_and_task(raw_preview, user_goal, data_description=data_description)
    target_col = initial_inference["target_column"]
    preview = parse_numeric_like_columns(raw_preview, target_col=target_col)
    if _should_parse_target_for_inference(raw_preview, target_col, user_goal, data_description):
        preview = parse_numeric_like_columns(preview, target_col=target_col, parse_target=True)
    inference = infer_target_and_task(preview, user_goal, data_description=data_description)
    target_col = inference["target_column"] or target_col
    task_type = inference["task_type"]
    workflow_type = _workflow_type(task_type, target_col)
    supported_by_current_pipeline = workflow_type == "supervised"
    metric = select_metric(task_type) if supported_by_current_pipeline else None
    candidate_models = candidate_model_plans(task_type) if supported_by_current_pipeline else []
    intake["preview_metadata"] = build_preview_metadata(data_path, preview)
    if test_intake:
        test_preview = normalize_column_names(read_preview(test_data_path, test_intake["data_reading_options"]))
        test_intake["preview_metadata"] = build_preview_metadata(test_data_path, test_preview)
    risk_notes = _risk_notes(task_type, target_col, supported_by_current_pipeline, intake)
    plan = {
        "dataset_intake": intake,
        "input_config": {
            "input_mode": input_mode,
            "train_data_path": data_path,
            "test_data_path": test_data_path,
            "test_data_reading_options": test_intake["data_reading_options"] if test_intake else None,
            "test_has_target": bool(test_intake and target_col in test_intake["preview_metadata"]["column_names"]),
        },
        "test_dataset_intake": test_intake,
        "task_summary": {
            "business_goal": user_goal,
            "data_description": data_description,
            "task_type": task_type,
            "workflow_type": workflow_type,
            "supported_by_current_pipeline": supported_by_current_pipeline,
            "target_column": target_col,
            "initial_target_inference": initial_inference,
            "target_inference": inference,
            "primary_metric": metric,
            "candidate_model_ids": [model["model_id"] for model in candidate_models],
            "candidate_models": candidate_models,
            "risk_notes": risk_notes,
            "data_reading_options": read_options,
            "test_data_reading_options": test_intake["data_reading_options"] if test_intake else None,
            "input_mode": input_mode,
            "initial_feature_engineering_policy": {
                "mode": "light",
                "can_be_skipped": True,
                "decision_required_after_cleaning": True,
            },
        },
        "stage_contracts": {
            "data_cleaning": {
                "required_artifacts": ["cleaned_data.csv", "train_data.csv", "test_data.csv", "cleaning_report.json"],
                "required_report_fields": [
                    "stage",
                    "status",
                    "target_column",
                    "profile_before",
                    "profile_after",
                    "row_loss",
                    "row_loss_breakdown",
                    "duplicate_policy",
                    "numeric_like_conversions",
                    "split",
                    "recommended_tool_gaps",
                ],
                "validation_rules": [
                    {"id": "target_preserved", "check": "target column exists after cleaning when workflow_type is supervised"},
                    {"id": "non_empty_columns", "check": "cleaned dataset has at least one non-empty column"},
                    {"id": "row_loss_reported", "check": "cleaning_report.json includes row_loss"},
                ],
                "allowed_tools": [
                    "basic_clean_dataframe",
                    "decide_duplicate_removal",
                    "drop_duplicate_rows",
                    "drop_empty_columns",
                    "numeric_like_conversion_report",
                    "profile_dataframe",
                    "split_cleaned_data",
                ],
                "recommended_tool_gap_field": "recommended_tool_gaps",
            },
            "feature_engineering": {
                "required_artifacts": ["fe_train.csv", "fe_test.csv", "feature_policy_report.json", "feature_report.json"],
                "required_report_fields": [
                    "stage",
                    "status",
                    "policy",
                    "pipelines",
                    "execution",
                    "train_shape",
                    "test_shape",
                    "target_column",
                    "recommended_tool_gaps",
                ],
                "validation_rules": [
                    {"id": "policy_justified", "check": "feature policy mode and reason are reported"},
                    {"id": "matching_columns", "check": "feature train/test columns match in order"},
                    {"id": "target_preserved", "check": "target column remains present when workflow_type is supervised"},
                    {"id": "model_preprocessing_requirements", "check": "feature policy accounts for candidate_models preprocessing requirements"},
                ],
                "allowed_tools": [
                    "recommend_feature_engineering_policy",
                    "apply_model_feature_policy",
                    "align_and_validate_feature_matrices",
                    "clip_numeric_outliers_with_report",
                    "convert_boolean_like_features",
                    "create_datetime_features_with_report",
                    "create_missing_indicators_with_report",
                    "drop_target_leakage_features_with_report",
                    "encode_categorical_features_with_report",
                    "impute_missing_features",
                    "impute_missing_features_with_report",
                    "profile_features",
                    "remove_highly_correlated_features_with_report",
                    "remove_low_information_features_with_report",
                    "scale_numeric_features_with_report",
                    "validate_matching_columns",
                ],
                "recommended_tool_gap_field": "recommended_tool_gaps",
            },
            "eda_analysis": {
                "required_artifacts": ["eda_report.json", "eda_summary.md"],
                "required_report_fields": [
                    "stage",
                    "status",
                    "leakage_policy",
                    "dataset_overview",
                    "target_analysis",
                    "feature_target_relationships",
                    "feature_engineering_recommendations",
                ],
                "validation_rules": [
                    {"id": "train_only_target_analysis", "check": "feature-target analysis uses train_data only"},
                    {"id": "eda_summary_present", "check": "eda_summary.md exists for business reporting"},
                ],
                "allowed_tools": ["build_eda_report", "build_eda_summary"],
                "recommended_tool_gap_field": "recommended_tool_gaps",
            },
            "model_selection": {
                "required_artifacts": ["model_report.json", "feature_importance.json"],
                "required_report_fields": [
                    "task_type",
                    "target_column",
                    "candidate_model_ids",
                    "metric",
                    "candidate_results",
                    "selected_model_id",
                    "best_model",
                    "best_cv_score",
                    "test_metrics",
                    "tuning",
                    "recommended_tool_gaps",
                ],
                "validation_rules": [
                    {"id": "metric_matches_task", "check": "selected metric matches task_summary.primary_metric"},
                    {"id": "test_metrics_reported", "check": "model_report.json includes final test metrics"},
                    {"id": "supervised_only", "check": "model selection runs only when supported_by_current_pipeline is true"},
                    {"id": "model_id_tracked", "check": "model_report.json identifies evaluated and selected model IDs"},
                ],
                "allowed_tools": [
                    "infer_task_type",
                    "default_tuning_plan",
                    "llm_tuning_plan",
                    "tune_and_evaluate_model",
                    "train_and_evaluate_model",
                    "validate_json_object",
                ],
                "recommended_tool_gap_field": "recommended_tool_gaps",
            },
        },
    }
    write_json(MANAGER_PLAN_PATH, plan)
    return {
        "global_plan": plan,
        "plan_version": 1,
        "plan_patches": [],
        "stage_reports": {},
        "manager_plan_path": MANAGER_PLAN_PATH,
    }


def _workflow_type(task_type: str, target_col: str | None) -> str:
    """Map inferred task type to a pipeline-level workflow category."""

    if task_type in SUPPORTED_SUPERVISED_TASKS and target_col:
        return "supervised"
    if task_type == "descriptive":
        return "descriptive"
    return "unsupported"


def _should_parse_target_for_inference(
    df,
    target_col: str | None,
    user_goal: str,
    data_description: str = "",
) -> bool:
    """Return whether an inferred target should be numeric-parsed for task inference."""

    if not target_col or target_col not in df.columns:
        return False
    conversions = numeric_like_conversion_report(df[[target_col]], target_col=target_col, parse_target=True)
    if not conversions:
        return False
    target_hint = target_col.lower()
    goal = f"{user_goal}\n{data_description}".lower()
    numeric_goal_words = ("financial", "sales", "revenue", "profit", "price", "cost", "amount", "units", "predict")
    numeric_target_words = ("sales", "revenue", "profit", "price", "cost", "amount", "units", "gross", "cogs")
    return any(word in goal for word in numeric_goal_words) or any(word in target_hint for word in numeric_target_words)


def _risk_notes(
    task_type: str,
    target_col: str | None,
    supported_by_current_pipeline: bool,
    intake: dict[str, Any],
) -> list[str]:
    """Return concise planning risks visible to downstream agents."""

    notes: list[str] = []
    if not target_col:
        notes.append("No reliable supervised target was inferred from the goal and preview.")
    if not supported_by_current_pipeline:
        notes.append(f"Current pipeline supports supervised classification/regression, not task_type={task_type}.")
    if intake.get("encoding_detection", {}).get("confidence") == "low":
        notes.append("Encoding detection used a low-confidence fallback.")
    if intake.get("delimiter_detection", {}).get("confidence") == "low":
        notes.append("Delimiter detection used a low-confidence fallback.")
    return notes


def manager_replanning_checkpoint(state: WorkflowState, completed_stage: str) -> dict:
    """Patch downstream contracts after a completed stage.

    Replanning is intentionally limited: it only uses structured stage reports,
    never rewrites completed facts, and only updates downstream requirements.
    """

    patch = _build_plan_patch(state, completed_stage)
    if not patch:
        return {"current_stage": f"manager_replan:{completed_stage}", "current_error": ""}
    plan = _apply_patch(state["global_plan"], patch)
    version = int(state.get("plan_version", 1)) + 1
    patch_record = {"plan_version": version, **patch}
    patches = list(state.get("plan_patches", []))
    patches.append(patch_record)
    write_json(PLAN_PATCH_LOG_PATH, patches)
    write_json(MANAGER_PLAN_PATH, plan)
    return {
        "global_plan": plan,
        "plan_version": version,
        "plan_patches": patches,
        "current_stage": f"manager_replan:{completed_stage}",
        "current_error": "",
    }


def _build_plan_patch(state: WorkflowState, completed_stage: str) -> dict:
    """Create a downstream-only plan patch from current structured reports."""

    if completed_stage == "data_cleaning":
        return _patch_after_data_cleaning(state)
    if completed_stage == "feature_engineering":
        return _patch_after_feature_engineering(state)
    return {}


def _patch_after_data_cleaning(state: WorkflowState) -> dict:
    """Update FE/model contracts based on cleaned data profile."""

    report = state.get("stage_reports", {}).get("data_cleaning", {})
    profile = report.get("profile_after", {})
    task = state["global_plan"]["task_summary"]
    target_col = task.get("target_column")
    updates: dict = {}
    reasons: list[str] = []

    if target_col and target_col in profile.get("column_names", []):
        distribution = profile.get("target_distribution", {})
        if distribution and _minority_ratio(distribution) < 0.2 and task.get("task_type") == "classification":
            updates.setdefault("model_selection", {})["imbalance_strategy"] = "class_weight_or_resampling"
            updates.setdefault("model_selection", {}).setdefault("validation_rules", []).append(
                {"id": "class_imbalance_strategy", "check": "model_report.json reports class imbalance handling strategy"}
            )
            updates.setdefault("feature_engineering", {}).setdefault("validation_rules", []).append(
                {"id": "target_balance_reported", "check": "feature_report.json reports target class balance before modelling"}
            )
            reasons.append("classification target appears imbalanced after cleaning")

    high_missing = [
        col for col, ratio in profile.get("missing_ratio", {}).items()
        if col != target_col and isinstance(ratio, (int, float)) and ratio >= 0.3
    ]
    if high_missing:
        updates.setdefault("feature_engineering", {})["high_missing_columns"] = high_missing
        updates.setdefault("feature_engineering", {}).setdefault("validation_rules", []).append(
            {
                "id": "high_missingness_handled",
                "check": "feature_report.json creates or justifies missingness indicators for high-missingness columns",
            }
        )
        reasons.append(f"high missingness detected in columns: {high_missing}")

    if not updates:
        return {}
    return {
        "after_stage": "data_cleaning",
        "reason": "; ".join(reasons),
        "updates": updates,
    }


def _patch_after_feature_engineering(state: WorkflowState) -> dict:
    """Update model contract based on feature output shape and FE policy."""

    report = state.get("stage_reports", {}).get("feature_engineering", {})
    train_shape = report.get("train_shape", [0, 0])
    policy = report.get("policy", {})
    updates: dict = {}
    reasons: list[str] = []

    if len(train_shape) == 2 and train_shape[1] > 100:
        updates.setdefault("model_selection", {})["candidate_model_constraint"] = "prefer_regularized_or_high_dimensional_capable_models"
        updates.setdefault("model_selection", {}).setdefault("validation_rules", []).append(
            {"id": "high_dimensional_model", "check": "selected model can handle high-dimensional feature tables"}
        )
        reasons.append("feature engineering produced high-dimensional data")

    if policy.get("mode") == "skip":
        updates.setdefault("model_selection", {}).setdefault("validation_rules", []).append(
            {
                "id": "skipped_fe_model_ready",
                "check": "model_report.json confirms skipped feature engineering left model-ready numeric features",
            }
        )
        reasons.append("feature engineering was skipped")

    if not updates:
        return {}
    return {
        "after_stage": "feature_engineering",
        "reason": "; ".join(reasons),
        "updates": updates,
    }


def _minority_ratio(distribution: dict) -> float:
    """Return the minority-class ratio from a target distribution."""

    counts = [int(v) for v in distribution.values() if isinstance(v, (int, float))]
    if not counts or sum(counts) == 0:
        return 1.0
    return min(counts) / sum(counts)


def _apply_patch(plan: dict, patch: dict) -> dict:
    """Apply a limited downstream contract patch to a copied plan."""

    updated = {
        "dataset_intake": dict(plan.get("dataset_intake", {})),
        "input_config": dict(plan.get("input_config", {})),
        "test_dataset_intake": dict(plan.get("test_dataset_intake", {})) if plan.get("test_dataset_intake") else None,
        "task_summary": dict(plan.get("task_summary", {})),
        "stage_contracts": {
            stage: dict(contract)
            for stage, contract in plan.get("stage_contracts", {}).items()
        },
    }
    for stage, changes in patch.get("updates", {}).items():
        contract = updated["stage_contracts"].setdefault(stage, {})
        for key, value in changes.items():
            if key in {"validation_rules", "required_report_fields", "allowed_tools", "required_artifacts"}:
                existing = list(contract.get(key, []))
                for item in value:
                    if item not in existing:
                        existing.append(item)
                contract[key] = existing
            else:
                contract[key] = value
    return updated
