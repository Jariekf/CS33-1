import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.data_clean_agent import _normalize_cleaning_plan


def test_normalize_cleaning_plan_maps_manager_style_fields():
    plan = {
        "summary": "test plan",
        "type_conversions": [
            {"column_name": "MSSubClass", "new_type": "object"},
            {"column": "YrSold", "to": "category"},
            {"column_name": "MoSold", "conversion_type": "object"},
        ],
        "missing_value_actions": [
            {"column_name": "LotFrontage", "imputation_strategy": "median"},
            {"column_name": "Alley", "imputation_strategy": "fill", "fill_value": "missing"},
            {"column_name": "GarageCars", "strategy": "fill", "imputation_strategy": "median"},
            {"column": "LotFrontage", "action": "impute_group_median", "group_by": ["Neighborhood"]},
            {"column": "GarageCars", "action": "fill_with_zero_if_no_garage", "value": 0},
            {"column": "GarageYrBlt", "action": "fill_with_constant_or_median", "value": "GarageMissingFlag"},
        ],
        "outlier_actions": [{"column_name": "SalePrice", "strategy": "clip_iqr"}],
        "drop_columns": [{"column_name": "Id"}],
        "notes": "single-note",
    }

    normalized = _normalize_cleaning_plan(plan)

    assert normalized["type_conversions"][0] == {
        "column_name": "MSSubClass",
        "new_type": "object",
        "column": "MSSubClass",
        "target_type": "object",
    }
    assert normalized["type_conversions"][1]["target_type"] == "category"
    assert normalized["type_conversions"][2]["target_type"] == "object"

    assert normalized["missing_value_actions"][0]["column"] == "LotFrontage"
    assert normalized["missing_value_actions"][0]["strategy"] == "median"
    assert normalized["missing_value_actions"][1]["column"] == "Alley"
    assert normalized["missing_value_actions"][1]["strategy"] == "fill"
    assert normalized["missing_value_actions"][1]["value"] == "missing"
    assert "method" not in normalized["missing_value_actions"][1]
    assert normalized["missing_value_actions"][2]["method"] == "median"
    assert normalized["missing_value_actions"][3]["strategy"] == "impute_group_median"
    assert normalized["missing_value_actions"][3]["group_by"] == ["Neighborhood"]
    assert normalized["missing_value_actions"][4]["strategy"] == "fill_with_zero_if_no_garage"
    assert normalized["missing_value_actions"][4]["condition_column"] == "GarageType"
    assert normalized["missing_value_actions"][4]["condition_value"] == "NoGarage"
    assert normalized["missing_value_actions"][5]["strategy"] == "fill_with_constant_or_median"
    assert normalized["missing_value_actions"][5]["flag_column"] == "GarageMissingFlag"
    assert normalized["missing_value_actions"][5]["condition_column"] == "GarageType"
    assert normalized["missing_value_actions"][5]["condition_value"] == "NoGarage"
    assert normalized["missing_value_actions"][5]["condition_fill_value"] == 0
    assert normalized["missing_value_actions"][5]["fallback_method"] == "median"

    assert normalized["outlier_actions"][0]["column"] == "SalePrice"
    assert normalized["drop_columns"] == ["Id"]
    assert normalized["notes"] == ["single-note"]
