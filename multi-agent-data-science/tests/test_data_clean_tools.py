import json

import pandas as pd

from core.dataset_store import get_dataset, put_dataset
from tools.dataclean_tools.clean_data_types import clean_data_types
from tools.dataclean_tools.clean_duplicates import clean_duplicates
from tools.dataclean_tools.clean_drop_columns import clean_drop_columns
from tools.dataclean_tools.clean_missing_values import clean_missing_values
from tools.dataclean_tools.clean_outliers import clean_outliers
from tools.dataclean_tools.load_dataset_to_store import load_dataset_to_store
from tools.dataclean_tools.save_dataset_from_store import save_dataset_from_store


def test_load_and_save_dataset_store_tools(tmp_path):
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "cleaned.csv"
    pd.DataFrame({"amount": [1, 2], "category": ["a", "b"]}).to_csv(
        raw_path, index=False
    )

    load_report = json.loads(load_dataset_to_store.invoke({"file_path": str(raw_path)}))
    dataset_id = load_report["dataset_id"]

    assert load_report["shape"] == {"rows": 2, "cols": 2}
    assert get_dataset(dataset_id).shape == (2, 2)

    save_report = json.loads(
        save_dataset_from_store.invoke(
            {"dataset_id": dataset_id, "output_path": str(output_path)}
        )
    )

    assert save_report["output_path"] == str(output_path.resolve())
    assert save_report["shape"] == {"rows": 2, "cols": 2}
    assert output_path.exists()


def test_data_clean_tools_update_dataset_store():
    dataset_id = put_dataset(
        pd.DataFrame(
            {
                "amount": [" $1,000.00 ", " $-   ", None, " $10,000.00 "],
                "date": ["01/01/2014", "01/02/2014", "bad", "01/04/2014"],
                "category": ["a", "a", None, "a"],
            }
        )
    )

    type_report = json.loads(
        clean_data_types.invoke(
            {
                "dataset_id": dataset_id,
                "conversions": [
                    {"column": "amount", "target_type": "numeric"},
                    {"column": "date", "target_type": "datetime"},
                ],
            }
        )
    )
    assert type_report["converted_columns"] == ["amount", "date"]

    missing_report = json.loads(
        clean_missing_values.invoke(
            {
                "dataset_id": dataset_id,
                "actions": [
                    {"column": "amount", "strategy": "fill", "value": 0},
                    {"column": "category", "strategy": "fill", "value": "unknown"},
                ],
            }
        )
    )
    assert missing_report["filled_columns"] == ["amount", "category"]

    outlier_report = json.loads(
        clean_outliers.invoke(
            {
                "dataset_id": dataset_id,
                "actions": [{"column": "amount", "strategy": "clip_iqr"}],
            }
        )
    )
    assert outlier_report["processed_columns"] == ["amount"]

    duplicate_report = json.loads(
        clean_duplicates.invoke(
            {
                "dataset_id": dataset_id,
                "subset": ["category"],
                "keep": "first",
            }
        )
    )
    assert duplicate_report["dropped_rows"] == 2

    drop_report = json.loads(
        clean_drop_columns.invoke(
            {
                "dataset_id": dataset_id,
                "columns": ["date", "not_exists"],
            }
        )
    )
    assert drop_report["dropped_columns"] == ["date"]
    assert drop_report["missing_columns"] == ["not_exists"]

    cleaned = get_dataset(dataset_id)
    assert cleaned["amount"].isna().sum() == 0
    assert cleaned["category"].isna().sum() == 0
    assert "date" not in cleaned.columns
    assert len(cleaned) == 2


def test_clean_missing_values_strategy_aliases():
    dataset_id = put_dataset(
        pd.DataFrame(
            {
                "num_median": [1.0, None, 3.0],
                "num_mean": [2.0, None, 6.0],
                "cat_mode": ["a", None, "a"],
                "cat_mf": ["x", None, "x"],
            }
        )
    )

    report = json.loads(
        clean_missing_values.invoke(
            {
                "dataset_id": dataset_id,
                "actions": [
                    {"column": "num_median", "strategy": "median"},
                    {"column": "num_mean", "strategy": "mean"},
                    {"column": "cat_mode", "strategy": "mode"},
                    {"column": "cat_mf", "strategy": "most_frequent"},
                ],
                "fallback_strategy": "none",
            }
        )
    )

    cleaned = get_dataset(dataset_id)
    assert cleaned["num_median"].isna().sum() == 0
    assert cleaned["num_mean"].isna().sum() == 0
    assert cleaned["cat_mode"].isna().sum() == 0
    assert cleaned["cat_mf"].isna().sum() == 0
    assert report["has_warnings"] is False


def test_clean_missing_values_soft_warning_and_fallback_scope():
    dataset_id = put_dataset(
        pd.DataFrame(
            {
                "invalid_strategy_col": [1.0, None, 3.0],
                "fill_without_value_col": [None, "x", None],
                "no_action_col": [None, 2.0, 3.0],
            }
        )
    )

    report = json.loads(
        clean_missing_values.invoke(
            {
                "dataset_id": dataset_id,
                "actions": [
                    {"column": "invalid_strategy_col", "strategy": "median_x"},
                    {"column": "fill_without_value_col", "strategy": "fill"},
                ],
                "fallback_strategy": "fill_zero",
            }
        )
    )

    cleaned = get_dataset(dataset_id)

    # configured but invalid => keep NaN, do not get fallback-covered
    assert cleaned["invalid_strategy_col"].isna().sum() == 1
    assert cleaned["fill_without_value_col"].isna().sum() == 2
    # unconfigured column => fallback applies
    assert cleaned["no_action_col"].isna().sum() == 0
    assert "no_action_col" in report["fallback_filled"]
    assert report["fallback_filled"]["no_action_col"] == 0

    assert report["has_warnings"] is True
    reasons = {item["reason"] for item in report["unsupported_actions"]}
    assert "unsupported_strategy:median_x" in reasons
    assert "fill_value_missing" in reasons


def test_clean_tools_allow_missing_optional_action_lists():
    dataset_id = put_dataset(
        pd.DataFrame(
            {
                "num_col": [1.0, None, 3.0],
                "cat_col": ["a", None, "b"],
            }
        )
    )

    type_report = json.loads(clean_data_types.invoke({"dataset_id": dataset_id}))
    assert "error" not in type_report
    assert type_report["dataset_id"] == dataset_id

    missing_report = json.loads(clean_missing_values.invoke({"dataset_id": dataset_id}))
    assert "error" not in missing_report
    assert missing_report["dataset_id"] == dataset_id

    outlier_report = json.loads(clean_outliers.invoke({"dataset_id": dataset_id}))
    assert "error" not in outlier_report
    assert outlier_report["dataset_id"] == dataset_id


def test_clean_data_types_supports_manager_alias_and_category_object():
    dataset_id = put_dataset(
        pd.DataFrame(
            {
                "MSSubClass": [20, 30, None],
                "SaleType": ["WD", "New", None],
                "YrSold": [2006, 2007, 2008],
            }
        )
    )

    report = json.loads(
        clean_data_types.invoke(
            {
                "dataset_id": dataset_id,
                "conversions": [
                    {"column_name": "MSSubClass", "new_type": "object"},
                    {"column_name": "SaleType", "new_type": "category"},
                    {"column": "YrSold", "to": "category"},
                ],
            }
        )
    )

    cleaned = get_dataset(dataset_id)
    assert report["converted_columns"] == ["MSSubClass", "SaleType", "YrSold"]
    assert str(cleaned["MSSubClass"].dtype) == "object"
    assert str(cleaned["SaleType"].dtype) == "category"
    assert str(cleaned["YrSold"].dtype) == "category"


def test_clean_missing_values_supports_manager_alias_keys():
    dataset_id = put_dataset(
        pd.DataFrame(
            {
                "num_col": [1.0, None, 3.0],
                "cat_col": [None, "a", None],
            }
        )
    )

    report = json.loads(
        clean_missing_values.invoke(
            {
                "dataset_id": dataset_id,
                "actions": [
                    {"column_name": "num_col", "imputation_strategy": "median"},
                    {"column_name": "cat_col", "imputation_strategy": "fill", "fill_value": "unknown"},
                ],
                "fallback_strategy": "none",
            }
        )
    )

    cleaned = get_dataset(dataset_id)
    assert cleaned["num_col"].isna().sum() == 0
    assert cleaned["cat_col"].isna().sum() == 0
    assert report["has_warnings"] is False


def test_clean_missing_values_supports_group_median_legacy_action():
    dataset_id = put_dataset(
        pd.DataFrame(
            {
                "Neighborhood": ["A", "A", "B", "B", "B"],
                "LotFrontage": [60.0, None, 80.0, 100.0, None],
            }
        )
    )

    report = json.loads(
        clean_missing_values.invoke(
            {
                "dataset_id": dataset_id,
                "actions": [
                    {
                        "column": "LotFrontage",
                        "action": "impute_group_median",
                        "group_by": ["Neighborhood"],
                    }
                ],
                "fallback_strategy": "none",
            }
        )
    )

    cleaned = get_dataset(dataset_id)
    assert cleaned.loc[1, "LotFrontage"] == 60.0
    assert cleaned.loc[4, "LotFrontage"] == 90.0
    assert "LotFrontage" in report["filled_columns"]


def test_clean_missing_values_supports_conditional_fill_no_garage():
    dataset_id = put_dataset(
        pd.DataFrame(
            {
                "GarageType": ["NoGarage", "Attchd", "NoGarage", "Detchd"],
                "GarageCars": [None, None, 2.0, 1.0],
            }
        )
    )

    report = json.loads(
        clean_missing_values.invoke(
            {
                "dataset_id": dataset_id,
                "actions": [
                    {
                        "column": "GarageCars",
                        "action": "fill_with_zero_if_no_garage",
                        "value": 0,
                    }
                ],
                "fallback_strategy": "none",
            }
        )
    )

    cleaned = get_dataset(dataset_id)
    assert cleaned.loc[0, "GarageCars"] == 0
    assert pd.isna(cleaned.loc[1, "GarageCars"])
    assert "GarageCars" in report["filled_columns"]


def test_clean_missing_values_supports_conditional_fill_with_flag():
    dataset_id = put_dataset(
        pd.DataFrame(
            {
                "GarageType": ["NoGarage", "Attchd", "Detchd", "Attchd"],
                "GarageYrBlt": [None, None, 2005.0, 2001.0],
            }
        )
    )

    report = json.loads(
        clean_missing_values.invoke(
            {
                "dataset_id": dataset_id,
                "actions": [
                    {
                        "column": "GarageYrBlt",
                        "action": "fill_with_constant_or_median",
                        "value": "GarageMissingFlag",
                    }
                ],
                "fallback_strategy": "none",
            }
        )
    )

    cleaned = get_dataset(dataset_id)
    assert cleaned.loc[0, "GarageYrBlt"] == 0
    assert cleaned.loc[1, "GarageYrBlt"] == 2003.0
    assert "GarageMissingFlag" in cleaned.columns
    assert int(cleaned.loc[0, "GarageMissingFlag"]) == 1
    assert int(cleaned.loc[1, "GarageMissingFlag"]) == 1
    assert int(cleaned.loc[2, "GarageMissingFlag"]) == 0
    assert report["has_warnings"] is False
