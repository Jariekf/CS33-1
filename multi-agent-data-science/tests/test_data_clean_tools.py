import json

import pandas as pd

from core.dataset_store import get_dataset, put_dataset
from tools.dataclean_tools.clean_data_types import clean_data_types
from tools.dataclean_tools.clean_duplicates import clean_duplicates
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

    cleaned = get_dataset(dataset_id)
    assert cleaned["amount"].isna().sum() == 0
    assert cleaned["category"].isna().sum() == 0
    assert len(cleaned) == 2
