import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("sklearn")

from core.dataset_split import perform_train_test_file_split, split_dataset_node


def test_regression_random_split_writes_paths(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"x": range(100), "y": range(100)}).to_csv(csv_path, index=False)
    train_p = tmp_path / "train.csv"
    test_p = tmp_path / "test.csv"

    report = perform_train_test_file_split(
        source_path=csv_path,
        train_out=train_p,
        test_out=test_p,
        task_type="regression",
        target_column=None,
        test_size=0.2,
        random_state=0,
    )

    assert report["train_rows"] == 80
    assert report["test_rows"] == 20
    assert report["stratify_used"] is False
    assert train_p.exists() and test_p.exists()


def test_classification_stratified_split(tmp_path):
    csv_path = tmp_path / "cls.csv"
    pd.DataFrame(
        {
            "f": list(range(100)),
            "target": ["a"] * 50 + ["b"] * 50,
        }
    ).to_csv(csv_path, index=False)

    report = perform_train_test_file_split(
        source_path=csv_path,
        train_out=tmp_path / "tr.csv",
        test_out=tmp_path / "te.csv",
        task_type="classification",
        target_column="target",
        test_size=0.2,
        random_state=0,
    )

    assert report["stratify_used"] is True
    assert report["stratify_fallback_to_random"] is False
    tr = pd.read_csv(report["train_path"])
    te = pd.read_csv(report["test_path"])
    assert set(tr["target"].unique()) == {"a", "b"}
    assert set(te["target"].unique()) == {"a", "b"}


def test_split_dataset_node_prefers_cleaned_path(tmp_path):
    raw = tmp_path / "raw.csv"
    cleaned = tmp_path / "cleaned.csv"
    pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]}).to_csv(raw, index=False)
    pd.DataFrame({"a": [10, 20, 30, 40], "b": [0, 1, 0, 1]}).to_csv(cleaned, index=False)

    update = split_dataset_node(
        {
            "data_path": str(raw),
            "cleaned_data_path": str(cleaned),
            "task_type": "regression",
        },
        test_size=0.25,
        random_state=1,
    )

    assert update["current_stage"] == "split_done"
    assert Path(update["training_data_path"]).exists()
    assert Path(update["testing_data_path"]).exists()
    loaded = pd.read_csv(update["training_data_path"])
    combined = pd.concat([loaded, pd.read_csv(update["testing_data_path"])])
    assert set(combined["a"].tolist()) == {10, 20, 30, 40}


def test_split_dataset_node_classification_requires_target_column(tmp_path):
    csv_path = tmp_path / "c.csv"
    pd.DataFrame({"x": [1, 2, 3, 4]}).to_csv(csv_path, index=False)

    update = split_dataset_node(
        {
            "cleaned_data_path": str(csv_path),
            "task_type": "classification",
            "target_column": "",
        }
    )

    assert update["current_stage"] == "split_failed"
    assert "errors" in update
