import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.agentmanager_tools.profile_dataset import profile_dataset

TESTENV = Path(__file__).resolve().parent / "testenv"
RAW_CSV = TESTENV / "raw" / "sample.csv"


def test_profile_dataset_basic_shape():
    profile = json.loads(profile_dataset.invoke({"file_path": str(RAW_CSV)}))

    assert "error" not in profile
    assert profile["dataset"]["rows"] == 40
    assert profile["dataset"]["cols"] == 5
    assert "subscribed" in profile["dataset"]["columns"]
    assert len(profile["sample_rows"]) == 5


def test_profile_dataset_detects_missing_values():
    profile = json.loads(profile_dataset.invoke({"file_path": str(RAW_CSV)}))
    # balance 有2个缺失值
    assert profile["missing_values"]["total_missing_cells"] == 2
    assert profile["missing_values"]["columns"]["balance"]["missing_count"] == 2


def test_profile_dataset_detects_category_columns():
    profile = json.loads(profile_dataset.invoke({"file_path": str(RAW_CSV)}))
    category_cols = profile["type_issues"]["category_columns"]
    # job、education、subscribed 应被识别为类别列
    assert "job" in category_cols or "education" in category_cols


def test_profile_dataset_missing_file():
    profile = json.loads(profile_dataset.invoke({"file_path": str(TESTENV / "nonexistent.csv")}))
    assert "error" in profile
