"""Shared fixtures and path constants."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TESTENV = Path(__file__).parent / "testenv"

RAW_CSV     = TESTENV / "raw"     / "sample.csv"
CLEAN_CSV   = TESTENV / "clean"   / "sample_cleaned.csv"
TRAIN_CSV   = TESTENV / "split"   / "sample_train.csv"
TEST_CSV    = TESTENV / "split"   / "sample_test.csv"
FEATURE_DIR = TESTENV / "feature"

# Feature-engineered train/test sets for three candidate models (already numeric; model agent uses directly)
CANDIDATE_MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]

FEATURE_TRAIN_PATHS = {m: FEATURE_DIR / f"{m}_train_featured.csv" for m in CANDIDATE_MODELS}
FEATURE_TEST_PATHS  = {m: FEATURE_DIR / f"{m}_test_featured.csv"  for m in CANDIDATE_MODELS}

# Sample dataset description (used by agent system prompts)
SAMPLE_DESCRIPTION = (
    "Binary classification dataset: predict whether a customer subscribes to a bank term deposit (subscribed: yes/no). "
    "Features: age, job (occupation), education (education level), balance (account balance)."
)

# Cleaning plan: fill 2 missing values in the balance column
SAMPLE_CLEANING_PLAN = {
    "summary": "The balance column has 2 missing values; fill with median.",
    "column_name_actions": [],
    "type_conversions": [],
    "missing_value_actions": [{"column": "balance", "strategy": "median"}],
    "outlier_actions": [],
    "drop_columns": [],
    "notes": "",
}

# Feature engineering plan
SAMPLE_FEATURE_PLAN = {
    "LogisticRegression": {"scaling": True,  "encoding": "onehot", "notes": "linear models require standardization"},
    "RandomForest":       {"scaling": False, "encoding": "label",  "notes": "tree models do not require scaling"},
    "XGBoost":            {"scaling": False, "encoding": "label",  "notes": "tree models do not require scaling"},
}
