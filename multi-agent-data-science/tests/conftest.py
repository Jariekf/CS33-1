"""共享 fixtures 和路径常量。"""

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

# 三个候选模型的特征工程后训练集 / 测试集（已数值化，model agent 直接使用）
CANDIDATE_MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]

FEATURE_TRAIN_PATHS = {m: FEATURE_DIR / f"{m}_train_featured.csv" for m in CANDIDATE_MODELS}
FEATURE_TEST_PATHS  = {m: FEATURE_DIR / f"{m}_test_featured.csv"  for m in CANDIDATE_MODELS}

# 样本数据集描述（agent system prompt 使用）
SAMPLE_DESCRIPTION = (
    "二分类数据集：预测客户是否订阅银行定期存款（subscribed: yes/no）。"
    "特征：age（年龄）、job（职业）、education（教育程度）、balance（余额）。"
)

# 清洗计划：填充 balance 的2个缺失值
SAMPLE_CLEANING_PLAN = {
    "summary": "balance 列有2个缺失值，用中位数填充。",
    "column_name_actions": [],
    "type_conversions": [],
    "missing_value_actions": [{"column": "balance", "strategy": "median"}],
    "outlier_actions": [],
    "drop_columns": [],
    "notes": "",
}

# 特征工程计划
SAMPLE_FEATURE_PLAN = {
    "LogisticRegression": {"scaling": True,  "encoding": "onehot", "notes": "线性模型需标准化"},
    "RandomForest":       {"scaling": False, "encoding": "label",  "notes": "树模型无需缩放"},
    "XGBoost":            {"scaling": False, "encoding": "label",  "notes": "树模型无需缩放"},
}
