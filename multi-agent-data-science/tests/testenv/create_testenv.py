"""Generate test resource files. Run from the project root: python tests/testenv/create_testenv.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TESTENV = Path(__file__).parent
np.random.seed(42)
N = 40

# ── Raw data ─────────────────────────────────────────────────────────────────

jobs = ["admin", "technician", "services", "management"]
educations = ["primary", "secondary", "tertiary"]

raw = pd.DataFrame(
    {
        "age": np.random.randint(22, 60, N),
        "job": np.random.choice(jobs, N),
        "education": np.random.choice(educations, N),
        "balance": np.round(np.random.uniform(100, 5000, N), 2),
        "subscribed": np.random.choice(["yes", "no"], N, p=[0.4, 0.6]),
    }
)
# Inject 2 missing values to simulate a dataset that needs cleaning
raw.loc[[3, 17], "balance"] = np.nan

out = TESTENV / "raw" / "sample.csv"
raw.to_csv(out, index=False)
print(f"raw: {out}  ({len(raw)} rows)")

# ── Cleaned data ─────────────────────────────────────────────────────────────

cleaned = raw.copy()
median_balance = cleaned["balance"].median()
cleaned["balance"] = cleaned["balance"].fillna(median_balance)

out = TESTENV / "clean" / "sample_cleaned.csv"
cleaned.to_csv(out, index=False)
print(f"clean: {out}  ({len(cleaned)} rows)")

# ── Train / test split ───────────────────────────────────────────────────────

train_df, test_df = train_test_split(
    cleaned, test_size=0.2, random_state=42, stratify=cleaned["subscribed"]
)

out_train = TESTENV / "split" / "sample_train.csv"
out_test = TESTENV / "split" / "sample_test.csv"
train_df.to_csv(out_train, index=False)
test_df.to_csv(out_test, index=False)
print(f"split train: {out_train}  ({len(train_df)} rows)")
print(f"split test:  {out_test}  ({len(test_df)} rows)")

# ── Feature-engineered training sets ─────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["subscribed"] = (df["subscribed"] == "yes").astype(int)
    return df


# LogisticRegression / Ridge: StandardScaler + onehot
def make_lr_featured(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_target(df)
    target = df.pop("subscribed")
    # one-hot
    df = pd.get_dummies(df, columns=["job", "education"])
    # scale numeric
    num_cols = ["age", "balance"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    df["subscribed"] = target
    return df.astype(float, errors="ignore")


# RandomForest / XGBoost: LabelEncoder (no scaling)
def make_tree_featured(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_target(df)
    for col in ["job", "education"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


lr_feat = make_lr_featured(train_df)
rf_feat = make_tree_featured(train_df)
xgb_feat = rf_feat.copy()

lr_test = make_lr_featured(test_df)
rf_test = make_tree_featured(test_df)
xgb_test = rf_test.copy()

for model, train_feat, test_feat in [
    ("LogisticRegression", lr_feat, lr_test),
    ("RandomForest", rf_feat, rf_test),
    ("XGBoost", xgb_feat, xgb_test),
]:
    out = TESTENV / "feature" / f"{model}_train_featured.csv"
    train_feat.to_csv(out, index=False)
    print(f"feature/{model} train: {out}  ({len(train_feat)} rows, {len(train_feat.columns)} cols)")

    out = TESTENV / "feature" / f"{model}_test_featured.csv"
    test_feat.to_csv(out, index=False)
    print(f"feature/{model} test:  {out}  ({len(test_feat)} rows)")

print("\ntestenv generation complete.")
