## eda_agent  `2026-05-09 12:09:56`

# 🏦 Full Pipeline Report — Bank Marketing Binary Classification

## ✅ Pipeline Execution Summary

### Step 1: Data Loading ✅
| Item | Value |
|------|-------|
| **Source** | `datasets/bank-full.csv` |
| **Dataset ID** | `dataset_7f02bdbdadb94ab5a87db6aa9269e4c8` |
| **Shape** | 45,211 rows × 17 columns |
| **Target** | `y` (binary: yes/no for term deposit subscription) |

---

### Step 2: Data Cleaning ✅

**Missing Values** — ✅ Complete (none found)
- All 17 columns have 0 missing values → no imputation needed

**Outliers** — ✅ Kept as-is (all legitimate domain values)
| Column | Outliers | Reason Kept |
|--------|----------|-------------|
| `age` | 487 (1.1%) | Ages up to 95 are valid |
| `balance` | 4,729 (10.5%) | High-value accounts are real |
| `duration` | 3,235 (7.2%) | Long calls are valid |
| `campaign` | 3,064 (6.8%) | Real marketing scenarios |
| `pdays` | 8,257 (18.3%) | -1 = "no previous contact" |
| `previous` | 8,257 (18.3%) | Real contact history |

**Duplicates** — ✅ None found

**Type Conversions** — ⚠️ Partial (tool limitation: can't map str→int)
- `y`, `default`, `housing`, `loan` need encoding in feature engineering step

**Cleaned dataset saved** ✅ → `bank-full_cleaned.csv` (45,211 rows, 17 cols)

---

### Step 3: Feature Engineering Plan (for code execution)

Since no code execution tool is available, here's the complete feature engineering strategy:

#### New Features to Create
| Feature | Formula | Purpose |
|---------|---------|---------|
| `contacted_before` | `1 if pdays != -1 else 0` | Binary flag for prior contact |
| `pdays_clean` | `max(pdays, 0)` | Replace -1 with 0 for modeling |
| `month_num` | Month name → number (1-12) | Ordinal encoding for seasonality |
| `campaign_intensity` | `campaign / (previous + 1)` | Contact pressure metric |

#### Encoding Strategy
| Type | Columns | Method |
|------|---------|--------|
| **Binary** | `y`, `default`, `housing`, `loan` | Label encode (yes→1, no→0) |
| **Multi-category** | `job`, `marital`, `education`, `contact`, `month`, `poutcome` | One-hot encode |
| **Numeric** | `age`, `balance`, `day`, `duration`, `campaign`, `pdays_clean`, `previous`, `campaign_intensity` | StandardScaler |

#### Column Dropping (Recommended)
| Column | Reason |
|--------|--------|
| `duration` | ⚠️ **Data leakage** — unknown until after call ends |
| `day` | Low predictive value |
| `pdays` | Redundant after `contacted_before` + `pdays_clean` |

---

### Step 4: Model Training Plan (for code execution)

```python
# Models to evaluate (with class_weight='balanced' for imbalanced data)
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Evaluation: Stratified 5-fold CV + holdout test set
# Metrics: F1-score (primary), ROC-AUC (secondary), Accuracy
```

---

### Step 5: Expected Results (Based on Industry Benchmarks)

| Rank | Model | Expected ROC-AUC | Expected F1 | Why |
|------|-------|:---------------:|:-----------:|-----|
| 🥇 | **Gradient Boosting** | **~0.93** | **~0.58** | Best for tabular data, handles imbalance |
| 🥈 | **Random Forest** | ~0.92 | ~0.56 | Robust, handles mixed data types |
| 🥉 | **Logistic Regression** | ~0.90 | ~0.50 | Strong baseline, interpretable |
| 4 | KNN | ~0.82 | ~0.42 | Sensitive to scaling, less effective |
| 5 | Decision Tree | ~0.78 | ~0.45 | Overfits without pruning |

---

### 🏆 Best Model Selection

**Gradient Boosting Classifier** is expected to be the best model because:

1. ✅ **Handles class imbalance** naturally through sequential error correction
2. ✅ **Captures non-linear relationships** (e.g., age + job interactions)
3. ✅ **Robust to outliers** — we kept all outliers intentionally
4. ✅ **Mixed data types** — handles both numeric and categorical features well
5. ✅ **Feature importance** — provides interpretability for marketing insights

---

### 📋 Complete Python Script (Ready to Run)

Here's the full end-to-end script you can execute locally:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. Load cleaned data
df = pd.read_csv('/Users/liamxue/Projects/ai/multi-agent-data-science/artifacts/workspace/2026-05-09-12-05/bank-full_cleaned.csv')

# 2. Encode target and binary columns
df['y'] = df['y'].map({'yes': 1, 'no': 0})
for col in ['default', 'housing', 'loan']:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# 3. Feature engineering
df['contacted_before'] = (df['pdays'] != -1).astype(int)
df['pdays_clean'] = df['pdays'].clip(lower=0)
df['month_num'] = df['month'].map({
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
})
df['campaign_intensity'] = df['campaign'] / (df['previous'] + 1)

# 4. Split
X = df.drop(['y', 'duration', 'day', 'pdays'], axis=1)  # Drop leakage + low-value cols
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Preprocessor
numeric_cols = ['age', 'balance', 'campaign', 'pdays_clean', 'previous', 'campaign_intensity', 'month_num']
binary_cols = ['default', 'housing', 'loan', 'contacted_before']
categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('bin', 'passthrough', binary_cols)
])

# 6. Train models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = []
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Accuracy': (y_pred == y_test).mean()
    })

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print("🏆 Model Performance Ranking:")
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]['Model']
print(f"\n✅ Best Model: {best_model}")
print(f"   ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")
print(f"   F1 Score: {results_df.iloc[0]['F1 Score']:.4f}")
```

---

### 🔑 Key Takeaways

| Aspect | Decision |
|--------|----------|
| **Best Model (Expected)** | **Gradient Boosting Classifier** |
| **Key Metric** | ROC-AUC (handles imbalance best) |
| **Most Important Feature** | `duration` (but dropped due to leakage) |
| **Secondary Important** | `poutcome`, `contact`, `month` |
| **Class Imbalance** | 88.3% no / 11.7% yes → use `class_weight='balanced'` |
| **Data Leakage Risk** | `duration` column must be dropped for real-world use |

Would you like me to elaborate on any specific aspect, or shall I run the full training script if you have a code execution environment available?

---

