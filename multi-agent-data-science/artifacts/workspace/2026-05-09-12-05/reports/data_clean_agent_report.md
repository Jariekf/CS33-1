## data_clean_agent  `2026-05-09 12:09:25`

The cleaned dataset has been saved. Now let me provide the comprehensive full pipeline report.

---

# 🏦 Full Pipeline Report — Bank Marketing Binary Classification

## ✅ Completed Steps

### 1. Data Loading ✅
- **Source**: `datasets/bank-full.csv`
- **Loaded into**: `dataset_7f02bdbdadb94ab5a87db6aa9269e4c8`
- **Shape**: 45,211 rows × 17 columns

### 2. Data Cleaning ✅

| Step | Status | Details |
|------|--------|---------|
| **Type Conversions** | ⚠️ Partial | The tool doesn't support mapping `str` → `int` for binary columns (y, default, housing, loan). These need to be encoded in the feature engineering step. |
| **Missing Values** | ✅ Complete | No missing values found — 0 actions needed |
| **Outliers** | ✅ Kept as-is | All 6 outlier columns (age, balance, duration, campaign, pdays, previous) kept — values are legitimate domain data |
| **Duplicates** | ✅ Complete | No duplicate rows found |
| **Saved** | ✅ Complete | `bank-full_cleaned.csv` → workspace directory |

### 3. Feature Engineering (Recommended Code)

Since I don't have a code execution tool, here's the complete feature engineering + model training pipeline you can run:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
df = pd.read_csv('/Users/liamxue/Projects/ai/multi-agent-data-science/artifacts/workspace/2026-05-09-12-05/bank-full_cleaned.csv')

# Encode target
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Encode binary categoricals
for col in ['default', 'housing', 'loan']:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Feature engineering
df['contacted_before'] = (df['pdays'] != -1).astype(int)
df['pdays_clean'] = df['pdays'].clip(lower=0)  # -1 → 0 for modeling
df['month_num'] = df['month'].map({
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
})
df['campaign_intensity'] = df['campaign'] / (df['previous'] + 1)

# Split features and target
X = df.drop('y', axis=1)
y = df['y']

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define column types
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 
                'pdays_clean', 'previous', 'campaign_intensity']
binary_cols = ['default', 'housing', 'loan', 'contacted_before']
categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('bin', 'passthrough', binary_cols)
])

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate
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
print(results_df)

# Best model details
best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name}")
```

### 4. Expected Model Performance (Based on Industry Benchmarks)

| Model | Expected ROC-AUC | Expected F1 |
|-------|-----------------|-------------|
| 🥇 **Gradient Boosting** | **~0.93** | **~0.58** |
| 🥈 **Random Forest** | ~0.92 | ~0.56 |
| 🥉 **Logistic Regression** | ~0.90 | ~0.50 |
| Decision Tree | ~0.78 | ~0.45 |
| KNN | ~0.82 | ~0.42 |

### 5. 📋 Recommendations for Column Dropping

Since the toolset doesn't include a `drop_columns` tool, here are columns I recommend **dropping** (not executed):

| Column | Reason |
|--------|--------|
| **`duration`** | ⚠️ **Data leakage risk** — this is the duration of the last contact, which is unknown until after the call ends. Using it in a real-time prediction model would make it unusable for actual marketing campaigns. However, for benchmarking it's often kept. |
| **`day`** | Low predictive value — day of month (1-31) has minimal correlation with subscription behavior |
| **`pdays`** | Redundant after creating `contacted_before` feature — the raw -1/positive values can be replaced by the binary flag + a cleaned version |

### 6. 📊 EDA Key Visualizations (Suggested)

If you had matplotlib, I'd generate:
1. **Target distribution bar chart** — shows 88.3% / 11.7% imbalance
2. **Correlation heatmap** — `duration` most correlated with `y`
3. **Box plots** — `balance`, `campaign`, `age` by target class
4. **Monthly subscription rates** — May and August typically have higher rates
5. **Job type vs subscription rate** — students and retired have higher conversion

---

### 🏆 Best Classifier Selection (Expected)

**Gradient Boosting Classifier** is expected to be the best model because:
1. ✅ Best handles the class imbalance naturally
2. ✅ Captures non-linear relationships in the data
3. ✅ Handles mixed data types (numeric + categorical) well
4. ✅ Robust to outliers (which we intentionally kept)
5. ✅ Provides feature importance for interpretability

Would you like me to elaborate on any specific part of the pipeline, or do you have a code execution environment where I can run the full training script?

---

