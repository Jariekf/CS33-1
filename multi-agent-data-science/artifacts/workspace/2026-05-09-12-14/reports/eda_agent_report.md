## eda_agent  `2026-05-09 12:16:18`

## ✅ EDA & Model Selection Complete

Here's a full summary of everything I've done and decided:

---

### 📊 Data Profile Summary

| Metric | Value |
|--------|-------|
| **Rows** | 36,168 (train set) |
| **Columns** | 17 (7 numerical, 10 categorical) |
| **Missing Values** | ✅ None |
| **Outliers** | Clipped (age, balance, duration, campaign) |
| **Target** | `y` (binary: yes/no) — imbalanced |
| **Key Risk** | `duration` is a **data leakage** variable |

### 📈 Correlation Insights
- `pdays` ↔ `previous`: r = 0.44 (moderate correlation — both measure prior contact)
- `campaign` ↔ `day`: r = 0.15 (more calls later in month)
- No strong multicollinearity issues among features

### 🏆 Selected Candidate Models

| Rank | Model | Encoding | Scaling | Imbalance Strategy | Why This Model |
|:----:|-------|----------|:-------:|--------------------|----------------|
| 1️⃣ | **XGBoost** | Label | ❌ No | `scale_pos_weight` | Best performance on structured data, handles non-linearity, early stopping |
| 2️⃣ | **RandomForest** | Label | ❌ No | `class_weight='balanced'` | Robust ensemble, captures interactions, no scaling needed |
| 3️⃣ | **LogisticRegression** | One-hot | ✅ Yes | `class_weight='balanced'` | Interpretable baseline, coefficient insights for marketing |

### ❌ Models Rejected
- **SVM**: Too slow on 36K rows with high-dimensional one-hot features; sensitive to imbalance
- **KNN**: Curse of dimensionality with categorical encoding; poor with imbalanced data

### 📁 Artifacts Generated
| Artifact | Path |
|----------|------|
| Training Data | `.../bank-full_cleaned_train.csv` |
| Distribution Plots | `.../plots/` (7 PNG files) |
| Model Selection Decision | ✅ Submitted |

The next step would be to proceed with **model training & tuning** using these three models with their respective feature engineering strategies!

---

