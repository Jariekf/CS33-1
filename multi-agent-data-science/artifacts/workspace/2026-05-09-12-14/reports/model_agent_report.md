## model_agent — XGBoost  `2026-05-09 12:18:47`

## ✅ All 3 Models — Training & Evaluation Complete!

Here is the final comparison:

---

## 🏆 Model Selection — Final Results

| Rank | Model | Train Accuracy | Test Accuracy | F1 Macro | F1 Weighted | Notes |
|:----:|-------|:--------------:|:-------------:|:--------:|:-----------:|-------|
| **🥇** | **RandomForest** | **1.0000** | **0.9053** | **0.7270** | **0.8960** | Highest accuracy, but overfitting (100% train) |
| **🥈** | **XGBoost** | 0.9568 | 0.9019 | **0.7395** | **0.8971** | Best F1 Macro (best minority class handling), best generalization gap |
| **🥉** | **LogisticRegression** | 0.8845 | 0.8834 | 0.5925 | 0.8563 | Weakest on minority class, but most interpretable |

### 🏆 Best Model: **RandomForest** (by accuracy) or **XGBoost** (by F1 Macro)

| Criteria | Winner | Reason |
|----------|:------:|--------|
| **Highest Accuracy** | 🥇 RandomForest (0.9053) | +0.34% over XGBoost |
| **Best Minority Class Handling** | 🥇 XGBoost (0.7395 F1 Macro) | +1.25% over RandomForest |
| **Best Generalization** | 🥇 XGBoost (gap: 5.5%) | RandomForest gap: 9.5% (overfitting) |
| **Most Interpretable** | 🥇 LogisticRegression | Coefficients directly explainable |

### 📁 Final Artifacts

| Model | Model Path | Feature Columns |
|-------|-----------|-----------------|
| LogisticRegression | `models/LogisticRegression_model.pkl` | `models/LogisticRegression_feature_columns.json` |
| RandomForest | `models/RandomForest_model.pkl` | `models/RandomForest_feature_columns.json` |
| XGBoost | `models/XGBoost_model.pkl` | `models/XGBoost_feature_columns.json` |

**Recommendation**: If deployment priority is **robustness to imbalance** → use **XGBoost**. If **raw accuracy** is the goal → use **RandomForest** (with hyperparameter tuning to reduce overfitting).

---

