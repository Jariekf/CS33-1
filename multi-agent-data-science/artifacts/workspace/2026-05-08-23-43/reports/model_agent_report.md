## model_agent — XGBoost  `2026-05-08 23:46:51`

---

## ✅ XGBoost 训练与评估完成！所有模型训练完毕！

### 📊 最终评估结果

| 模型 | Accuracy | F1 Macro | F1 Weighted | 训练集 Score | 泛化差距 |
|------|:--------:|:--------:|:-----------:|:-----------:|:--------:|
| **LogisticRegression** | 0.8876 | 0.5972 | 0.8592 | 0.8891 | **0.15%** ✅ |
| **RandomForest** | 0.9049 | 0.7249 | 0.8953 | 1.0000 | **9.51%** ⚠️ |
| **XGBoost** 🏆 | **0.9057** | **0.7481** | **0.9007** | 0.9586 | **5.29%** |

### 🏆 最佳模型：XGBoost

| 指标 | 表现 |
|------|:----:|
| **Accuracy** | **0.9057**（三项最高） |
| **F1 Macro** | **0.7481**（对少数类识别最好） |
| **F1 Weighted** | **0.9007**（综合表现最佳） |
| 泛化能力 | 训练0.9586 → 测试0.9057，差距5.29%，合理可控 |

### 📋 模型保存路径

| 模型 | 模型文件 | 特征列配置 |
|------|---------|-----------|
| LogisticRegression | `models/LogisticRegression_model.pkl` | `models/LogisticRegression_feature_columns.json` |
| RandomForest | `models/RandomForest_model.pkl` | `models/RandomForest_feature_columns.json` |
| XGBoost | `models/XGBoost_model.pkl` | `models/XGBoost_feature_columns.json` |

### 💡 关键洞察

1. **XGBoost 综合最优** — Accuracy 0.9057，F1 Macro 0.7481，在三个指标上全面领先
2. **RandomForest 过拟合明显** — 训练集1.0但测试集0.9049，泛化差距9.51%
3. **LogisticRegression 最稳定** — 泛化差距仅0.15%，但性能最低
4. **类别不平衡问题** — 所有模型的F1 Macro都低于F1 Weighted，说明对少数类（`yes`）的识别仍有提升空间

需要我进一步进行超参数调优或生成最终报告吗？

---

