## eda_agent  `2026-05-08 23:44:25`

---

## ✅ EDA 分析与候选模型提交完成

### 📊 EDA 关键发现

| 维度 | 结果 |
|------|------|
| **数据集** | 36,168 行 × 17 列（训练集） |
| **缺失值** | ✅ 无缺失值 |
| **数值特征** | 7个：age, balance, day, duration, campaign, pdays, previous |
| **分类特征** | 10个：job, marital, education, default, housing, loan, contact, month, poutcome, y(目标) |
| **异常值处理** | age/campaign已封顶，balance基本处理完毕，duration/pdays/previous保留 |
| **关键相关** | pdays↔previous 相关性 0.439（中度相关），其余特征间相关性较弱 |

### 🏆 选定的三个候选模型

| 模型 | 缩放 | 编码 | 核心理由 |
|------|:----:|:----:|---------|
| **LogisticRegression** | ✅ 标准化 | One-Hot | 线性可解释基线模型 |
| **RandomForest** | ❌ 不需要 | Label | 处理混合数据类型，鲁棒性好 |
| **XGBoost** | ❌ 不需要 | Label | 表格数据SOTA，性能上限高 |

### 📈 分布图已生成（7张）
保存在：`C:\Projects\multi-agent-data-science\artifacts\workspace\2026-05-08-23-43\plots\`

---

