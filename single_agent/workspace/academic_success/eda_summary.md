# EDA Summary

## Dataset Overview
- Rows: 76518
- Columns: 38
- Numeric features: 37
- Categorical features: 1

## Target Analysis
- Target column: target
- Task type: classification
- Summary: 3 classes; majority ratio 0.47

## Feature Quality
- High-missing columns: []
- High-cardinality columns: []
- ID-like columns: ['id']

## Top Feature Relationships
- curricular_units_2nd_sem_(approved): abs_corr_with_target_code=0.783
- curricular_units_1st_sem_(approved): abs_corr_with_target_code=0.7262
- curricular_units_2nd_sem_(grade): abs_corr_with_target_code=0.7202
- curricular_units_1st_sem_(grade): abs_corr_with_target_code=0.6628
- tuition_fees_up_to_date: abs_corr_with_target_code=0.4153

## Feature Engineering Recommendations
- Drop ID-like columns from model features to reduce memorization risk.