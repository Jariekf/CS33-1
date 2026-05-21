# EDA Summary

## Dataset Overview
- Rows: 145
- Columns: 33
- Numeric features: 32
- Categorical features: 1

## Target Analysis
- Target column: 1
- Task type: classification
- Summary: 3 classes; majority ratio 0.48

## Feature Quality
- High-missing columns: []
- High-cardinality columns: ['student_id']
- ID-like columns: ['student_id', 'course_id']

## Top Feature Relationships
- 4: abs_corr_with_target_code=0.327
- course_id: abs_corr_with_target_code=0.3163
- 3: abs_corr_with_target_code=0.3069
- 28: abs_corr_with_target_code=0.2891
- 15: abs_corr_with_target_code=0.2408

## Feature Engineering Recommendations
- Use frequency encoding or drop high-cardinality categorical columns instead of wide one-hot encoding.
- Drop ID-like columns from model features to reduce memorization risk.