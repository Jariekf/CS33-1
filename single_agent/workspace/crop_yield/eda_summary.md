# EDA Summary

## Dataset Overview
- Rows: 2200
- Columns: 9
- Numeric features: 8
- Categorical features: 1

## Target Analysis
- Target column: yield
- Task type: regression
- Summary: mean=2702.844, median=1825.000, skew=16.411

## Feature Quality
- High-missing columns: []
- High-cardinality columns: ['crop']
- ID-like columns: ['temperature', 'humidity', 'ph_value', 'rainfall']

## Top Feature Relationships
- rainfall: abs_corr=0.0822
- temperature: abs_corr=0.0364
- humidity: abs_corr=0.0349
- ph_value: abs_corr=0.0302
- phosphorus: abs_corr=0.0215
- crop: target_mean_spread=0.9742

## Feature Engineering Recommendations
- Use frequency encoding or drop high-cardinality categorical columns instead of wide one-hot encoding.
- Drop ID-like columns from model features to reduce memorization risk.
- Consider robust metrics or transformations for highly skewed regression targets.