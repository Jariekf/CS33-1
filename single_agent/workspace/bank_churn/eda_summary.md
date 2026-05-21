# EDA Summary

## Dataset Overview
- Rows: 165034
- Columns: 14
- Numeric features: 11
- Categorical features: 3

## Target Analysis
- Target column: exited
- Task type: classification
- Summary: 2 classes; majority ratio 0.79

## Feature Quality
- High-missing columns: []
- High-cardinality columns: ['surname']
- ID-like columns: ['id', 'customerid']

## Top Feature Relationships
- age: abs_corr_with_target_code=0.3417
- numofproducts: abs_corr_with_target_code=0.2153
- isactivemember: abs_corr_with_target_code=0.211
- balance: abs_corr_with_target_code=0.13
- creditscore: abs_corr_with_target_code=0.0266
- surname: target_rate_spread=1.0
- geography: target_rate_spread=0.2146
- gender: target_rate_spread=0.1221

## Feature Engineering Recommendations
- Use frequency encoding or drop high-cardinality categorical columns instead of wide one-hot encoding.
- Drop ID-like columns from model features to reduce memorization risk.