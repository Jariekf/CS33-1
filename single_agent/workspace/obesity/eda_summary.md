# EDA Summary

## Dataset Overview
- Rows: 20758
- Columns: 18
- Numeric features: 9
- Categorical features: 9

## Target Analysis
- Target column: nobeyesdad
- Task type: classification
- Summary: 7 classes; majority ratio 0.19

## Feature Quality
- High-missing columns: []
- High-cardinality columns: []
- ID-like columns: ['id']

## Top Feature Relationships
- weight: abs_corr_with_target_code=0.4311
- age: abs_corr_with_target_code=0.2808
- ch2o: abs_corr_with_target_code=0.1913
- faf: abs_corr_with_target_code=0.0909
- ncp: abs_corr_with_target_code=0.09

## Feature Engineering Recommendations
- Drop ID-like columns from model features to reduce memorization risk.