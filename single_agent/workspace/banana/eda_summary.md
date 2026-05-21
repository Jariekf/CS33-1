# EDA Summary

## Dataset Overview
- Rows: 1000
- Columns: 16
- Numeric features: 11
- Categorical features: 5

## Target Analysis
- Target column: quality_score
- Task type: regression
- Summary: mean=2.474, median=2.440, skew=0.002

## Feature Quality
- High-missing columns: []
- High-cardinality columns: ['harvest_date']
- ID-like columns: ['sample_id', 'weight_g', 'altitude_m', 'rainfall_mm']

## Top Feature Relationships
- ripeness_index: abs_corr=0.679
- sugar_content_brix: abs_corr=0.6005
- length_cm: abs_corr=0.447
- soil_nitrogen_ppm: abs_corr=0.0733
- firmness_kgf: abs_corr=0.0511
- quality_category: target_mean_spread=4.2844
- ripeness_category: target_mean_spread=1.9743
- harvest_date: target_mean_spread=1.5814
- variety: target_mean_spread=0.3664
- region: target_mean_spread=0.244

## Feature Engineering Recommendations
- Use frequency encoding or drop high-cardinality categorical columns instead of wide one-hot encoding.
- Drop ID-like columns from model features to reduce memorization risk.