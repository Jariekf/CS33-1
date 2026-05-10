# EDA Summary

## Dataset Overview
- Rows: 642
- Columns: 16
- Numeric features: 10
- Categorical features: 6

## Target Analysis
- Target column: profit
- Task type: regression
- Summary: mean=27289.276, median=11135.000, skew=2.754

## Feature Quality
- High-missing columns: []
- High-cardinality columns: []
- ID-like columns: []

## Top Feature Relationships
- sales: abs_corr=0.8533
- gross_sales: abs_corr=0.8351
- cogs: abs_corr=0.7941
- sale_price: abs_corr=0.6868
- discounts: abs_corr=0.4714
- segment: target_mean_spread=0.8756
- date: target_mean_spread=0.5193
- discount_band: target_mean_spread=0.3621
- month_name: target_mean_spread=0.31
- product: target_mean_spread=0.2904

## Feature Engineering Recommendations
- Consider robust metrics or transformations for highly skewed regression targets.