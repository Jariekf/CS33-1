# EDA Summary

## Dataset Overview
- Rows: 1460
- Columns: 81
- Numeric features: 38
- Categorical features: 43

## Target Analysis
- Target column: id
- Task type: classification
- Summary: 1168 classes; majority ratio 0.00

## Feature Quality
- High-missing columns: ['alley', 'masvnrtype', 'fireplacequ', 'poolqc', 'fence', 'miscfeature']
- High-cardinality columns: ['neighborhood']
- ID-like columns: []

## Top Feature Relationships
- poolarea: abs_corr_with_target_code=0.056
- lowqualfinsf: abs_corr_with_target_code=0.0508
- 3ssnporch: abs_corr_with_target_code=0.0494
- bedroomabvgr: abs_corr_with_target_code=0.0399
- lotarea: abs_corr_with_target_code=0.0331

## Feature Engineering Recommendations
- Create missing indicators and impute high-missingness columns using train-fitted values.
- Use frequency encoding or drop high-cardinality categorical columns instead of wide one-hot encoding.