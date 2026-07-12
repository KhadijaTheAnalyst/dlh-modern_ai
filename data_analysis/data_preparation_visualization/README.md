# Telco Customer Churn — EDA & Data Preprocessing

Exploratory data analysis and preprocessing pipeline for the Telco Customer
Churn dataset, built as part of the Digital Learning Hub Luxembourg (DLH) AI
Academy, following the Holberton School ML Engineering curriculum.

## Overview

This project takes the raw Telco Customer Churn dataset through a complete
pipeline — from initial inspection through cleaning, visualization,
statistical testing, feature engineering, and model-ready preprocessing.
Each task is a standalone, importable Python module with a single function,
chained together in later tasks to form the full pipeline.

## Requirements

- Ubuntu 20.04 LTS, `python3` (3.11)
- All files start with `#!/usr/bin/env python3`, end with a newline, and
  are executable
- `pycodestyle` (2.14.0) compliant
- Every module, class, and function is documented
- No unauthorized imports beyond what each task specifies

| Package | Version |
|---|---|
| numpy | 2.0.2 |
| pandas | 2.2.2 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| scipy | 1.16.0 |
| pillow | 11.3.0 |

## Datasets

- **Tasks 0–5**: `Telco-Customer-Churn.csv` (raw)
- **Tasks 6–17**: `precleaned-Telco-Customer-Churn.csv`

Datasets are not committed to this repository — download them separately
and place them in this directory before running any task.

## Tasks

| # | File | Description |
|---|---|---|
| 0 | `0-describe_data.py` | Shape, dtypes, head, missing counts, duplicate count |
| 1 | `1-plot_missingness.py` | Scatter plot visualizing missing values across the dataset |
| 2 | `2-convert_columns.py` | Converts `TotalCharges` to numeric, maps `SeniorCitizen` to Yes/No |
| 3 | `3-clean_total_charges.py` | Handles missing `TotalCharges` via drop, median fill, or imputation |
| 4 | `4-remove_duplicates.py` | Drops duplicate rows |
| 5 | `5-drop_customerID.py` | Removes the non-predictive `customerID` column |
| 6 | `6-plot_churn_distribution.py` | Bar plot of the target class distribution |
| 7 | `7-plot_categorical_distributions.py` | Grid of bar plots for all categorical features |
| 8 | `8-plot_continuous_distributions.py` | Histogram+KDE and boxplot pairs for numeric features |
| 9 | `9-plot_correlation_heatmap.py` | Annotated correlation heatmap for numeric features |
| 10 | `10-plot_categorical_vs_churn.py` | Churn rate per category for a given categorical column |
| 11 | `11-plot_numeric_vs_churn.py` | Overlapping histograms comparing a numeric feature by churn status |
| 12 | `12-chi_square_tests.py` | Chi-square independence tests: categorical features vs. Churn |
| 13 | `13-ttest_numeric.py` | Welch's t-tests: numeric features vs. Churn |
| 14 | `14-create_features.py` | Engineers `NumServices` and `TenureGroup`; drops source columns |
| 15 | `15-encode_features.py` | Label, ordinal, and one-hot encoding for modeling |
| 16 | `16-scale_numeric.py` | Standardizes `MonthlyCharges` and `TotalCharges` |
| 17 | `17-split_data.py` | Stratified train/test split |

## Pipeline Usage

Tasks 14–17 are designed to chain together into a single preprocessing
pipeline:

```python
import pandas as pd

create_features = __import__('14-create_features').create_features
encode_features = __import__('15-encode_features').encode_features
scale_numeric = __import__('16-scale_numeric').scale_numeric
split_data = __import__('17-split_data').split_data

df = pd.read_csv('precleaned-Telco-Customer-Churn.csv')
df.drop(columns=['gender', 'PhoneService'], inplace=True)

df = create_features(df)
df_enc, churn_le, binary_oe, tenure_oe = encode_features(df)
df_scaled = scale_numeric(df_enc)

X_train, X_test, y_train, y_test = split_data(df_scaled)
```

## Key Findings

- `TotalCharges` contains 11 missing values, stored as whitespace strings
  (`' '`) rather than `NaN` — invisible to `.isnull()` until explicitly
  converted with `pd.to_numeric(..., errors='coerce')`.
- `gender` and `PhoneService` show no statistically significant
  association with churn (Chi-square p-values ≈ 0.49 and ≈ 0.35), and are
  excluded from the feature set accordingly.
- `Contract`, `OnlineSecurity`, `TechSupport`, and `InternetService` show
  extremely strong associations with churn (p-values effectively at 0).
- Customer tenure differs significantly between churned and non-churned
  customers (Welch's t-test), consistent with churn being concentrated
  among newer customers.

## Author

**Khadija** — DLH AI Academy / Holberton ML Engineering
GitHub: [KhadijaTheAnalyst](https://github.com/KhadijaTheAnalyst)