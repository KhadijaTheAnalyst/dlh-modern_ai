#!/usr/bin/env python3
"""
Module for testing whether continuous numeric features differ
significantly between churned and non-churned customers using
Welch's t-test.
"""
from scipy import stats


def ttest_numeric(df):
    """
    Performs Welch's t-test comparing each numeric feature
    between Churn=Yes and Churn=No groups.

    Args:
        df (pd.DataFrame): DataFrame with a 'Churn' column and
            numeric feature columns.

    Returns:
        dict: A dictionary mapping each numeric feature name to
            its Welch's t-test p-value.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    result = {}
    for column in numeric_cols:
        churned = df[df['Churn'] == 'Yes'][column]
        not_churned = df[df['Churn'] == 'No'][column]
        t_stat, p = stats.ttest_ind(not_churned, churned, equal_var=False)
        result[column] = p

    return result
