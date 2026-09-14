#!/usr/bin/env python3
"""
Module for testing independence between categorical features
and the Churn target using the Chi-square test of independence.
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """
    Performs Chi-square tests of independence between each
    categorical feature and the Churn target variable.

    Args:
        df (pd.DataFrame): DataFrame with a 'Churn' column and
            other categorical columns.

    Returns:
        dict: A dictionary mapping each feature name to its
            Chi-square test p-value.
    """
    col_type = df.select_dtypes(include='object').columns
    categorical_cols = [c for c in col_type if c != 'Churn']

    result = {}
    for column in categorical_cols:
        contingency_table = pd.crosstab(df[column], df['Churn'])
        chi2, p, dof, expected = stats.chi2_contingency(
            contingency_table
        )
        result[column] = p

    return result
