#!/usr/bin/env python3
"""
Module for handling missing values in the TotalCharges column
using dropping, median filling, or imputation strategies.
"""


def clean_total_charges(df, method='drop'):
    """
    Handles missing values in the 'TotalCharges' column of a
    DataFrame using the specified strategy.

    Args:
        df (pd.DataFrame): DataFrame with missing values in
            'TotalCharges'.
        method (str): Strategy to use. One of:
            'drop'   - Remove rows with missing TotalCharges.
            'median' - Fill missing values with the column median.
            'impute' - Fill missing values with
                       MonthlyCharges * tenure.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    elif method == 'median':
        median_value = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
    elif method == 'impute':
        df['TotalCharges'] = df['TotalCharges'].fillna(
            df['MonthlyCharges'] * df['tenure']
        )
    return df
