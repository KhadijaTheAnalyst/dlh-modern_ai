#!/usr/bin/env python3
"""
Module for standardizing numeric features using scikit-learn's
StandardScaler.
"""
from sklearn import preprocessing


def scale_numeric(df):
    """
    Scales MonthlyCharges and TotalCharges to have mean 0 and
    standard deviation 1.

    Args:
        df (pd.DataFrame): DataFrame containing 'MonthlyCharges'
            and 'TotalCharges' columns.

    Returns:
        pd.DataFrame: The modified DataFrame with scaled columns.
    """
    scaler = preprocessing.StandardScaler()
    numeric_cols = ['MonthlyCharges', 'TotalCharges']
    scaled_values = scaler.fit_transform(df[numeric_cols])
    df[numeric_cols] = scaled_values
    return df
