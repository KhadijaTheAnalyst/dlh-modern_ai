#!/usr/bin/env python3
"""
Module for converting specific column types in the
Telco Customer Churn DataFrame.
"""
import pandas as pd


def convert_columns(df):
    """
    Converts specific columns in the DataFrame to appropriate types.

    Args:
        df (pd.DataFrame): DataFrame containing 'TotalCharges' and
            'SeniorCitizen' columns.

    Returns:
        pd.DataFrame: The modified DataFrame with:
            - 'TotalCharges' converted to numeric (non-numeric
              entries become NaN).
            - 'SeniorCitizen' mapped from 0/1 to 'No'/'Yes'.
    """
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df
