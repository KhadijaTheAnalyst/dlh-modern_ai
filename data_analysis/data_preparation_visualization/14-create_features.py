#!/usr/bin/env python3
"""
Module for engineering new features from the Telco dataset:
NumServices (count of subscribed services) and TenureGroup
(binned tenure categories).
"""
import pandas as pd


def create_features(df):
    """
    Creates NumServices and TenureGroup features, then drops
    the original columns used to build them.

    Args:
        df (pd.DataFrame): DataFrame containing service-related
            columns and a 'tenure' column.

    Returns:
        pd.DataFrame: The modified DataFrame with new features
            and original source columns removed.
    """
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV',
                     'StreamingMovies']
    mapped_services = df[service_cols].map(
        lambda x: 1 if x == 'Yes' else 0)
    internet_mapped = df['InternetService'].map(
        {'DSL': 1, 'Fiber optic': 1, 'No': 0}
    )
    mapped_services['InternetService'] = internet_mapped
    df['NumServices'] = mapped_services.sum(axis=1)

    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, float('inf')],
        labels=['0-12', '13-24', '25-48', '49-60', '60+']
    )

    cols_to_drop = service_cols + ['InternetService', 'tenure']
    df = df.drop(cols_to_drop, axis=1)

    return df
