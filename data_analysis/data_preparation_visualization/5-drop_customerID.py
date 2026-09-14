#!/usr/bin/env python3
"""
Module for dropping the customerID column, since unique
identifiers have no predictive value for modeling.
"""


def drop_customerID(df):
    """
    Removes the customerID column from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a customerID column.

    Returns:
        pd.DataFrame: The modified DataFrame without customerID.
    """
    df = df.drop('customerID', axis=1)
    return df
