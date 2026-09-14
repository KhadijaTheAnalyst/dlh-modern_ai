#!/usr/bin/env python3
"""
Module for encoding categorical and binary features for
modeling, using scikit-learn's LabelEncoder and OrdinalEncoder,
plus pandas one-hot encoding.
"""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    Encodes categorical features for modeling.

    Args:
        df (pd.DataFrame): DataFrame containing 'Churn', binary
            columns ('Partner', 'Dependents', 'PaperlessBilling',
            'SeniorCitizen'), 'Contract', 'PaymentMethod', and
            'TenureGroup'.

    Returns:
        tuple: (
            pd.DataFrame: The encoded DataFrame,
            LabelEncoder: Fitted encoder for 'Churn',
            OrdinalEncoder: Fitted encoder for binary columns,
            OrdinalEncoder: Fitted encoder for 'TenureGroup'
        )
    """
    le = preprocessing.LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])

    binary_cols = ['Partner', 'Dependents', 'PaperlessBilling',
                   'SeniorCitizen']
    oe = preprocessing.OrdinalEncoder(categories=[['No', 'Yes']])
    for column in binary_cols:
        df[[column]] = oe.fit_transform(df[[column]])
        df[column] = df[column].astype(int)

    df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod'],
                        drop_first=True, dtype=int)

    oe_tenure = preprocessing.OrdinalEncoder()
    df['TenureGroup'] = oe_tenure.fit_transform(df[['TenureGroup']])
    df['TenureGroup'] = df['TenureGroup'].astype(int)

    return df, le, oe, oe_tenure
