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

    oe = preprocessing.OrdinalEncoder(categories=[['No', 'Yes']] * 4)
    binary_cols = ['Partner', 'Dependents', 'PaperlessBilling',
                    'SeniorCitizen']
    encoded_binary = oe.fit_transform(df[binary_cols])
    df[binary_cols] = encoded_binary
    df[binary_cols] = df[binary_cols].astype(int)

    df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod'],
                         drop_first=True, dtype=int)

    oe_tenure = preprocessing.OrdinalEncoder()
    df['TenureGroup'] = oe_tenure.fit_transform(df[['TenureGroup']])
    df['TenureGroup'] = df['TenureGroup'].astype(int)

    return df, le, oe, oe_tenure
