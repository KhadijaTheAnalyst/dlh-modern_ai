#!/usr/bin/env python3
"""
Module for splitting a DataFrame into stratified train/test
sets for modeling.
"""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """
    Splits a DataFrame into stratified train and test sets.

    Args:
        df (pd.DataFrame): DataFrame containing features and
            the target column.
        target (str): Name of the target column.
        test_size (float): Proportion of data to use as the
            test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
