#!/usr/bin/env python3
"""
Module for visualizing churn rate broken down by a categorical
feature.
"""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """
    Generates a bar plot of churn rate per category for a
    given categorical column.

    Args:
        df (pd.DataFrame): DataFrame with a 'Churn' column.
        col (str): Name of the categorical column to group by.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    churn_numeric = df['Churn'].map({'No': 0, 'Yes': 1})
    churn_rate = churn_numeric.groupby(df[col]).mean()

    plt.bar(churn_rate.index, churn_rate.values)

    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Churn Rate')
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.show()
