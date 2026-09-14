#!/usr/bin/env python3
"""
Module for comparing the distribution of a continuous numeric
feature between churned and non-churned customers.
"""
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """
    Generates overlapping/grouped histograms comparing a numeric
    column's distribution for churned vs non-churned customers.

    Args:
        df (pd.DataFrame): DataFrame with a 'Churn' column.
        col (str): Name of the numeric column to compare.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    churned = df[df['Churn'] == 'Yes'][col]
    not_churned = df[df['Churn'] == 'No'][col]

    plt.hist([not_churned, churned], bins=30, label=['No', 'Yes'])

    plt.title(f'{col} Distribution by Churn')
    plt.xlabel(col)
    plt.legend(title='Churn')
    plt.show()
