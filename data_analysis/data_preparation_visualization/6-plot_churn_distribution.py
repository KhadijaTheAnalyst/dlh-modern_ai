#!/usr/bin/env python3
"""
Module for visualizing the distribution of the Churn target
variable using a bar plot.
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Generates a bar plot showing the distribution of the
    Churn column.

    Args:
        df (pd.DataFrame): DataFrame with a 'Churn' column.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    counts = df['Churn'].value_counts().reindex(['No', 'Yes'])
    plt.bar(counts.index, counts.values, color=['skyblue', 'salmon'])
    plt.title('Churn Distribution')
    plt.ylabel('Count')
    plt.show()
