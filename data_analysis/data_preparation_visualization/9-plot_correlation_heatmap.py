#!/usr/bin/env python3
"""
Module for visualizing pairwise correlations between continuous
numeric features using a heatmap.
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Generates an annotated correlation heatmap for the numeric
    columns of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to visualize.

    Returns:
        None
    """
    plt.figure(figsize=(6, 5))

    df_corr = df.select_dtypes(include=['int64', 'float64'])

    sns.heatmap(
        df_corr.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1
    )
    plt.title('Correlation Heatmap')
    plt.show()
