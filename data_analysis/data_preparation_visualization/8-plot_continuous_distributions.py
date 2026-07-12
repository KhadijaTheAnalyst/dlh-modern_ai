#!/usr/bin/env python3
"""
Module for visualizing distributions of continuous numerical
features using histograms with KDE and boxplots.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Generates histogram+KDE and boxplot pairs for continuous
    numerical columns.

    Args:
        df (pd.DataFrame): DataFrame to visualize.
        columns_to_plot (list): Optional list of continuous
            numeric columns. Defaults to all numeric columns.

    Returns:
        None
    """
    if columns_to_plot is None:
        columns_to_plot = df.select_dtypes(
            include=['int64', 'float64']
        ).columns

    n_rows = len(columns_to_plot)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, column in enumerate(columns_to_plot):
        row = i
        hist_ax = axes[row, 0]
        box_ax = axes[row, 1]
        hist_ax.hist(
            df[column], bins=30, density=True,
            alpha=0.7, edgecolor='black'
        )
        sns.kdeplot(df[column], ax=hist_ax, color='red')
        hist_ax.set_title(f'{column} Histogram + KDE')
        box_ax.boxplot(df[column], vert=False)
        box_ax.set_title(f'{column} Boxplot')

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
