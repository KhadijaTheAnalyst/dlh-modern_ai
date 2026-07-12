#!/usr/bin/env python3
"""
Module for visualizing distributions of continuous numerical
features using histograms with KDE and boxplots.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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
        data = df[column].dropna()
        hist_ax = axes[i, 0]
        box_ax = axes[i, 1]

        hist_ax.hist(
            data, bins=30, density=True,
            alpha=0.7, edgecolor='black'
        )
        kde = stats.gaussian_kde(data)
        x_values = np.linspace(data.min(), data.max(), 200)
        hist_ax.plot(x_values, kde(x_values), color='red')
        hist_ax.set_title(f'{column} Histogram + KDE')

        box_ax.boxplot(data, vert=False)
        box_ax.set_title(f'{column} Boxplot')

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
