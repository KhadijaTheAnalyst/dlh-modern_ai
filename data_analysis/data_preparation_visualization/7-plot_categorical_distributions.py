#!/usr/bin/env python3
"""
Module for visualizing distributions of categorical features
in a grid of bar plots.
"""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    Generates bar plots for categorical columns in a grid layout.

    Args:
        df (pd.DataFrame): DataFrame to visualize.
        columns_to_plot (list): Optional list of categorical
            columns. Defaults to all object-dtype columns,
            excluding 'Churn'.

    Returns:
        None
    """
    if columns_to_plot is None:
        col_type = df.select_dtypes(include='object').columns
        columns_to_plot = [c for c in col_type if c != 'Churn']

    n_cols = 3
    n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    for i, column in enumerate(columns_to_plot):
        counts = df[column].value_counts()
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col]
        ax.bar(counts.index, counts.values)
        ax.set_title(column)
        ax.tick_params(axis='x', rotation=45)

    for i in range(len(columns_to_plot), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
