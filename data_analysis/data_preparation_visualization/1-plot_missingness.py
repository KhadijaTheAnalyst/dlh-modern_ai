#!/usr/bin/env python3
"""
Module for visualizing missing values in a DataFrame using a
scatter plot.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Generates a scatter plot showing the location of missing
    values in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to analyze.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    x_positions, y_positions = np.where(df.isnull())

    plt.scatter(x_positions, y_positions, marker='|')

    plt.yticks(range(len(df.columns)), df.columns)

    plt.tight_layout()
    plt.show()
