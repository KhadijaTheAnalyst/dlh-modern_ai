#!/usr/bin/env python3
"""
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    """
    plt.figure(figsize=(12, 8))

    y_positions, x_positions = np.where(df.isnull().T)

    plt.scatter(x_positions, y_positions, marker='|', color='C0')

    plt.yticks(range(len(df.columns)), df.columns)
    plt.xlabel('Row Index')
    plt.ylabel('Column')
    plt.title('Missing Data Visualization')

    plt.tight_layout()
    plt.show()
    