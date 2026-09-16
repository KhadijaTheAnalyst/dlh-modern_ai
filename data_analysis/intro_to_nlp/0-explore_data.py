#!/usr/bin/env python3
"""
This module provides basic exploration functions for text datasets.

Functions:
    explore_data(df): Performs initial dataset exploration with visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """
    Performs initial dataset exploration on a DataFrame containing messages.

    Creates a figure with two side-by-side subplots:
    - Left: Bar chart showing ham vs spam counts
    - Right: Histogram of raw message lengths

    Args:
        df (pandas.DataFrame): DataFrame with at least 'label' and 'message' columns

    Returns:
        None

    Side Effects:
        Displays a matplotlib figure with two subplots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left subplot: bar chart of ham vs spam counts
    sns.countplot(data=df, x='label', ax=ax1, palette='Set2')
    ax1.set_title("Ham vs Spam Counts")
    ax1.set_xlabel("label")
    ax1.set_ylabel("count")

    # Right subplot: histogram of raw message lengths
    ax2.hist(df['message'].str.len(), bins=50, color='steelblue', edgecolor='black')
    ax2.set_title("Histogram of Raw Message Lengths")
    ax2.set_xlabel("length")
    ax2.set_ylabel("count")

    plt.tight_layout()
