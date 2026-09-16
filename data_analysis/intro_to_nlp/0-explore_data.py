#!/usr/bin/env python3
"""
This module performs initial dataset exploration.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """
    Performs initial dataset exploration:
    Creates a figure with two subplots side by side:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    Left subplot: bar chart of ham vs spam counts using sns.barplot:
        - title: "Ham vs Spam Counts", xlabel: "label", ylabel: "count"
    Right subplot: histogram of raw message lengths using sns.histplot:
        - bins: 50
        - title: "Histogram of Raw Message Lengths",
          xlabel: "length", ylabel: "count"

    Args:
        df: Dataset containing 'label' and 'message' columns.
    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left subplot - bar chart of label distribution
    label_data = df['label'].value_counts()
    sns.barplot(x=label_data.index, y=label_data.values, ax=ax1)
    ax1.set(title="Ham vs Spam Counts", xlabel="label", ylabel="count")

    # Right subplot - histogram of message lengths
    msg_lens = df['message'].str.len()
    sns.histplot(msg_lens, bins=50, ax=ax2, stat='count', kde=False)
    ax2.set(title="Histogram of Raw Message Lengths", xlabel="length",
            ylabel="count")

    plt.tight_layout()
