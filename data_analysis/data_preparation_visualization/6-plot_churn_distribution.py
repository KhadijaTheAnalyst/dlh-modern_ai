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
    x = df['Churn'].value_counts()
    color_mapping = {'No': 'skyblue', 'Yes': 'salmon'}
    colors = [color_mapping[val] for val in x.index]
    x.plot(kind='bar', color=colors)
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.title('Distribution of Churn')
    plt.show()
