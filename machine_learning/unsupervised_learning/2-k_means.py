#!/usr/bin/env python3
"""Clustering with K-Means module."""

from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """Create and fit a K-Means clustering model on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters to find.
        random_state (int): Random seed for reproducibility.

    Returns:
        sklearn.cluster.KMeans: K-Means instance fitted on the input data.
    """
    model = cluster.KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    model.fit(X)

    return model
