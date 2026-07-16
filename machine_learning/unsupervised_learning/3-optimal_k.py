#!/usr/bin/env python3
"""Choosing the optimal k for K-Means module."""

from sklearn import metrics

K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """Evaluate K-Means clustering quality across a range of k values.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        max_clusters (int): Maximum number of clusters to evaluate (>= 2).
        random_state (int): Random seed for reproducibility.

    Returns:
        list[int]: Evaluated cluster numbers.
        list[float]: Inertia values corresponding to each cluster number,
            for the elbow method.
        list[float]: Silhouette scores corresponding to each cluster
            number, for cluster quality evaluation.
    """
    ks = []
    inertia_values = []
    silhouette_values = []

    for k in range(2, max_clusters + 1):
        model = K_Means(X, n_clusters=k, random_state=random_state)

        ks.append(k)
        inertia_values.append(model.inertia_)
        silhouette_values.append(
            metrics.silhouette_score(X, model.labels_)
        )

    return ks, inertia_values, silhouette_values
