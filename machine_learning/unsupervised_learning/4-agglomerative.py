#!/usr/bin/env python3
"""Agglomerative hierarchical clustering module."""

from sklearn import cluster
from sklearn import metrics

Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state, n_components,
                              use_pca_data=True):
    """Perform Agglomerative hierarchical clustering on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility (used by PCA).
        n_components (int): Number of PCA components to reduce the data
            to (used only if use_pca_data=True).
        use_pca_data (bool): Whether to apply PCA to reduce
            dimensionality before clustering. Default is True.

    Returns:
        sklearn.cluster.AgglomerativeClustering: Fitted Agglomerative
            Clustering instance.
        numpy.ndarray: Data used for fitting (PCA-reduced or original).
        float: Silhouette score of the clustering (None if n_clusters=1).
    """
    if use_pca_data:
        X_used, _ = Apply_PCA(X, n_components, random_state)
    else:
        X_used = X

    model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    model.fit(X_used)

    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)
    else:
        score = None

    return model, X_used, score
