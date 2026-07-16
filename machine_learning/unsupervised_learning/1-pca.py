#!/usr/bin/env python3
"""Dimensionality reduction with PCA module."""

from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """Perform Principal Component Analysis (PCA) on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_components (int, float, or None):
            - int: number of principal components to keep.
            - float (between 0 and 1): minimum fraction of variance
              to preserve.
            - None: keep all components.
        random_state (int): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Data transformed into principal component space.
        sklearn.decomposition.PCA: Fitted PCA instance.
    """
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )
    X_pca = pca.fit_transform(X)

    return X_pca, pca
