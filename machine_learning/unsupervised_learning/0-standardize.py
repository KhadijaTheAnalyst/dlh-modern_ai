#!/usr/bin/env python3
"""Feature standardization module."""

from sklearn import preprocessing


def Standardize(X):
    """Standardize tabular data using Scikit-learn.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The standardized version of the input data, with the
            same shape as X.
    """
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled
