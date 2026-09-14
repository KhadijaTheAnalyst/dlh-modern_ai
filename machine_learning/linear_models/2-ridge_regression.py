#!/usr/bin/env python3
"""Defines a function that creates a Ridge Regression model."""
from sklearn import linear_model


def ridge_regression(random_state):
    """
    Creates a Ridge Regression model using scikit-learn.

    Ridge Regression extends ordinary linear regression by adding
    L2 regularization, which helps stabilize the model by shrinking
    large coefficients.

    Args:
        random_state: an integer used to set the random seed for
            reproducibility.

    Returns:
        model: an untrained Ridge regression model instance.
    """
    model = linear_model.Ridge(random_state=random_state)
    return model
