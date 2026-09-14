#!/usr/bin/env python3
"""Defines a function that creates a Lasso Regression model."""
from sklearn import linear_model


def lasso_regression(random_state):
    """
    Creates a Lasso Regression model using scikit-learn.

    Lasso Regression extends ordinary linear regression by adding
    L1 regularization, which helps simplify the model by forcing
    some coefficients to zero, enabling automatic feature selection.

    Args:
        random_state: an integer used to set the random seed for
            reproducibility.

    Returns:
        model: an untrained Lasso regression model instance.
    """
    model = linear_model.Lasso(random_state=random_state)
    return model
