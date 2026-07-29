#!/usr/bin/env python3
"""Defines a function that creates a Logistic Regression model."""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """
    Creates a Logistic Regression model using scikit-learn.

    Logistic Regression performs binary classification by fitting
    a logistic function to the data.

    Args:
        random_state: an integer used to set the random seed for
            reproducibility.

    Returns:
        model: an untrained LogisticRegression instance.
    """
    model = linear_model.LogisticRegression(random_state=random_state)
    return model
