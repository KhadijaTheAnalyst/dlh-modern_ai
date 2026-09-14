#!/usr/bin/env python3
"""Defines a function that creates a Linear Regression model."""
from sklearn import linear_model


def Linear_Regression():
    """
    Creates a Linear Regression model using scikit-learn.

    Returns:
        model: an untrained LinearRegression instance.
    """
    model = linear_model.LinearRegression()
    return model
