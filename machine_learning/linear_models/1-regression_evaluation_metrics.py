#!/usr/bin/env python3
"""Defines a function that computes regression evaluation metrics."""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluation_metrics_for_regression(y_true, y_pred):
    """
    Computes common regression evaluation metrics.

    Args:
        y_true: array-like of true target values.
        y_pred: array-like of predicted target values.

    Returns:
        mse, rmse, mae, r2
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mae, r2
