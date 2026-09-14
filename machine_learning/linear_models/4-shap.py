#!/usr/bin/env python3
"""Defines a function that generates SHAP explanations for a model."""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """
    Creates a SHAP explainer and computes SHAP values.

    Args:
        model: a trained regression model.
        X_train: input data used to initialize the explainer
            (background dataset).
        X_test: input data to explain.

    Returns:
        explainer: SHAP explainer object.
        shap_values: SHAP values for the predictions on X_test.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return explainer, shap_values
