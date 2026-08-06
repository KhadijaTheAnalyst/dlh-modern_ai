#!/usr/bin/env python3
"""Evaluate a trained Keras model."""


def evaluate_model(model, X, Y, verbose=0):
    """
    Assesses a trained Keras model's performance on given data.

    Arguments:
        model: A trained Keras model.
        X: Input data, shape (number of examples, input features).
        Y: True labels, shape (number of examples, 1).
        verbose: Verbosity mode (0 = silent, 1 = progress bar).

    Returns:
        loss: The calculated loss on the provided data.
        accuracy: The accuracy of the model on the provided data.
    """
    loss, accuracy = model.evaluate(X, Y, verbose=verbose)
    return loss, accuracy
