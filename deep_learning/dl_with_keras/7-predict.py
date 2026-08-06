#!/usr/bin/env python3
"""Generate class predictions from a trained Keras model."""
import tensorflow as tf


def predict(model, X, verbose=0):
    """
    Makes predictions on a given dataset using a trained Keras model.

    Arguments:
        model: A trained Keras model.
        X: Input data, shape (number of examples, input features).
        verbose: Verbosity level during predictions (0, 1, or 2).

    Returns:
        predictions: A list of predicted class labels for the input data.
    """
    probabilities = model.predict(X, verbose=verbose)
    predictions = tf.argmax(probabilities, axis=1).numpy().tolist()
    return predictions
