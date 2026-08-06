#!/usr/bin/env python3
"""Save and load only the weights of a Keras model."""


def save_model_weights(model, filepath):
    """
    Saves only the weights of a trained Keras model.

    Arguments:
        model: A trained Keras model whose weights need to be saved.
        filepath: Path (including file name) where the weights
                  will be saved.

    Returns:
        None.
    """
    model.save_weights(filepath)
    return None


def load_model_weights(model, filepath):
    """
    Loads saved weights into a compatible Keras model instance.

    Arguments:
        model: A compatible Keras model instance where the weights
               will be loaded.
        filepath: Path (including file name) from where the weights
                  will be loaded.

    Returns:
        None.
    """
    model.load_weights(filepath)
    return None
