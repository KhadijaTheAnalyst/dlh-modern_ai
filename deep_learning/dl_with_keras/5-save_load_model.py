#!/usr/bin/env python3
"""Save and load a full Keras model
(architecture, weights, optimizer state)."""
from tensorflow import keras


def save_model(model, filepath):
    """
    Saves a trained Keras model to the specified filepath.

    Arguments:
        model: A trained Keras model to be saved.
        filepath: Path (including file name) where the model will be saved.

    Returns:
        None.
    """
    model.save(filepath)
    return None


def load_model(filepath):
    """
    Loads a full Keras model from the specified filepath.

    Arguments:
        filepath: Path (including file name) from where the model
                  will be loaded.

    Returns:
        model: The reloaded Keras model.
    """
    model = keras.models.load_model(filepath)
    return model
