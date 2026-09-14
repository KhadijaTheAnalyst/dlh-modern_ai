#!/usr/bin/env python3
"""Build a shallow neural network with single hidden layer
    using the Keras sequential API."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim, neurons_h):
    """
    Builds a shallow neural network for multi-class classification
    using the Keras Sequential API.

    Arguments:
        input_dim: Number of input features.
        neurons_h: Number of neurons for the hidden layer.

    Returns:
        model: Keras model.
    """
    model = keras.Sequential([
        layers.Dense(neurons_h, activation='sigmoid',
                     input_shape=(input_dim,)),
        layers.Dense(10, activation='softmax')
    ])
    return model
