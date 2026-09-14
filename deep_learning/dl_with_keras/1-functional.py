#!/usr/bin/env python3
"""Build a shallow neural network using the Keras functional API."""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """
    Builds a shallow neural network for multi-class classification
    using the Keras functional API (keras.Model), without Sequential.

    Arguments:
        input_dim: Number of input features.
        neurons_h: Number of neurons for the hidden layer.

    Returns:
        model: Keras model.
    """
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(neurons_h, activation='sigmoid')(inputs)
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
