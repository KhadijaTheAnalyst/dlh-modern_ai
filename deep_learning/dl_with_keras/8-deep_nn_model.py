#!/usr/bin/env python3
"""Build a deep neural network using the Keras Sequential class."""
from tensorflow import keras


def build_deep_model(input_dim, hidden_layers):
    """
    Builds a deep neural network for multi-class classification
    using the Keras Sequential class.

    Arguments:
        input_dim: Number of input features.
        hidden_layers: List of integers representing the number of
                       neurons in each hidden layer,
                       e.g. [16, 8, 4] for three hidden layers.

    Returns:
        model: Keras model.
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
