#!/usr/bin/env python3
"""
Task 4: Weight Initialization

This module implements the build_model_initializer_by_activation function
which builds a Keras model with appropriate weight initializers based on
the activation function used in the hidden layer.

Weight initialization strategies:
- sigmoid/tanh: Glorot Uniform (Xavier) initializer
- relu/leaky_relu: He Normal initializer
"""

from tensorflow import keras


def build_model_initializer_by_activation(input_dim, hidden_units, activation):
    """
    Build a compiled Keras model with appropriate weight initialization
    based on the activation function.

    This function creates a neural network with a hidden layer using
    an appropriate weight initializer for the specified activation function,
    followed by a softmax output layer.

    Args:
        input_dim (int): The number of input features.
        hidden_units (int): The number of neurons in the hidden layer.
        activation (str): The activation function for the hidden layer.
                         Options: 'sigmoid', 'tanh', 'relu', 'leaky_relu'

    Returns:
        model (keras.Model): A Keras Sequential model with:
            - Input layer with shape (input_dim,)
            - Dense hidden layer with specified activation and initializer
            - Dense output layer (10 units) with softmax activation
    """

    # Determine initializer based on activation function
    if activation in ['sigmoid', 'tanh']:
        initializer = keras.initializers.GlorotUniform()
    elif activation in ['relu', 'leaky_relu']:
        initializer = keras.initializers.HeNormal()
    else:
        raise ValueError(
            "activation must be 'sigmoid', 'tanh', 'relu', or 'leaky_relu'"
        )

    # Build the model
    model = keras.Sequential([
        keras.layers.Dense(
            hidden_units,
            activation=activation,
            kernel_initializer=initializer,
            input_shape=(input_dim,)
        ),
        keras.layers.Dense(10, activation='softmax')
    ])

    return model
