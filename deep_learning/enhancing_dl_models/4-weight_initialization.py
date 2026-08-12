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
                        Example: 784 for MNIST (28x28 flattened)
        hidden_units (int): The number of neurons in the hidden layer.
                           Typical range: [4, 256]
        activation (str): The activation function for the hidden layer.
                         Options: 'sigmoid', 'tanh', 'relu', 'leaky_relu'

    Returns:
        model (keras.Model): A compiled Keras Sequential model with:
            - Input layer with shape (input_dim,)
            - Dense hidden layer with specified activation and initializer
            - Dense output layer (10 units) with softmax activation

    Raises:
        ValueError: If activation is not one of the valid options.

    Examples:
        >>> model = build_model_initializer_by_activation(784, 128, 'relu')
        >>> # Uses HeNormal initializer for relu

        >>> model = build_model_initializer_by_activation(784, 128, 'sigmoid')
        >>> # Uses GlorotUniform initializer for sigmoid

    Weight Initialization Strategies:
    ---------------------------------
    'sigmoid' or 'tanh':
        - Initializer: Glorot Uniform (Xavier Uniform)
        - Range: [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
        - Pros: Works well for sigmoid/tanh, prevents saturation
        - Why: These activations are more sensitive to weight magnitude
        - Effect: Maintains similar variance across layers

    'relu' or 'leaky_relu':
        - Initializer: He Normal
        - Distribution: Normal with std = sqrt(2 / fan_in)
        - Pros: Accounts for ReLU dead neuron problem
        - Why: ReLU kills negative values, needs careful initialization
        - Effect: Maintains signal propagation in deep networks

    Why Weight Initialization Matters:
    ---------------------------------
    - Too small: Vanishing gradients, slow learning
    - Too large: Exploding gradients, training instability
    - Appropriate: Balanced signal flow, efficient learning
    - Different activations need different strategies
    """

    # Determine initializer based on activation function
    if activation in ['sigmoid', 'tanh']:
        # Glorot Uniform (Xavier) for sigmoid and tanh
        initializer = keras.initializers.GlorotUniform()
    elif activation in ['relu', 'leaky_relu']:
        # He Normal for ReLU and LeakyReLU
        initializer = keras.initializers.HeNormal()
    else:
        raise ValueError(
            "activation must be 'sigmoid', 'tanh', 'relu', or 'leaky_relu'"
        )

    # Build the model
    model = keras.Sequential()

    # Add input layer
    model.add(keras.layers.Input(shape=(input_dim,)))

    # Add hidden layer with appropriate initializer and activation
    model.add(
        keras.layers.Dense(
            hidden_units,
            activation=activation,
            kernel_initializer=initializer
        )
    )

    # Add output layer with softmax activation
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
