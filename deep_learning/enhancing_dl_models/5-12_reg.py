#!/usr/bin/env python3
"""
Task 5: L2 Regularization

This module implements the build_model_with_L2_regularization function
which builds a Keras model with L2 regularization applied to the kernel
weights of all hidden layers.

L2 regularization penalizes large weights to prevent overfitting and
encourage the model to learn simpler, more generalizable patterns.
"""

from tensorflow import keras


def build_model_with_L2_regularization(input_dim, hidden_units,
                                       n_layers, lambda_l2):
    """
    Build a Keras model with L2 regularization on hidden layers.

    This function creates a neural network with multiple hidden layers,
    each using ReLU activation and L2 regularization on kernel weights,
    followed by a softmax output layer.

    Args:
        input_dim (int): The number of input features.
                        Example: 784 for MNIST (28x28 flattened)
        hidden_units (int): The number of neurons in each hidden layer.
                           Typical range: [32, 256]
        n_layers (int): The number of hidden layers to include.
                       Typical range: [1, 5]
        lambda_l2 (float): The strength of L2 regularization.
                          Typical range: [1e-6, 1e-2]
                          - 0: No regularization
                          - Higher values: Stronger regularization

    Returns:
        model (keras.Model): A compiled Keras model with:
            - InputLayer with shape (input_dim,)
            - n_layers Dense hidden layers with ReLU activation
                and L2 regularization
            - Dense output layer (10 units) with softmax activation

    Raises:
        ValueError: If n_layers < 1 or lambda_l2 < 0.

    Examples:
        >>> # Model without regularization
        >>> model = build_model_with_L2_regularization(784, 64, 3, lambda_l2=0)

        >>> # Model with strong regularization
        >>> model = build_model_with_L2_regularization
                    (784, 64, 3, lambda_l2=1e-6)

    What is L2 Regularization:
    -------------------------
    L2 regularization adds a penalty term to the loss function:

    Total Loss = Original Loss + lambda * sum(weights^2)

    This encourages the model to learn smaller weights, which:
    - Simplifies the model
    - Reduces overfitting
    - Improves generalization to test data

    Why L2 Regularization Works:
    ---------------------------
    - Large weights can lead to complex, overfitted models
    - Penalizing large weights forces the model to use smaller weights
    - Smaller weights mean simpler decision boundaries
    - Simpler models generalize better to new data

    L2 vs Other Regularization:
    --------------------------
    - L2 (Ridge): Penalizes magnitude of weights (preferred for most cases)
    - L1 (Lasso): Forces some weights to zero (sparse models)
    - L2 is smoother: Distributes the penalty across all weights
    - L1 is sharper: Can eliminate less important features

    Lambda Tuning:
    ---------------
    - lambda_l2 = 0: No regularization (may overfit)
    - lambda_l2 = 1e-6: Weak regularization (most common)
    - lambda_l2 = 1e-4: Medium regularization
    - lambda_l2 = 1e-2: Strong regularization (may underfit)

    Effect on Weights:
    -----------------
    Without L2:  Weights can grow large → Complex decision boundaries
    With L2:     Weights stay small → Simple decision boundaries
    """

    # Input validation
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if lambda_l2 < 0:
        raise ValueError("lambda_l2 must be non-negative")

    # Create L2 regularizer (or None if lambda_l2 is 0)
    if lambda_l2 > 0:
        l2_regularizer = keras.regularizers.L2(lambda_l2)
    else:
        l2_regularizer = None

    # Step 1: Build model using Functional API
    # Create input layer
    inputs = keras.Input(shape=(input_dim,))

    # Step 2: Add hidden layers with L2 regularization
    x = inputs
    for i in range(n_layers):
        x = keras.layers.Dense(
            units=hidden_units,
            activation='relu',
            kernel_regularizer=l2_regularizer,
            name=f'hidden_layer_{i+1}'
        )(x)

    # Step 3: Add output layer (no regularization on output)
    outputs = keras.layers.Dense(
        units=10,
        activation='softmax',
        name='output_layer'
    )(x)

    # Step 4: Bundle layers into a Model object
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Step 5: Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
