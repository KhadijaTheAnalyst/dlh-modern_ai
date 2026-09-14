#!/usr/bin/env python3
"""
Task 6: Dropout Regularization

This module implements the build_model_with_dropout function which builds
a Keras model with dropout regularization applied after the input layer
and after each hidden layer.

Dropout randomly deactivates neurons during training to prevent co-adaptation
and reduce overfitting, leading to better generalization.
"""

from tensorflow import keras


def build_model_with_dropout(input_dim, hidden_units, n_layers,
                             dropout_rate_input, dropout_rate_hidden):
    """
    Build a Keras model with dropout regularization.

    This function creates a neural network with dropout layers applied after
    the input layer and after each hidden layer to prevent overfitting.

    Args:
        input_dim (int): The number of input features.
                        Example: 784 for MNIST (28x28 flattened)
        hidden_units (int): The number of neurons in each hidden layer.
                           Typical range: [256, 512]
        n_layers (int): The number of hidden layers to include.
                       Typical range: [1, 5]
        dropout_rate_input (float): Dropout rate after the input layer.
                                   Typical range: [0.1, 0.3]
        dropout_rate_hidden (float): Dropout rate after each hidden layer.
                                    Typical range: [0.3, 0.7]

    Returns:
        model (keras.Model): A compiled Keras model with:
            - InputLayer with shape (input_dim,)
            - Dropout layer after input
            - n_layers hidden layers (Dense + ReLU + Dropout)
            - Dense output layer (10 units) with softmax activation

    Raises:
        ValueError: If n_layers < 1, dropout rates not in [0, 1].

    Examples:
        >>> # Model without dropout
        >>> model = build_model_with_dropout(784, 512, 2, 0, 0)

        >>> # Model with dropout
        >>> model = build_model_with_dropout(784, 512, 2, 0.2, 0.5)

    What is Dropout:
    ----------------
    Dropout randomly deactivates neurons during training with a specified rate:

    During training:
    - Each neuron has a probability (dropout_rate) of being "dropped"
    - Dropped neurons don't participate in forward/backward pass
    - Remaining neurons are scaled by 1/(1-dropout_rate)

    During inference (testing):
    - All neurons are active (no dropout)
    - Predictions use the full network

    Why Dropout Works:
    ------------------
    1. Prevents co-adaptation: Neurons can't rely on specific other neurons
    2. Creates implicit ensemble: Each training step uses a different network
    3. Reduces overfitting: Forces network to learn robust features
    4. Acts as regularization: Similar effect to L2 but different mechanism

    Dropout Rates:
    ---------------
    Input layer: 0.1-0.3 (lower, since input is important)
    Hidden layers: 0.3-0.7 (higher, can afford to drop more)
    Output layer: 0 (never use dropout on output)

    Higher dropout_rate = Stronger regularization = Simpler model
    Lower dropout_rate = Weaker regularization = Complex model

    Difference from L2:
    ------------------
    L2 Regularization:
    - Penalizes weight magnitude directly
    - Smooth effect: all weights reduced slightly
    - Computationally efficient

    Dropout:
    - Removes neurons probabilistically
    - Sharper effect: neurons either active or inactive
    - Creates ensembles implicitly
    - Can be combined with L2 for best results

    Architecture:
    ---------------
    Input → Dropout(rate_input) → Dense(ReLU) → Dropout(rate_hidden)
    → ... repeat n_layers times ... → Dense(Softmax)
    """

    # Input validation
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if not (0 <= dropout_rate_input <= 1):
        raise ValueError("dropout_rate_input must be between 0 and 1")
    if not (0 <= dropout_rate_hidden <= 1):
        raise ValueError("dropout_rate_hidden must be between 0 and 1")

    # Step 1: Build model using Functional API
    # Create input layer
    inputs = keras.Input(shape=(input_dim,))

    # Step 2: Add dropout after input layer
    x = keras.layers.Dropout(rate=dropout_rate_input)(inputs)

    # Step 3: Add hidden layers with dropout after each
    for i in range(n_layers):
        # Dense layer with ReLU activation
        x = keras.layers.Dense(
            units=hidden_units,
            activation='relu',
            name=f'hidden_layer_{i+1}'
        )(x)

        # Dropout layer after hidden layer
        x = keras.layers.Dropout(
            rate=dropout_rate_hidden,
            name=f'dropout_hidden_{i+1}'
        )(x)

    # Step 4: Add output layer (no dropout on output)
    outputs = keras.layers.Dense(
        units=10,
        activation='softmax',
        name='output_layer'
    )(x)

    # Step 5: Bundle layers into a Model object
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Step 6: Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
