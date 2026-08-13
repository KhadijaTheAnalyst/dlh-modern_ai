#!/usr/bin/env python3
"""
Task 8: Build a Model to be Tuned

This module implements the build_model function which creates a Keras model
for multi-class classification with tunable hyperparameters using Keras Tuner.

The function defines a search space for hyperparameters
like the number of layers,
units per layer, activation functions, and learning rate.
"""

from tensorflow import keras


def build_model(hp):
    """
    Build a Keras model for multi-class classification with
    tunable hyperparameters.

    This function creates a Sequential model where the architecture
    and training parameters are defined through Keras Tuner's
    HyperParameters object, allowing for automated hyperparameter search.

    Args:
        hp (keras_tuner.HyperParameters): HyperParameters object that defines
                                    the search space for tunable parameters.

    Returns:
        model (keras.Model): A compiled Keras Sequential model configured with
                            hyperparameters from the hp object.

    Tunable Hyperparameters:
    ------------------------
    `num_layers` (int):
        - Range: 1-2 hidden layers
        - Controls model depth (complexity)
        - More layers = deeper network, higher capacity
        - Usage: hp.Int('num_layers', min_value=1, max_value=2)

    `units` (int):
        - Range: 4-12 neurons per layer (step of 4)
        - Possible values: 4, 8, 12
        - Controls layer width (capacity)
        - More units = wider layer, higher capacity
        - Usage: hp.Int('units', min_value=4, max_value=12, step=4)

    `activation` (str):
        - Choices: 'relu' or 'sigmoid'
        - Affects hidden layer activation functions
        - relu: Fast, prevents vanishing gradients (better for deep networks)
        - sigmoid: Smooth, but can suffer from vanishing gradients
        - Usage: hp.Choice('activation', ['relu', 'sigmoid'])

    `learning_rate` (float):
        - Choices: 1e-2 (0.01) or 1e-3 (0.001)
        - Controls optimizer step size
        - 1e-2: Faster learning, higher overshoot risk
        - 1e-3: Slower learning, more stable
        - Usage: hp.Choice('learning_rate', [1e-2, 1e-3])

    Model Architecture:
    -------------------
    Input Layer:
        - Shape: (784,) for MNIST 28x28 images flattened

    Hidden Layers:
        - Number: num_layers (1 or 2)
        - Each hidden layer:
            * Dense(units, activation)
            * activation: 'relu' or 'sigmoid'
            * units: 4, 8, or 12

    Output Layer:
        - Dense(10, 'softmax')
        - 10 units for 10 classes (MNIST digits 0-9)
        - softmax activation for multi-class classification

    Optimizer:
        - Adam with tunable learning_rate
        - learning_rate: 1e-2 or 1e-3

    Example Model Architectures (different tuning choices):
    -------------------------------------------------------
    Configuration 1:
        - num_layers=1, units=12, activation='relu', lr=1e-2
        - Input(784) → Dense(12, relu) → Dense(10, softmax)

    Configuration 2:
        - num_layers=2, units=8, activation='sigmoid', lr=1e-3
        - Input(784) → Dense(8, sigmoid) → Dense(8, sigmoid)
                                        → Dense(10, softmax)

    Configuration 3:
        - num_layers=2, units=4, activation='relu', lr=1e-2
        - Input(784) → Dense(4, relu) → Dense(4, relu) → Dense(10, softmax)

    Keras Tuner Integration:
    -------------------------
    This function is designed to be used with Keras Tuner for automated
    hyperparameter search:

```python
    from keras_tuner import RandomSearch

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10
    )
    tuner.search(x_train, y_train, validation_data=(x_val, y_val))
```

    Why These Hyperparameters:
    ---------------------------
    `num_layers`: 1-2 layers allows comparison of shallow vs deeper networks
    `units`: 4-12 range keeps network reasonably sized for MNIST
    `activation`: relu vs sigmoid shows impact of different activations
    `learning_rate`: 1e-2 vs 1e-3 shows impact of learning rate scale
    """

    # Define tunable hyperparameters using HyperParameters object

    # Number of hidden layers (1 or 2)
    num_layers = hp.Int(
        'num_layers',
        min_value=1,
        max_value=2
    )

    # Number of units per hidden layer (4, 8, or 12 with step of 4)
    units = hp.Int(
        'units',
        min_value=4,
        max_value=12,
        step=4
    )

    # Activation function for hidden layers
    activation = hp.Choice(
        'activation',
        values=['relu', 'sigmoid']
    )

    # Learning rate for Adam optimizer
    learning_rate = hp.Choice(
        'learning_rate',
        values=[1e-2, 1e-3]
    )

    # Build the model
    model = keras.Sequential()

    # Add input layer
    model.add(keras.layers.Input(shape=(784,)))

    # Add hidden layers based on tunable num_layers
    for i in range(num_layers):
        model.add(
            keras.layers.Dense(
                units=units,
                activation=activation,
                name=f'hidden_layer_{i+1}'
            )
        )

    # Add output layer (10 units for 10 classes)
    model.add(
        keras.layers.Dense(
            units=10,
            activation='softmax',
            name='output_layer'
        )
    )

    # Compile model with tunable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
