#!/usr/bin/env python3
"""Module that builds a configurable Convolutional Neural Network (CNN)."""
from tensorflow import keras


def create_cnn_model(
        input_shape, filters, kernel_sizes, activations,
        pooling_type='max'):
    """Create and compile a Convolutional Neural Network model.

    Args:
        input_shape (tuple): shape of the input data (excluding batch size),
            e.g. (28, 28, 1).
        filters (list): number of filters for each convolutional layer.
        kernel_sizes (list): kernel size for each convolutional layer.
        activations (list): activation function for each convolutional
            layer.
        pooling_type (str): type of pooling to apply after each
            convolutional layer, either 'max' or 'avg'. Defaults to 'max'.

    Returns:
        keras.Model: a compiled Keras CNN model.
    """
    if pooling_type not in ('max', 'avg'):
        raise ValueError("pooling_type must be 'max' or 'avg'")
    if not (len(filters) == len(kernel_sizes) == len(activations)):
        raise ValueError(
            "filters, kernel_sizes and activations must have the same "
            "length"
        )

    pooling_layer = (
        keras.layers.MaxPooling2D if pooling_type == 'max'
        else keras.layers.AveragePooling2D
    )

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    for n_filters, kernel_size, activation in zip(
            filters, kernel_sizes, activations):
        model.add(keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=kernel_size,
            activation=activation,
            padding='same',
        ))
        model.add(pooling_layer(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
