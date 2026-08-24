#!/usr/bin/env python3
"""Module that compiles and trains a CNN model with a configurable
optimizer.
"""
from tensorflow import keras


def _get_optimizer(optimizer_name, optimizer_params):
    """Build a Keras optimizer instance from a name and its parameters.

    Args:
        optimizer_name (str): name of the optimizer ('adam', 'sgd',
            'rmsprop', 'adagrad', 'adadelta' or 'nadam').
        optimizer_params (dict): keyword arguments passed to the
            optimizer's constructor (e.g. learning_rate).

    Returns:
        keras.optimizers.Optimizer: the instantiated optimizer.
    """
    optimizers_map = {
        'adam': keras.optimizers.Adam,
        'sgd': keras.optimizers.SGD,
        'rmsprop': keras.optimizers.RMSprop,
        'adagrad': keras.optimizers.Adagrad,
        'adadelta': keras.optimizers.Adadelta,
        'nadam': keras.optimizers.Nadam,
    }

    name = optimizer_name.lower()
    if name not in optimizers_map:
        raise ValueError(
            "optimizer_name must be one of: " +
            ", ".join(optimizers_map.keys())
        )

    return optimizers_map[name](**optimizer_params)


def compile_and_train_cnn(
        model, epochs, batch_size, x_train, y_train, x_val=None,
        y_val=None, optimizer_name='adam', optimizer_params=None):
    """Compile a CNN model with the given optimizer and train it.

    Args:
        model (keras.Model): the CNN model to compile and train.
        epochs (int): number of training epochs.
        batch_size (int): size of the batches used during training.
        x_train (np.ndarray): training data.
        y_train (np.ndarray): training labels (one-hot encoded).
        x_val (np.ndarray): validation data. Defaults to None.
        y_val (np.ndarray): validation labels (one-hot encoded).
            Defaults to None.
        optimizer_name (str): name of the optimizer to use. Defaults
            to 'adam'.
        optimizer_params (dict): additional parameters for the
            optimizer (e.g. {'learning_rate': 0.01}). Defaults to None.

    Returns:
        tuple: (model, history) the trained model and its training
            history object.
    """
    if optimizer_params is None:
        optimizer_params = {}

    optimizer = _get_optimizer(optimizer_name, optimizer_params)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    validation_data = None
    if x_val is not None and y_val is not None:
        validation_data = (x_val, y_val)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
    )

    return model, history
