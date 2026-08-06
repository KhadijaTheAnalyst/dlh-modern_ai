#!/usr/bin/env python3
"""Log a Keras model's training metrics to TensorBoard."""
from tensorflow import keras
import datetime


def log_to_tensorboard(log_dir, model, X, Y, epochs, verbose=1):
    """
    Trains a Keras model while logging metrics, weight histograms,
    and activation histograms to TensorBoard.

    Arguments:
        log_dir: Base directory where logs should be saved.
        model: Keras model.
        X: Input data, shape (number of examples, input features).
        Y: labels, shape (number of examples, 1).
        epochs: Number of training epochs.
        verbose: Verbosity mode (0 = silent, 1 = progress bar).

    Returns:
        None
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    full_log_dir = log_dir + "/" + timestamp

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=full_log_dir,
        histogram_freq=1
    )

    model.fit(X, Y, epochs=epochs, verbose=verbose,
              callbacks=[tensorboard_callback])

    return None
