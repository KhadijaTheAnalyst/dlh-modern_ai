#!/usr/bin/env python3
"""
Task 10: Search and Return Best Model
"""

import keras_tuner as kt


def search_and_return_best_model(tuner, x_train, y_train, epochs,
                                 validation_split, verbose=0):
    """
    Perform hyperparameter tuning and return best hyperparameters.

    Args:
        tuner: A Keras Tuner object (Hyperband, RandomSearch, or
               BayesianOptimization).
        x_train: Training input data.
        y_train: Training target data.
        epochs: Number of training epochs for each trial.
        validation_split: Fraction of training data for validation.
        verbose: Verbosity mode (0=silent, 1=progress bar).

    Returns:
        best_hyperparameters: HyperParameters object with best config.
    """

    # Perform hyperparameter search
    tuner.search(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=validation_split,
        verbose=verbose
    )

    # Retrieve the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(
        num_trials=1
    )[0]

    return best_hyperparameters
