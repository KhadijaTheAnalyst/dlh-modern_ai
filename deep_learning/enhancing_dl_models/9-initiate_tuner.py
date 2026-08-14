#!/usr/bin/env python3
"""
Task 9: Initiate the Tuner

This module implements the initiate_tuner function which creates and
returns a configured Keras Tuner for hyperparameter optimization.

Supports three tuner types: Hyperband, RandomSearch, and
BayesianOptimization.
"""

import keras_tuner


def initiate_tuner(tuner_type, build_model, seed, hyperband_iterations,
                   max_trials, objective, overwrite=True):
    """
    Initialize and return a Keras Tuner for hyperparameter tuning.
    """

    # Normalize tuner_type
    tuner_type = tuner_type.strip()

    # Create tuner based on type
    if tuner_type == 'Hyperband':
        # Hyperband tuner: Fast, eliminates bad configs early
        tuner = keras_tuner.Hyperband(
            hypermodel=build_model,
            objective=objective,
            seed=seed,
            max_epochs=100,
            factor=3,
            overwrite=overwrite,
            directory='tuner_results',
            project_name='hyperband'
        )
        # Store the hyperband_iterations value on the tuner object
        tuner.hyperband_iterations = hyperband_iterations

    elif tuner_type == 'RandomSearch':
        # RandomSearch tuner: Random sampling from search space
        tuner = keras_tuner.RandomSearch(
            hypermodel=build_model,
            objective=objective,
            seed=seed,
            max_trials=max_trials,
            overwrite=overwrite,
            directory='tuner_results',
            project_name='random_search'
        )

    elif tuner_type == 'BayesianOptimization':
        # BayesianOptimization tuner: Smart search
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            objective=objective,
            seed=seed,
            max_trials=max_trials,
            overwrite=overwrite,
            directory='tuner_results',
            project_name='bayesian_optimization'
        )

    else:
        raise ValueError(
            f"tuner_type must be 'Hyperband', 'RandomSearch', "
            f"or 'BayesianOptimization', got: {tuner_type}"
        )

    return tuner
