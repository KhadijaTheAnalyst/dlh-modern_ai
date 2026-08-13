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

    This function creates a tuner based on the specified type,
    configured with the given hyperparameters and optimization
    settings.

    Args:
        tuner_type (str): Type of tuner to create.
                         Options: 'Hyperband', 'RandomSearch',
                         'BayesianOptimization'
        build_model (function): Function that builds and compiles
                                a Keras model.
                                Takes hp (HyperParameters) as
                                argument.
        seed (int): Random seed for reproducibility.
        hyperband_iterations (int): Number of iterations for
                                    Hyperband tuner.
                                    Typical range: [1, 10]
        max_trials (int): Maximum number of trials for RandomSearch
                         and BayesianOptimization.
                         Typical range: [5, 100]
        objective (str): Metric to optimize.
                        Common options: 'val_accuracy', 'val_loss'
        overwrite (bool): Whether to overwrite previous tuning
                         project. Default: True

    Returns:
        tuner (keras_tuner.Tuner): A configured Keras Tuner
                                   instance ready for hyperparameter
                                   search. Type depends on
                                   tuner_type argument.

    Raises:
        ValueError: If tuner_type is not one of the valid options.

    Tuner Types Explained:
    ----------------------
    `Hyperband`:
        - Algorithm: Hyperband (fast, eliminates bad configs early)
        - Speed: Very fast ⭐⭐⭐⭐⭐
        - Quality: Good (eliminates poor configurations)
        - Best for: Quick tuning, limited compute resources
        - Key parameter: hyperband_iterations

    `RandomSearch`:
        - Algorithm: Random sampling from search space
        - Speed: Medium ⭐⭐⭐
        - Quality: Good (but random, less efficient)
        - Best for: Baseline comparisons, simple problems
        - Key parameter: max_trials

    `BayesianOptimization`:
        - Algorithm: Bayesian optimization (smart sampling)
        - Speed: Slow ⭐⭐
        - Quality: Excellent (learns from previous trials)
        - Best for: Complex problems, maximize accuracy
        - Key parameter: max_trials
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
            iterations=hyperband_iterations,
            overwrite=overwrite,
            directory='tuner_results',
            project_name='hyperband'
        )

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
        # BayesianOptimization tuner: Smart, probabilistic search
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
