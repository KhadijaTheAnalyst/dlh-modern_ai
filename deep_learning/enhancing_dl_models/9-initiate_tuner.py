#!/usr/bin/env python3
"""
Task 9: Initiate the Tuner

This module implements the initiate_tuner function which creates and returns
a configured Keras Tuner for hyperparameter optimization.

Supports three tuner types: Hyperband, RandomSearch, and BayesianOptimization.
"""

import keras_tuner


def initiate_tuner(tuner_type, build_model, seed, hyperband_iterations,
                   max_trials, objective, overwrite=True):
    """
    Initialize and return a Keras Tuner for hyperparameter tuning.

    This function creates a tuner based on the specified type, configured
    with the given hyperparameters and optimization settings.

    Args:
        tuner_type (str): Type of tuner to create.
                         Options: 'Hyperband', 'RandomSearch',
                         'BayesianOptimization'
        build_model (function): Function that builds and compiles
                                a Keras model.
                               Takes hp (HyperParameters) as argument.
        seed (int): Random seed for reproducibility.
        hyperband_iterations (int): Number of iterations for Hyperband tuner.
                                   Typical range: [1, 10]
        max_trials (int): Maximum number of trials for RandomSearch and
                         BayesianOptimization.
                         Typical range: [5, 100]
        objective (str): Metric to optimize.
                        Common options: 'val_accuracy', 'val_loss'
        overwrite (bool): Whether to overwrite previous tuning project.
                         Default: True

    Returns:
        tuner (keras_tuner.Tuner): A configured Keras Tuner instance
                                  ready for hyperparameter search.
                                  Type depends on tuner_type argument.

    Raises:
        ValueError: If tuner_type is not one of the valid options.

    Examples:
        >>> # Create Hyperband tuner
        >>> tuner = initiate_tuner('Hyperband', build_model, seed=0,
        ...                         hyperband_iterations=5, max_trials=5,
        ...                         objective='val_accuracy')

        >>> # Create RandomSearch tuner
        >>> tuner = initiate_tuner('RandomSearch', build_model, seed=0,
        ...                         hyperband_iterations=5, max_trials=10,
        ...                         objective='val_accuracy')

        >>> # Create BayesianOptimization tuner
        >>> tuner = initiate_tuner('BayesianOptimization', build_model,
        ...                         seed=0, hyperband_iterations=5,
        ...                         max_trials=20, objective='val_accuracy')

    Tuner Types Explained:
    ----------------------
    `Hyperband`:
        - Algorithm: Hyperband (fast, eliminates bad configs early)
        - Speed: Very fast ⭐⭐⭐⭐⭐
        - Quality: Good (eliminates poor configurations)
        - Best for: Quick tuning, limited compute resources
        - Key parameter: hyperband_iterations (more = more thorough)
        - How it works:
            * Trains all configurations briefly
            * Eliminates worst performers
            * Trains remaining configurations longer
            * Repeat until best found

    `RandomSearch`:
        - Algorithm: Random sampling from search space
        - Speed: Medium ⭐⭐⭐
        - Quality: Good (but random, less efficient)
        - Best for: Baseline comparisons, simple problems
        - Key parameter: max_trials (more trials = better search)
        - How it works:
            * Randomly sample hyperparameter combinations
            * Train each combination fully
            * Track best found
            * Try max_trials different combinations

    `BayesianOptimization`:
        - Algorithm: Bayesian optimization (smart sampling)
        - Speed: Slow ⭐⭐
        - Quality: Excellent (learns from previous trials)
        - Best for: Complex problems, maximize accuracy
        - Key parameter: max_trials (more = better tuning)
        - How it works:
            * Use probabilistic model of performance
            * Choose next trial likely to be best
            * Update model with results
            * Repeat to find optimal hyperparameters

    Comparison:
    -----------
    ┌──────────────────┬────────┬─────────┬──────────────┐
    │ Aspect           │ Hyperb │ Random  │ Bayesian     │
    ├──────────────────┼────────┼─────────┼──────────────┤
    │ Speed            │ Very F │ Medium  │ Slow         │
    │ Quality          │ Good   │ Good    │ Excellent    │
    │ Best For         │ Quick  │ Simple  │ Complex      │
    │ Compute Cost     │ Low    │ Medium  │ High         │
    │ Learning Curve   │ Fast   │ None    │ Adaptive     │
    └──────────────────┴────────┴─────────┴──────────────┘

    When to Use Each:
    ------------------
    Use Hyperband when:
    - You have limited compute resources
    - You need results quickly
    - You want good-enough solutions fast
    - You don't want to wait for full optimization

    Use RandomSearch when:
    - You need a baseline to compare against
    - The problem is relatively simple
    - You want easy-to-understand results
    - You don't have prior knowledge

    Use BayesianOptimization when:
    - You want the best possible model
    - Compute resources are available
    - You have a complex hyperparameter space
    - Training time per trial is acceptable

    Common Configuration:
    ---------------------
    Quick tuning: Hyperband with hyperband_iterations=5
    Standard tuning: RandomSearch or Hyperband with max_trials=20
    Thorough tuning: BayesianOptimization with max_trials=50-100
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
