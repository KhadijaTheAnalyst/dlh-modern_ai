#!/usr/bin/env python3
"""
Task 3: SGD with Learning Rate Schedules

This module implements the get_optimizer_SGD_with_schedule function which
configures an SGD optimizer with momentum and a learning rate schedule.
"""

from tensorflow import keras


def get_optimizer_SGD_with_schedule(schedule_type, initial_lr,
                                    decay_steps, decay_rate, momentum):
    """
    Configure and return an SGD optimizer with momentum and
    a learning rate schedule.

    Args:
        schedule_type (str): The learning rate schedule type.
                            Options: 'exponential' or 'inverse_time'
        initial_lr (float): The initial learning rate.
        decay_steps (int): Number of steps before applying decay.
        decay_rate (float): The decay rate factor.
        momentum (float): The momentum factor for SGD.

    Returns:
        tuple: A tuple containing:
            - optimizer (keras.optimizers.SGD):
              SGD optimizer with the learning rate schedule
            - lr_schedule: The learning rate schedule object
    """

    # Create the appropriate learning rate schedule
    if schedule_type == 'exponential':
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    elif schedule_type == 'inverse_time':
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    else:
        raise ValueError(
            "schedule_type must be 'exponential' or 'inverse_time'"
        )

    # Create SGD optimizer with the learning rate schedule and momentum
    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=momentum
    )

    return optimizer, lr_schedule
