#!/usr/bin/env python3
"""
Task 3: SGD with Learning Rate Schedules

This module implements the get_optimizer_SGD_with_schedule function which
configures an SGD optimizer with momentum and a learning rate schedule.

The function supports two schedule types:
- exponential: Applies exponential decay to the learning rate
- inverse_time: Applies inverse time decay to the learning rate
"""

from tensorflow import keras


def get_optimizer_SGD_with_schedule(schedule_type, initial_lr,
                                    decay_steps, decay_rate, momentum):
    """
    Configure and return an SGD optimizer with momentum and
    a learning rate schedule.

    This function creates an SGD optimizer with momentum and
    applies a learning rate
    schedule that decays the learning rate over time in a stepwise fashion.

    Args:
        schedule_type (str): The learning rate schedule type.
                            Options: 'exponential' or 'inverse_time'
        initial_lr (float): The initial learning rate.
                           Typical range: [0.01, 0.1]
        decay_steps (int): Number of steps before applying decay.
                          Typical range: [1000, 10000]
        decay_rate (float): The decay rate factor.
                           Typical range: [0.9, 0.99]
        momentum (float): The momentum factor for SGD.
                         Typical range: [0.0, 0.99]

    Returns:
        tuple: A tuple containing:
            - optimizer (keras.optimizers.SGD):
                SGD optimizer with the learning rate schedule
            - lr_schedule (keras.optimizers.schedules.LearningRateSchedule):
              The learning rate schedule object

    Raises:
        ValueError: If schedule_type is not 'exponential' or 'inverse_time'.

    Examples:
        >>> optimizer, lr_schedule = get_optimizer_SGD_with_schedule(
        ...     'exponential', 0.1, 1000, 0.96, 0.9)
        >>> # Returns: (SGD with ExponentialDecay schedule,
                        ExponentialDecay schedule object)

        >>> optimizer, lr_schedule = get_optimizer_SGD_with_schedule(
        ...     'inverse_time', 0.1, 1000, 0.5, 0.9)
        >>> # Returns: (SGD with InverseTimeDecay schedule,
                        InverseTimeDecay schedule object)

    Schedule Descriptions:
    ----------------------
    'exponential':
        - Exponential Decay
        - Learning rate decays exponentially:
            lr = initial_lr * decay_rate^(step/decay_steps)
        - Pros: Smooth decay, works well in practice
        - Cons: Decay may be too rapid initially
        - Formula: decayed_learning_rate =
            initial_lr * decay_rate ^ (global_step / decay_steps)
        - Use when: Want smooth, continuous learning rate decay

    'inverse_time':
        - Inverse Time Decay
        - Learning rate decays inversely:
            lr = initial_lr / (1 + decay_rate * step / decay_steps)
        - Pros: Slower decay, prevents learning rate from becoming too small
        - Cons: May decay too slowly initially
        - Formula: decayed_learning_rate =
            initial_lr / (1 + decay_rate * global_step / decay_steps)
        - Use when: Want gradual decay that doesn't drop too fast

    Why Use Learning Rate Schedules:
    --------------------------------
    - Initial phase: Large learning rate for fast initial progress
    - Middle phase: Moderate learning rate for convergence
    - Final phase: Small learning rate for fine-tuning and stability
    - Benefits: Better final accuracy, faster convergence, more stable training
    """

    # Create the appropriate learning rate schedule
    if schedule_type == 'exponential':
        # Exponential decay schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
    elif schedule_type == 'inverse_time':
        # Inverse time decay schedule
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
    else:
        # Raise error for invalid schedule type
        raise ValueError(
            "schedule_type must be 'exponential' or 'inverse_time'"
        )

    # Create SGD optimizer with the learning rate schedule and momentum
    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=momentum
    )

    return optimizer, lr_schedule
