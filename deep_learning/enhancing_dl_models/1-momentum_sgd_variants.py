#!/usr/bin/env python3
"""
Task 1: Momentum-Based SGD Variants Optimization

This module implements the get_optimizer_SGD function which configures
and returns an SGD-based optimizer with different momentum strategies.

The function supports three variants:
- SGD: Standard stochastic gradient descent
- SGD+Momentum: SGD with classical momentum
- SGD+Momentum+Nesterov: SGD with momentum and Nesterov acceleration
"""

from tensorflow import keras


def get_optimizer_SGD(name, lr, momentum=0.0, nesterov=False):
    """
    Configure and return an SGD-based optimizer with momentum variants.

    This function creates an SGD optimizer with the specified learning rate
    and momentum settings based on the selected variant.

    Args:
        name (str): The optimizer variant to use.
                   Options: 'SGD', 'SGD+Momentum', or 'SGD+Momentum+Nesterov'
        lr (float): The learning rate for the optimizer.
                   Typical range: [0.001, 0.1]
        momentum (float): The momentum factor (default: 0.0).
                         Typical range: [0.0, 0.99]
                         Higher values = stronger momentum effect
        nesterov (boolean): Whether to apply Nesterov acceleration
        (default: False).
                           Only effective when momentum > 0

    Returns:
        optimizer (keras.optimizers.SGD): Configured SGD optimizer instance
                                          with specified settings

    Raises:
        ValueError: If name is not one of the accepted variants.

    Examples:
        >>> optimizer = get_optimizer_SGD('SGD', lr=0.01)
        >>> # Returns: SGD(lr=0.01, momentum=0.0, nesterov=False)

        >>> optimizer = get_optimizer_SGD('SGD+Momentum', lr=0.01,
                                                    momentum=0.9)
        >>> # Returns: SGD(lr=0.01, momentum=0.9, nesterov=False)

        >>> optimizer = get_optimizer_SGD('SGD+Momentum+Nesterov',
        ...                                lr=0.01, momentum=0.9,
                                                    nesterov=True)
        >>> # Returns: SGD(lr=0.01, momentum=0.9, nesterov=True)

    Variants Description:
    --------------------
    'SGD':
        - Standard Stochastic Gradient Descent
        - No momentum, no acceleration
        - momentum parameter is ignored
        - nesterov parameter is ignored
        - Pros: Simple, baseline optimizer
        - Cons: Slow convergence, oscillatory
        - Use when: Baseline comparisons, simple problems

    'SGD+Momentum':
        - SGD with classical momentum
        - Accumulates velocity from past gradients
        - Typically uses momentum=0.9
        - Formula: velocity = momentum × velocity + gradient
        - Pros: Faster convergence, smooths oscillations
        - Cons: May overshoot optimum
        - Use when: Need faster convergence than vanilla SGD

    'SGD+Momentum+Nesterov':
        - SGD with momentum and Nesterov acceleration
        - "Look-ahead" gradient computation
        - Typically uses momentum=0.9
        - Pros: Fastest of SGD variants, better convergence
        - Cons: Slightly more complex, may overshoot
        - Use when: Want best SGD variant performance (RECOMMENDED)
    """

    # Configure SGD optimizer based on selected variant
    if name == 'SGD':
        # Standard SGD: no momentum, no Nesterov
        optimizer = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.0,
            nesterov=False
        )
    elif name == 'SGD+Momentum':
        # SGD with classical momentum
        optimizer = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=False
        )
    elif name == 'SGD+Momentum+Nesterov':
        # SGD with momentum and Nesterov acceleration
        optimizer = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=True
        )
    else:
        # Raise error for invalid variant
        raise ValueError(
            "name must be 'SGD', 'SGD+Momentum', or 'SGD+Momentum+Nesterov'"
        )

    return optimizer
