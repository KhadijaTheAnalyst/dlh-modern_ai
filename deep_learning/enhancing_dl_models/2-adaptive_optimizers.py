#!/usr/bin/env python3
"""
Task 2: Adaptive Optimizers vs. SGD

This module implements the get_optimizer function which configures
and returns different Keras optimizers based on the specified variant.

The function supports three optimizers:
- sgd: Stochastic Gradient Descent with optional momentum
- adam: Adaptive Moment Estimation (combines momentum and RMSprop)
- rmsprop: Root Mean Square Propagation (adapts learning rate per parameter)
"""

from tensorflow import keras


def get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho):
    """
    Configure and return a Keras optimizer based on the specified variant.

    This function creates and configures different Keras optimizers
    with their respective hyperparameters based on the optimizer name.

    Args:
        name (str): The optimizer variant to use.
                   Options: 'sgd', 'adam', or 'rmsprop'
        learning_rate (float): The learning rate for the optimizer.
                              Typical range: [0.001, 0.1]
        momentum (float): The momentum factor (only used for SGD).
                         Typical range: [0.0, 0.99]
        beta_1 (float): Exponential decay rate for first moment estimate
                        (only for Adam).
                       Typical value: 0.9
        beta_2 (float): Exponential decay rate for second moment estimate
                        (only for Adam).
                       Typical value: 0.999
        rho (float): Decay factor for RMSprop (only used for RMSprop).
                    Typical value: 0.9

    Returns:
        optimizer (keras.optimizers): Configured Keras optimizer instance
                                      (SGD, Adam, or RMSprop)

    Raises:
        ValueError: If name is not one of the accepted optimizer variants.

    Examples:
        >>> optimizer = get_optimizer('sgd', lr=0.01, momentum=0.9,
        ...                           beta_1=0.9, beta_2=0.999, rho=0.9)
        >>> # Returns: SGD(lr=0.01, momentum=0.9)

        >>> optimizer = get_optimizer('adam', lr=0.001, momentum=0.9,
        ...                           beta_1=0.9, beta_2=0.999, rho=0.9)
        >>> # Returns: Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

        >>> optimizer = get_optimizer('rmsprop', lr=0.001, momentum=0.9,
        ...                           beta_1=0.9, beta_2=0.999, rho=0.9)
        >>> # Returns: RMSprop(lr=0.001, rho=0.9)

    Optimizer Descriptions:
    ----------------------
    'sgd':
        - Stochastic Gradient Descent with optional momentum
        - Updates weights with gradient and accumulated velocity
        - Pros: Simple, interpretable, good generalization
        - Cons: Can be slow, oscillatory convergence
        - Uses: learning_rate, momentum
        - Ignores: beta_1, beta_2, rho

    'adam':
        - Adaptive Moment Estimation
        - Combines momentum and adaptive learning rates
        - Maintains first moment (momentum) and second moment (RMSprop)
        - Pros: Fast convergence, works well in practice, minimal tuning
        - Cons: May converge to sharp minima, high memory usage
        - Uses: learning_rate, beta_1, beta_2
        - Ignores: momentum, rho
        - Formula:
            m_t = beta_1 * m_(t-1) + (1 - beta_1) * g_t
            v_t = beta_2 * v_(t-1) + (1 - beta_2) * g_t^2
            theta = theta - lr * m_t / (sqrt(v_t) + epsilon)

    'rmsprop':
        - Root Mean Square Propagation
        - Adapts learning rate based on historical gradient magnitudes
        - Divides learning rate by root mean square of gradients
        - Pros: Handles non-stationary problems, adaptive learning rates
        - Cons: Still requires tuning, more complex than SGD
        - Uses: learning_rate, rho
        - Ignores: momentum, beta_1, beta_2
        - Formula:
            v_t = rho * v_(t-1) + (1 - rho) * g_t^2
            theta = theta - lr * g_t / (sqrt(v_t) + epsilon)
    """

    # Configure optimizer based on selected variant
    if name == 'sgd':
        # Stochastic Gradient Descent with momentum
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum
        )
    elif name == 'adam':
        # Adaptive Moment Estimation
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )
    elif name == 'rmsprop':
        # Root Mean Square Propagation
        optimizer = keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho
        )
    else:
        # Raise error for invalid optimizer name
        raise ValueError(
            "name must be 'sgd', 'adam', or 'rmsprop'"
        )

    return optimizer
