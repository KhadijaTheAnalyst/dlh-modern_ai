#!/usr/bin/env python3
"""
Task 0: Gradient Descent Variants Optimization

This module implements the train_with_gradient_descent_variant function
which configures and returns an SGD optimizer with the appropriate batch size
based on the selected gradient descent variant.

The function supports three variants:
- batch: Full dataset updates (stable but slow)
- stochastic: Single sample updates (fast but noisy)
- mini_batch: Small batch updates (best practical performance)
"""

from tensorflow.keras.optimizers import SGD


def train_with_gradient_descent_variant(variant, learning_rate,
                                        x_train, batch_size):
    """
    Configure and return a gradient descent optimizer based on the variant.

    This function creates an SGD optimizer with the specified learning rate
    and determines the appropriate batch size based on the selected gradient
    descent variant strategy.

    Args:
        variant (str): The gradient descent variant to use.
                      Options: 'batch', 'stochastic', or 'mini_batch'
        learning_rate (float): The learning rate for the optimizer.
                              Typical range: [0.001, 0.1]
        x_train (np.ndarray): The training dataset (input data).
                             Used to determine full dataset size
                             for 'batch' variant.
        batch_size (int): The batch size to use when variant='mini_batch'.
                         Typical range: 16-256

    Returns:
        tuple: A tuple containing:
            - optimizer (SGD): Configured SGD optimizer with
            specified learning_rate
            - bs (int): The batch size to use for training

    Raises:
        ValueError: If variant is not one of the accepted options.

    Examples:
        >>> optimizer, bs = train_with_gradient_descent_variant(
        ...     'batch', 0.01, x_train, batch_size=32)
        >>> # Returns: (SGD optimizer, len(x_train))

        >>> optimizer, bs = train_with_gradient_descent_variant(
        ...     'mini_batch', 0.001, x_train, batch_size=64)
        >>> # Returns: (SGD optimizer, 64)

    Variants Description:
    --------------------
    'batch':
        - Batch size = full dataset
        - Updates after processing ALL samples
        - Pros: Stable, smooth convergence
        - Cons: Slow, high memory, may get stuck in local minima
        - Use when: Small datasets, need guaranteed convergence

    'stochastic':
        - Batch size = 1 (one sample at a time)
        - Updates after EACH sample
        - Pros: Fast, escapes local minima, low memory
        - Cons: Noisy/oscillatory convergence, high variance
        - Use when: Very large datasets, need quick initial progress

    'mini_batch':
        - Batch size = custom value (typically 32-128)
        - Updates after processing a small batch
        - Pros: Best practical performance, balanced speed/stability
        - Cons: Requires tuning batch size
        - Use when: Default choice for most real-world scenarios (RECOMMENDED)
    """

    # Create SGD optimizer with specified learning rate
    optimizer = SGD(learning_rate=learning_rate)

    # Determine batch size based on selected variant
    if variant == 'batch':
        # Use entire dataset as one batch
        bs = len(x_train)
    elif variant == 'stochastic':
        # Use single sample per update
        bs = 1
    elif variant == 'mini_batch':
        # Use custom batch size provided by user
        bs = batch_size
    else:
        # Raise error for invalid variant
        raise ValueError(
            "variant must be 'batch', 'stochastic', or 'mini_batch'"
        )

    return optimizer, bs
