#!/usr/bin/env python3
"""
Task 7: Early Stopping

This module implements the get_early_stopping_callback function which creates
a configured Early Stopping callback for Keras training.

Early stopping prevents overfitting by monitoring a metric
and stopping training.
when the metric stops improving, then restoring the best model weights.
"""

from tensorflow import keras


def get_early_stopping_callback(patience, monitor='val_loss', verbose=1):
    """
    Create a customizable Early Stopping callback for Keras training.

    This callback monitors a specified metric during training and stops
    training if no improvement is seen after a defined number of epochs.
    The best model weights are automatically restored.

    Args:
        patience (int): Number of epochs to wait without improvement before
                       stopping training.
                       Typical range: [3, 10]
                       - Lower values: Stop sooner, risk underfitting
                       - Higher values: Train longer, risk overfitting
        monitor (str): Metric to monitor during training.
                      Common options: 'val_loss', 'val_accuracy'
                      Can also monitor: 'loss', 'accuracy',
                                        'val_precision', etc.
                      Default: 'val_loss' (most common)
        verbose (int): Verbosity mode for displaying callback messages.
                      0: Silent (no messages)
                      1: Print messages (recommended)
                      2: Detailed messages
                      Default: 1

    Returns:
        callback (keras.callbacks.EarlyStopping): A configured Early Stopping
                                                  callback ready to use in
                                                  model.fit()

    Examples:
        >>> # Create callback with default parameters
        >>> callback = get_early_stopping_callback(patience=3)

        >>> # Monitor validation accuracy instead of loss
        >>> callback = get_early_stopping_callback(
        ...     patience=5,
        ...     monitor='val_accuracy',
        ...     verbose=1
        ... )

        >>> # Use in model training
        >>> model.fit(
        ...     x_train, y_train,
        ...     epochs=100,
        ...     callbacks=[callback]
        ... )

    What is Early Stopping:
    ----------------------
    Early stopping is a regularization technique that stops training when
    the model stops improving on a validation metric.

    Training Process:
    1. Monitor a metric (e.g., validation loss) each epoch
    2. Keep track of the best value seen
    3. If no improvement for `patience` epochs, stop training
    4. Restore model weights from the best epoch

    Why Early Stopping Works:
    -------------------------
    Overfitting typically follows this pattern:

    Epoch 1-10:   Validation loss decreases ✓ (good)
    Epoch 10-20:  Validation loss plateaus (no improvement)
    Epoch 20-30:  Validation loss increases ✗ (overfitting!)

    Without early stopping:
    - Keep training through epoch 30
    - End with overfitted model (good train, bad validation)

    With early stopping:
    - Stop at epoch 20 (no improvement for `patience` epochs)
    - End with best model (good train, good validation)

    Key Parameters Explained:
    -------------------------
    `monitor`:
    - 'val_loss': Most common, stops when validation loss stops improving
    - 'val_accuracy': Alternative, stops when validation accuracy plateaus
    - Choose based on your goal (loss = balanced, accuracy = classification)

    `patience`:
    - Number of epochs to wait for improvement
    - patience=3: Stop if no improvement for 3 consecutive epochs
    - patience=5: More tolerant, wait longer for improvement
    - patience=1: Stop immediately on first non-improvement epoch

    `verbose`:
    - 0: Silent (good for scripts)
    - 1: Print when stopping (good for training monitoring)
    - 2: Very detailed output

    Typical Behavior:
    -----------------
    Epoch 1:  val_loss = 0.50 (best so far) ✓ Save weights
    Epoch 2:  val_loss = 0.48 (best so far) ✓ Save weights
    Epoch 3:  val_loss = 0.49 (no improvement) wait 1/3
    Epoch 4:  val_loss = 0.49 (no improvement) wait 2/3
    Epoch 5:  val_loss = 0.48 (IMPROVEMENT!) ✓ Save weights, reset counter
    Epoch 6:  val_loss = 0.50 (no improvement) wait 1/3
    Epoch 7:  val_loss = 0.51 (no improvement) wait 2/3
    Epoch 8:  val_loss = 0.52 (no improvement) wait 3/3 → STOP!
    → Restore weights from Epoch 5 (best: val_loss = 0.48)

    Best Practices:
    ----------------
    1. Always monitor validation metric, not training metric
    2. Set patience based on noise level (noisy → higher patience)
    3. Combine with learning rate scheduling for best results
    4. Use restore_best_weights=True (automatically done)

    Common Mistakes:
    -----------------
    ✗ Monitoring training loss instead of validation loss
    ✗ patience too small (stops too early, underfitting)
    ✗ patience too large (defeats purpose, overfits)
    ✓ Monitor 'val_loss' or 'val_accuracy' with patience=3-5
    """

    # Create and return EarlyStopping callback
    callback = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=verbose,
        restore_best_weights=True
    )

    return callback
