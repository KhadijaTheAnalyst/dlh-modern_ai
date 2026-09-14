#!/usr/bin/env python3
"""
3-generate_predictions.py

Task 3: Generate Predictions

This module defines a function that uses an already-trained
Scikit-learn classifier to predict class labels for new samples.

Key concept to remember:
- .fit(X, y) (Task 1) taught the tree the decision rules.
- .predict(X) applies those already-learned rules to new samples:
  for each row in X, it walks down the tree (the same structure you
  saw printed in Task 2) until it reaches a leaf, then returns that
  leaf's predicted class.
- Unlike .fit(), .predict() does NOT mutate the classifier - it's
  a pure, read-only operation that returns something new (an array
  of predictions), which is why (unlike train_tree) this function
  DOES have a meaningful return value.
"""


def generate_predictions(clf, X):
    """
    Generate class predictions for a set of samples using a trained
    tree-based classifier.

    Args:
        clf: A trained Scikit-learn classifier instance (already
            fit on training data, e.g. via train_tree).
        X: Feature matrix (NumPy array or pandas DataFrame) of
            samples to predict labels for. These can be samples
            the model has never seen before (e.g. a test set).

    Returns:
        numpy.ndarray: Predicted class labels, one per row of X.
    """
    # clf.predict walks each sample down the tree's learned
    # if/else structure and returns the class label found at
    # whichever leaf it lands on.
    predictions = clf.predict(X)

    return predictions
