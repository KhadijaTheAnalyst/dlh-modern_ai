#!/usr/bin/env python3
"""
9-random_forest.py

Task 9: Random Forest Classifier

This module defines a function that builds (but does not train) a
Scikit-learn RandomForestClassifier - an ensemble of decision trees.

Key concepts to remember:
- A single decision tree, even pruned, is one rigid set of rules
  learned from one dataset - it can still overfit or be sensitive
  to the specific data it saw.
- A Random Forest builds MANY trees and combines their predictions
  (via majority vote for classification), which tends to
  generalize better than any single tree.
- Two sources of randomness keep the trees diverse, so their
  errors don't all line up the same way:
    1. Bagging: each tree trains on a random bootstrap sample
       (drawn with replacement) of the training rows.
    2. Random feature subsets: each split only considers a random
       subset of features (default max_features='sqrt'), not all
       of them.
- n_estimators controls how many trees are in the forest - more
  trees generally means more stable predictions, at the cost of
  more compute.
- Like Task 0, this function only BUILDS the model; it isn't
  trained until train_tree() (Task 1) is called on it.
"""
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """
    Build an (untrained) Random Forest classifier.

    Args:
        n_estimators (int): Number of trees in the forest.
        random_state (int): Seed for reproducibility - controls
            both the bootstrap sampling and the random feature
            selection at each split.

    Returns:
        sklearn.ensemble.RandomForestClassifier: An untrained
        random forest classifier instance.
    """
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    return model
