#!/usr/bin/env python3
"""
1-train.py

Task 1: Train a Tree-Based Classifier

This module defines a function that trains an already-built
Scikit-learn classifier (like the one from Task 0) on a given
dataset.

Key concept to remember:
- Building a model (Task 0) and training a model (this task) are
  two separate steps in Scikit-learn.
- Calling .fit(X, y) is what actually grows the tree: it looks at
  the data, tries different splits, and picks the ones that reduce
  Gini impurity the most, until a stopping condition is reached.
- .fit() modifies the classifier IN PLACE (it mutates the same
  object you pass in). It also happens to return `self` by
  convention (so you can chain calls like clf.fit(X, y).predict(...)),
  but this function deliberately ignores that return value and
  returns None instead, since the caller already holds a reference
  to the same, now-trained, clf object.
"""


def train_tree(clf, X, y):
    """
    Train a Scikit-learn tree-based classifier on the given data.

    Args:
        clf: A Scikit-learn classifier instance (e.g. an untrained
            DecisionTreeClassifier from build_decision_tree).
        X: Input features (training data).
        y: Target labels corresponding to X.

    Returns:
        None. The classifier `clf` is trained in place; no new
        object is created or returned.
    """
    # .fit() mutates clf directly, building out the tree structure.
    # We intentionally don't return its result (which is just `self`).
    clf.fit(X, y)

    return None
