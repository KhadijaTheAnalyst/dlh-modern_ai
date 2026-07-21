#!/usr/bin/env python3
"""
0-build.py

Task 0: Decision Tree Classifier

This module defines a function that builds (but does not train) a
Scikit-learn DecisionTreeClassifier with custom regularization settings.

Key concepts to remember:
- A decision tree splits data by asking yes/no questions about features.
- 'gini' impurity measures how mixed the classes are at a node
  (0 = pure node, higher = more mixed). The tree picks splits that
  reduce this impurity the most.
- max_depth=None means the tree can grow until nodes are pure or
  another stopping condition (like min_samples_leaf/split) kicks in.
  Since depth isn't capped, min_samples_leaf and min_samples_split
  become the main defenses against overfitting.
- random_state makes the tree's behavior reproducible across runs.
"""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """
    Build a DecisionTreeClassifier (untrained) using Gini impurity,
    with no maximum depth restriction.

    Args:
        min_samples_leaf (int): Minimum number of samples required
            to be at a leaf node. Prevents the tree from creating
            leaves based on just one or two noisy samples.
        min_samples_split (int): Minimum number of samples an
            internal node must have before it's allowed to split.
            Prevents splitting on tiny, statistically weak subsets.
        random_state (int): Seed for the random number generator,
            so results are reproducible.

    Returns:
        sklearn.tree.DecisionTreeClassifier: An untrained decision
        tree classifier instance configured with the given params.
    """
    # Always pass hyperparameters as keyword arguments (not positional!)
    # to avoid accidentally mismatching them to the wrong constructor slot.
    model = tree.DecisionTreeClassifier(
        criterion='gini',                    # split quality measure
        max_depth=None,                      # no depth limit
        min_samples_leaf=min_samples_leaf,   # regularization knob #1
        min_samples_split=min_samples_split, # regularization knob #2
        random_state=random_state            # reproducibility
    )

    return model
