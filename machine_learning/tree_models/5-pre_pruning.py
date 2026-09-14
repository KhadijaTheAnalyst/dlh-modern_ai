#!/usr/bin/env python3
"""
5-pre_pruning.py

Task 5: Pre-Pruning

This module defines a function that performs a Grid Search to find
the best pre-pruning hyperparameters for a decision tree classifier.

Key concepts to remember:
- Pre-pruning restricts how large a tree is allowed to grow WHILE
  it's being built (via max_depth, min_samples_leaf, etc.), as
  opposed to post-pruning, which grows a full tree and cuts it
  back afterward.
- Grid Search (GridSearchCV) automates hyperparameter tuning: it
  trains and cross-validates a model for every combination of the
  parameter values you provide, then reports which combination
  scored best - so you don't have to guess values by hand like in
  Task 0.
- GridSearchCV.fit(X, y) does all of this internally; the object's
  .best_params_ attribute holds the winning combination afterward.
"""
from sklearn import model_selection


def prepruning(X, y, clf):
    """
    Perform a grid search over pre-pruning hyperparameters for a
    decision tree classifier.

    Args:
        X: Input features (training data).
        y: Target labels corresponding to X.
        clf: An untrained DecisionTreeClassifier instance, used as
            the base estimator for the grid search.

    Returns:
        dict: The best combination of hyperparameters found
        (criterion, max_depth, min_samples_leaf, min_samples_split).
    """
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(2, 5)),
        'min_samples_leaf': list(range(2, 5)),
        'min_samples_split': list(range(2, 5)),
    }

    grid_search = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid
    )

    grid_search.fit(X, y)

    return grid_search.best_params_
