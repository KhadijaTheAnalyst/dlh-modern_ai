#!/usr/bin/env python3
"""
11-boosting.py

Task 11: Boosting

This module defines a function that builds (but does not train) a
boosting classifier, selected by name, from among four common
boosting algorithms.

Key concepts to remember:
- Unlike Random Forest (bagging), which trains many trees
  independently and averages their votes, boosting trains trees
  SEQUENTIALLY: each new tree is trained specifically to correct
  the mistakes of the trees built before it, and their outputs are
  combined into a single weighted prediction.
- Because each tree depends on the previous ones, boosting can't be
  parallelized the way bagging can - often making it slower to
  train, in exchange for typically higher accuracy.
- AdaBoost: reweights misclassified samples so later trees focus
  on them.
- GradientBoosting: each new tree fits the residual errors (loss
  gradient) of the current ensemble, rather than reweighted samples.
- XGBoost / LightGBM: optimized, regularized gradient boosting
  implementations built for speed and performance at scale.
- As with Task 0 and Task 9, this function only BUILDS the model;
  it isn't trained until train_tree() (Task 1) is called on it.
"""
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """
    Build an (untrained) boosting classifier of the requested type.

    Args:
        name (str): Which boosting algorithm to build. One of:
            'adaboost', 'gradientboosting', 'xgboost', 'lightgbm'.
        n_estimators (int): Number of boosting iterations (trees).
        random_state (int): Seed for reproducibility.

    Returns:
        An untrained boosting classifier instance matching `name`.

    Raises:
        ValueError: If `name` is not one of the supported model
            names.
    """
    if name == "adaboost":
        model = ensemble.AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
    elif name == "gradientboosting":
        model = ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
    elif name == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
    elif name == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown model name '{name}'")

    return model
