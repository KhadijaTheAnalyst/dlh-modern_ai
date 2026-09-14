#!/usr/bin/env python3
"""
10-feature_importance.py

Task 10: Feature Importance with Random Forest

This module defines a function that extracts feature importance
scores from an already-trained Random Forest classifier.

Key concepts to remember:
- During training, every split in every tree picks a feature that
  reduces impurity. A trained RandomForestClassifier automatically
  tracks how much each feature contributed to impurity reduction,
  averaged across all trees in the forest, exposed via the
  .feature_importances_ attribute (values sum to 1).
- This is a major practical advantage of tree-based models: you
  get interpretability (which inputs actually mattered) as a
  side effect of training, unlike many other model types.
- np.argsort() returns the indices that would sort an array in
  ASCENDING order (least important first). Combined with barh's
  bottom-to-top drawing order, this makes the most important
  feature end up visually at the top of the resulting chart.
"""
import numpy as np


def feature_importance(rf):
    """
    Compute feature importance scores from a trained random forest.

    Args:
        rf: A trained sklearn RandomForestClassifier instance
            (already fit on data, e.g. via train_tree).

    Returns:
        tuple:
            importances (numpy.ndarray): Importance score for each
                feature, in original feature order. Values sum to 1.
            indices (numpy.ndarray): Feature indices, sorted so
                that importances[indices] is in ascending order
                (least important feature first).
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    return importances, indices
