#!/usr/bin/env python3
"""
6-pruning_path.py

Task 6: (Post-Pruning) Retrieve the Pruning Path of a Decision Tree

This module defines a function that computes the cost-complexity
pruning path for a decision tree - the sequence of ccp_alpha values
at which the tree's optimal pruning changes, along with the total
leaf impurity at each of those points.

Key concepts to remember:
- Post-pruning (unlike pre-pruning in Task 5) lets a tree grow
  fully first, then trims branches back afterward.
- ccp_alpha is a "cost per branch" threshold: at alpha=0, no
  pruning happens (full tree). As alpha increases, the algorithm
  becomes less willing to keep branches that don't sufficiently
  reduce impurity, so the tree gets progressively simpler.
- cost_complexity_pruning_path() finds every alpha "breakpoint"
  where the next branch would get pruned, rather than making you
  guess a single alpha value manually.
- As ccp_alpha increases, impurities also increase - simpler trees
  merge previously-pure leaves back together, so total leaf
  impurity goes up. This is the classic complexity-vs-fit tradeoff.
- This function trains clf as a side effect of computing the path,
  but its real purpose is to hand back candidate alpha values for
  choosing a good pruned tree later (not to produce a usable
  trained model itself).
"""


def get_pruning_path(clf, X, y):
    """
    Compute the cost-complexity pruning path for a decision tree.

    Args:
        clf: A DecisionTreeClassifier instance (untrained is fine;
            it will be fit internally as part of computing the
            path).
        X: Input features (training data).
        y: Target labels corresponding to X.

    Returns:
        tuple:
            ccp_alphas (numpy.ndarray): Effective alpha values,
                increasing, at which the optimally pruned tree
                changes.
            impurities (numpy.ndarray): Total leaf impurity of the
                tree pruned at each corresponding ccp_alpha.
    """
    path = clf.cost_complexity_pruning_path(X, y)

    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    return ccp_alphas, impurities
