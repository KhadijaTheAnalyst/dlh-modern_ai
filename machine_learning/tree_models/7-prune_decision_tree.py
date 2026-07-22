#!/usr/bin/env python3
"""
7-prune_decision_tree.py

Task 7: (Post-Pruning) Train and Evaluate Decision Trees with Pruning

This module defines a function that trains one decision tree per
ccp_alpha value (from Task 6's pruning path) and records train/test
accuracy for each, so the effect of pruning strength on
generalization can be compared.

Key concepts to remember:
- Each ccp_alpha corresponds to a differently-pruned tree: alpha=0
  is the full, unpruned tree; larger alpha means more aggressive
  pruning (a simpler tree).
- Training accuracy tends to DECREASE as alpha increases - a
  simpler, more pruned tree fits the training data less exactly.
- Test accuracy typically goes up at first (pruning removes
  overfit noise-fitting branches, improving generalization), peaks
  at some "sweet spot" alpha, then eventually collapses as
  over-pruning removes real signal, not just noise.
- This divergence-then-convergence pattern between train and test
  accuracy is a direct illustration of the bias-variance tradeoff.
- clf.score(X, y) returns accuracy: the fraction of samples whose
  predicted label matches the true label.
"""
from sklearn import tree

train_tree = __import__('1-train').train_tree


def prune_and_evaluate_trees(X_train, y_train, X_test, y_test, ccp_alphas,
                              random_state, min_samples_leaf,
                              min_samples_split):
    """
    Train a decision tree classifier for each given ccp_alpha value
    and record its training and testing accuracy.

    Args:
        X_train, y_train: Training features and labels.
        X_test, y_test: Testing features and labels.
        ccp_alphas: Array of cost-complexity pruning alpha values,
            e.g. from get_pruning_path, to train one tree per value.
        random_state: Seed for reproducibility.
        min_samples_leaf: Minimum samples required at a leaf node.
        min_samples_split: Minimum samples required to split a
            node.

    Returns:
        tuple:
            clfs (list): Trained DecisionTreeClassifier instances,
                one per ccp_alpha value, in the same order.
            train_scores (list): Training accuracy for each tree.
            test_scores (list): Testing accuracy for each tree.
    """
    clfs = []
    train_scores = []
    test_scores = []

    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            ccp_alpha=ccp_alpha
        )

        train_tree(clf, X_train, y_train)

        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
        clfs.append(clf)

    return clfs, train_scores, test_scores
