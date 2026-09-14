#!/usr/bin/env python3
"""
8-best_ccp_alpha.py

Task 8: (Post-Pruning) Best ccp_alpha for Pruning

This module defines a function that selects the best-performing
pruned decision tree (and its corresponding ccp_alpha) from a set
of candidates trained in Task 7, using a three-step tie-breaking
rule.

Key concepts to remember:
- Selection priority, in order:
    1. Highest test accuracy (the main goal).
    2. Smallest |train_score - test_score| gap (favors models that
       generalize well, not just ones that got lucky on the test
       set).
    3. Largest ccp_alpha (Occam's razor: prefer the simpler, more
       regularized tree when performance is otherwise tied).
- Python compares tuples element-by-element, so building a sort key
  tuple in this exact priority order lets a single max() call
  implement all three rules at once, without nested if/else logic.
"""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """
    Select the best ccp_alpha (and corresponding trained classifier)
    from a set of pruned decision trees.

    Args:
        clfs: List of trained DecisionTreeClassifier instances, one
            per ccp_alpha value.
        train_scores: List of training accuracy scores, aligned
            with clfs.
        test_scores: List of testing accuracy scores, aligned with
            clfs.
        ccp_alphas: List/array of ccp_alpha values, aligned with
            clfs.

    Returns:
        tuple:
            best_alpha (float): The chosen ccp_alpha value.
            best_clf: The trained classifier corresponding to
                best_alpha.
    """
    n = len(clfs)

    # Sort key per candidate, in priority order:
    #   1. test_score            -> higher is better
    #   2. -|train - test| gap   -> smaller gap is better, so negate
    #                               it to keep "higher is better"
    #   3. ccp_alpha              -> higher is better (simpler tree)
    def sort_key(i):
        gap = abs(train_scores[i] - test_scores[i])
        return (test_scores[i], -gap, ccp_alphas[i])

    best_index = max(range(n), key=sort_key)

    best_alpha = ccp_alphas[best_index]
    best_clf = clfs[best_index]

    return best_alpha, best_clf
