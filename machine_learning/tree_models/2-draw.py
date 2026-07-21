#!/usr/bin/env python3
"""
2-draw.py

Task 2: View the Decision Rules of a Trained Tree

This module defines a function that prints the human-readable,
nested if/else structure of an already-trained decision tree.

Key concept to remember:
- A trained DecisionTreeClassifier stores its structure internally
  (which feature/threshold each node split on, and what class each
  leaf predicts), but that structure isn't visible to a human until
  you export it somehow.
- sklearn.tree.export_text() walks that internal structure and
  builds a readable text version of it: one indented line per
  node/leaf, showing the split condition or the final class.
- This is a *read-only* operation - it doesn't change the tree,
  it just reports what .fit() already built.
"""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """
    Print the text representation of a trained decision tree's
    decision rules.

    Args:
        clf: A trained sklearn DecisionTreeClassifier instance
            (already fit on data, e.g. via train_tree).
        feature_names: List of names corresponding to the columns
            of X, used to make split conditions human-readable
            (e.g. "proline <= 755.00" instead of "X[12] <= 755.00").
        class_names: List of names corresponding to the target
            classes, used to make leaf predictions human-readable
            (e.g. "class: class_1" instead of "class: 1").

    Returns:
        None. The tree structure is printed directly to stdout.
    """
    # class_names must be a list of strings for export_text;
    # wine.target_names (and similar sklearn datasets) come back
    # as a NumPy array, so convert defensively.
    class_names = [str(name) for name in class_names]

    tree_rules = tree.export_text(
        clf,
        feature_names=list(feature_names),
        class_names=class_names
    )

    print(tree_rules)

    return None
