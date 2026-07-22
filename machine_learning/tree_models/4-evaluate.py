#!/usr/bin/env python3
"""
4-evaluate.py

Task 4: Evaluate Classifier Performance

This module defines a function that summarizes how well a
classifier's predictions match the ground truth, broken down
per class.

Key concepts to remember:
- Precision (per class): of all samples predicted as this class,
  what fraction actually were? High false positives -> low precision.
- Recall (per class): of all samples that truly are this class,
  what fraction did the model catch? High false negatives -> low
  recall.
- F1-score: harmonic mean of precision and recall - a single
  balanced score per class.
- Support: how many true samples of that class appear in the data.
- metrics.classification_report() computes all of this automatically
  by comparing true_labels against predicted_labels, and returns it
  as a formatted string when output_dict=False (the default).
"""
from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """
    Generate a classification report summarizing precision, recall,
    and F1-score per class.

    Args:
        true_labels: Ground truth class labels (e.g. y_test).
        predicted_labels: Model-predicted class labels (e.g. the
            output of generate_predictions, applied to the same
            samples as true_labels).
        class_names: List of human-readable names corresponding to
            the label indices, used to label rows in the report.

    Returns:
        str: A formatted classification report, as produced by
        sklearn.metrics.classification_report.
    """
    report = metrics.classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )

    return report
