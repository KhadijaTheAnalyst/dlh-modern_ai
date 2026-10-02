#!/usr/bin/env python3
"""Compute evaluation metrics for a classification model."""
import numpy as np
import sklearn.metrics


def compute_metrics(predictions):
    """
    Compute accuracy, precision, recall and F1-score.

    Args:
        predictions: EvalPrediction object with model logits in
            `predictions` and true labels in `label_ids`

    Returns:
        dict: accuracy, weighted precision, weighted recall
            and weighted F1-score
    """
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=-1)

    accuracy = sklearn.metrics.accuracy_score(labels, preds)
    precision = sklearn.metrics.precision_score(
        labels, preds, average="weighted", zero_division=0
    )
    recall = sklearn.metrics.recall_score(
        labels, preds, average="weighted", zero_division=0
    )
    f1 = sklearn.metrics.f1_score(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
