#!/usr/bin/env python3
"""Attaches a classification head to a pretrained feature extractor."""
from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """
    Attaches a custom classification head on top of a pretrained
    feature extractor model.

    Args:
        base_model (keras.Model): a model whose output is a pooled
            feature vector (e.g., the output of build_feature_extractor).
        num_classes (int): number of output classes for classification.

    Returns:
        keras.Model: a new model that takes the same input as
            base_model and outputs class probabilities.
    """
    inputs = base_model.input
    x = base_model.output
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model
