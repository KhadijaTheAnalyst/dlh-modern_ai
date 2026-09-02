#!/usr/bin/env python3
"""Builds a frozen feature extractor from a pretrained CNN."""
from tensorflow import keras


def build_feature_extractor():
    """
    Loads a pretrained MobileNetV2 model, removes its classification
    head, freezes its weights, and adds a GlobalAveragePooling2D layer
    on top to produce a feature extractor.

    Returns:
        keras.Model: a model that maps input images of shape
            (224, 224, 3) to a feature vector using the frozen
            MobileNetV2 base.
    """
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_top=False
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    outputs = keras.layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs, outputs)
    return model
