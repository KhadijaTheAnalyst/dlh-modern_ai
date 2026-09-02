#!/usr/bin/env python3
"""Unfreezes the top layers of a pretrained backbone for fine-tuning."""
from tensorflow import keras


def unfreeze_top_layers(model, n_layers):
    """
    Unfreezes the last n_layers of a base model, leaving earlier
    layers frozen. Works whether `model` is the base model itself,
    or a wrapper model that contains the base model as one of its
    layers.

    Args:
        model (keras.Model): the base model, or a model containing
            the base model as one of its layers.
        n_layers (int): number of layers (counting from the end) to
            unfreeze in the base model.

    Returns:
        None
    """
    base_model = model
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break

    base_model.trainable = True

    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    return None
