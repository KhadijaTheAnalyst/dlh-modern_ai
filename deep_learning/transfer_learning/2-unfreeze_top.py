#!/usr/bin/env python3
"""2-unfreeze_top.py"""


def unfreeze_top_layers(model, n_layers):
    """unfreeze_top_layers"""
    base_model = model
    for layer in model.layers:
        if hasattr(layer, "layers"):
            base_model = layer
            break

    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    for layer in base_model.layers[-n_layers:]:
        layer.trainable = True

    return None
