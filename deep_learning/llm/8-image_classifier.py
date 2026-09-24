#!/usr/bin/env python3
"""
High-level image classification interface using a Hugging Face pipeline.
"""
import transformers


def image_classifier(model):
    """
    Creates a high-level interface to perform image classification using
    a pre-trained model adapted for computer-vision applications.

    Args:
        model (str): Name of the pre-trained model to use.

    Returns:
        classifier: A Hugging Face pipeline object.
    """
    classifier = transformers.pipeline("image-classification", model=model)

    return classifier
