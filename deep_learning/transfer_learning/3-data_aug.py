#!/usr/bin/env python3
"""Builds a data augmentation pipeline for training images."""
from tensorflow import keras


def build_data_augmentation():
    """
    Creates a Keras Sequential model containing common image data
    augmentation layers, to be applied to training images before
    they are fed into a pretrained CNN.

    Returns:
        keras.Sequential: a model that applies random horizontal
            flips, rotations, zooms, and contrast changes to its
            input images.
    """
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=42),
        keras.layers.RandomRotation(0.15, seed=42),
        keras.layers.RandomZoom(0.15, seed=42),
        keras.layers.RandomContrast(0.1, seed=42),
    ])

    return data_augmentation
