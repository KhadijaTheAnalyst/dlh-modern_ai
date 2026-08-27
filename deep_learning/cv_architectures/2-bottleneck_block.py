#!/usr/bin/env python3
"""Module that implements a ResNet bottleneck residual block."""
from tensorflow import keras


def _layer_name(name, suffix):
    """Build a layer name from a block prefix and a suffix.

    Args:
        name (str): the block's name prefix, or None.
        suffix (str): the suffix identifying the specific layer.

    Returns:
        str or None: '{name}_{suffix}' if name is given, otherwise
            None so Keras auto-generates a unique name.
    """
    return f'{name}_{suffix}' if name else None


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """Build a ResNet bottleneck residual block.

    The block reduces channels with a 1x1 convolution, processes the
    reduced representation with a 3x3 convolution, then expands the
    channels back by a factor of 4 with a final 1x1 convolution. Each
    convolution is followed by Batch Normalization, with ReLU
    activations after the first two. The block output is the sum of
    this path and a shortcut connection, followed by a final ReLU.

    Args:
        x (tf.Tensor): input tensor.
        filters (int): number of filters for the 3x3 convolution (the
            1x1 convolutions reduce to this many filters and expand
            back to filters * 4).
        stride (int): stride applied to the first convolution, used
            for spatial downsampling. Defaults to 1.
        downsample (bool): whether to apply a projection shortcut
            (1x1 convolution + Batch Normalization) instead of an
            identity shortcut. Defaults to False.
        name (str): optional prefix used to name the block's layers.
            Defaults to None.

    Returns:
        tf.Tensor: the output tensor of the bottleneck residual block.
    """
    shortcut = x

    y = keras.layers.Conv2D(
        filters, 1, strides=stride, use_bias=False,
        name=_layer_name(name, 'conv1'))(x)
    y = keras.layers.BatchNormalization(
        name=_layer_name(name, 'bn1'))(y)
    y = keras.layers.ReLU(name=_layer_name(name, 'relu1'))(y)

    y = keras.layers.Conv2D(
        filters, 3, padding='same', use_bias=False,
        name=_layer_name(name, 'conv2'))(y)
    y = keras.layers.BatchNormalization(
        name=_layer_name(name, 'bn2'))(y)
    y = keras.layers.ReLU(name=_layer_name(name, 'relu2'))(y)

    y = keras.layers.Conv2D(
        filters * 4, 1, use_bias=False,
        name=_layer_name(name, 'conv3'))(y)
    y = keras.layers.BatchNormalization(
        name=_layer_name(name, 'bn3'))(y)

    if downsample:
        shortcut = keras.layers.Conv2D(
            filters * 4, 1, strides=stride, use_bias=False,
            name=_layer_name(name, 'shortcut_conv'))(x)
        shortcut = keras.layers.BatchNormalization(
            name=_layer_name(name, 'shortcut_bn'))(shortcut)

    out = keras.layers.Add(name=_layer_name(name, 'add'))([y, shortcut])
    out = keras.layers.ReLU(name=_layer_name(name, 'out'))(out)

    return out
