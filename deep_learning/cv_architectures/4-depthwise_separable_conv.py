#!/usr/bin/env python3
"""Module that implements a depthwise separable convolution block,
the core building block of MobileNetV1.
"""
from tensorflow import keras


def depthwise_separable_conv(X, filters, stride=1):
    """Build a depthwise separable convolution block.

    Factorizes a standard convolution into a depthwise convolution
    (one 3x3 filter per input channel, capturing spatial patterns)
    followed by a pointwise convolution (a 1x1 convolution mixing
    channels), each followed by Batch Normalization and a ReLU
    activation. This factorization drastically reduces the number of
    parameters and computations compared to a standard convolution.

    Args:
        X (tf.Tensor): input tensor.
        filters (int): number of output channels for the pointwise
            convolution.
        stride (int): stride applied to the depthwise convolution.
            Defaults to 1.

    Returns:
        tf.Tensor: the output tensor of the depthwise separable
            convolution block.
    """
    x = keras.layers.DepthwiseConv2D(
        kernel_size=3, strides=stride, padding='same',
        use_bias=False)(X)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(
        filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x
