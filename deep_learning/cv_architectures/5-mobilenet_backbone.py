#!/usr/bin/env python3
"""Module that builds the MobileNetV1 feature extraction backbone."""
from tensorflow import keras
depthwise_separable_conv = __import__(
    '4-depthwise_separable_conv').depthwise_separable_conv


def mobilenet_backbone(inputs):
    """Build the MobileNetV1 feature extraction backbone.

    Follows the original MobileNetV1 architecture: a standard 3x3
    convolution with stride 2, followed by 13 depthwise separable
    convolution blocks. Spatial downsampling (stride 2) happens at
    the start of each new "stage" (when the channel count increases),
    while every other block uses stride 1. A 224x224x3 input is
    reduced to a 7x7x1024 feature map.

    Args:
        inputs (tf.Tensor): input tensor to the network.

    Returns:
        tf.Tensor: the output tensor of the MobileNet backbone
            (before global average pooling and classification).
    """
    x = keras.layers.Conv2D(
        32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # (filters, stride) for each of the 13 depthwise separable blocks
    stage_config = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    for filters, stride in stage_config:
        x = depthwise_separable_conv(x, filters=filters, stride=stride)

    return x
