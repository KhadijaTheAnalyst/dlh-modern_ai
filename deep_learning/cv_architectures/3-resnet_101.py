#!/usr/bin/env python3
"""Module that builds the ResNet-101 architecture."""
from tensorflow import keras
bottleneck_block = __import__('2-bottleneck_block').bottleneck_block


def make_layer(x, blocks, filters, stride=1, name=None):
    """Stack a sequence of bottleneck residual blocks into one stage.

    The first block applies the given stride and a projection
    shortcut (to both downsample spatially, when stride > 1, and to
    match the expanded channel count). Every subsequent block uses a
    stride of 1 and an identity shortcut.

    Args:
        x (tf.Tensor): input tensor.
        blocks (int): number of bottleneck blocks in this stage.
        filters (int): number of filters for the 3x3 convolution in
            each block (the stage's output channels are filters * 4).
        stride (int): stride applied by the stage's first block.
            Defaults to 1.
        name (str): prefix used to name the stage's blocks. Defaults
            to None.

    Returns:
        tf.Tensor: the output tensor of the stage.
    """
    x = bottleneck_block(
        x, filters, stride=stride, downsample=True,
        name=f'{name}_block1')
    for i in range(1, blocks):
        x = bottleneck_block(
            x, filters, stride=1, downsample=False,
            name=f'{name}_block{i + 1}')
    return x


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """Build the ResNet-101 architecture.

    Follows "Deep Residual Learning for Image Recognition" (2015):
    a 7x7 stem convolution and max pooling, followed by four stages
    of bottleneck residual blocks (3, 4, 23 and 3 blocks respectively
    for conv2_x through conv5_x), global average pooling, and a
    fully connected classification layer.

    Args:
        input_shape (tuple): shape of the input image. Defaults to
            (224, 224, 3).
        num_classes (int): number of output classes. Defaults to
            1000.

    Returns:
        keras.Model: the ResNet-101 model.
    """
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(
        64, 7, strides=2, padding='same', use_bias=False,
        name='conv1')(inputs)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.ReLU(name='relu1')(x)
    x = keras.layers.MaxPooling2D(
        3, strides=2, padding='same', name='maxpool')(x)

    x = make_layer(x, blocks=3, filters=64, stride=1, name='conv2')
    x = make_layer(x, blocks=4, filters=128, stride=2, name='conv3')
    x = make_layer(x, blocks=23, filters=256, stride=2, name='conv4')
    x = make_layer(x, blocks=3, filters=512, stride=2, name='conv5')

    x = keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    outputs = keras.layers.Dense(
        num_classes, activation='softmax', name='fc')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='resnet101')

    return model
