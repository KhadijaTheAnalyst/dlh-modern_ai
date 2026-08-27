#!/usr/bin/env python3
"""Module that builds the full MobileNetV1 classification model."""
from tensorflow import keras
mobilenet_backbone = __import__(
    '5-mobilenet_backbone').mobilenet_backbone


def mobilenet(input_shape=(224, 224, 3), num_classes=1000):
    """Build the MobileNetV1 architecture.

    Follows "MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications" (2017): the MobileNet backbone (a
    standard convolution followed by 13 depthwise separable
    convolution blocks), reduced to a single feature vector by global
    average pooling, then classified by a fully connected layer with
    a softmax activation.

    Args:
        input_shape (tuple): shape of the input image. Defaults to
            (224, 224, 3).
        num_classes (int): number of output classes. Defaults to
            1000.

    Returns:
        keras.Model: the MobileNetV1 model.
    """
    inputs = keras.Input(shape=input_shape)

    x = mobilenet_backbone(inputs)

    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(
        num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='mobilenet')

    return model
