import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, losses, metrics
from tensorflow.keras.layers import (
    Rescaling, Conv2D, Lambda, MaxPooling2D, Flatten,
    Dropout, Dense, GlobalAvgPool2D, Activation
)

from layers import *


def AlexNet(model=None, optimizer=optimizers.Nadam):
    if model is None:
        model = Sequential([Rescaling(1. / 255)])

    for layer in [
        Conv2D(48, 7, padding='valid', strides=4, activation='relu'),
        Lambda(tf.nn.local_response_normalization),
        MaxPooling2D(pool_size=3, strides=2, padding='valid'),
        Conv2D(116, 5, padding='same', activation='relu'),
        Lambda(tf.nn.local_response_normalization),
        MaxPooling2D(pool_size=3, strides=2, padding='valid'),
        Conv2D(184, 3, padding='same', activation='relu'),
        Conv2D(184, 3, padding='same', activation='relu'),
        Conv2D(116, 3, padding='same', activation='relu'),
        Flatten(),
        Dense(124, activation='relu'),
        Dropout(0.5),
        Dense(66, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='softmax'),
    ]:
        model.add(layer)

    model.compile(
        optimizer=optimizer(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    return model


def ResNet(model=None, optimizer=optimizers.Nadam):
    if model is None:
        model = Sequential([Rescaling(1. / 255)])

    for layer in [
        Conv2D(32, 7, strides=2, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPool2D(pool_size=3, strides=2, padding="same")
    ]:
        model.add(layer)

    prev_filters = 32
    for filters in [32] * 3 + [64] * 4 + [128] * 6 + [256] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Dense(6, activation="softmax"))

    model.compile(
        optimizer=optimizer(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    return model
    

def GoogLeNet(model=None, optimizer=optimizers.Nadam):
    if model is None:
        model = Sequential([Rescaling(1. / 255)])

    for layer in [
        Conv2D(64, 7, strides=2, padding="same", activation="relu"),
        MaxPool2D(pool_size=3, strides=2, padding="same"),
        Lambda(tf.nn.local_response_normalization),
        Conv2D(64, 1, strides=1, padding="same", activation="relu"),
        Conv2D(192, 3, strides=1, padding="same", activation="relu"),
        Lambda(tf.nn.local_response_normalization),
        MaxPool2D(pool_size=3, strides=2, padding="same"),
        InceptionModule([96, 16, 64, 128, 32, 32]),
        InceptionModule([128, 32, 128, 192, 96, 64]),
        MaxPool2D(pool_size=3, strides=2, padding="same"),
        InceptionModule([96, 16, 192, 208, 48, 64]),
        InceptionModule([112, 24, 160, 224, 46, 64]),
        InceptionModule([128, 24, 128, 256, 64, 64]),
        InceptionModule([144, 32, 112, 288, 64, 64]),
        InceptionModule([160, 32, 256, 320, 128, 128]),
        MaxPool2D(pool_size=3, strides=2, padding="same"),
        InceptionModule([160, 32, 256, 320, 128, 128]),
        InceptionModule([192, 48, 384, 384, 128, 128]),
        GlobalAvgPool2D(),
        Flatten(),
        Dropout(0.4),
        Dense(6, activation="softmax")
    ]:
        model.add(layer)

    model.compile(
        optimizer=optimizer(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    return model


def Xception(model=None, optimizer=optimizers.Nadam):
    if model is None:
        model = Sequential([Rescaling(1. / 255)])

    for layer in [
        Conv2D(16, 3, strides=2, padding="same"),
        BatchNormalization(),
        ReLU(),
        Conv2D(32, 3, strides=1, padding="same"),
        BatchNormalization(),
        ReLU(),
        XceptionEntryRU(64, start_with_relu=False),
        XceptionEntryRU(128),
        XceptionEntryRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionEntryRU(256),
        Conv2D(512, 3, strides=1, padding="same"),
        BatchNormalization(),
        ReLU(),
        Conv2D(1024, 3, strides=1, padding="same"),
        BatchNormalization(),
        ReLU(),
        GlobalAvgPool2D(),
        Dropout(0.5),
        Dense(6, activation="softmax")
    ]:
        model.add(layer)

    model.compile(
        optimizer=optimizer(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    return model


def SEResNet(model=None, optimizer=optimizers.Nadam):
    if model is None:
        model = Sequential([Rescaling(1. / 255)])

    for layer in [
        Conv2D(32, 7, strides=2, padding="same"),
        BatchNormalization(),
        ReLU(),
        MaxPool2D(pool_size=3, strides=2, padding="same")
    ]:
        model.add(layer)

    prev_filters = 32
    for filters in [32] * 3 + [64] * 4 + [128] * 6 + [256] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(SEResidualUnit(filters, strides=strides))
        prev_filters = filters

    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(6, activation="softmax"))

    model.compile(
        optimizer=optimizer(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    return model


def SqueezedXception(model=None, optimizer=optimizers.Nadam):
    if model is None:
        model = Sequential([Rescaling(1. / 255)])

    for layer in [
        Conv2D(16, 3, strides=2, padding="same"),
        ReLU(),
        BatchNormalization(),
        Conv2D(32, 3, strides=1, padding="same"),
        ReLU(),
        BatchNormalization(),
        SqueezedXceptionEntryRU(64, start_with_relu=False),
        SqueezedXceptionEntryRU(128),
        SqueezedXceptionEntryRU(256),
        SqueezedXceptionMiddleRU(256),
        SqueezedXceptionMiddleRU(256),
        SqueezedXceptionMiddleRU(256),
        SqueezedXceptionMiddleRU(256),
        SqueezedXceptionMiddleRU(256),
        SqueezedXceptionMiddleRU(256),
        SqueezedXceptionMiddleRU(256),
        SqueezedXceptionMiddleRU(256),
        SqueezedXceptionEntryRU(256),
        Conv2D(512, 3, strides=1, padding="same"),
        BatchNormalization(),
        ReLU(),
        Conv2D(1024, 3, strides=1, padding="same"),
        BatchNormalization(),
        ReLU(),
        GlobalAvgPool2D(),
        Dropout(0.5),
        Dense(6, activation="softmax")
    ]:
        model.add(layer)

    model.compile(
        optimizer=optimizer(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    return model
