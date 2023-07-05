from functools import partial

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import (
    Dense, Conv2D, GlobalAvgPool2D, MaxPool2D, Layer,
    BatchNormalization, ReLU, Reshape, SeparableConv2D
)


class ResidualUnit(Layer):
    
    def __init__(self, filters, strides, activation="relu", **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)

        self.filters = filters
        self.strides = strides
        self.activation = activation

        self.activation = activations.get(activation)
        
        self.main_layers = [
            Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            BatchNormalization(),
        ]

        self.skip_layers = []

        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                BatchNormalization(),
            ]

    def call(self, inputs):
        Y = inputs
        for layer in self.main_layers:
            Y = layer(Y)

        skip_Y = inputs
        for layer in self.skip_layers:
            skip_Y = layer(skip_Y)

        return self.activation(Y + skip_Y)

    def get_config(self):
        config = super(ResidualUnit, self).get_config()
        config.update({
            "filters": self.filters,
            "strides": self.strides,
            "activation": self.activation
        })
        return config


class InceptionModule(Layer):

    def __init__(self, filters_list, **kwargs):
        super(InceptionModule, self).__init__(**kwargs)

        self.filters_list = filters_list

        conv_2d = partial(
            Conv2D,
            strides=2,
            padding="same",
            activation="relu"
        )
        
        self.conv1ml = conv_2d(filters_list[0], 1)
        self.conv1mr = conv_2d(filters_list[1], 1)
        self.pool1r = MaxPool2D(pool_size=3, strides=2, padding="same")

        self.conv2l = conv_2d(filters_list[2], 1)
        self.conv2ml = conv_2d(filters_list[3], 3)
        self.conv2mr = conv_2d(filters_list[4], 5)
        self.conv2r = conv_2d(filters_list[5], 1)

    def call(self, inputs):
        return tf.concat([
            self.conv2l(inputs),
            self.conv2ml(self.conv1ml(inputs)),
            self.conv2mr(self.conv1mr(inputs)),
            self.conv2r(self.pool1r(inputs)),
        ], axis=3)

    def get_config(self):
        config = super(InceptionModule, self).get_config()
        config.update({
            "filters_list": self.filters_list
        })


class NaiveInceptionModule(Layer):

    def __init__(self, filters_list, **kwargs):
        super(NaiveInceptionModule, self).__init__(**kwargs)

        self.filters_list = filters_list

        conv_2d = partial(
            Conv2D,
            strides=1, padding="same",
            activation="relu",
            use_bias=False
        )
        
        self.conv1 = conv_2d(filters_list[0], 1)
        self.conv3 = conv_2d(filters_list[1], 3)
        self.conv5 = conv_2d(filters_list[2], 5)
        self.pool = MaxPool2D(pool_size=3, strides=1, padding="same")

    def call(self, inputs):
        return tf.concat([
            self.conv1(inputs),
            self.conv3(inputs),
            self.conv5(inputs),
            self.pool(inputs),
        ], axis=3)

    def get_config(self):
        config = super(NaiveInceptionModule, self).get_config()
        config.update({
            "filters_list": self.filters_list
        })


class XceptionEntryRU(Layer):

    def __init__(self, filters, start_with_relu=True, **kwargs):
        super(XceptionEntryRU, self).__init__(**kwargs)

        self.filters = filters
        self.start_with_relu = start_with_relu

        separable_conv_2d = partial(
            SeparableConv2D,
            strides=1,
            padding="same",
            use_bias=False
        )

        self.layers = []

        if start_with_relu == True:
            self.layers.append(ReLU())

        self.layers += [
            separable_conv_2d(filters, 3),
            ReLU(),
            BatchNormalization(),
            separable_conv_2d(filters, 3),
            BatchNormalization(),
            MaxPool2D(pool_size=3, strides=2, padding="same")
        ]

        self.skip_layers = [
            Conv2D(filters, 1, strides=2, padding="same", use_bias=False),
            BatchNormalization(),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y = layer(Y)

        skip_Y = inputs
        for layer in self.skip_layers:
            skip_Y = layer(skip_Y)

        return Y + skip_Y

    def get_config(self):
        config = super(XceptionEntryRU, self).get_config()
        config.update({
            "filters": self.filters,
            "start_with_relu": self.start_with_relu
        })
        return config


class XceptionMiddleRU(Layer):

    def __init__(self, filters, **kwargs):
        super(XceptionMiddleRU, self).__init__(**kwargs)

        self.filters = filters

        separable_conv_2d = partial(
            SeparableConv2D,
            strides=1,
            padding="same",
            use_bias=False
        )

        self.layers = [
            ReLU(),
            BatchNormalization(),
            separable_conv_2d(filters, 3),
            ReLU(),
            BatchNormalization(),
            separable_conv_2d(filters, 3),
            ReLU(),
            BatchNormalization(),
            separable_conv_2d(filters, 3),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y = layer(Y)

        return Y + inputs

    def get_config(self):
        config = super(XceptionMiddleRU, self).get_config()
        config.update({
            "filters": self.filters,
        })
        return config


class SEResidualUnit(Layer):
    
    def __init__(self, filters, strides, ratio=16, activation="relu", **kwargs):
        super(SEResidualUnit, self).__init__(**kwargs)

        self.filters = filters
        self.strides = strides
        self.ratio = ratio
        self.activation_ = activation

        self.activation = activations.get(activation)
        
        self.layers = [
            Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            BatchNormalization(),
        ]

        self.skip_layers = []

        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                BatchNormalization(),
            ]

        self.se_layers = [
            GlobalAvgPool2D(),
            Reshape((1, 1, filters)),
            Dense(filters // ratio, activation="relu"),
            Dense(filters, activation="sigmoid"),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y = layer(Y)

        y = Y
        for layer in self.se_layers:
            y = layer(y)

        Y = Y * y 

        skip_Y = inputs
        for layer in self.skip_layers:
            skip_Y = layer(skip_Y)

        return self.activation(Y + skip_Y)

    def get_config(self):
        config = super(SEResidualUnit, self).get_config()
        config.update({
            "filters": self.filters,
            "strides": self.strides,
            "ratio": self.ratio,
            "activation": self.activation_
        })
        return config


class SqueezedXceptionEntryRU(XceptionEntryRU):

    def __init__(self, filters, ratio=16, start_with_relu=True, **kwargs):
        super(SqueezedXceptionEntryRU, self).__init__(filters, start_with_relu, **kwargs)

        self.ratio = ratio

        self.se_layers = [
            GlobalAvgPool2D(),
            Reshape((1, 1, filters)),
            Dense(filters // ratio, activation="relu"),
            Dense(filters, activation="sigmoid"),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y = layer(Y)

        y = Y
        for layer in self.se_layers:
            y = layer(y)

        Y = Y * y 

        skip_Y = inputs
        for layer in self.skip_layers:
            skip_Y = layer(skip_Y)

        return Y + skip_Y

    def get_config(self):
        config = super(SqueezedXceptionEntryRU, self).get_config()
        config.update({
            "ratio": self.ratio
        })


class SqueezedXceptionMiddleRU(XceptionMiddleRU):

    def __init__(self, filters, ratio=16, **kwargs):
        super(SqueezedXceptionMiddleRU, self).__init__(filters, **kwargs)

        self.ratio = ratio

        self.se_layers = [
            GlobalAvgPool2D(),
            Reshape((1, 1, filters)),
            Dense(filters // ratio, activation="relu"),
            Dense(filters, activation="sigmoid"),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y = layer(Y)

        y = Y
        for layer in self.se_layers:
            y = layer(y)

        Y = Y * y 

        return Y + inputs

    def get_config(self):
        config = super(SqueezedXceptionMiddleRU, self).get_config()
        config.update({
            "ratio": self.ratio
        })
        return config
