import tensorflow as tf
import numpy as np

class ConvANDLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding="VALID", kernel_regularizer=None, bias=True):
        super(ConvANDLayer, self).__init__()
        self.use_bias = bias
        self.filters = filters
        self.kernelSize = kernel_size
        self.padding = padding

        self.convA = tf.keras.layers.Conv2D(filters, self.kernelSize, use_bias=False, padding=padding, kernel_regularizer=kernel_regularizer)
        self.convB = tf.keras.layers.Conv2D(filters, self.kernelSize, use_bias=False, padding=padding, kernel_regularizer=kernel_regularizer)
        self.bias = None
        if self.use_bias: self.bias = self.add_weight("bias", shape=[self.filters])

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, inputs):
        if not self.use_bias: return self.convA(inputs) * self.convB(inputs)
        return (self.convA(inputs) + self.bias) * (self.convB(inputs) + self.bias)