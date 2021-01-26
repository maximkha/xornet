import tensorflow as tf
import numpy as np

class X2Layer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, bias=True):
        super(X2Layer, self).__init__()
        self.num_outputs = num_outputs
        self.use_bias = bias

    def build(self, input_shape):
        self.weightA = self.add_weight("weightA", shape=[int(input_shape[-1]), self.num_outputs])#, initializer=tf.random_normal_initializer(0, .16)) #np.sqrt(2. / input_shape[-1])))
        self.bias = None
        if self.use_bias: self.bias = self.add_weight("bias", shape=[int(input_shape[-1])])

    @tf.function
    def call(self, inputs):
        if self.use_bias: biased = inputs + self.bias
        else: biased = inputs

        return (biased @ self.weightA)**2