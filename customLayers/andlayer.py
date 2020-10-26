import tensorflow as tf
import numpy as np

class ANDLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, bias=True):
        super(ANDLayer, self).__init__()
        self.num_outputs = num_outputs
        self.use_bias = bias

    def build(self, input_shape):
        self.weightA = self.add_weight("weightA", shape=[int(input_shape[-1]), self.num_outputs])#, initializer=tf.random_normal_initializer(0, .16)) #np.sqrt(2. / input_shape[-1])))
        self.weightB = self.add_weight("weightB", shape=[int(input_shape[-1]), self.num_outputs])#, initializer=tf.random_normal_initializer(0, .16)) #np.sqrt(2. / input_shape[-1])))
        self.bias = None
        if self.use_bias: self.bias = self.add_weight("bias", shape=[int(input_shape[-1])])
        #else: self.bias = self.add_weight("bias", shape=[int(input_shape[-1])], trainable=False, kernel_initializer=tf.keras.initializers.Zeros())

    @tf.function
    def call(self, inputs):
        #if self.use_bias: inputs = inputs + self.bias

        # xVals = tf.matmul(inputs, self.weightA)
        # yVals = tf.matmul(inputs, self.weightB)
        # return xVals * yVals
        #return ((inputs + self.bias) @ self.weightA) * ((inputs + self.bias) @ self.weightB)
        
        if self.use_bias: biased = inputs + self.bias
        else: biased = inputs

        return (biased @ self.weightA) * (biased @ self.weightB)