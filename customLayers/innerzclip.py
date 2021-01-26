import tensorflow as tf
import numpy as np

#forwards input values but clips out the middle bound of z scores
class InnerZClip(tf.keras.layers.Layer):
    def __init__(self, _zClip):
        super(InnerZClip, self).__init__()
        self.zClip = _zClip

    @tf.function
    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs)
        std = tf.math.reduce_std(inputs)
        Zs = (inputs - mean) / std
        # for now, I'll use a naive clipping method
        # I suspect that a smooth clip (not just setting invalid values to 0)
        
        #first approach: set clipped values to 0 (not it's z score, if z score that would just be setting it to the mean)
        mask = (tf.math.abs(Zs) < self.zClip)
        floatMask = tf.cast(mask, tf.float32)
        floatNotMask = tf.cast(~mask, tf.float32)
        return (inputs * floatMask) + (floatNotMask * mean)

        # coefMat = tf.math.abs(Zs) - self.zClip
