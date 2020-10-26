import tensorflow as tf
import tensorflow.keras as keras

import sys
sys.path.append(r'C:\Users\maxim\Desktop\js\xornet')
from xorlayer import XORLayer

def GetModel(n=10):
    model = keras.Sequential()
    
    # model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(10))
    
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(XORLayer(n))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    #model.build(input_shape = [28, 28])
    return model