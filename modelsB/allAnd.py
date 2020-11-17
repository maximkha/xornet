import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers

import sys
sys.path.append(r'C:\Users\maxim\Desktop\js\xornet')
from customLayers.convandlayer import ConvANDLayer
from customLayers.convxorlayer import ConvXORLayer
from customLayers.andlayer import ANDLayer
from customLayers.xorlayer import XORLayer

def GetModel(n=32):
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            ConvANDLayer(n, kernel_size=(3, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            ConvANDLayer(2*n, kernel_size=(3, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            ANDLayer(n),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model