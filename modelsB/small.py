import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers

import sys
sys.path.append(r'C:\Users\maxim\Desktop\js\xornet')
from customLayers.andlayer import ANDLayer
from customLayers.x2layer import X2Layer
from customLayers.innerzclip import InnerZClip

def GetModel(n=32):
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(n, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(2*n, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),

            #X2Layer(n*2),
            #InnerZClip(1.),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model