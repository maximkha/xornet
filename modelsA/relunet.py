import tensorflow as tf
import tensorflow.keras as keras

def GetModel(n=10):
    model = keras.Sequential()
    
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(n, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    #model.build(input_shape = [28, 28])
    return model