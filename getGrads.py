import libs.importdir
import numpy as np
import functools
import time

import libs.ConfidenceInterval

import tqdm

import tensorflow as tf

#NOTE: to disable gpu:
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

vEnv = dict()
libs.importdir.do(r"C:\Users\maxim\Desktop\js\xornet\modelsB", vEnv)

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

NUM_EPOCH = 16
NUM_EVALS = 5

ALL_LOSS_DATA = dict()
ALL_TIME_DATA = dict()

VIEW = "accuracy" #loss

# CROSS_POINTS = []
def model_train(features, labels):
   # Define the GradientTape context
   with tf.GradientTape() as tape:
       # Get the probabilities
       predictions = model(features)
       # Calculate the loss
       loss = loss_func(labels, predictions)
   # Get the gradients
   gradients = tape.gradient(loss, model.trainable_variables)
   # Update the weights
   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

   # Update the loss and accuracy
   train_loss(loss)
   train_acc(labels, predictions)

for modelName, modelModule in libs.ConfidenceInterval.tqdmProgress(list(vEnv.items())[::-1], False):
    if modelName in ["xornet", "andnet", "allAnd", "beta", "x_2net"]: continue
    print("\nEval Model: " + str(modelName))
    start = time.time()
    multiSampleTrain = np.zeros((NUM_EVALS, NUM_EPOCH))
    multiSampleVal = np.zeros((NUM_EVALS, NUM_EPOCH))
    for i in libs.ConfidenceInterval.tqdmProgress(range(NUM_EVALS), False):
        model = modelModule.GetModel(16)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        historyObj = model.fit(x_train, y_train, epochs=NUM_EPOCH, verbose=1, use_multiprocessing=True, validation_data=(x_test, y_test), batch_size=128)
        
        multiSampleTrain[i] = historyObj.history[VIEW]
        multiSampleVal[i] = historyObj.history['val_' + VIEW]

    end = time.time()
    ALL_TIME_DATA[modelName] = end - start
    #ALL_LOSS_DATA[modelName+"_train"] = [libs.ConfidenceInterval.generateMeanData(multiSampleTrain), libs.ConfidenceInterval.generateConfidenceInterval(multiSampleTrain)]
    ALL_LOSS_DATA[modelName+"_acc"] = [libs.ConfidenceInterval.generateMeanData(multiSampleVal), libs.ConfidenceInterval.generateConfidenceInterval(multiSampleVal)]

import matplotlib.pyplot as plt

plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(ALL_LOSS_DATA.items())))))

for model_name, modelData in ALL_LOSS_DATA.items():
    x = range(len(modelData[0]))
    plt.plot(x, modelData[0], label=model_name)
    plt.fill_between(x, (modelData[0] - modelData[1]), (modelData[0] + modelData[1]), color=plt.gca().lines[-1].get_color(), alpha=.2)

plt.legend()

plt.figure()

SortedTime = list(sorted(ALL_TIME_DATA.items(), key=lambda x: x[1], reverse=True))
plt.xticks(range(len(SortedTime)), rotation=45)
plt.bar([timeEntry[0] for timeEntry in SortedTime], [timeEntry[1] for timeEntry in SortedTime])

plt.show()