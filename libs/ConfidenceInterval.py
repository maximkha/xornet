from matplotlib import pyplot as plt
import numpy as np
import tqdm

def tqdmProgress(iterator, progress):
    if progress: return tqdm.tqdm(iterator)
    else: return iterator

def multiSample(f, Xs, numSample, progressBar=False):
    nx = len(Xs)
    fevals = np.zeros((numSample, nx))
    for sample in tqdmProgress(range(numSample), progressBar):
        for ix in range(nx):
            fevals[sample, ix] = f(Xs[ix])
    return fevals

def multiSampleNoX(f, numX, numSample, progressBar=False):
    nx = numX
    fevals = np.zeros((numSample, nx))
    for sample in tqdmProgress(range(numSample), progressBar):
        data = f()
        for ix in range(numX):
            fevals[sample, ix] = data[ix]
    return fevals

def generateConfidenceInterval(multiSampleData, Z = 1.96):
    #means = multiSampleData.mean(axis=1)
    #print(multiSampleData)
    standardDevs = multiSampleData.std(axis=0)
    sqrtN = np.sqrt(multiSampleData.shape[0])
    return Z*(standardDevs/sqrtN)

def generateMeanData(multiSampleData):
    return multiSampleData.mean(axis=0)

# def noisyF(x):
#     return (x ** 2) + np.random.uniform(-10, 10)

# x = np.linspace(0.1, 9.9, 20)

# data = multiSample(noisyF, x, 200)
# y = generateMeanData(data)
# ci = generateConfidenceInterval(data)

# plt.plot(x, y)
# plt.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)
# plt.plot(x, x**2)

# #fig, ax = plt.subplots()
# #ax.plot(x, y)
# #ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)

# #ax.plot(x, x**2)

# plt.show()


#print(generateConfidenceInterval(data))
#print(data.shape)

# #y = [noisyF(ix) for ix in x]
# #some confidence interval
# #ci = 1.96 * np.std(y)/np.mean(y)

# fig, ax = plt.subplots()
# ax.plot(x, y)
# #ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)

# plt.show()