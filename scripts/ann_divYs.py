# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:46:28 2018

@author: Aldo Contreras
"""

# Importing the libraries
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

# Importing sklearn tools for prepare data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Importing the dataset
import usrInput

def importAndPrepare(X, y):
    y0 = y[:,[0]]
    y1 = y[:,[1]]
    y2 = y[:,[2]]
    try:
        y3 = y[:,[3]]
    except IndexError:
        y3 = 0
    return(X, y0, y1, y2, y3)

def classifierAnn(X, y):
    # [0.2,0.8] or from [0.3,0.7]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
    # sc = StandardScaler()
    sc = MinMaxScaler(feature_range = (0.2, 0.8))
    X_train = sc.fit_transform(X_train)
    y_train = sc.fit_transform(y_train)

    [rowX, colX] = np.shape(X_train)
    colY = 1
    unit = ceil((colX + colY)/2)
    optim, lossT = 'adam', 'mean_squared_error'
    batchS, nEpochs = 32, 100
    classifier = Sequential()
    classifier.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = colX)) # relu relu sigmoid
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = colY, kernel_initializer = 'uniform', activation = 'linear'))          # tanh tanh linear
    classifier.compile(optimizer = optim, loss = lossT, metrics=['mae', 'acc'])
    classifier.fit(X_train, y_train, batch_size = batchS, epochs = nEpochs)
    # Predicting the Test set results
    scores = classifier.evaluate(X_test, y_test)
    print('PARAMETERS --> dense units: {0}, optimizer: {1} loss: {2} batch_size: {3} epochs: {4}'
        .format(unit, optim, lossT, batchS, nEpochs))
    print("{0}: {1} --> {2}: {3} %"
        .format(classifier.metrics_names[1], scores[1], classifier.metrics_names[2], scores[2]*100))
    y_pred = classifier.predict(X)
    return(classifier, y_pred)

def plotData(y0, y1, y2, y3, y_pred0, y_pred1, y_pred2, y_pred3):
    fig, axs = plt.subplots(2, 2)
    axs[0,0].set_title("Q1")
    axs[0,0].plot(y0, color = 'red', label = 'Test')
    axs[0,0].plot(y_pred0, color = 'blue', label = 'Predicted')
    axs[1,0].set_title("Q2")
    axs[1,0].plot(y1, color = 'red', label = 'Test')
    axs[1,0].plot(y_pred1, color = 'blue', label = 'Predicted')
    axs[0,1].set_title("Q3")
    axs[0,1].plot(y2, color = 'red', label = 'Test')
    axs[0,1].plot(y_pred2, color = 'blue', label = 'Predicted')
    axs[1,1].set_title("Q4")
    axs[1,1].plot(y3, color = 'red', label = 'Test')
    axs[1,1].plot(y_pred3, color = 'blue', label = 'Predicted')
    plt.show()

def main():
    [input, target] = usrInput.getDataset()
    [X, y0, y1, y2, y3] = importAndPrepare(input, target)
    [classifier_y0, y_pred0] = classifierAnn(X, y0)                 # y0
    [classifier_y1, y_pred1] = classifierAnn(X, y1)                 # y1
    [classifier_y2, y_pred2] = classifierAnn(X, y2)                 # y2
    if y3 != 0:[classifier_y3, y_pred3] = classifierAnn(X, y3)      # y3
    else: y_pred3 = 0
    plotData(y0, y1, y2, y3, y_pred0, y_pred1, y_pred2, y_pred3)    # Plot all

if __name__ == "__main__":
    main()
