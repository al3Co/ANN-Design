# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:47:44 2018

@author: Aldo Contreras
"""
# # import os to TensorFlow warnings, doesn't enable AVX/FMA
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

# Importing sklearn tools for prepare data
from sklearn.preprocessing import MinMaxScaler

# Importing the Keras libraries and packages
# import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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

def classifierRnn(X, y):
    from math import ceil
    # Feature Scaling
    training_set = np.append(y, X, axis = 1)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    # training_set_scaled = training_set
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    timeSteps = 2
    for elements in range(len(training_set_scaled[0])):
        for i in range(timeSteps, len(training_set_scaled)):
            X_train.append(training_set_scaled[i-timeSteps:i, elements])
            y_train.append(training_set_scaled[i, elements])

    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # Initialising the RNN
    [rowX, colX, le] = np.shape(X_train)
    #[rowy, coly,] = np.shape(y_train)
    unit = ceil((colX + 1)/2)
    regressor = Sequential()
    regressor.add(LSTM(units = unit, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = unit, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = unit, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = unit))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])
    regressor.fit(X_train, y_train, epochs = 2, batch_size = 32)

    X_test = X_train
    predicted = regressor.predict(X_test)
    scores = regressor.evaluate(X_train, y_train)
    # predicted_stock_price = sc.inverse_transform(predicted)
    return(regressor, predicted)

def plotData(y0, y1, y2, y3, y_pred0, y_pred1, y_pred2, y_pred3):
    fig, axs = plt.subplots(2, 2)
    plt.suptitle('Test -> Red Predicted -> Blue', fontsize=12)
    axs[0,0].set_title('Q1')
    axs[0,0].plot(y0, color = 'red', label = 'Test')
    axs[0,0].plot(y_pred0, color = 'blue', label = 'Predicted')
    axs[1,0].set_title('Q2')
    axs[1,0].plot(y1, color = 'red', label = 'Test')
    axs[1,0].plot(y_pred1, color = 'blue', label = 'Predicted')
    axs[0,1].set_title('Q3')
    axs[0,1].plot(y2, color = 'red', label = 'Test')
    axs[0,1].plot(y_pred2, color = 'blue', label = 'Predicted')
    axs[1,1].set_title('Q4')
    axs[1,1].plot(y3, color = 'red', label = 'Test')
    axs[1,1].plot(y_pred3, color = 'blue', label = 'Predicted')
    plt.show()

def main():
    [input, target] = usrInput.getDataset()
    [X, y0, y1, y2, y3] = importAndPrepare(input, target)
    [regressor_y0, y_pred0] = classifierRnn(X, y0)          # y0
    [regressor_y1, y_pred1] = classifierRnn(X, y1)          # y1
    [regressor_y2, y_pred2] = classifierRnn(X, y2)          # y2
    if y3 != 0:[regressor_y3, y_pred3] = classifierRnn(X, y3)
    else: y_pred3 = 0
    plotData(y0, y1, y2, y3, y_pred0, y_pred1, y_pred2, y_pred3)

if __name__ == "__main__":
    main()
