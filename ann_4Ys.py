# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:05:10 2018

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

# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import usrInput

# Data Preprocessing
def importAndPrepare(X, y):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
    # Feature Scaling
    #sc = MinMaxScaler(feature_range = (0, 1))
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    y_train = sc.fit_transform(y_train)
    # print(X_test, y_test)
    return(X, y, X_train, X_test, y_train, y_test, sc)
    #    return(X, X, y, y, sc)

# Create the ANN
def createANN(X, X_train, X_test, y_train):
    # size of tests
    [rowX, colX] = np.shape(X_train)
    [row, col] = np.shape(y_train)
    unit = ceil((colX + col)/2)

    classifier = Sequential()
    classifier.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'relu', input_dim = colX)) # relu relu sigmoid
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = unit, kernel_initializer = 'uniform', activation = 'relu'))
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = col, kernel_initializer = 'uniform', activation = 'sigmoid')) #tanh tanh linear
    classifier.compile(optimizer = 'nadam', loss = 'mean_squared_error', metrics=['mae', 'acc'])
    classifier.fit(X_train, y_train, batch_size = 27, epochs = 300)
    # Predicting the Test set results
    y_pred = classifier.predict(X)
    # y_pred = sc.inverse_transform(y_pred)
    # print(y_final)
    return(classifier, y_pred)

# Single prediction function
def singlePrediction(classifier, sc, y_test, y_pred):
	# Feature Scaling
    # TODO: singleObservation = first line of input data
	singleObservation = np.array([[1.54, 2.08, 1.53, 1.7, 1.5, 1.14, 2.29, 1.8, 1.51, 1.81]])
	#singleObservation = sc.transform(singleObservation)
	new_prediction = classifier.predict(singleObservation)
	print(new_prediction)
	# Making the Confusion Matrix
	# cm = confusion_matrix(y_test, y_pred)
	# print accuracy_score(expected, y_test)
	# print classification_report(expected, y_test)
	# print(cm)

def plotData(y_pred, y):
    fig, axs = plt.subplots(2, 2)
    axs[0,0].set_title("Q1")
    axs[0,0].plot(y[:, 0], color = 'red', label = 'Test')
    axs[0,0].plot(y_pred[:, 0], color = 'blue', label = 'Predicted')
    axs[1,0].set_title("Q2")
    axs[1,0].plot(y[:, 1], color = 'red', label = 'Test')
    axs[1,0].plot(y_pred[:, 1], color = 'blue', label = 'Predicted')
    axs[0,1].set_title("Q3")
    axs[0,1].plot(y[:, 2], color = 'red', label = 'Test')
    axs[0,1].plot(y_pred[:, 2], color = 'blue', label = 'Predicted')
    axs[1,1].set_title("Q4")
    axs[1,1].plot(y[:, 3], color = 'red', label = 'Test')
    axs[1,1].plot(y_pred[:, 3], color = 'blue', label = 'Predicted')
    plt.show()

def main():
    [input, target] = usrInput.getDataset()
    X, y, X_train, X_test, y_train, y_test, sc = importAndPrepare(input, target)
    [classifier, y_pred] = createANN(X, X_train, X_test, y_train)
    # singlePrediction(classifier, sc, y_test, y_pred)
    plotData(y_pred, y)

if __name__ == "__main__":
    main()
