# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:46:28 2018

@author: Aldo Contreras
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing sklearn tools for prepare data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Importing the dataset
def importData():
    dataset = pd.read_csv('all_27.csv')
    X = dataset.iloc[:, 4:14].values	# flex sensor dataset
    y0 = dataset.iloc[:, 0:1].values		# IMU Quat1
    y1 = dataset.iloc[:, 1:2].values		# IMU Quat2
    y2 = dataset.iloc[:, 2:3].values		# IMU Quat3
    y3 = dataset.iloc[:, 3:4].values		# IMU Quat4
    return(X, y0, y1, y2, y3)

def classifierAnn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    y_train = sc.fit_transform(y_train)
    
    classifier = Sequential()
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 10))
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'tanh'))
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'tanh')) #tanh linear
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['acc'])
    classifier.fit(X_train, y_train, batch_size = 14, epochs = 100)
    # Predicting the Test set results
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
    [X, y0, y1, y2, y3] = importData()
    # y0
    [classifier_y0, y_pred0] = classifierAnn(X, y0)
    # y1
    [classifier_y1, y_pred1] = classifierAnn(X, y1)
    # y2
    [classifier_y2, y_pred2] = classifierAnn(X, y2)
    # y3
    [classifier_y3, y_pred3] = classifierAnn(X, y3)
    # Plot all
    plotData(y0, y1, y2, y3, y_pred0, y_pred1, y_pred2, y_pred3)
    
if __name__ == "__main__":
    main()
    
    