# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:03:13 2018

@author: Aldo Contreras
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing sklearn tools for prepare data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Importing the Keras libraries and packages
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# Data set

dataset = pd.read_csv('myData.csv')
X = dataset.iloc[:, 12:22].values	# flex sensor dataset
y0 = dataset.iloc[:, 5:6].values		# IMU Quat
y1 = dataset.iloc[:, 6:7].values		# IMU Quat
y2 = dataset.iloc[:, 7:8].values		# IMU Quat
y3 = dataset.iloc[:, 8:9].values		# IMU Quat

def y_first(y_out):
    X_train, X_test, y_train, y_test = train_test_split(X, y_out, test_size = 0.10, random_state = 0)
    sc = MinMaxScaler(feature_range = (0, 1))
    X_train = sc.fit_transform(X_train)
    y_train = sc.fit_transform(y_train)
    classifierANN(X_train, X_test, y_train, y_test, sc)
    
def classifierANN(X_train, X_test, y_train, y_test, sc):
    classifier = Sequential()
    classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
    # classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'elu'))
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae', 'acc'])
    classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
    # Predicting the Test set results
    y_final = classifier.predict(X_test)
    y_final = sc.inverse_transform(y_final)
    plotValues(y_test, y_final)
    
     
def plotValues(y_test, y_final):
    plt.plot(y_test, color = 'red', label = 'Test')
    plt.plot(y_final, color = 'blue', label = 'Predicted')
    plt.title('Data Prediction')
    plt.xlabel('Sample')
    plt.ylabel('Data')
    plt.legend()
    plt.show() 
 
if __name__ == "__main__":
    y_first(y3)